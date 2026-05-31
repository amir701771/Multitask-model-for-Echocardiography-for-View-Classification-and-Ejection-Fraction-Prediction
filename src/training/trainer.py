import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os
import time
import numpy as np
import collections

from src.utils.metrics import compute_metrics_multitask
from src.utils.helpers import save_checkpoint
from src.utils.visualize import plot_losses, plot_metrics

def train_one_epoch(model, loader, view_criterion, ef_criterion, optimizer, device, epoch_num, num_epochs, alpha, beta):
    """Runs a single training epoch with separate loss calculations."""
    model.train()
    running_loss = 0.0
    running_loss_c = 0.0
    running_loss_r = 0.0
    processed_samples = 0
    logging.info(f"[Trainer] Starting train_one_epoch {epoch_num}")

    train_loop = tqdm(loader, desc=f"Epoch {epoch_num}/{num_epochs} [Train]", leave=False)

    for batch_idx, batch_data in enumerate(train_loop):
        logging.debug(f"[Trainer] Train Batch {batch_idx}: Loading data...")

        if len(batch_data) != 3:
             logging.error(f"[Trainer] Expected 3 items in batch (inputs, view_labels, ef_labels), got {len(batch_data)}. Skipping batch {batch_idx}.")
             continue
        inputs, view_labels, ef_labels = batch_data

        if not isinstance(inputs, torch.Tensor):
            logging.warning(f"[Trainer] Train Batch {batch_idx} Received non-tensor input: {type(inputs)}")
            continue

        try:
            inputs = inputs.to(device)
            view_labels = view_labels.to(device)
            ef_labels = ef_labels.to(device)
        except Exception as e:
             logging.error(f"[Trainer] Error moving train batch {batch_idx} to device: {e}", exc_info=True)
             continue

        optimizer.zero_grad()

        try:
            logging.debug(f"[Trainer] Train Batch {batch_idx}: Starting model forward pass...")

            model_output = model(inputs)
            if not (isinstance(model_output, tuple) and len(model_output) == 2):
                 logging.error(f"[Trainer] Expected model output tuple (view_logits, ef_preds), got {type(model_output)}. Skipping batch {batch_idx}.")
                 continue
            view_logits, ef_preds = model_output
            logging.debug(f"[Trainer] Train Batch {batch_idx}: Model forward pass complete.")

            loss_c = view_criterion(view_logits, view_labels)
            
            # MASKED EF LOSS (Only train on A4C samples, Class 1)
            ef_preds = ef_preds.squeeze(-1)
            ef_mask = (view_labels == 1)
            
            if ef_mask.sum() > 0:
                 loss_r = ef_criterion(ef_preds[ef_mask], ef_labels[ef_mask].float())
            else:
                 loss_r = torch.tensor(0.0, device=device)

            loss = alpha * loss_c + beta * loss_r
            
            logging.debug(f"[Trainer] Train Batch {batch_idx}: Losses calculated: Total={loss.item():.4f}, C={loss_c.item():.4f}, R={loss_r.item():.4f} (A4C count: {ef_mask.sum()}). Starting backward...")

            loss.backward()
            logging.debug(f"[Trainer] Train Batch {batch_idx}: Backward pass complete. Starting optimizer step...")
            optimizer.step()
            logging.debug(f"[Trainer] Train Batch {batch_idx}: Optimizer step complete.")

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_loss_c += loss_c.item() * batch_size
            running_loss_r += loss_r.item() * batch_size
            processed_samples += batch_size

            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        except Exception as e:
            logging.error(f"[Trainer] Error during training batch {batch_idx} (model/loss/optim phase): {e}", exc_info=True)

    logging.info(f"[Trainer] Finished train_one_epoch {epoch_num}")

    epoch_loss = running_loss / processed_samples if processed_samples > 0 else 0.0
    epoch_loss_c = running_loss_c / processed_samples if processed_samples > 0 else 0.0
    epoch_loss_r = running_loss_r / processed_samples if processed_samples > 0 else 0.0

    return epoch_loss, epoch_loss_c, epoch_loss_r

def validate_one_epoch(model, loader, view_criterion, ef_criterion, device, epoch_num, num_epochs, alpha, beta, config={}):
    """Runs a single validation epoch with separate loss calculations."""
    model.eval()
    running_loss = 0.0
    running_loss_c = 0.0 # Store classification component
    running_loss_r = 0.0 # Store regression component
    processed_samples = 0
    logging.info(f"[Trainer] Starting validate_one_epoch {epoch_num}")

    all_view_logits_list = []
    all_ef_preds_list = []
    all_view_labels_list = []
    all_ef_labels_list = []

    val_loop = tqdm(loader, desc=f"Epoch {epoch_num}/{num_epochs} [Val]", leave=False)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loop):
            logging.debug(f"[Trainer] Val Batch {batch_idx}: Loading data...")

            if len(batch_data) != 3:
                 logging.error(f"[Trainer] Expected 3 items in val batch, got {len(batch_data)}. Skipping batch {batch_idx}.")
                 continue
            inputs, view_labels, ef_labels = batch_data

            if not isinstance(inputs, torch.Tensor):
                logging.warning(f"[Trainer] DEBUG VAL Batch {batch_idx} Received non-tensor input: {type(inputs)}")
                continue

            try:
                inputs = inputs.to(device)
                view_labels = view_labels.to(device)
                ef_labels = ef_labels.to(device)
            except Exception as e:
                 logging.error(f"[Trainer] Error moving validation batch {batch_idx} to device: {e}", exc_info=True)
                 continue

            try:
                logging.debug(f"[Trainer] Val Batch {batch_idx}: Starting model forward pass...")

                model_output = model(inputs)
                if not (isinstance(model_output, tuple) and len(model_output) == 2):
                     logging.error(f"[Trainer] Expected model output tuple (view_logits, ef_preds), got {type(model_output)}. Skipping batch {batch_idx}.")

                     if view_labels is not None: all_view_labels_list.append(view_labels.cpu())
                     if ef_labels is not None: all_ef_labels_list.append(ef_labels.cpu())
                     continue
                view_logits, ef_preds = model_output
                logging.debug(f"[Trainer] Val Batch {batch_idx}: Model forward pass complete.")

                loss_c = view_criterion(view_logits, view_labels)
                
                # MASKED EF LOSS (Validation on A4C samples only)
                ef_preds = ef_preds.squeeze(-1)
                ef_mask = (view_labels == 1)
                
                if ef_mask.sum() > 0:
                     loss_r = ef_criterion(ef_preds[ef_mask], ef_labels[ef_mask].float())
                else:
                     loss_r = torch.tensor(0.0, device=device)

                loss = alpha * loss_c + beta * loss_r
                logging.debug(f"[Trainer] Val Batch {batch_idx}: Losses calculated: Total={loss.item():.4f}, C={loss_c.item():.4f}, R={loss_r.item():.4f}.")

                batch_size = inputs.size(0)
                if not torch.isnan(loss):
                    running_loss += loss.item() * batch_size
                    running_loss_c += loss_c.item() * batch_size
                    running_loss_r += loss_r.item() * batch_size
                    processed_samples += batch_size

                if view_logits is not None: all_view_logits_list.append(view_logits.cpu())
                if view_labels is not None: all_view_labels_list.append(view_labels.cpu())
                if ef_preds is not None: all_ef_preds_list.append(ef_preds.cpu())
                if ef_labels is not None: all_ef_labels_list.append(ef_labels.cpu())

                if not torch.isnan(loss): val_loop.set_postfix(loss=f"{loss.item():.4f}")
                else: val_loop.set_postfix(loss=f"NaN")

            except Exception as e:
                 logging.error(f"[Trainer] Error during validation batch {batch_idx} (model/loss phase): {e}", exc_info=True)

    logging.info(f"[Trainer] Finished validate_one_epoch {epoch_num}")
    if processed_samples == 0:
        logging.warning("[Trainer] Validation epoch completed with 0 successfully processed samples.")
        default_metrics = {'accuracy': 0.0, 'f1_weighted': 0.0, 'kappa': 0.0, 'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'r2': -float('inf')}
        return 0.0, 0.0, 0.0, default_metrics

    epoch_loss = running_loss / processed_samples
    epoch_loss_c = running_loss_c / processed_samples
    epoch_loss_r = running_loss_r / processed_samples

    final_view_logits = torch.cat(all_view_logits_list, dim=0) if all_view_logits_list else None
    final_view_labels = torch.cat(all_view_labels_list, dim=0) if all_view_labels_list else None
    final_ef_preds = torch.cat(all_ef_preds_list, dim=0) if all_ef_preds_list else None
    final_ef_labels = torch.cat(all_ef_labels_list, dim=0) if all_ef_labels_list else None

    logging.info(f"[Trainer] Calculating metrics for epoch {epoch_num}...")
    epoch_metrics = compute_metrics_multitask( view_logits=final_view_logits, ef_preds=final_ef_preds, view_labels=final_view_labels,
        ef_labels=final_ef_labels, config=config.get('data', {}))
    logging.info(f"[Trainer] Metrics calculation complete for epoch {epoch_num}.")

    return epoch_loss, epoch_loss_c, epoch_loss_r, epoch_metrics

def train_model(model, train_loader, val_loader,
                optimizer, scheduler, num_epochs, device, config,
                checkpoint_dir="results", start_epoch=0,
                best_val_metric=float('inf'), metric_to_monitor='val_loss',
                monitor_mode='min'):

    history = collections.defaultdict(list)
    train_cfg = config.get('training', {})
    output_dir = train_cfg.get('output_dir', 'results/default_run')
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    vis_cfg = config.get('visualization', {})
    plot_freq = vis_cfg.get('plot_freq', 5)

    alpha = train_cfg.get('loss_alpha', 0.5)
    beta = train_cfg.get('loss_beta', 0.5)
    view_weights_list = train_cfg.get('view_loss_class_weights', None)

    view_weights_tensor = None
    if view_weights_list:
        try:
            view_weights_tensor = torch.tensor(view_weights_list, dtype=torch.float32).to(device)
            logging.info(f"[Trainer] Using View Class Weights: {view_weights_tensor.cpu().numpy()}")
        except Exception as e:
            logging.error(f"[Trainer] Failed to create view weights tensor from config: {view_weights_list}. Error: {e}. Using unweighted loss.", exc_info=True)
            view_weights_tensor = None
    else:
        logging.info("[Trainer] No 'view_loss_class_weights' found in config. Using unweighted view loss.")

    view_criterion = nn.CrossEntropyLoss(weight=view_weights_tensor).to(device) # Pass weights tensor (or None)

    ef_criterion = nn.MSELoss().to(device)
    logging.info(f"[Trainer] Using View Loss: {type(view_criterion).__name__} | EF Loss: {type(ef_criterion).__name__}")
    logging.info(f"[Trainer] Loss Combination Weights: Alpha (View)={alpha}, Beta (EF)={beta}")

    logging.info(f"[Trainer] Inside train_model: Starting training from epoch {start_epoch + 1} to {num_epochs} on {device}.")
    logging.info(f"[Trainer] Inside train_model: Monitoring validation metric: '{metric_to_monitor}' (mode: {monitor_mode})")
    logging.info(f"[Trainer] Inside train_model: Initial best validation metric value: {best_val_metric:.6f}")

    if not train_loader or not val_loader:
        logging.error("[Trainer] Training or validation loader is None. Cannot train.")
        return dict(history)

    try:
        for epoch in range(start_epoch, num_epochs):
            epoch_num = epoch + 1
            start_time = time.time()
            logging.info(f"[Trainer] Starting Epoch {epoch_num}/{num_epochs}")

            logging.debug(f"[Trainer] Epoch {epoch_num}: Calling train_one_epoch...")

            train_loss, train_loss_c, train_loss_r = train_one_epoch(
                model, train_loader, view_criterion, ef_criterion, optimizer, device, epoch_num, num_epochs, alpha, beta
            )
            history['train_loss'].append(train_loss); history['train_loss_c'].append(train_loss_c); history['train_loss_r'].append(train_loss_r)
            logging.debug(f"[Trainer] Epoch {epoch_num}: train_one_epoch finished.")

            logging.debug(f"[Trainer] Epoch {epoch_num}: Calling validate_one_epoch...")

            val_loss, val_loss_c, val_loss_r, val_metrics = validate_one_epoch(
                model, val_loader, view_criterion, ef_criterion, device, epoch_num, num_epochs, alpha, beta, config=config
            )
            history['val_loss'].append(val_loss); history['val_loss_c'].append(val_loss_c); history['val_loss_r'].append(val_loss_r)
            logging.debug(f"[Trainer] Epoch {epoch_num}: validate_one_epoch finished.")

            for metric_name, metric_value in val_metrics.items():
                history[f'val_{metric_name}'].append(metric_value)

            current_val_metric = history.get(metric_to_monitor, [None])[-1]
            if current_val_metric is None:
                 logging.warning(f"[Trainer] Could not retrieve value for monitored metric '{metric_to_monitor}' at epoch {epoch_num}.")

            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            if scheduler:
                logging.debug(f"[Trainer] Epoch {epoch_num}: Stepping scheduler...")
                try:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                         if current_val_metric is not None: scheduler.step(current_val_metric)
                         else: logging.warning(f"[Trainer] Scheduler ReduceLROnPlateau step skipped: Monitored metric '{metric_to_monitor}' value not available.")
                    else: scheduler.step()
                except Exception as e: logging.error(f"[Trainer] Error during scheduler step at epoch {epoch_num}: {e}", exc_info=True)
                logging.debug(f"[Trainer] Epoch {epoch_num}: Scheduler step finished.")

            epoch_time = time.time() - start_time
            log_msg = (f"Epoch {epoch_num}/{num_epochs} | Time: {epoch_time:.2f}s | LR: {current_lr:.6e} | "
                       f"Train Loss: {train_loss:.4f} (C:{train_loss_c:.4f}, R:{train_loss_r:.4f}) | "
                       f"Val Loss: {val_loss:.4f} (C:{val_loss_c:.4f}, R:{val_loss_r:.4f})")
            metrics_log = " | ".join([f"Val {k.replace('_', ' ').title()}: {v:.4f}" for k, v in val_metrics.items()])
            logging.info(log_msg + " | " + metrics_log)

            is_best = False
            if current_val_metric is not None:
                if monitor_mode == 'min' and current_val_metric < best_val_metric: is_best = True; best_val_metric = current_val_metric
                elif monitor_mode == 'max' and current_val_metric > best_val_metric: is_best = True; best_val_metric = current_val_metric

            if is_best: logging.info(f"[Trainer] Epoch {epoch_num}: Found new best {metric_to_monitor} ({best_val_metric:.6f}). Saving best model.")

            checkpoint_state = { 'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None, 'best_val_metric_value': best_val_metric,
                'monitored_metric': metric_to_monitor, 'monitor_mode': monitor_mode, 'config': config }

            last_checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth.tar")
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth.tar")
            save_checkpoint(state=checkpoint_state, is_best=is_best, filename=last_checkpoint_path, best_filename=best_checkpoint_path)

            if history['train_loss'] and (epoch_num % plot_freq == 0 or epoch_num == num_epochs):
                try:
                    logging.info(f"[Trainer] Epoch {epoch_num}: Generating and saving plots...")
                    plot_losses(history, save_path=os.path.join(plots_dir, f"loss_curves_epoch_{epoch_num}.png"))
                    metrics_to_plot = sorted([k for k in history if k.startswith('val_') and 'loss' not in k])
                    if metrics_to_plot: plot_metrics(history, metrics=metrics_to_plot, save_path=os.path.join(plots_dir, f"metrics_curves_epoch_{epoch_num}.png"))
                except Exception as e: logging.error(f"[Trainer] Error generating plots at epoch {epoch_num}: {e}", exc_info=True)

            logging.info(f"[Trainer] Finished Epoch {epoch_num}/{num_epochs}")

    except Exception as e_outer:
        logging.error(f"[Trainer] CRITICAL FAILURE inside train_model loop: {e_outer}", exc_info=True)
        logging.error("[Trainer] Returning history collected before the critical failure.")
        return dict(history)

    logging.info(f"[Trainer] Training finished after {num_epochs} epochs.")
    logging.info(f"[Trainer] Best validation '{metric_to_monitor}' achieved: {best_val_metric:.6f}")

    return dict(history)