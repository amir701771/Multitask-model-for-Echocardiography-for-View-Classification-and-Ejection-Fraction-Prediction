
import argparse
import logging
import os
import sys
import time

import json
from multiprocessing import freeze_support
import importlib # For dynamic loading


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.helpers import load_config, setup_logging, set_seed, save_checkpoint, load_checkpoint

from src.data_handling.dataset_echonet import EchoNetViewDataset as EchonetPseudoDataset, video_collate_fn

from src.models.cnn_multitask import MultiTaskCNN # Assuming this is your multitask model class

from src.training.trainer import train_model

from src.utils.metrics import compute_metrics_multitask
from src.utils.visualize import plot_confusion_matrix, plot_ef_scatter

def run_evaluation(model, loader, device, config, split_name="Test"):
    """Runs evaluation on a given dataset loader."""
    model.eval()
    all_view_logits, all_ef_preds, all_view_labels, all_ef_labels = [], [], [], []
    processed_samples = 0

    logging.info(f"--- Starting Evaluation on {split_name} Set ---")
    if not loader:
        logging.warning(f"{split_name} loader is None or empty. Skipping evaluation.")
        return None
    eval_loop = tqdm(loader, desc=f"Evaluating {split_name} Set", leave=False)

    eval_criterion = None
    train_cfg = config.get('training', {})
    try:

        pass
    except Exception as e:
        logging.error(f"Failed to initialize CombinedLoss for evaluation metrics: {e}. Eval loss won't be calculated.", exc_info=True)
        eval_criterion = None

    with torch.no_grad():
        for batch in eval_loop:
            if not isinstance(batch, (list, tuple)) or len(batch) != 3:
                logging.error("Unexpected batch structure in evaluation. Expected (inputs, view_labels, ef_labels).")
                continue
            try:
                inputs, view_labels, ef_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            except Exception as e:
                logging.error(f"Error moving batch to device: {e}. Skipping batch.")
                continue

            try:
                view_logits, ef_preds = model(inputs)

                all_view_logits.append(view_logits.cpu())
                all_ef_preds.append(ef_preds.cpu())
                all_view_labels.append(view_labels.cpu())
                all_ef_labels.append(ef_labels.cpu())
                processed_samples += inputs.size(0)

            except Exception as e:
                 logging.error(f"Error during model forward pass in eval: {e}. Skipping batch.", exc_info=True)
                 continue

    if processed_samples == 0:
        logging.warning(f"{split_name} set evaluation skipped: No samples processed successfully.")
        return None

    eval_metrics = {}
    try:
        final_view_logits = torch.cat(all_view_logits, dim=0) if all_view_logits else None
        final_ef_preds = torch.cat(all_ef_preds, dim=0) if all_ef_preds else None
        final_view_labels = torch.cat(all_view_labels, dim=0) if all_view_labels else None
        final_ef_labels = torch.cat(all_ef_labels, dim=0) if all_ef_labels else None

        eval_metrics = compute_metrics_multitask(
            final_view_logits, final_ef_preds, final_view_labels, final_ef_labels,
            config=config.get('data', {})
        )
    except Exception as e:
        logging.error(f"Error computing metrics for {split_name} set: {e}", exc_info=True)
        eval_metrics = {}

    logging.info(f"--- {split_name} Set Evaluation Results ---")

    if eval_metrics:
        metrics_log = " | ".join([f"{split_name} {k.replace('_',' ').title()}: {v:.4f}" for k, v in eval_metrics.items()])
        logging.info(metrics_log)
    else:
        logging.warning(f"No metrics computed for {split_name} set.")

    results = {

        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer, np.number)) else v for k,v in eval_metrics.items()},
    }

    output_dir = config['training']['output_dir']
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    view_map = config.get('data', {}).get('view_mapping', {})
    view_class_names = [k for k, v in sorted(view_map.items(), key=lambda item: item[1])] if view_map else None

    try:
        can_plot_cm = 'accuracy' in eval_metrics and final_view_labels is not None and final_view_labels.numel() > 0 and final_view_logits is not None and final_view_logits.numel() > 0
        can_plot_scatter = 'mae' in eval_metrics and final_ef_labels is not None and final_ef_labels.numel() > 0 and final_ef_preds is not None and final_ef_preds.numel() > 0

        if can_plot_cm:
             plot_confusion_matrix(final_view_labels, final_view_logits, class_names=view_class_names,
                                   save_path=os.path.join(plots_dir, f"{split_name.lower()}_confusion_matrix.png"))
        if can_plot_scatter:
            plot_ef_scatter(final_ef_labels, final_ef_preds,
                            save_path=os.path.join(plots_dir, f"{split_name.lower()}_ef_scatter.png"))
        if can_plot_cm or can_plot_scatter: logging.info(f"{split_name} set evaluation plots saved in {plots_dir}")
        else: logging.info(f"{split_name} set evaluation plots skipped due to missing metrics or data.")
    except Exception as e:
        logging.error(f"Error generating evaluation plots for {split_name}: {e}", exc_info=True)

    return results

def main():
    parser = argparse.ArgumentParser(description="Train or Evaluate Multi-Task Model (View+EF) on Echonet using Pseudo-Labels.")
    parser.add_argument("--config", type=str, default="config/config_echonet_multitask.yaml", help="Path to config file.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from (e.g., .../last_checkpoint.pth.tar).")
    parser.add_argument("--evaluate", action='store_true', help="Run evaluation only on TEST set using best/specified checkpoint.")
    parser.add_argument("--eval_checkpoint", type=str, default=None, help="Specify checkpoint for --evaluate mode (defaults to best/last).")
    parser.add_argument("--load_view_backbone", type=str, default=None, help="Initialize backbone from a pre-trained view classifier checkpoint.")
    args = parser.parse_args()

    try: config = load_config(args.config)
    except Exception as e: print(f"Error loading config {args.config}: {e}"); sys.exit(1)
    if 'training' not in config or 'model' not in config or 'data' not in config:
        print("Error: Config file must contain 'training', 'model', and 'data' sections."); sys.exit(1)
    train_cfg = config['training']; model_cfg = config['model']; data_cfg = config['data']

    output_dir = train_cfg.get('output_dir', 'results/multitask_default')
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'logs', 'train_multitask.log')
    setup_logging(log_file=log_file, level=logging.INFO)

    mode = "Evaluation" if args.evaluate else "Training"
    logging.info("="*50); logging.info(f"--- {mode} Multi-Task Model ---"); logging.info("="*50)
    logging.info(f"Using config: {args.config}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Echonet pseudo-label data source: {data_cfg.get('csv_path', 'N/A')}")
    if not args.evaluate and args.resume: logging.info(f"Attempting to resume training from: {args.resume}")
    if args.evaluate: logging.info(f"Evaluation Mode Enabled. Checkpoint source: {args.eval_checkpoint or 'Default (best/last)'}")

    set_seed(train_cfg.get('seed', 42))
    device = torch.device(train_cfg.get('device', 'cuda') if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu': logging.warning("CUDA not available, running on CPU. This will be slow.")
    logging.info(f"Using device: {device}")

    logging.info("Loading Echonet datasets...")
    train_loader, val_loader, test_loader = None, None, None
    try:
        num_workers = train_cfg.get('num_workers', 0)
        logging.info(f"Using num_workers = {num_workers} for DataLoaders.")
        batch_size = train_cfg.get('batch_size', 8)

        if not args.evaluate:
            logging.info("Loading TRAIN dataset...")
            train_dataset = EchonetPseudoDataset(config=data_cfg, split='TRAIN')
            if len(train_dataset) == 0: raise RuntimeError("Training dataset is empty after initialization.")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=(device.type=='cuda'),
                                      drop_last=True, collate_fn=video_collate_fn)

            logging.info("Loading VALIDATE dataset...")
            val_dataset = EchonetPseudoDataset(config=data_cfg, split='VALIDATE')
            if len(val_dataset) == 0: raise RuntimeError("Validation dataset is empty after initialization.")
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=(device.type=='cuda'),
                                    collate_fn=video_collate_fn)
            logging.info(f"Echonet Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        logging.info("Loading TEST dataset...")
        test_dataset = EchonetPseudoDataset(config=data_cfg, split='TEST')
        if len(test_dataset) > 0:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=(device.type=='cuda'),
                                     collate_fn=video_collate_fn)
            logging.info(f"Echonet Test samples: {len(test_dataset)}")
        else:
            logging.warning("Test dataset is empty or failed to load. Final evaluation will be skipped.")
            test_loader = None

    except Exception as e:
        logging.error(f"FATAL: Failed loading Echonet data: {e}", exc_info=True)
        sys.exit(1)

    logging.info("Initializing Multi-Task model...")
    model = None
    try:
        num_views = data_cfg.get('num_views', None)
        if num_views is None: raise ValueError("Config missing 'data.num_views'")

        model = MultiTaskCNN(
            backbone_name=model_cfg.get('backbone', 'r2plus1d_18'),
            pretrained=model_cfg.get('pretrained', True),
            num_view_classes=num_views,
            dropout_rate=model_cfg.get('dropout_rate', 0.5)
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters()); trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model: {model.__class__.__name__} | Backbone: {model_cfg.get('backbone')} | Total Params: {total_params:,} | Trainable: {trainable_params:,}")
    except Exception as e: logging.error(f"FATAL: Failed initializing MultiTaskCNN model: {e}", exc_info=True); sys.exit(1)

    if args.load_view_backbone and not args.resume and not args.evaluate:
        logging.info(f"Attempting to initialize backbone weights from: {args.load_view_backbone}")
        if os.path.exists(args.load_view_backbone):
            try:
                view_checkpoint = torch.load(args.load_view_backbone, map_location=device, weights_only=False)
                view_state_dict = view_checkpoint.get('state_dict', view_checkpoint if isinstance(view_checkpoint, dict) else None)
                if view_state_dict:
                    backbone_weights = {k.replace('backbone.', ''): v for k, v in view_state_dict.items() if k.startswith('backbone.')}
                    if backbone_weights:
                        missing, unexpected = model.backbone.load_state_dict(backbone_weights, strict=False)
                        logging.info(f"Loaded view classifier backbone weights. Missing: {missing or 'None'}, Unexpected: {unexpected or 'None'}")
                        if unexpected: logging.warning(f"Unexpected keys found when loading backbone: {unexpected}")
                    else: logging.warning("Could not find keys prefixed with 'backbone.' in the view checkpoint.")
                else: logging.warning("Could not extract state_dict from the view checkpoint file.")
            except Exception as e: logging.error(f"Failed to load backbone weights from {args.load_view_backbone}: {e}", exc_info=True)
        else: logging.warning(f"Specified view backbone checkpoint not found: {args.load_view_backbone}")

    optimizer, scheduler = None, None
    if not args.evaluate:
        logging.info("Initializing optimizer and scheduler...")
        lr = train_cfg.get('learning_rate', 1e-4); wd = train_cfg.get('weight_decay', 1e-5); wd_float=float(wd)
        opt_name = train_cfg.get('optimizer', 'adam').lower()
        try:
            params_to_optimize = model.parameters()
            if opt_name == 'adam': optimizer = optim.Adam(params_to_optimize, lr=lr, weight_decay=wd_float)
            elif opt_name == 'adamw': optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=wd_float)
            elif opt_name == 'sgd': optimizer = optim.SGD(params_to_optimize, lr=lr, momentum=train_cfg.get('momentum', 0.9), weight_decay=wd_float)
            else: raise ValueError(f"Unsupported optimizer: {opt_name}")
            logging.info(f"Using optimizer: {type(optimizer).__name__} (LR={lr}, WD={wd_float})")
        except Exception as e: logging.error(f"FATAL: Optimizer setup failed: {e}", exc_info=True); sys.exit(1)

        scheduler_type = train_cfg.get('scheduler', 'null').lower()
        if scheduler_type != 'null':
            sch_params = train_cfg.get('scheduler_params', {})
            try:
                if scheduler_type == 'steplr':
                    if 'step_size' not in sch_params or 'gamma' not in sch_params: raise ValueError("StepLR requires 'step_size' and 'gamma'")
                    scheduler = optim.lr_scheduler.StepLR(optimizer, **sch_params)
                elif scheduler_type == 'reducelronplateau':
                    sch_params.setdefault('patience', 5); sch_params.setdefault('factor', 0.1)
                    sch_monitor_mode = train_cfg.get('monitor_mode', 'min')
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=sch_monitor_mode, verbose=True, **sch_params)
                elif scheduler_type == 'cosineannealinglr':
                     if 'T_max' not in sch_params: raise ValueError("CosineAnnealingLR requires 'T_max'")
                     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **sch_params)
                else: logging.warning(f"Unsupported scheduler type '{scheduler_type}'. No scheduler used.")
                if scheduler: logging.info(f"Using scheduler: {type(scheduler).__name__} with params: {sch_params}")
            except Exception as e: logging.error(f"Scheduler initialization failed: {e}. Continuing without scheduler."); scheduler = None

    start_epoch = 0
    metric_to_monitor = train_cfg.get('monitor_metric', 'val_mae')
    monitor_mode = train_cfg.get('monitor_mode', 'min')
    best_val_metric = float('inf') if monitor_mode == 'min' else float('-inf')
    logging.debug(f"Initialized: start_epoch={start_epoch}, monitor='{metric_to_monitor}', mode='{monitor_mode}', best_metric={best_val_metric}")

    checkpoint_to_load = None
    default_best_model_path = os.path.join(checkpoint_dir, "best_model.pth.tar")
    default_last_model_path = os.path.join(checkpoint_dir, "last_checkpoint.pth.tar")

    if args.evaluate:
        checkpoint_to_load = args.eval_checkpoint or default_best_model_path
        if not os.path.exists(checkpoint_to_load): checkpoint_to_load = default_last_model_path
        if not os.path.exists(checkpoint_to_load): logging.error(f"Evaluation mode: Checkpoint not found in {checkpoint_dir}. Exiting."); sys.exit(1)
        logging.info(f"Evaluation mode: Loading checkpoint '{checkpoint_to_load}'")
    elif args.resume:
        checkpoint_to_load = args.resume
        if not os.path.exists(checkpoint_to_load):
             logging.warning(f"Resume checkpoint not found: '{checkpoint_to_load}'. Checking last checkpoint.")
             checkpoint_to_load = default_last_model_path
             if not os.path.exists(checkpoint_to_load): logging.warning(f"Last checkpoint not found. Starting fresh."); checkpoint_to_load = None
             else: logging.info(f"Resuming from last checkpoint instead: '{checkpoint_to_load}'")
        else: logging.info(f"Training mode: Attempting resume from specified checkpoint: '{checkpoint_to_load}'")

    if checkpoint_to_load:
        load_optimizer = optimizer if args.resume else None
        load_scheduler = scheduler if args.resume else None
        checkpoint = load_checkpoint(checkpoint_to_load, model, optimizer=load_optimizer, scheduler=load_scheduler, device=device)
        if checkpoint and args.resume:
             try:
                 start_epoch = checkpoint.get('epoch', -1) + 1
                 saved_metric = checkpoint.get('monitored_metric', 'N/A'); saved_mode = checkpoint.get('monitor_mode', 'N/A'); saved_best_val = checkpoint.get('best_val_metric_value', None)
                 if saved_best_val is not None:
                     if saved_metric == metric_to_monitor and saved_mode == monitor_mode: best_val_metric = saved_best_val; logging.info(f"Restored best metric '{metric_to_monitor}': {best_val_metric:.6f}")
                     else: logging.warning(f"Checkpoint monitor mismatch. Resetting best metric tracking."); best_val_metric = float('inf') if monitor_mode == 'min' else float('-inf')
                 else: logging.warning("No best metric found in checkpoint. Resetting."); best_val_metric = float('inf') if monitor_mode == 'min' else float('-inf')
                 logging.info(f"Successfully resumed training from epoch {start_epoch}.")
             except Exception as e: logging.error(f"Error processing checkpoint data: {e}. Starting fresh.", exc_info=True); start_epoch = 0; best_val_metric = float('inf') if monitor_mode == 'min' else float('-inf')
        elif not checkpoint:
            logging.error(f"Failed loading checkpoint content: {checkpoint_to_load}");
            if args.evaluate: sys.exit(1)
            else: logging.warning("Starting fresh training."); start_epoch = 0; best_val_metric = float('inf') if monitor_mode == 'min' else float('-inf')
    elif not args.evaluate:
        logging.info("No resume checkpoint specified or found. Starting fresh training run.")
        start_epoch = 0; best_val_metric = float('inf') if monitor_mode == 'min' else float('-inf')

    if not args.evaluate:
        if not train_loader or not val_loader or model is None or optimizer is None:
            logging.error("FATAL: Setup incomplete (loaders/model/optimizer is None). Cannot start training."); sys.exit(1)

        logging.info(f"--- Starting Multi-Task Training (Epochs: {start_epoch+1} to {train_cfg.get('epochs', 50)}) ---")
        logging.info(f"Monitoring Metric: '{metric_to_monitor}' (Mode: {monitor_mode}, Initial Best: {best_val_metric})")

        history = None
        try:

            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,

                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=train_cfg.get('epochs', 50),
                device=device,
                config=config,
                checkpoint_dir=checkpoint_dir,
                start_epoch=start_epoch,
                best_val_metric=best_val_metric,
                metric_to_monitor=metric_to_monitor,
                monitor_mode=monitor_mode
            )

            logging.info("--- Multi-Task Training Finished ---")

            if history is not None and isinstance(history, dict):
                history_path = os.path.join(output_dir, "multitask_training_history.json")

                try:
                    serializable_history = {}
                    for key, value_list in history.items():
                        if isinstance(value_list, (list, tuple, np.ndarray)):
                            serializable_history[key] = [ v.item() if hasattr(v, 'item') else v for v in value_list ]
                        else: serializable_history[key] = value_list.item() if hasattr(value_list, 'item') else value_list
                    with open(history_path, 'w') as f: json.dump(serializable_history, f, indent=4)
                    logging.info(f"Training history saved to {history_path}")
                except Exception as e: logging.error(f"Could not save training history: {e}", exc_info=True)

            else: logging.warning("train_model did not return a valid history dictionary.")


            if os.path.exists(default_best_model_path):
                logging.info(f"Loading best model from {default_best_model_path} for final test evaluation.")
                load_checkpoint(default_best_model_path, model, device=device)
            else: logging.warning(f"Best model checkpoint not found after training. Evaluating using the last model state.")

        except KeyboardInterrupt: logging.warning("--- Training Interrupted By User ---"); sys.exit(0)
        except Exception as e: logging.error(f"Training loop failed critically: {e}", exc_info=True); sys.exit(1)

    if test_loader:
        model.eval()
        logging.info("Proceeding to evaluate final model on Test Set...")

        test_results = run_evaluation(
                model=model,
                loader=test_loader,

                device=device,
                config=config,
                split_name="Test"
        )


        if test_results:
            results_path = os.path.join(output_dir, "multitask_test_results.json")

            try:
                 serializable_results = {}
                 for k, v in test_results.items():
                      if k == 'metrics' and isinstance(v, dict):
                           serializable_results['metrics'] = {mk: float(mv) if isinstance(mv, (np.floating, np.number)) else mv for mk, mv in v.items()}
                      elif isinstance(v, (np.floating, np.number)):
                           serializable_results[k] = float(v)
                      else:
                           serializable_results[k] = v
                 with open(results_path, 'w') as f: json.dump(serializable_results, f, indent=4)
                 logging.info(f"Test evaluation summary saved to {results_path}")
            except Exception as e: logging.error(f"Could not save test results: {e}", exc_info=True)

    else:
        logging.info("Skipping evaluation on test set as test loader was not available or empty.")

    logging.info("--- Script Finished ---")


if __name__ == "__main__":
    freeze_support()
    main()