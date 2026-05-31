import logging
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import traceback

from src.utils.helpers import load_config, setup_logging, set_seed, save_checkpoint, load_checkpoint
from src.data_handling.dataset_echonet import EchoNetViewDataset as EchonetPseudoDataset, video_collate_fn
from src.models.cnn_multitask import MultiTaskCNN
from src.training.trainer import train_model

def run_training_pipeline(config_path, status_callback=None, stop_flag_callback=None):
    """
    Runs the full training pipeline with status updates via callback.
    
    Args:
        config_path (str): Path to YAML config.
        status_callback (callable): Function(status_dict) to update global state.
        stop_flag_callback (callable): Function() that returns True if training should stop.
    """
    try:
        if status_callback:
            status_callback({"status": "initializing", "message": "Loading configuration..."})

        # 1. Load Config
        try:
            config = load_config(config_path)
            train_cfg = config['training']
            model_cfg = config['model']
            data_cfg = config['data']
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")

        # Setup Logging just for file output (stdout is handled by app)
        output_dir = train_cfg.get('output_dir', 'results/multitask_ui_run')
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
        setup_logging(log_file=os.path.join(output_dir, 'logs', 'train_ui.log'), level=logging.INFO)

        # 2. Setup Device & Seed
        set_seed(train_cfg.get('seed', 42))
        device = torch.device(train_cfg.get('device', 'cuda') if torch.cuda.is_available() else "cpu")
        
        if status_callback:
            status_callback({"status": "initializing", "message": f"Device: {device}. Loading Data..."})

        # 3. Load Datasets
        try:
            num_workers = train_cfg.get('num_workers', 0)
            batch_size = train_cfg.get('batch_size', 8)

            train_dataset = EchonetPseudoDataset(config=data_cfg, split='TRAIN')
            if len(train_dataset) == 0: raise RuntimeError("Training dataset is empty.")
            
            val_dataset = EchonetPseudoDataset(config=data_cfg, split='VALIDATE')
            if len(val_dataset) == 0: raise RuntimeError("Validation dataset is empty.")

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=(device.type=='cuda'),
                                      collate_fn=video_collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=(device.type=='cuda'),
                                    collate_fn=video_collate_fn)
        except FileNotFoundError as fnf:
            raise RuntimeError(f"Dataset file not found: {fnf}. Please check if 'FileList_camus_pseudolabels.csv' exists.")
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {e}")

        # 4. Initialize Model
        if status_callback:
            status_callback({"status": "initializing", "message": "Initializing Model..."})

        model = MultiTaskCNN(
            backbone_name=model_cfg.get('backbone', 'r2plus1d_18'),
            pretrained=model_cfg.get('pretrained', True),
            num_view_classes=data_cfg.get('num_views', 2),
            dropout_rate=model_cfg.get('dropout_rate', 0.5)
        ).to(device)

        # 5. Optimizer
        optimizer = optim.Adam(model.parameters(), lr=train_cfg.get('learning_rate', 1e-4))
        
        # 6. Training Loop Wrapper
        # We wrap the train_model call or reimplement the loop to provide granular updates?
        # The existing train_model does NOT accept a callback.
        # To strictly follow the requirement of "SHOW OUTPUT STATUS" live, we need to inject checking.
        # However, modifying the shared `src/training/trainer.py` is risky if other scripts depend on it.
        # But wait, `train_model` is just a function. I can copy the loop logic OR update `train_model` 
        # to accept an optional callback. Updating `train_model` is cleaner.
        
        # NOTE: For now, I will use `train_model` as is, but since it assumes console logging,
        # I won't get per-batch updates in the UI unless I modify it.
        # The user wants "Current epoch, Loss & accuracy updating live".
        # So I MUST modify `train_model` or reimplement the loop here.
        # I will reimplement a simplified loop here to ensure UI compatibility without breaking legacy.
        
        num_epochs = train_cfg.get('epochs', 5) # Default to 5 for UI safety
        alpha = train_cfg.get('loss_alpha', 0.5)
        beta = train_cfg.get('loss_beta', 0.5)
        
        view_criterion = torch.nn.CrossEntropyLoss().to(device)
        ef_criterion = torch.nn.MSELoss().to(device)

        checkpoint_dir = os.path.join(train_cfg.get('output_dir', 'results'), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth.tar')
        
        start_epoch = 0
        best_val_acc = 0.0
        min_delta = train_cfg.get('min_delta', 1e-4)
        resume_training = train_cfg.get('resume_training', True)
        
        # Initialize scaler for mixed precision if CUDA is available
        scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get('use_amp', False)) if torch.cuda.is_available() else None
        
        if resume_training and os.path.exists(latest_checkpoint_path):
            if status_callback:
                status_callback({"status": "initializing", "message": "Resuming from checkpoint..."})
            checkpoint = load_checkpoint(latest_checkpoint_path, model, optimizer, scaler=scaler, device=device)
            if checkpoint:
                start_epoch = checkpoint.get('epoch', -1) + 1
                best_val_acc = checkpoint.get('best_val_acc', 0.0)
                logging.info(f"Resuming training from epoch {start_epoch + 1} with best_val_acc {best_val_acc:.4f}")

        if status_callback:
            status_callback({
                "status": "training", 
                "total_epochs": num_epochs, 
                "message": f"Starting Training from Epoch {start_epoch+1}..."
            })

        save_freq = train_cfg.get('save_freq', 10)  # Periodic checkpoint frequency

        for epoch in range(start_epoch, num_epochs):
            # Train Phase
            model.train()
            running_loss = 0.0
            processed = 0
            
            if status_callback:
                status_callback({
                    "status": "training", 
                    "epoch": epoch + 1, 
                    "total_epochs": num_epochs, 
                    "message": f"Training Epoch {epoch+1}..."
                })
            
            # Check stop flag at start of each epoch
            if stop_flag_callback and stop_flag_callback():
                logging.info("Training stop requested at start of epoch. Exiting. Resumption will use the last completed epoch.")
                if status_callback:
                    status_callback({"status": "stopped", "message": f"Training stopped before epoch {epoch+1}"})
                return

            for batch_idx, (inputs, view_labels, ef_labels) in enumerate(train_loader):
                inputs, view_labels, ef_labels = inputs.to(device), view_labels.to(device), ef_labels.to(device)
                
                optimizer.zero_grad()
                v_logits, ef_preds = model(inputs)
                
                loss_c = view_criterion(v_logits, view_labels)
                ef_preds = ef_preds.squeeze(-1)
                ef_mask = (view_labels == 1)
                loss_r = ef_criterion(ef_preds[ef_mask], ef_labels[ef_mask].float()) if ef_mask.sum() > 0 else torch.tensor(0.0).to(device)
                
                loss = alpha * loss_c + beta * loss_r
                loss.backward()
                optimizer.step()
                
                # Check stop flag immediately after batch step
                if stop_flag_callback and stop_flag_callback():
                    logging.info(f"Training stop requested during batch {batch_idx}. Exiting. Resumption will discard this partial epoch.")
                    if status_callback:
                        status_callback({"status": "stopped", "message": f"Training stopped at epoch {epoch+1}, batch {batch_idx}"})
                    return
                
                running_loss += loss.item() * inputs.size(0)
                processed += inputs.size(0)
                
                # Update UI every N batches to avoid flooding
                if batch_idx % 5 == 0 and status_callback:
                     current_avg_loss = running_loss / processed
                     status_callback({
                        "status": "training",
                        "epoch": epoch + 1,
                        "loss": round(current_avg_loss, 4),
                        "message": f"Training Batch {batch_idx}/{len(train_loader)}"
                     })

            # Validation Phase (Simplified)
            model.eval()
            val_loss = 0.0
            correct = 0
            total_val = 0
            
            if status_callback:
                 status_callback({"message": "Validating..."})

            with torch.no_grad():
                for inputs, view_labels, ef_labels in val_loader:
                     inputs, view_labels = inputs.to(device), view_labels.to(device)
                     v_logits, _ = model(inputs)
                     _, preds = torch.max(v_logits, 1)
                     correct += (preds == view_labels).sum().item()
                     total_val += inputs.size(0)
            
            val_acc = correct / total_val if total_val > 0 else 0.0
            
            if status_callback:
                status_callback({
                    "status": "training",
                    "accuracy": round(val_acc, 4),
                    "loss": round(running_loss / processed, 4)
                })

            # Checkpointing at end of epoch
            is_best = val_acc > (best_val_acc + min_delta)
            if is_best:
                best_val_acc = val_acc
            
            save_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'val_loss': running_loss / processed
            }
            if scaler:
                save_state['scaler'] = scaler.state_dict()
                
            save_checkpoint(save_state, is_best=is_best, filename=latest_checkpoint_path, 
               best_filename=os.path.join(checkpoint_dir, 'best_model.pth.tar'))
            
            # Periodic checkpoints
            if (epoch + 1) % save_freq == 0:
                periodic_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth.tar')
                save_checkpoint(save_state, is_best=False, filename=periodic_path)

        if status_callback:
            status_callback({"status": "completed", "message": "Training Successfully Completed"})
            
    except KeyboardInterrupt:
        logging.warning("Training interrupted by KeyboardInterrupt (Ctrl+C). Exiting without saving partial epoch.")
        if status_callback:
            status_callback({"status": "stopped", "message": "Training stopped via KeyboardInterrupt"})
    except Exception as e:
        logging.error(f"Training Pipeline Failed: {e}", exc_info=True)
        if status_callback:
            status_callback({"status": "failed", "message": f"Error: {str(e)}"})

