
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import copy
import time
import os
import numpy as np
import sys

from src.data_handling.dataset_echonet import EchonetPseudoDataset

from src.models.cnn_multitask import MultiTaskCNN

from src.training.losses import CombinedLoss
from src.training.trainer import train_model
from src.utils.helpers import set_seed

def objective_function_multitask(hyperparameters, base_config, run_id="ea_run"):

    start_time = time.time()

    logging.info(f"--- Starting EA Objective (MultiTask): {run_id} ---")
    logging.info(f"Hyperparameters Received: {hyperparameters}")

    config = copy.deepcopy(base_config)

    try:

        if 'learning_rate' in hyperparameters:
            config['training']['learning_rate'] = float(hyperparameters['learning_rate']) # Ensure float
        if 'dropout_rate' in hyperparameters:
            config['model']['dropout_rate'] = float(hyperparameters['dropout_rate']) # Ensure float

        if 'loss_alpha' in hyperparameters:
            alpha = float(hyperparameters['loss_alpha'])

            alpha = max(0.0, min(1.0, alpha))
            config['training']['loss_alpha'] = alpha
            config['training']['loss_beta'] = 1.0 - alpha
            logging.info(f"Applying loss weights: alpha={alpha:.4f}, beta={1.0 - alpha:.4f}")

    except KeyError as e:
        logging.error(f"EA Run {run_id}: Error applying hyperparameters. Missing key in config: {e}. Check config structure and ea_config param_space names.")
        return float('inf'),
    except Exception as e:
        logging.error(f"EA Run {run_id}: Error applying hyperparameters: {e}")
        return float('inf'),

    lr_eff = config['training']['learning_rate']
    drop_eff = config['model']['dropout_rate']
    alpha_eff = config['training']['loss_alpha']
    logging.info(f"Effective Run Config: LR={lr_eff:.6f}, Dropout={drop_eff:.3f}, LossAlpha={alpha_eff:.3f}") # Add others if optimized

    train_cfg = config['training']
    model_cfg = config['model']
    data_cfg = config['data']

    set_seed(train_cfg['seed'])

    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else "cpu")

    run_output_dir = os.path.join(train_cfg['output_dir'], "ea_runs", run_id)
    try:
        os.makedirs(run_output_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"EA Run {run_id}: Failed to create run output directory {run_output_dir}: {e}")
        return float('inf'),

    try:

        train_dataset = EchonetPseudoDataset(config=data_cfg, split='TRAIN')
        val_dataset = EchonetPseudoDataset(config=data_cfg, split='VALIDATE')

        if len(train_dataset) == 0 or len(val_dataset) == 0:
             raise RuntimeError("Training or Validation dataset (Echonet/Pseudo) is empty. Check CSV and paths.")

        train_loader = DataLoader(
            train_dataset, batch_size=train_cfg['batch_size'], shuffle=True,
            num_workers=train_cfg['num_workers'], pin_memory=True, drop_last=True)
        val_loader = DataLoader(
            val_dataset, batch_size=train_cfg['batch_size'], shuffle=False,
            num_workers=train_cfg['num_workers'], pin_memory=True)
        logging.info(f"EA Run {run_id}: Loaded Train ({len(train_dataset)}) and Val ({len(val_dataset)}) datasets.")

    except Exception as e:
         logging.error(f"EA Run {run_id}: FATAL - Failed during dataset loading or DataLoader creation - {e}", exc_info=True)
         return float('inf'),

    try:
        model = MultiTaskCNN(
            backbone_name=model_cfg['backbone'],
            pretrained=model_cfg['pretrained'],
            num_view_classes=data_cfg['num_views'],
            dropout_rate=config['model']['dropout_rate']
        ).to(device)
    except Exception as e:
         logging.error(f"EA Run {run_id}: Failed to initialize MultiTaskCNN model: {e}", exc_info=True)
         return float('inf'),

    try:
        criterion = CombinedLoss(
            alpha=config['training']['loss_alpha'], # Use updated alpha
            beta=config['training']['loss_beta']   # Use updated beta
        ).to(device)

        lr = config['training']['learning_rate']
        optimizer = None
        opt_name = train_cfg['optimizer'].lower()
        if opt_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # Example WD
        elif opt_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4) # Example WD
        else:
            raise ValueError(f"Unsupported optimizer '{opt_name}'")

        scheduler = None
        if train_cfg.get('scheduler') and train_cfg['scheduler'].lower() != 'null':
            sch_params = train_cfg.get('scheduler_params', {})
            sch_type = train_cfg['scheduler'].lower()
            if sch_type == 'steplr':
                if 'step_size' not in sch_params or 'gamma' not in sch_params: raise ValueError("StepLR needs step_size/gamma")
                scheduler = optim.lr_scheduler.StepLR(optimizer, **sch_params)
            elif sch_type == 'reducelronplateau':
                sch_params.setdefault('patience', 5)

                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=False, **sch_params)
            else:
                 logging.warning(f"EA Run {run_id}: Unsupported scheduler type '{sch_type}'.")
    except Exception as e:
        logging.error(f"EA Run {run_id}: Failed setting up Loss/Optimizer/Scheduler: {e}", exc_info=True)
        return float('inf'),

    if device == torch.device('cuda'):
        torch.cuda.empty_cache()

    fitness = float('inf')
    try:
        logging.info(f"EA Run {run_id}: Starting training loop...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=train_cfg['epochs'],
            device=device,
            config=config,
            checkpoint_dir=run_output_dir,
            metric_to_monitor='val_loss'
        )
        logging.info(f"EA Run {run_id}: Training loop completed.")

        if history and 'val_loss' in history and history['val_loss']:
            fitness = min(history['val_loss'])
            best_epoch_idx = np.argmin(history['val_loss'])
            logging.info(f"EA Run {run_id}: Fitness = Best Val Loss: {fitness:.6f} (found at epoch {best_epoch_idx + 1})")
        else:
            logging.warning(f"EA Run {run_id}: No validation loss found in history after training. Assigning high fitness.")
            fitness = float('inf')

    except KeyboardInterrupt:
        logging.warning(f"EA Run {run_id}: Training interrupted by user (KeyboardInterrupt). Assigning high fitness.")
        fitness = float('inf')

    except Exception as e:

        logging.error(f"EA Run {run_id}: *** CRITICAL ERROR DURING TRAINING ***: {e}", exc_info=True)
        fitness = float('inf')

    finally:

        del model, optimizer, criterion, train_loader, val_loader
        if device == torch.device('cuda'):
            torch.cuda.empty_cache()

    end_time = time.time()
    logging.info(f"--- Finished EA Objective (MultiTask): {run_id}. Time: {end_time - start_time:.2f}s. Fitness Returned: {fitness:.6f} ---")

    return fitness,