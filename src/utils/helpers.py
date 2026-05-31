
import yaml
import logging
import random
import numpy as np
import torch
import os
import shutil
import time

def load_config(config_path):

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        raise

def setup_logging(log_file="log.log", level=logging.INFO):

    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[ logging.FileHandler(log_file, mode='a'), logging.StreamHandler() ] )
    logging.info(f"Logging setup complete. Log file: {log_file} | Level: {logging.getLevelName(level)}")

def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", best_filename="best_model.pth.tar", max_retries=3):
    checkpoint_dir = os.path.dirname(filename)
    os.makedirs(checkpoint_dir, exist_ok=True)
    temp_filename = filename + ".tmp"
    
    for attempt in range(max_retries):
        try:
            torch.save(state, temp_filename)
            os.replace(temp_filename, filename)
            
            # Integrity check
            if not os.path.exists(filename) or os.path.getsize(filename) == 0:
                raise IOError(f"Checkpoint file {filename} is missing or empty after save.")
                
            logging.debug(f"Saved current checkpoint atomically to {filename} (Attempt {attempt + 1})")
            
            if is_best:
                best_filepath = os.path.join(checkpoint_dir, os.path.basename(best_filename))
                shutil.copyfile(filename, best_filepath)
                logging.info(f"Updated best model checkpoint at {best_filepath}")
                
            return # Success
            
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed to save checkpoint {filename}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1) # wait before retry
            else:
                logging.error(f"Failed to save checkpoint after {max_retries} attempts.", exc_info=True)
        finally:
            # Cleanup temp file if it exists
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except OSError:
                    pass

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None, device='cuda'):

    if not os.path.isfile(checkpoint_path):
        logging.error(f"Checkpoint file not found at '{checkpoint_path}'")
        return None
    try:
        logging.info(f"Loading checkpoint '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            is_data_parallel = False
            for k, v in state_dict.items():
                if k.startswith('module.'): is_data_parallel = True; name = k[7:]; new_state_dict[name] = v
                else: new_state_dict[k] = v
            if is_data_parallel: logging.info("DataParallel 'module.' prefix removed from state_dict keys.")

            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            if missing_keys: logging.warning(f"Missing keys when loading model state_dict: {missing_keys}")
            if unexpected_keys: logging.warning(f"Unexpected keys when loading model state_dict: {unexpected_keys}")
            logging.info("Model state loaded successfully.")
        else:
            logging.warning(f"Checkpoint '{checkpoint_path}' does not contain 'state_dict' for the model.")

        if optimizer is not None and 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                logging.info("Optimizer state loaded successfully.")
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor): state[k] = v.to(device)
                logging.debug("Moved optimizer state tensors to target device if necessary.")
            except Exception as e: logging.error(f"Could not load optimizer state: {e}. Optimizer state might be reset.", exc_info=True)
        elif optimizer is not None: logging.warning(f"Optimizer state not found or is None in checkpoint '{checkpoint_path}'. Optimizer not loaded.")

        if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
                logging.info("Scheduler state loaded successfully.")
            except Exception as e: logging.error(f"Could not load scheduler state: {e}. Scheduler state might be reset.", exc_info=True)
        elif scheduler is not None: logging.warning(f"Scheduler state not found or is None in checkpoint '{checkpoint_path}'. Scheduler not loaded.")

        if scaler is not None and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
            try:
                scaler.load_state_dict(checkpoint['scaler'])
                logging.info("Scaler state loaded successfully.")
            except Exception as e: logging.error(f"Could not load scaler state: {e}. Scaler state might be reset.", exc_info=True)
        elif scaler is not None: logging.warning(f"Scaler state not found or is None in checkpoint '{checkpoint_path}'. Scaler not loaded.")

        logging.info(f"Checkpoint loaded successfully from '{checkpoint_path}'.")
        return checkpoint

    except Exception as e:
        logging.error(f"Failed to load checkpoint '{checkpoint_path}': {e}", exc_info=True)
        return None

