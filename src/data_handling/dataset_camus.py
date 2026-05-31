
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from PIL import Image
import logging
from torchvision import transforms as T
import glob

def get_transforms(img_size, split='TRAIN'):

    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    if isinstance(img_size, list): img_size = tuple(img_size) # Ensure tuple (H, W)

    if split.upper() == 'TRAIN':
        return T.Compose([
            T.Resize(img_size),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    elif split.upper() in ['VALIDATE', 'VAL', 'TEST']: # Allow common names
        return T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    else:
        raise ValueError(f"Unknown split name for transforms: {split}")


class CamusViewDataset(Dataset):
    def __init__(self, config, split='train', transform=None):

        self.config = config
        # --- Get required paths from config ---
        self.base_dir = config.get('base_dir') # e.g., data/camus/database_nifti
        self.processed_data_dir = config.get('processed_dir') # e.g., processed_data/camus
        # Optionally get split_dir from config, otherwise derive it
        self.split_dir_cfg = config.get('split_dir', None) # e.g., data/camus/database_split

        self.view_mapping = config.get('view_mapping', {'2CH': 0, '4CH': 1}) # Use default if missing
        self.split = split.lower() # Ensure lowercase for matching keys
        self.transform = transform

        logging.info(f"Initializing CAMUS dataset for split: {self.split}")
        if not self.base_dir: raise ValueError("FATAL: 'data.base_dir' path is missing in config.")
        if not self.processed_data_dir: raise ValueError("FATAL: 'data.processed_dir' path is missing in config.")

        logging.info(f"Reading patient data structure expected within: {self.base_dir}") # Base_dir likely unused if reading from processed_dir
        logging.info(f"Looking for processed PNGs in: {self.processed_data_dir}")


        all_samples = []
        if not os.path.isdir(self.processed_data_dir):
             raise FileNotFoundError(f"Processed data directory not found: {self.processed_data_dir}. Run preprocessing first.")

        patient_folders = sorted([d for d in os.listdir(self.processed_data_dir) if d.startswith('patient') and os.path.isdir(os.path.join(self.processed_data_dir, d))])
        logging.info(f"Found {len(patient_folders)} patient subfolders in processed directory.")

        if not patient_folders: raise ValueError("No patient folders found in processed data directory.")

        for patient_id in patient_folders:
            patient_processed_path = os.path.join(self.processed_data_dir, patient_id)
            png_files = glob.glob(os.path.join(patient_processed_path, "*.png"))

            for img_path in png_files:
                filename = os.path.basename(img_path)
                parts = filename.split('_')
                if len(parts) >= 3:
                    view_name = parts[1]
                    frame_type = parts[2].split('.')[0]
                    if view_name in self.view_mapping:
                        all_samples.append({'ImagePath': img_path, 'View': view_name, 'PatientID': patient_id, 'FrameType': frame_type})
                else:
                    logging.warning(f"Could not parse info from filename: {filename}. Skipping.")

        if not all_samples: raise ValueError("No valid processed PNG samples found matching expected views.")

        full_df = pd.DataFrame(all_samples)
        full_df['ViewLabel'] = full_df['View'].map(self.view_mapping)
        full_df['ViewLabel'] = full_df['ViewLabel'].astype(int)

        logging.info(f"Found {len(full_df)} total valid processed CAMUS samples (ED/ES frames).")


        split_filename_map = { 'train': 'subgroup_training.txt', 'val': 'subgroup_validation.txt',
                               'validation': 'subgroup_validation.txt', 'test': 'subgroup_testing.txt' }
        subgroup_filename = split_filename_map.get(self.split)
        if not subgroup_filename: raise ValueError(f"Invalid split name '{self.split}'. Expected 'train', 'val', or 'test'.")


        if self.split_dir_cfg and os.path.isdir(self.split_dir_cfg):

             split_dir_path = self.split_dir_cfg
             logging.debug(f"Using split directory from config: {split_dir_path}")
        elif self.base_dir:

             parent_dir = os.path.dirname(self.base_dir)
             split_dir_path = os.path.join(parent_dir, 'database_split')
             logging.debug(f"Derived split directory path: {split_dir_path}")
        else:

             raise ValueError("Cannot determine split file directory. Set 'data.base_dir' or 'data.split_dir' in config.")

        subgroup_filepath = os.path.join(split_dir_path, subgroup_filename)

        logging.info(f"Attempting to load patient IDs for split '{self.split}' from: {subgroup_filepath}")

        if not os.path.exists(subgroup_filepath):
            raise FileNotFoundError(f"Subgroup file for split '{self.split}' not found at: {subgroup_filepath}")

        try:
            with open(subgroup_filepath, 'r') as f:
                split_patient_ids = {line.strip() for line in f if line.strip()}
            logging.info(f"Loaded {len(split_patient_ids)} unique patient IDs for split '{self.split}'.")
            if not split_patient_ids: logging.warning(f"Subgroup file {subgroup_filepath} is empty!")

            # Filter the main DataFrame
            self.df = full_df[full_df['PatientID'].isin(split_patient_ids)].reset_index(drop=True)
            logging.info(f"Applied predefined split '{self.split}'. Kept {len(self.df)} samples.")

        except Exception as e:
            logging.error(f"Error reading or applying splits from {subgroup_filepath}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to process subgroup file {subgroup_filepath}") from e


        logging.info(f"Final CAMUS dataset size for split '{self.split}': {len(self.df)} samples.")
        if len(self.df) == 0: logging.warning(f"CAMUS dataset split '{self.split}' is empty after filtering!")

        if self.transform is None:
            img_size = config.get('preprocessing', {}).get('img_size', [128, 128])
            self.transform = get_transforms(img_size, split=self.split.upper())
            logging.info(f"Using default transforms for CAMUS split {self.split}")
        else:
            logging.info(f"Using provided external transforms for CAMUS split {self.split}")

    def __len__(self):
        return len(self.df) if hasattr(self, 'df') and isinstance(self.df, pd.DataFrame) else 0

    def __getitem__(self, idx):
        if not hasattr(self, 'df') or idx >= len(self.df):
            raise IndexError("Dataset index out of range or DataFrame not initialized.")

        row = self.df.iloc[idx]
        img_path = row['ImagePath']
        view_label = torch.tensor(row['ViewLabel'], dtype=torch.long)

        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform: img = self.transform(img)

            return img, view_label
        except FileNotFoundError:
            logging.error(f"Image file not found at path: {img_path} for index {idx}.")
            raise FileNotFoundError(f"Image file not found: {img_path}") from None
        except Exception as e:
            logging.error(f"Error loading or transforming CAMUS item idx {idx}, path {img_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load/transform CAMUS data for index {idx}") from e