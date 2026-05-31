# src/data_handling/dataset_hmcqu.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import re
import numpy as np

from PIL import Image
import logging
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
import glob # Keep glob import, although not used directly in this version

# --- Standard Transforms ---
# (Assuming get_transforms is defined correctly as before)
def get_transforms(img_size, split='TRAIN'):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    if isinstance(img_size, list): img_size = tuple(img_size)
    if split.upper() == 'TRAIN':
        return T.Compose([ T.Resize(img_size), T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10), T.ColorJitter(brightness=0.2, contrast=0.2), T.RandomHorizontalFlip(p=0.5), T.ToTensor(), T.Normalize(mean=mean, std=std) ])
    elif split.upper() in ['VALIDATE', 'VAL', 'TEST']:
        return T.Compose([ T.Resize(img_size), T.ToTensor(), T.Normalize(mean=mean, std=std) ])
    else: raise ValueError(f"Unknown split name for transforms: {split}")

class HMCQUViewDataset(Dataset):
    def __init__(self, config, split='train', transform=None):
        """
        Dataset for HMC-QU view classification. Reads metadata, finds preprocessed frames,
        splits data randomly, and loads single frames.
        """
        self.config = config
        # Use .get() for safety, raise error if essential paths missing
        self.base_data_dir = config.get('base_data_dir')
        self.processed_dir = config.get('processed_dir')
        if not self.base_data_dir: raise ValueError("Config missing 'data.base_data_dir'")
        if not self.processed_dir: raise ValueError("Config missing 'data.processed_dir'")

        self.view_mapping = config.get('view_mapping', {'A2C': 0, 'A4C': 1})
        self.split = split.lower()
        self.transform = transform
        # Ensure frame_type is uppercase for consistency in filename construction
        self.frame_to_use = config.get('frame_type_to_use', 'ED').upper()
        self.filename_col = config.get('filename_col', 'Unnamed: 0') # Default to 'Unnamed: 0'
        self.split_strategy = config.get('split_strategy', 'random')
        self.split_ratio = config.get('split_ratio', 0.8)
        self.seed = config.get('training', {}).get('seed', 42)
        prep_cfg = config.get('preprocessing', {})
        self.output_format = prep_cfg.get('output_format', 'png')

        logging.info(f"Initializing HMC-QU dataset for split: {self.split}")
        if not os.path.isdir(self.processed_dir):
             raise FileNotFoundError(f"Processed data directory missing or invalid: {self.processed_dir}")
        if not os.path.isdir(self.base_data_dir):
             raise FileNotFoundError(f"Base data directory not found or invalid: {self.base_data_dir}")

        # --- 1. Load Info & Find Existing Processed Frames ---
        all_samples_info = []
        view_files = {view_name: f"{view_name}.xlsx" for view_name in self.view_mapping.keys()}

        for view_name_excel, excel_file in view_files.items(): # view_name_excel is 'A2C' or 'A4C'
            excel_path = os.path.join(self.base_data_dir, excel_file)
            if not os.path.exists(excel_path):
                 logging.warning(f"Metadata file not found: {excel_path}. Skipping view {view_name_excel}.")
                 continue
            try:
                df_view = pd.read_excel(excel_path, header=1) # Skip header row
                logging.debug(f"Columns found in {excel_file}: {list(df_view.columns)}")

                # Check if the expected column name exists
                if self.filename_col not in df_view.columns:
                     logging.warning(f"Expected column '{self.filename_col}' not found in {excel_path}. Skipping.")
                     continue # Skip this file

                logging.info(f"Scanning {len(df_view)} potential samples from {excel_file} for view {view_name_excel}")
                processed_in_file_count = 0
                for id_str in df_view[self.filename_col]:
                    if pd.isna(id_str): continue
                    id_str = str(id_str).strip().replace('*','')

                    patient_id_match = re.match(r"(ES\d+)", id_str) # Assuming ESXXXX... format
                    if not patient_id_match:
                        logging.debug(f"Could not parse PatientID from '{id_str}'. Skipping.")
                        continue
                    patient_id = patient_id_match.group(1)

                    # =========================================================
                    # <<< CONSTRUCT EXPECTED FILENAME FOR PROCESSED PNG >>>
                    # Ensure this matches EXACTLY how preprocess_hmcqu.py saved the files
                    processed_img_filename = f"{patient_id}_{view_name_excel}_{self.frame_to_use}.{self.output_format}"
                    # Example: ES00001_A4C_ED.png (if frame_to_use is 'ED')
                    # Example: ES00001_A2C_MIDDLE.png (if frame_to_use is 'MIDDLE')
                    # <<< ADJUST IF YOUR PREPROCESSING NAMING WAS DIFFERENT >>>
                    # =========================================================
                    img_path = os.path.join(self.processed_dir, patient_id, processed_img_filename)

                    # Check if this specific processed frame exists
                    if os.path.exists(img_path):
                         all_samples_info.append({
                             'PatientID': patient_id,
                             'View': view_name_excel, # Keep original view string
                             'ImagePath': img_path    # Store full path to the existing PNG
                         })
                         processed_in_file_count += 1
                    else:
                         # Log only if DEBUG level is enabled to avoid excessive output
                         logging.debug(f"Processed PNG not found for ID {id_str} at expected path: {img_path}")

                logging.info(f"Found {processed_in_file_count} samples with existing processed {self.frame_to_use} frames for view {view_name_excel}.")

            except ImportError: logging.error(f"Reading Excel file {excel_path} failed. Install 'openpyxl'."); raise
            except Exception as e: logging.error(f"Error processing Excel file {excel_path}: {e}", exc_info=True)

        if not all_samples_info:
             # Raise error if NO samples with existing processed frames were found
             raise ValueError(f"Failed to find any existing processed frames in '{self.processed_dir}' matching Excel entries and expected naming convention (PatientID_View_{self.frame_to_use}.{self.output_format}). Ensure HMC-QU preprocessing ran successfully and saved files with this name.")

        combined_df = pd.DataFrame(all_samples_info)
        combined_df = combined_df.drop_duplicates(subset=['PatientID', 'View'], keep='first')
        logging.info(f"Found {len(combined_df)} unique PatientID-View combinations with existing processed {self.frame_to_use} frames.")

        # --- Create Splits ---
        if self.split_strategy == 'random':
             unique_patient_ids = combined_df['PatientID'].unique()
             try:
                 train_ratio = min(max(0.01, self.split_ratio), 0.99)
                 train_ids, val_ids = train_test_split(unique_patient_ids, train_size=train_ratio, random_state=self.seed)
                 logging.info(f"Randomly splitting {len(unique_patient_ids)} patients into {len(train_ids)} train / {len(val_ids)} val")
             except Exception as e: logging.error(f"Failed to split patient IDs: {e}"); raise e

             if self.split == 'train': self.df = combined_df[combined_df['PatientID'].isin(train_ids)].reset_index(drop=True)
             elif self.split in ['val', 'validation']: self.df = combined_df[combined_df['PatientID'].isin(val_ids)].reset_index(drop=True)
             elif self.split == 'test': logging.warning("Using 'val' split for 'test'."); self.df = combined_df[combined_df['PatientID'].isin(val_ids)].reset_index(drop=True)
             else: raise ValueError(f"Invalid split name '{self.split}'.")
        else: raise ValueError(f"Unsupported split_strategy: {self.split_strategy}")

        logging.info(f"Applied split '{self.split}'. Kept {len(self.df)} samples.")

        # --- Map Labels ---
        if not self.df.empty:
            self.df['ViewLabel'] = self.df['View'].map(self.view_mapping)
            if self.df['ViewLabel'].isnull().any():
                logging.warning("Some view labels failed to map after filtering. Check 'view_mapping' and 'View' column data.")
                self.df = self.df.dropna(subset=['ViewLabel'])
            self.df['ViewLabel'] = self.df['ViewLabel'].astype(int)
        else: self.df['ViewLabel'] = pd.Series(dtype=int)

        # --- Transforms ---
        logging.info(f"Final HMC-QU dataset size for split '{self.split}': {len(self.df)} samples.")
        if len(self.df) == 0 and self.split != 'test':
             logging.warning(f"HMC-QU dataset split '{self.split}' is empty!")

        if self.transform is None:
            img_size_trans = config.get('preprocessing', {}).get('img_size', [112, 112])
            self.transform = get_transforms(img_size_trans, split=self.split.upper())
            logging.info(f"Using default transforms for HMC-QU split {self.split}")
        else:
            logging.info(f"Using provided external transforms for HMC-QU split {self.split}")

    def __len__(self):
        return len(self.df) if hasattr(self, 'df') and isinstance(self.df, pd.DataFrame) else 0

    def __getitem__(self, idx):
        if not hasattr(self, 'df') or idx >= len(self.df):
            raise IndexError("Dataset index out of range or DataFrame not initialized.")
        try:
            row = self.df.iloc[idx]
            img_path = row['ImagePath'] # Path to the specific processed PNG frame
            view_label = torch.tensor(row['ViewLabel'], dtype=torch.long)
            img = Image.open(img_path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, view_label # Return single frame image and label
        except FileNotFoundError:
            # This error indicates a mismatch between the df and actual files
            logging.error(f"HMC-QU image file not found in __getitem__ (should exist based on init checks): {img_path} for index {idx}.")
            raise FileNotFoundError(f"HMC-QU image file not found: {img_path}") from None
        except Exception as e:
            logging.error(f"Error loading/transforming HMC-QU item idx {idx}, path {img_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load/transform HMC-QU data for index {idx}") from e