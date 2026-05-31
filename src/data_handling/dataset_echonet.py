
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from PIL import Image
import logging
from torchvision import transforms as T
import traceback # Added for detailed error logging

def video_collate_fn(batch):

    batch = [item for item in batch if item is not None]

    if not batch:

        logging.warning("Collate function received an entirely empty batch after filtering None items.")

        return torch.empty(0), torch.empty(0, dtype=torch.long), torch.empty(0)


    video_tensors = [item[0] for item in batch]
    view_labels = [item[1] for item in batch]
    ef_labels = [item[2] for item in batch]

    video_batch = torch.stack(video_tensors, dim=0)

    view_label_batch = torch.stack([torch.as_tensor(v) for v in view_labels], dim=0)
    ef_label_batch = torch.stack([torch.as_tensor(e) for e in ef_labels], dim=0)


    return video_batch, view_label_batch, ef_label_batch

def get_transforms(img_size, split='TRAIN'):


    if isinstance(img_size, int): img_size = (img_size, img_size)
    elif isinstance(img_size, list) and len(img_size) == 2: img_size = tuple(img_size)
    elif not (isinstance(img_size, tuple) and len(img_size) == 2):
         default_size = (112, 112)
         logging.warning(f"Invalid img_size format: {img_size}. Defaulting to {default_size}.")
         img_size = default_size

    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]

    split_upper = split.upper()

    if split_upper == 'TRAIN':

        return T.Compose([
            T.Resize(img_size),
            T.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)), # Mild affine

            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    elif split_upper in ['VALIDATE', 'VAL', 'TEST']: # Accept 'VAL' as alias for 'VALIDATE'
        return T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    else:
        raise ValueError(f"Unknown split name for transforms: {split}. Expected 'TRAIN', 'VAL', 'VALIDATE', or 'TEST'.")


class EchoNetViewDataset(Dataset):
    def __init__(self, config, split='TRAIN', transform=None):

        self.config = config
        self.processed_data_dir = config.get('processed_dir', 'processed_data/echonet') # Get base dir for processed frames
        self.num_frames_per_sample = config.get('frames_per_video', 1)
        self.view_mapping = config.get('view_mapping', {})
        self.split = split.upper()
        self.filename_col = config.get('filename_col', 'FileName')
        self.ef_col = config.get('ef_col', 'EF')
        self.view_col = config.get('view_col', 'PredictedView')
        self.split_col = config.get('split_col', 'Split')

        if not self.view_mapping:
            logging.warning("View mapping is empty in config. View labels cannot be mapped.")


        csv_path = config.get('csv_path')
        if not csv_path: raise ValueError("FATAL: 'csv_path' not specified in data config.")
        logging.info(f"Loading Echonet dataset for split: {self.split} from {csv_path}")
        try:
            full_df = pd.read_csv(csv_path)
        except FileNotFoundError:
            logging.error(f"FATAL: Echonet pseudo-label CSV not found: {csv_path}")
            raise
        except Exception as e:
            logging.error(f"FATAL: Error reading CSV {csv_path}: {e}")
            raise


        required_columns = [self.filename_col, self.ef_col, self.view_col, self.split_col]
        missing_cols = [col for col in required_columns if col not in full_df.columns]
        if missing_cols:
            raise ValueError(f"FATAL: Missing required columns in {csv_path}: {missing_cols}. Expected: {required_columns}.")


        df_split_col_upper = full_df[self.split_col].astype(str).str.upper()
        if self.split == 'VALIDATE':
            self.df = full_df[df_split_col_upper.isin(['VAL', 'VALIDATE'])].copy()
        else:
            self.df = full_df[df_split_col_upper == self.split].copy()

        initial_split_len = len(self.df)
        logging.info(f"Filtered for split '{self.split}'. Kept {initial_split_len} entries initially.")

        if initial_split_len == 0:
            logging.warning(f"No entries found for split '{self.split}' in {csv_path}.")
            self.df = pd.DataFrame(columns=full_df.columns) # Ensure self.df exists but is empty
        else:

            logging.info(f"[{self.split}] Cleaning and preparing data...")
            len_before_clean = len(self.df)


            self.df[self.ef_col] = pd.to_numeric(self.df[self.ef_col], errors='coerce')
            len_after_ef_nan = len(self.df)
            self.df = self.df.dropna(subset=[self.ef_col], how='any') # Drop if EF is NaN
            if len_after_ef_nan > len(self.df):
                logging.info(f"[{self.split}] Dropped {len_after_ef_nan - len(self.df)} rows due to NaN in '{self.ef_col}'.")


            len_after_view_nan = len(self.df)
            self.df = self.df.dropna(subset=[self.view_col], how='any')
            # Ensure it's treated as string for stripping check
            self.df = self.df[self.df[self.view_col].astype(str).str.strip() != '']
            if len_after_view_nan > len(self.df):
                 logging.info(f"[{self.split}] Dropped {len_after_view_nan - len(self.df)} rows due to NaN/empty in '{self.view_col}'.")


            len_after_map_fail = len(self.df)
            if self.view_mapping:
                 try:
                     self.df['ViewLabel'] = self.df[self.view_col].map(self.view_mapping)

                     # STRICT FILTERING: Keep only mapped classes (0 and 1)
                     self.df = self.df.dropna(subset=['ViewLabel'])
                     self.df['ViewLabel'] = self.df['ViewLabel'].astype(int)
                     
                     # Double check to ensure NO other classes sneak in
                     self.df = self.df[self.df['ViewLabel'].isin([0, 1])]
                     
                     if not self.df.empty: 
                         logging.info(f"[{self.split}] Strictly filtered to A2C (0) and A4C (1). Count: {len(self.df)}")
                 except Exception as e:
                     logging.error(f"[{self.split}] Mapping pseudo-view labels failed: {e}", exc_info=True)
                     raise ValueError("Error mapping pseudo labels. Check view_mapping and CSV content.") from e
            else:

                 logging.error("FATAL: View mapping is empty in config. Cannot create 'ViewLabel' for classification.")
                 raise ValueError("View mapping missing in configuration.")

            if len_after_map_fail > len(self.df):
              logging.info(f"[{self.split}] Dropped {len_after_map_fail - len(self.df)} rows due to view mapping failure.")

            def check_frame_dir_valid(filename):
                if pd.isna(filename):

                    return False

                frame_dir = os.path.join(self.processed_data_dir, str(filename))

                try:
                    is_dir = os.path.isdir(frame_dir)
                    if not is_dir:
                        print(f"!!! [{self.split}] FAILED check_frame_dir_valid for '{filename}': Path '{frame_dir}' is NOT a directory or inaccessible.")
                        return False


                    has_frames = any(f.startswith('frame_') and f.endswith('.png') for f in os.listdir(frame_dir))
                    if not has_frames:
                        print(f"!!! [{self.split}] FAILED check_frame_dir_valid for '{filename}': Directory '{frame_dir}' exists but contains NO valid frames (e.g., 'frame_*.png').")
                        return False


                    return True

                except FileNotFoundError:
                    print(f"!!! [{self.split}] FAILED check_frame_dir_valid for '{filename}': Directory '{frame_dir}' NOT FOUND during listing.")
                    return False
                except PermissionError:
                    print(f"!!! [{self.split}] FAILED check_frame_dir_valid for '{filename}': PERMISSION DENIED for directory '{frame_dir}'.")
                    return False
                except Exception as e:
                    print(f"!!! [{self.split}] FAILED check_frame_dir_valid for '{filename}': Error checking directory '{frame_dir}': {e}")
                    return False



            len_before_frame_check = len(self.df)
            if not self.df.empty:
                logging.info(f"[{self.split}] Checking existence and content of processed frame directories for {len(self.df)} rows...")
                try:
                    valid_mask = self.df[self.filename_col].apply(check_frame_dir_valid)
                    num_removed = len(valid_mask) - valid_mask.sum()
                    if num_removed > 0:
                         logging.warning(f"[{self.split}] Will remove {num_removed} entries due to missing/empty/inaccessible frame directories based on checks above.")
                    self.df = self.df[valid_mask] # Keep only valid rows
                except Exception as e:
                    logging.error(f"[{self.split}] Frame existence check failed during apply: {e}", exc_info=True)
                    raise RuntimeError("Critical error during frame existence check.") from e



            dropped_count = len_before_clean - len(self.df)
            logging.info(f"[{self.split}] Cleaning dropped {dropped_count} rows (NaNs, mapping errors, missing frames, etc.).")


        final_len = len(self.df)
        logging.info(f"Split '{self.split}': Final usable size {final_len} samples.")
        if final_len == 0 and self.split != 'TEST': # Allow empty test set, error otherwise for train/val
            logging.error(f"FATAL: Split '{self.split}' is empty after cleaning/checks! Cannot proceed.")
            raise RuntimeError(f"Dataset split '{self.split}' became empty after filtering.")


        self.df = self.df.reset_index(drop=True)


        if transform is None:
            prep_cfg = config.get('preprocessing', {})
            img_size = prep_cfg.get('img_size', [112, 112])
            self.transform = get_transforms(img_size, split=self.split)
            logging.info(f"Using default transforms for split {self.split} with image size {img_size}.")
        else:
            self.transform = transform
            logging.info(f"Using provided custom transforms for Echonet split {self.split}")


    def __len__(self):

        return len(self.df) if hasattr(self, 'df') and not self.df.empty else 0

    def __getitem__(self, idx):

        if not hasattr(self, 'df') or self.df.empty:
             raise IndexError("Dataset is empty or not properly initialized.")
        if idx >= len(self.df):
             raise IndexError(f"Index {idx} out of range for dataset size {len(self.df)}.")

        video_name_no_ext = "UNKNOWN"
        try:

            row = self.df.iloc[idx]
            video_name_no_ext = str(row[self.filename_col])
            ef_value = torch.tensor(row[self.ef_col], dtype=torch.float32)
            view_label = torch.tensor(int(row['ViewLabel']), dtype=torch.long)

            frame_dir = os.path.join(self.processed_data_dir, video_name_no_ext)


            available_frames = sorted([f for f in os.listdir(frame_dir) if f.startswith('frame_') and f.endswith('.png')])
            num_available = len(available_frames)

            if num_available == 0:

                 logging.critical(f"CRITICAL: No frames found in {frame_dir} for index {idx}, video {video_name_no_ext} (should have been filtered in __init__).")
                 raise FileNotFoundError(f"No frames found in {frame_dir} for index {idx}")

            n_sample = max(1, self.num_frames_per_sample) # Number of frames to sample


            if num_available < n_sample:

                sampled_indices = np.random.choice(num_available, n_sample, replace=True)

            else:

                sampled_indices = np.random.choice(num_available, n_sample, replace=False)
            sampled_indices.sort()


            frames = []
            for frame_index in sampled_indices:
                frame_file = available_frames[frame_index]
                frame_path = os.path.join(frame_dir, frame_file)
                try:

                    img = Image.open(frame_path).convert('RGB')
                    transformed_img = self.transform(img)
                    frames.append(transformed_img)
                except FileNotFoundError:

                     logging.error(f"Frame file disappeared: {frame_path} for index {idx}.")
                     raise RuntimeError(f"Frame file {frame_file} missing for index {idx}") from None
                except Exception as img_err:
                     logging.error(f"Error loading/transforming image {frame_path} for index {idx}: {img_err}")
                     raise RuntimeError(f"Failed to load/transform image for index {idx}") from img_err


            if len(frames) != n_sample:
                raise RuntimeError(f"Logic Error: Expected {n_sample} frames but loaded {len(frames)} for index {idx}")


            stacked_tensor = torch.stack(frames, dim=0)      #[T, C, H, W]
            final_video_tensor = stacked_tensor.permute(1, 0, 2, 3) #[C, T, H, W]

            return final_video_tensor, view_label, ef_value

        except Exception as e:

            logging.error(f"CRITICAL FAILURE in __getitem__ for index {idx}, video {video_name_no_ext}: {e}", exc_info=True)

            logging.error(traceback.format_exc())

            raise RuntimeError(f"Failed to get data for index {idx}") from e