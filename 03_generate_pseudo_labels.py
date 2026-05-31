
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import argparse
import logging
import sys
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T

from multiprocessing import freeze_support
import glob

from src.utils.helpers import load_config, setup_logging, load_checkpoint

from src.models.cnn_view_classifier import ViewClassifierCNN
def get_inference_transforms(img_size):
    """ Basic transforms for inference (resize, tensor, normalize). """
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    # Ensure img_size is tuple (H, W)
    if isinstance(img_size, list) and len(img_size) == 2: img_size = tuple(img_size)
    elif not (isinstance(img_size, tuple) and len(img_size) == 2):
         default_size = (128, 128) # Default fallback size
         logging.warning(f"Invalid img_size {img_size} for transforms, using default {default_size}.")
         img_size = default_size
    return T.Compose([ T.Resize(img_size), T.ToTensor(), T.Normalize(mean=mean, std=std) ])


class EchonetInferenceDataset(Dataset):
    """ Dataset to load processed Echonet frames for view prediction inference. """
    def __init__(self, file_list, processed_dir, img_size, num_frames=1):
        self.file_list = file_list
        self.processed_dir = processed_dir
        self.num_frames = max(1, num_frames)
        self.transform = get_inference_transforms(img_size)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """ Loads frames for one video and returns tensor + filename. """
        video_name_no_ext = self.file_list[idx]
        frame_dir = os.path.join(self.processed_dir, str(video_name_no_ext))
        try:
            available_frames = sorted([f for f in os.listdir(frame_dir) if f.startswith('frame_') and f.endswith('.png')])
            if not available_frames:
                raise FileNotFoundError(f"No valid frame files found in {frame_dir}")

            num_available = len(available_frames)

            if self.num_frames == 1:
                sampled_indices = [num_available // 2]
            elif num_available < self.num_frames:
                sampled_indices = np.random.choice(num_available, self.num_frames, replace=True); sampled_indices.sort()
            else:
                indices = np.linspace(0, num_available - 1, num=self.num_frames, dtype=int)
                sampled_indices = sorted(list(np.unique(indices)))
                while len(sampled_indices) < self.num_frames: sampled_indices.append(sampled_indices[-1])
                sampled_indices = sampled_indices[:self.num_frames]


            sampled_frame_files = [available_frames[i] for i in sampled_indices]
            frames = [Image.open(os.path.join(frame_dir, f)).convert('RGB') for f in sampled_frame_files]
            transformed_frames = [self.transform(frame) for frame in frames]
            final_frame_tensor = torch.stack(transformed_frames, dim=0) # Shape: [T, C, H, W]


            if self.num_frames == 1:
                 final_frame_tensor = final_frame_tensor.squeeze(0)

            return final_frame_tensor, video_name_no_ext

        except FileNotFoundError:
             logging.warning(f"Frame directory not found or empty for {video_name_no_ext} at {frame_dir}. Returning None.")
             return None, video_name_no_ext
        except Exception as e:
            logging.error(f"Error loading Echonet frames for {video_name_no_ext} (idx {idx}): {e}", exc_info=True)
            return None, video_name_no_ext

class CamusInferenceDataset(Dataset):
    """ Dataset to load processed CAMUS PNG frames for view prediction inference. """
    def __init__(self, file_paths, img_size):
        """
        Args:
            file_paths (list): List of full paths to the processed PNG files.
            img_size (tuple/list): Target image size (H, W).
        """
        self.file_paths = file_paths
        self.transform = get_inference_transforms(img_size)

        self.identifiers = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """ Loads a single processed PNG image. """
        img_path = self.file_paths[idx]
        identifier = self.identifiers[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            transformed_img = self.transform(img)

            return transformed_img, identifier

        except FileNotFoundError:
             logging.warning(f"Image file not found for CAMUS: {img_path}. Returning None.")
             return None, identifier
        except Exception as e:
            logging.error(f"Error loading CAMUS frame {identifier} (idx {idx}): {e}", exc_info=True)
            return None, identifier

def collate_skip_none(batch):
    """ Filters out items where data is None, returns valid data and identifiers. """
    valid_batch = [item for item in batch if item[0] is not None]
    if not valid_batch:
        return None, []

    data_list = [item[0] for item in valid_batch]
    identifiers = [item[1] for item in valid_batch] # Filenames (Echonet) or Identifiers (CAMUS)

    try:

        data_batch = torch.stack(data_list, dim=0)
    except Exception as e:
        logging.error(f"Error stacking tensors in collate_fn: {e}. Items: {len(valid_batch)}")
        return None, []

    return data_batch, identifiers

def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-view-labels using a trained view classifier.")

    parser.add_argument("--config_dataset", type=str, required=True, help="Path to dataset config (e.g., config/config_echonet_multitask.yaml or config/config_camus_pseudolabels.yaml). Determines data source and handling.")
    parser.add_argument("--config_view", type=str, default="config/config_view_classifier.yaml", help="Path to VIEW CLASSIFIER model config (architecture, view mapping).")
    parser.add_argument("--view_model_path", type=str, required=True, help="Path to trained view classifier checkpoint (.pth.tar).")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV file with pseudo-labels.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference.")

    parser.add_argument("--num_frames", type=int, default=1, help="Number of frames per video for view prediction (Echonet ONLY, >=1). If >1, predictions are averaged. CAMUS always uses 1 frame.")
    parser.add_argument("--echonet_metadata_filename", type=str, default="FileList.csv", help="Filename of the original Echonet metadata (e.g., FileList.csv or FileList.xlsx) relative to Echonet's base_dir (Echonet ONLY).")

    args = parser.parse_args()

    log_dir = os.path.dirname(args.output_csv)
    if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir)
    log_file = os.path.join(log_dir or '.', 'generate_pseudo_labels.log')
    setup_logging(log_file=log_file, level=logging.INFO)

    logging.info("="*50); logging.info("--- Generate Pseudo Labels Script ---"); logging.info("="*50)
    logging.info(f"Dataset Config: {args.config_dataset}")
    logging.info(f"View Classifier Config: {args.config_view}")
    logging.info(f"View Model Checkpoint: {args.view_model_path}")
    logging.info(f"Output CSV: {args.output_csv}")
    logging.info(f"Batch Size: {args.batch_size}")

    try:
        config_dataset = load_config(args.config_dataset)
        config_view = load_config(args.config_view)

        dataset_type = config_dataset.get('dataset_type')
        if dataset_type not in ['echonet', 'camus']:
            raise ValueError(f"Config Error: 'dataset_type' in {args.config_dataset} must be 'echonet' or 'camus'. Found: {dataset_type}")
        logging.info(f"Processing Dataset Type: {dataset_type.upper()}")

        data_cfg = config_dataset['data']
        prep_cfg = config_dataset.get('preprocessing', {})
        exec_cfg = config_dataset.get('training', {})

        processed_dir = data_cfg['processed_dir']

        data_cfg_view = config_view['data']
        model_prep_cfg = data_cfg_view.get('preprocessing', {})
        img_size = model_prep_cfg.get('img_size')
        if img_size is None:
            img_size = prep_cfg.get('img_size', [128, 128])
            logging.warning(f"Could not find 'img_size' in view config preprocessing section ({args.config_view}), using value from dataset config or default: {img_size}")
        else:
             logging.info(f"Using img_size from view config ({args.config_view}): {img_size}")

        model_cfg_view = config_view['model']
        view_mapping = data_cfg_view.get('view_mapping')
        num_views = data_cfg_view.get('num_views')
        if not view_mapping or num_views is None:
             raise ValueError(f"Config Error: 'view_mapping' and 'num_views' must be defined in {args.config_view} ['data'] section.")

        device = torch.device(exec_cfg.get('device', 'cuda') if torch.cuda.is_available() else "cpu")
        num_workers = exec_cfg.get('num_workers', 0)

    except FileNotFoundError as e: logging.error(f"FATAL: Config file not found: {e}"); sys.exit(1)
    except KeyError as e: logging.error(f"FATAL: Missing expected key in config file: {e}"); sys.exit(1)
    except ValueError as e: logging.error(f"FATAL: Config error: {e}"); sys.exit(1)
    except Exception as e: logging.error(f"FATAL: Error loading config files: {e}", exc_info=True); sys.exit(1)

    logging.info(f"Using Device: {device}")
    logging.info(f"View Classifier Image Size: {img_size}")
    logging.info(f"View Mapping (from {args.config_view}): {view_mapping}")
    logging.info(f"DataLoader Workers: {num_workers}")

    logging.info("Loading trained view classifier model...")
    try:
        model = ViewClassifierCNN(
            backbone_name=model_cfg_view.get('backbone', 'resnet18'),
            pretrained=False,
            num_view_classes=num_views,
            dropout_rate=0.0
        ).to(device)

        checkpoint = load_checkpoint(args.view_model_path, model, device=device)
        if not checkpoint:
            raise FileNotFoundError(f"Failed to load or process checkpoint file: {args.view_model_path}")

        model.eval()
        logging.info(f"View classifier model loaded successfully from {args.view_model_path}")
    except FileNotFoundError: sys.exit(1)
    except Exception as e:
        logging.error(f"FATAL: Failed to initialize or load view classifier model: {e}", exc_info=True); sys.exit(1)

    inference_dataset = None
    original_metadata_df = None
    all_identifiers_to_process = []

    logging.info(f"Preparing data from processed directory: {processed_dir}")

    if not os.path.isabs(processed_dir):
        processed_dir = os.path.abspath(processed_dir)
        logging.info(f"Resolved relative processed_dir to absolute path: {processed_dir}")

    if not os.path.isdir(processed_dir):
        logging.error(f"FATAL: Processed data directory not found: {processed_dir}. Run preprocessing first for {dataset_type.upper()}.")
        sys.exit(1)

    try:
        if dataset_type == 'echonet':

            logging.info("Setting up Echonet inference...")
            if args.num_frames <= 0: logging.warning(f"--num_frames is {args.num_frames}, using 1 instead.")
            num_echonet_frames = max(1, args.num_frames)
            logging.info(f"Using num_frames per video: {num_echonet_frames}")

            echonet_base_dir = data_cfg.get('base_dir') # Path to folder containing metadata file
            echonet_filename_col = data_cfg.get('filename_col') # Col name in metadata
            if not echonet_base_dir or not echonet_filename_col:
                raise ValueError("Config Error: Echonet requires 'base_dir' and 'filename_col' in data config.")


            if not os.path.isabs(echonet_base_dir):
                echonet_base_dir = os.path.abspath(echonet_base_dir)

            echonet_orig_metadata_path = os.path.join(echonet_base_dir, args.echonet_metadata_filename)
            logging.info(f"Reading original Echonet metadata from: {echonet_orig_metadata_path}")
            if not os.path.exists(echonet_orig_metadata_path):
                 raise FileNotFoundError(f"Original Echonet metadata file not found: {echonet_orig_metadata_path}")

            try:
                 if args.echonet_metadata_filename.lower().endswith(('.xlsx', '.xls')):
                      original_metadata_df = pd.read_excel(echonet_orig_metadata_path)
                 elif args.echonet_metadata_filename.lower().endswith('.csv'):
                      original_metadata_df = pd.read_csv(echonet_orig_metadata_path)
                 else: raise ValueError(f"Unsupported metadata file format: {args.echonet_metadata_filename}.")
            except ImportError: logging.error("Reading Excel failed. Install 'openpyxl': `pip install openpyxl`"); sys.exit(1)
            except Exception as read_err: logging.error(f"Error reading metadata file {echonet_orig_metadata_path}: {read_err}"); sys.exit(1)

            logging.info(f"Found {len(original_metadata_df)} total entries in original Echonet metadata.")
            if echonet_filename_col not in original_metadata_df.columns:
                 raise ValueError(f"Original Echonet metadata missing specified filename column: '{echonet_filename_col}'")


            all_original_filenames = original_metadata_df[echonet_filename_col].astype(str).unique()
            valid_filenames = []
            logging.info(f"Checking for processed frame folders for {len(all_original_filenames)} unique Echonet filenames in {processed_dir}...")
            for filename in tqdm(all_original_filenames, desc="Checking Echonet Folders"):
                frame_dir = os.path.join(processed_dir, filename)
                if os.path.isdir(frame_dir):
                     valid_filenames.append(filename)


            all_identifiers_to_process = valid_filenames
            logging.info(f"Found {len(all_identifiers_to_process)} Echonet videos with processed frame folders for inference.")
            if not all_identifiers_to_process:
                 raise ValueError("No Echonet videos with processed frame folders found. Check 'processed_dir' and preprocessing output.")

            inference_dataset = EchonetInferenceDataset(
                file_list=all_identifiers_to_process,
                processed_dir=processed_dir,
                img_size=img_size,
                num_frames=num_echonet_frames
            )

        elif dataset_type == 'camus':

            logging.info("Setting up CAMUS inference...")
            camus_glob_pattern = data_cfg.get('processed_file_glob_pattern')
            if not camus_glob_pattern:
                raise ValueError("Config Error: CAMUS requires 'processed_file_glob_pattern' in data config.")

            search_path = os.path.join(processed_dir, camus_glob_pattern)
            logging.info(f"Scanning for CAMUS PNG files using pattern: {search_path}")

            processed_png_files = glob.glob(search_path, recursive=True)

            logging.info(f"Found {len(processed_png_files)} processed CAMUS PNG files for inference.")
            if not processed_png_files:
                raise ValueError(f"No processed CAMUS PNG files found matching pattern '{search_path}'. Check 'processed_dir', 'processed_file_glob_pattern' and preprocessing output.")

            all_identifiers_to_process = [os.path.splitext(os.path.basename(p))[0] for p in processed_png_files]

            inference_dataset = CamusInferenceDataset(
                file_paths=processed_png_files,
                img_size=img_size
            )

            if args.num_frames != 1:
                logging.warning(f"--num_frames argument ({args.num_frames}) is ignored for CAMUS dataset.")

        logging.info(f"Initializing DataLoader for {len(inference_dataset)} items...")
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' and num_workers > 0 else False,
            collate_fn=collate_skip_none
        )

    except Exception as e:
        logging.error(f"FATAL: Failed preparing {dataset_type.upper()} data: {e}", exc_info=True)
        sys.exit(1)

    predictions = {}

    reverse_view_mapping = {v: k for k, v in view_mapping.items()}
    logging.info(f"Using reverse view mapping for output: {reverse_view_mapping}")

    logging.info("Starting inference loop to predict views...")
    inference_errors = 0
    processed_count = 0
    try:
        with torch.no_grad():
            for inputs, identifiers_in_batch in tqdm(inference_loader, desc="Predicting Views"):
                if inputs is None or not identifiers_in_batch:

                    inference_errors += len(identifiers_in_batch) if isinstance(identifiers_in_batch, list) else 1
                    continue

                try:
                    inputs = inputs.to(device)

                    if dataset_type == 'echonet' and args.num_frames > 1:

                        batch_sz, num_frms, C, H, W = inputs.shape

                        inputs_reshaped = inputs.view(batch_sz * num_frms, C, H, W)
                        logits = model(inputs_reshaped)

                        logits = logits.view(batch_sz, num_frms, -1).mean(dim=1)
                    else:

                        logits = model(inputs)

                    preds_indices = torch.argmax(logits, dim=1).cpu().numpy()

                    for identifier, pred_idx in zip(identifiers_in_batch, preds_indices):
                        predictions[identifier] = reverse_view_mapping.get(pred_idx, "Unknown") # Default if index somehow invalid
                        processed_count += 1

                except Exception as batch_err:
                     logging.error(f"Error processing batch during inference: {batch_err}", exc_info=True)

                     inference_errors += len(identifiers_in_batch)

    except Exception as e:
        logging.error(f"FATAL: Error during inference loop: {e}", exc_info=True)

    logging.info(f"Inference finished. Generated predictions for {processed_count} items.")
    if inference_errors > 0:
         logging.warning(f"Encountered loading/processing errors for {inference_errors} items during inference.")

    logging.info("Saving results to CSV...")

    try:
        if dataset_type == 'echonet':

            logging.info("Merging predictions into Echonet metadata...")

            echonet_filename_col = data_cfg['filename_col']
            output_view_col = data_cfg.get('view_col', 'PredictedView')

            if original_metadata_df is None:
                 raise RuntimeError("Original Echonet metadata was not loaded correctly.")

            original_metadata_df[output_view_col] = original_metadata_df[echonet_filename_col].astype(str).map(predictions)

            missing_preds = original_metadata_df[output_view_col].isnull().sum()

            total_failed = missing_preds

            if total_failed > 0:
                 logging.warning(f"{total_failed} Echonet entries could not be mapped from predictions (likely loading/inference errors). Filling NaNs with 'Unknown'.")
                 original_metadata_df[output_view_col].fillna("Unknown", inplace=True)

            logging.info("Distribution of predicted pseudo-labels for Echonet:")

            logging.info(f"\n{original_metadata_df[output_view_col].value_counts(dropna=False).to_string()}")
            output_df = original_metadata_df

        elif dataset_type == 'camus':

            logging.info("Creating new CSV for CAMUS predictions...")

            output_filename_col = data_cfg['output_filename_col']
            output_viewlabel_col = data_cfg['output_viewlabel_col']

            results_list = [{output_filename_col: identifier, output_viewlabel_col: view}
                            for identifier, view in predictions.items()]
            output_df = pd.DataFrame(results_list)

            processed_identifiers_set = set(predictions.keys())
            all_identifiers_set = set(all_identifiers_to_process)
            missing_identifiers = list(all_identifiers_set - processed_identifiers_set)
            if missing_identifiers:
                logging.warning(f"{len(missing_identifiers)} CAMUS items were found but failed inference/loading. Adding them as 'Unknown'.")
                missing_df = pd.DataFrame([{output_filename_col: identifier, output_viewlabel_col: 'Unknown'}
                                           for identifier in missing_identifiers])
                output_df = pd.concat([output_df, missing_df], ignore_index=True)

            logging.info("Distribution of predicted pseudo-labels for CAMUS:")
            logging.info(f"\n{output_df[output_viewlabel_col].value_counts(dropna=False).to_string()}")

        else:
            raise RuntimeError(f"Internal Error: Unexpected dataset_type '{dataset_type}' for saving.")

        output_csv_path = args.output_csv
        if not os.path.isabs(output_csv_path):
             output_csv_path = os.path.abspath(output_csv_path)
        output_dir_csv = os.path.dirname(output_csv_path)
        if output_dir_csv: os.makedirs(output_dir_csv, exist_ok=True) # Ensure output dir exists

        output_df.to_csv(output_csv_path, index=False)
        logging.info(f"Successfully saved results CSV to: {output_csv_path}")

    except Exception as e:
        logging.error(f"FATAL: Failed to process results or save output CSV {args.output_csv}: {e}", exc_info=True)
        sys.exit(1)

    logging.info("--- Pseudo-Label Generation Finished ---")


if __name__ == "__main__":

    freeze_support()
    main()