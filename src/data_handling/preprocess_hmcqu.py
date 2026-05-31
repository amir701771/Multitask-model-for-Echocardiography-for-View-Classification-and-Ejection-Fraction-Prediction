# src/data_handling/preprocess_hmcqu.py
import os
import logging
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import re
import glob
import sys

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    logging.debug("SimpleITK not found. Will not be able to process MHD/NIFTI formats.")


def preprocess_data(config):
    """
    Preprocesses HMC-QU dataset. Reads metadata (Excel), finds raw video/image files,
    extracts a specified frame (e.g., middle), resizes, and saves as PNG using
    a consistent naming scheme: PatientID_View_FrameType.png

    Args:
        config (dict): Configuration dictionary loaded from config_hmcqu_preprocessing.yaml.
    """
    logging.info("--- Starting HMC-QU Preprocessing ---")

    # --- Get config values safely ---
    try:
        data_cfg = config['data']
        prep_cfg = config['preprocessing']
        metadata_base_dir = data_cfg['metadata_base_dir']
        raw_data_root = data_cfg['raw_data_root']
        processed_dir = data_cfg['processed_dir']
        filename_col = data_cfg['filename_col'] # <<< Get expected col name from config
        img_size_prep = tuple(prep_cfg['img_size'])
        frame_type_to_extract = prep_cfg.get('frame_type_to_extract', 'middle').upper()
        output_format = prep_cfg.get('output_format', 'png')
        view_mapping_from_excel = data_cfg.get('view_mapping', {'A2C': 0, 'A4C': 1})

        if not os.path.isdir(raw_data_root): raise FileNotFoundError(f"Raw data root dir not found: {raw_data_root}")
        os.makedirs(processed_dir, exist_ok=True)

    except KeyError as e: logging.error(f"Missing key: {e}"); raise ValueError(f"Config missing key: {e}") from e
    except FileNotFoundError as e: logging.error(f"{e}"); raise e
    except Exception as e: logging.error(f"Config error: {e}"); raise e

    logging.info(f"Metadata base directory: {metadata_base_dir}")
    logging.info(f"Raw data root directory: {raw_data_root}")
    logging.info(f"Processed frames output directory: {processed_dir}")
    logging.info(f"Target image size: {img_size_prep}")
    logging.info(f"Frame selection type: {frame_type_to_extract}")
    logging.info(f"Reading identifier column: '{filename_col}'") # Log expected name

    # --- Core Preprocessing Logic ---
    processed_count = 0; error_count = 0
    files_processed_successfully = set(); failed_base_ids = set()

    for view_name_actual, _ in view_mapping_from_excel.items():
        excel_file = f"{view_name_actual}.xlsx"
        excel_path = os.path.join(metadata_base_dir, excel_file)

        if not os.path.exists(excel_path):
            logging.warning(f"Metadata file not found: {excel_path}. Skipping view {view_name_actual}.")
            continue
        try:
            df_view = pd.read_excel(excel_path, header=1)
            logging.debug(f"Columns found in {excel_file}: {list(df_view.columns)}")

            # =============================================
            # <<< USE filename_col READ FROM CONFIG >>>
            # Check if the column name specified in the config exists in the DataFrame
            if filename_col not in df_view.columns:
                 # Log using the expected name from config and the actual columns found
                 logging.warning(f"Expected column '{filename_col}' (from config) not found in {excel_path} columns: {list(df_view.columns)}. Skipping file.")
                 continue # Skip this Excel file if the specified column is missing
            # =============================================

            # If we reach here, the column exists (using the name from config, e.g., "Unnamed: 0")
            logging.info(f"Processing {len(df_view)} entries from {excel_file} for view {view_name_actual} using column '{filename_col}'")

            unique_ids_in_excel = df_view[filename_col].dropna().astype(str).str.strip().unique()

            for id_str in tqdm(unique_ids_in_excel, desc=f"Processing {view_name_actual}"):
                base_id = id_str.replace('*','')
                if base_id in failed_base_ids: continue

                processed_key = f"{base_id}_{view_name_actual}_{frame_type_to_extract}"
                if processed_key in files_processed_successfully: continue

                patient_id_match = re.match(r"(ES\d+)", base_id)
                patient_id = patient_id_match.group(1) if patient_id_match else base_id

                raw_file_path = None
                try:
                    search_pattern = os.path.join(raw_data_root, view_name_actual, f"{base_id}.*")
                    logging.debug(f"Searching for pattern: {search_pattern}")
                    found_files = glob.glob(search_pattern)
                    logging.debug(f"Glob found for '{base_id}': {found_files}")

                    if not found_files:
                        logging.debug(f"Raw file not found. Skipping ID: {base_id}")
                        failed_base_ids.add(base_id); continue

                    if len(found_files) > 1:
                         video_files = [f for f in found_files if f.lower().endswith(('.avi', '.mp4', '.mov', '.wmv', '.mkv'))]
                         if video_files: raw_file_path = video_files[0]; logging.warning(f"Multiple files found for '{base_id}'. Using video: {raw_file_path}")
                         else: raw_file_path = found_files[0]; logging.warning(f"Multiple files found for '{base_id}'. Using first: {raw_file_path}")
                    else: raw_file_path = found_files[0]

                    raw_file_suffix_actual = os.path.splitext(raw_file_path)[1].lower()
                    logging.debug(f"Attempting to process raw file: {raw_file_path}")

                    output_patient_dir = os.path.join(processed_dir, patient_id)
                    os.makedirs(output_patient_dir, exist_ok=True)
                    output_filename = f"{patient_id}_{view_name_actual}_{frame_type_to_extract}.{output_format}"
                    output_path = os.path.join(output_patient_dir, output_filename)

                    if os.path.exists(output_path):
                        logging.debug(f"Output exists: {output_path}. Skipping.")
                        files_processed_successfully.add(processed_key); processed_count += 1
                        continue

                    # --- Load and Extract Frame ---
                    img_pil_rgb = None; frame_loaded = False; extracted_frame_info = frame_type_to_extract
                    # ... (Keep VIDEO, MHD/NIFTI, STATIC IMAGE loading logic as before) ...
                    # VIDEO
                    if raw_file_suffix_actual in ['.avi', '.mp4', '.mov', '.wmv', '.mkv']:
                        if 'cv2' not in sys.modules: raise ModuleNotFoundError("OpenCV (cv2) required.")
                        cap = cv2.VideoCapture(raw_file_path); frame_to_save = None; frame_idx = -1
                        if not cap.isOpened(): raise IOError(f"Cannot open video: {raw_file_path}")
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if total_frames > 0:
                            if frame_type_to_extract == 'middle': frame_idx = total_frames // 2
                            elif frame_type_to_extract == 'first': frame_idx = 0
                            else: frame_idx = total_frames // 2; logging.warning(f"Unsupported frame_type, using middle.")
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx); ret, frame = cap.read()
                            if ret: frame_to_save = frame; extracted_frame_info = f"frame {frame_idx}"
                        cap.release()
                        if frame_to_save is not None: img_pil_rgb = Image.fromarray(cv2.cvtColor(frame_to_save, cv2.COLOR_BGR2RGB)); frame_loaded = True
                        else: logging.warning(f"Could not extract frame {frame_idx} from video: {raw_file_path}")

                    # MHD/NIFTI
                    elif SITK_AVAILABLE and raw_file_suffix_actual in ['.mhd', '.nii', '.nii.gz']:
                        sitk_image = sitk.ReadImage(raw_file_path); img_array = sitk.GetArrayFromImage(sitk_image)
                        selected_slice_idx = 0; slice_info = "N/A"
                        if img_array.ndim == 3: # [D, H, W]
                            num_slices = img_array.shape[0]
                            if frame_type_to_extract == 'ED': selected_slice_idx = 0
                            elif frame_type_to_extract == 'ES': selected_slice_idx = num_slices - 1
                            else: selected_slice_idx = num_slices // 2
                            selected_slice_idx = max(0, min(selected_slice_idx, num_slices - 1))
                            img_array_2d = img_array[selected_slice_idx, :, :]; slice_info = f"slice {selected_slice_idx}/{num_slices}"
                        elif img_array.ndim == 2: img_array_2d = img_array; slice_info = "2D"
                        else: raise ValueError(f"Unsupported image dim {img_array.ndim}")
                        img_pil = Image.fromarray(img_array_2d); img_pil_rgb = img_pil.convert("RGB"); frame_loaded = True
                        extracted_frame_info = slice_info

                    # STATIC IMAGE
                    elif raw_file_suffix_actual in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
                         img_pil = Image.open(raw_file_path); img_pil_rgb = img_pil.convert("RGB"); frame_loaded = True
                         extracted_frame_info = "static image"
                    else:
                        logging.warning(f"Unsupported format: {raw_file_suffix_actual}. Skipping.")
                        failed_base_ids.add(base_id); continue

                    # --- Resize and Save ---
                    if frame_loaded and img_pil_rgb:
                         img_resized = img_pil_rgb.resize(img_size_prep[::-1], Image.Resampling.LANCZOS)
                         img_resized.save(output_path)
                         files_processed_successfully.add(processed_key)
                         processed_count += 1
                         logging.debug(f"Saved {output_filename} (extracted {extracted_frame_info})")
                    elif not frame_loaded: error_count += 1; failed_base_ids.add(base_id)

                except Exception as process_e:
                     logging.error(f"Error processing ID {id_str} (raw file: {raw_file_path}): {process_e}", exc_info=False)
                     error_count += 1
                     failed_base_ids.add(base_id)

        except ImportError: logging.error(f"Reading Excel file {excel_path} failed. Install 'openpyxl'."); raise
        except Exception as e: logging.error(f"Error processing entries from {excel_path}: {e}", exc_info=True)

    final_skipped_count = len(failed_base_ids)

    logging.info(f"--- HMC-QU Preprocessing Summary ---")
    logging.info(f"Successfully processed/verified frames for: {processed_count} PatientID-View-FrameType instances.")
    logging.info(f"Skipped/Errored unique IDs: {final_skipped_count}")
    # logging.info(f"Errors during frame processing/saving: {error_count}") # Covered by skipped count
    logging.info(f"Processed frames saved in: {processed_dir}")
    if final_skipped_count > 0:
        logging.warning("Some files were skipped or encountered errors. Check detailed (DEBUG level) logs.")