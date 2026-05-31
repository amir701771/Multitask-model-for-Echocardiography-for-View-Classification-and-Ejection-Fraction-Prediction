
import SimpleITK as sitk # For reading .nii/.mhd files
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import logging
import cv2 # For resizing
import glob # For finding files easily



def read_nifti_or_mhd(file_path):

    if not os.path.exists(file_path):
        logging.error(f"Image file not found: {file_path}")
        return None
    try:
        itk_image = sitk.ReadImage(file_path)
        np_image = sitk.GetArrayFromImage(itk_image)

        if np_image.ndim == 3:
            slice_idx = np_image.shape[0] // 2
            img_slice = np_image[slice_idx, :, :]
            logging.debug(f"Input NIfTI shape {np_image.shape}, taking slice {slice_idx}")
        elif np_image.ndim == 4:
            time_idx = np_image.shape[0] // 2
            slice_idx = np_image.shape[1] // 2
            img_slice = np_image[time_idx, slice_idx, :, :]
            logging.debug(f"Input NIfTI shape {np_image.shape}, taking time {time_idx}, slice {slice_idx}")
        elif np_image.ndim == 2: # Assume (Y, X)
            img_slice = np_image
        else:
            logging.warning(f"Unexpected image dimension {np_image.ndim} for {file_path}. Skipping.")
            return None


        return img_slice

    except Exception as e:
        logging.error(f"Failed to read/process image file {file_path}: {e}", exc_info=True)
        return None

def preprocess_camus_images(config):

    data_cfg = config['data']
    prep_cfg = config['preprocessing']
    view_mapping = data_cfg['view_mapping']

    base_dir = data_cfg['base_dir'] #  "data/camus"
    image_source_dir = os.path.join(base_dir, data_cfg['image_dir']) # e.g., "data/camus/database_nifti"
    processed_dir = data_cfg['processed_dir'] #  "processed_data/camus"
    target_size_hw = tuple(prep_cfg['img_size'])
    target_size_wh = tuple(prep_cfg['img_size'][::-1])
    input_ext = prep_cfg.get('file_extension', '.nii')

    logging.info("--- Starting CAMUS Preprocessing ---")
    logging.info(f"Reading images from: {image_source_dir}")
    logging.info(f"Saving processed PNGs to: {processed_dir}")
    logging.info(f"Target image size (HxW): {target_size_hw}")
    logging.info(f"Processing views: {list(view_mapping.keys())}")
    logging.info(f"Processing frames: ED, ES")

    if not os.path.isdir(image_source_dir):
        logging.error(f"Cannot find CAMUS image source directory: {image_source_dir}")
        return

    os.makedirs(processed_dir, exist_ok=True)
    processed_count = 0; failed_count = 0; skipped_count = 0


    patient_folders = sorted([d for d in os.listdir(image_source_dir) if d.startswith('patient') and os.path.isdir(os.path.join(image_source_dir, d))])
    logging.info(f"Found {len(patient_folders)} patient folders in {image_source_dir}.")

    if not patient_folders:
        logging.error("No patient folders found. Check 'image_dir' in config and data placement.")
        return


    for patient_id in tqdm(patient_folders, desc="Processing CAMUS Patients"):
        patient_path = os.path.join(image_source_dir, patient_id)

        for view_name in view_mapping.keys():
            for frame_type in ['ED', 'ES']:


                input_filename_stem = f"{patient_id}_{view_name}_{frame_type}"
                input_filename = input_filename_stem + input_ext
                input_path = os.path.join(patient_path, input_filename)


                output_subdir = os.path.join(processed_dir, patient_id)
                output_filename = f"{input_filename_stem}.png"
                output_path = os.path.join(output_subdir, output_filename)

                os.makedirs(output_subdir, exist_ok=True)


                if os.path.exists(output_path):
                    skipped_count += 1
                    logging.debug(f"Skipping existing: {output_path}")
                    continue


                if not os.path.exists(input_path):
                    logging.warning(f"Input file not found for {patient_id}, {view_name}, {frame_type}. Searched: {input_path}")
                    failed_count += 1
                    continue


                img_np = read_nifti_or_mhd(input_path)

                if img_np is None: # Reading failed
                    failed_count += 1
                    continue # Skip to next file


                try:

                    resized_img = cv2.resize(img_np, target_size_wh, interpolation=cv2.INTER_LINEAR)

                    if resized_img.dtype != np.uint8 or resized_img.max() > 255:

                         if resized_img.max() > 0:
                             resized_img_norm = cv2.normalize(resized_img, None, 0, 255, cv2.NORM_MINMAX)
                         else:
                             resized_img_norm = resized_img # Already all zero
                         resized_img_8u = resized_img_norm.astype(np.uint8)
                    else:
                         resized_img_8u = resized_img.astype(np.uint8)


                    cv2.imwrite(output_path, resized_img_8u)
                    processed_count += 1
                    logging.debug(f"Processed and saved: {output_path}")

                except Exception as e:
                    logging.warning(f"Failed to resize/save image {input_path} -> {output_path}: {e}")
                    failed_count += 1


    logging.info(f"--- CAMUS Preprocessing Finished ---")
    logging.info(f"Successfully processed: {processed_count} images.")
    logging.info(f"Skipped (already existed): {skipped_count} images.")
    if failed_count > 0:
        logging.warning(f"Failed/Not Found: {failed_count} images (check logs for details).")