
import cv2
import os
import pandas as pd
from tqdm import tqdm
import logging
import numpy as np
import shutil

def extract_frames(video_path, output_dir, target_size_wh, num_frames_to_extract=None, frame_rate=None):

    video_filename = os.path.basename(video_path)
    cap = None
    try:

        print(f"DEBUG [{video_filename}]: Attempting to open video: {video_path}")
        cap = cv2.VideoCapture(video_path)


        is_opened = cap.isOpened() if cap else False
        print(f"DEBUG [{video_filename}]: cv2.VideoCapture(...).isOpened(): {is_opened}")

        if not is_opened:
            logging.warning(f"Could not open video file using cv2.VideoCapture: {video_path}")
            return False


        os.makedirs(output_dir, exist_ok=True)
        width, height = target_size_wh

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames <= 0:
             logging.warning(f"Video file has invalid frame count ({total_frames}): {video_path}. Skipping.")
             if os.path.exists(output_dir) and not os.listdir(output_dir):
                 try: os.rmdir(output_dir)
                 except OSError: pass
             return False
        if video_fps <= 0:
             logging.warning(f"Video file has invalid FPS ({video_fps}): {video_path}. Assuming 30 FPS.")
             video_fps = 30


        frame_indices = []
        if num_frames_to_extract is not None and num_frames_to_extract > 0:
            if num_frames_to_extract >= total_frames: frame_indices = list(range(total_frames))
            else: indices = np.linspace(0, total_frames - 1, num=num_frames_to_extract, dtype=int); frame_indices = sorted(list(np.unique(indices)))
        elif frame_rate is not None and frame_rate > 0 and video_fps > 0:
            frame_interval = max(1, int(video_fps / frame_rate)); frame_indices = list(range(0, total_frames, frame_interval))
        else: default_num = min(16, total_frames); logging.warning(f"Defaulting to first {default_num} frames for {video_filename}."); indices = np.linspace(0, total_frames - 1, num=default_num, dtype=int); frame_indices = sorted(list(np.unique(indices)))

        if not frame_indices:
            logging.warning(f"No frames selected for extraction for {video_filename}."); cap.release(); return False

        frame_indices_set = set(frame_indices)
        extracted_count = 0
        frame_read_count = 0
        success_flag = True


        print(f"DEBUG [{video_filename}]: Starting frame reading loop. Target count: {len(frame_indices)}") # <<< ADDED
        while extracted_count < len(frame_indices):

            ret, frame = cap.read()

            if frame_read_count == 0:
                 print(f"DEBUG [{video_filename}]: First cap.read() attempt, ret = {ret}, frame is None: {frame is None}")

            if not ret:
                logging.warning(f"Video stream ended prematurely for {video_path} after reading {frame_read_count} frames. Got {extracted_count}/{len(frame_indices)}.")
                success_flag = extracted_count > 0
                break

            if frame_read_count in frame_indices_set:

                print(f"DEBUG [{video_filename}]: Processing target frame index {frame_read_count} (Extraction #{extracted_count}). Frame shape: {frame.shape if frame is not None else 'None'}")

                try:

                    if frame is None:
                        logging.error(f"Frame {frame_read_count} read as None for {video_path}. Skipping frame.")
                        success_flag = False

                        frame_read_count += 1
                        continue

                    resized_frame = cv2.resize(frame, target_size_wh, interpolation=cv2.INTER_LINEAR)
                    frame_filename = os.path.join(output_dir, f"frame_{extracted_count:04d}.png")

                    print(f"DEBUG [{video_filename}]: Attempting to write to: {frame_filename}")

                    write_success = cv2.imwrite(frame_filename, resized_frame)

                    print(f"DEBUG [{video_filename}]: cv2.imwrite() success: {write_success}")

                    if not write_success:
                        logging.error(f"Failed to write frame {extracted_count} to {frame_filename} for video {video_filename}")
                        success_flag = False; break

                    extracted_count += 1

                except Exception as e:
                    logging.error(f"Error processing/saving frame {frame_read_count} from {video_path}: {e}", exc_info=True) # Log traceback
                    success_flag = False; break

            frame_read_count += 1

        if success_flag and extracted_count > 0:
             logging.debug(f"Successfully processed {extracted_count} frames for {video_filename}")
             final_status = True
        else:
             if extracted_count == 0: logging.error(f"Extraction failed: 0 frames saved for {video_filename}.")
             else: logging.error(f"Extraction may be incomplete or failed during save for {video_filename}.")
             final_status = False

        return final_status

    except Exception as e:
        logging.error(f"General failure processing video {video_path}: {e}", exc_info=True)
        return False
    finally:

        if cap and cap.isOpened():
            cap.release()

def preprocess_videos(config):

    try: data_cfg = config['data']; prep_cfg = config['preprocessing']
    except KeyError as e: logging.error(f"Config missing section: {e}."); return


    original_metadata_filename = "FileList.csv"
    original_metadata_path = os.path.join(data_cfg.get('base_dir', 'data/EchoNetDynamic'), original_metadata_filename)


    video_dir = data_cfg.get('video_dir'); processed_dir = data_cfg.get('processed_dir', 'processed_data/echonet')
    if not video_dir: logging.error("Config 'data.video_dir' missing."); return
    if not processed_dir: logging.error("Config 'data.processed_dir' missing."); return

    try: target_size_hw = prep_cfg['img_size']; target_size_cv2 = tuple(target_size_hw[::-1])
    except Exception as e: logging.error(f"Error getting img_size from config: {e}"); return

    num_frames_extract = prep_cfg.get('num_frames_to_extract')
    frame_rate = prep_cfg.get('frame_rate') if num_frames_extract is None else None

    logging.info("--- Starting Echonet Video Preprocessing ---")
    logging.info(f"Reading video list from ORIGINAL Echonet Excel: {original_metadata_path}")
    logging.info(f"Source video directory: {video_dir}")
    logging.info(f"Output frame directory: {processed_dir}")
    if num_frames_extract: logging.info(f"Extraction strategy: {num_frames_extract} evenly spaced frames.")
    elif frame_rate: logging.info(f"Extraction strategy: Approx. {frame_rate} frames/sec.")
    else: logging.info("Extraction strategy: Default.")
    logging.info(f"Target frame size (WxH for OpenCV): {target_size_cv2}")

    if not os.path.exists(original_metadata_path): logging.error(f"Original Echonet Excel not found: {original_metadata_path}"); return
    if not os.path.exists(video_dir): logging.error(f"Video directory not found: {video_dir}"); return
    os.makedirs(processed_dir, exist_ok=True)

    try:

        df = pd.read_csv(original_metadata_path)
        logging.info(f"Successfully read csv file: {original_metadata_path}")

        if 'FileName' not in df.columns: logging.error(f"Excel file {original_metadata_path} missing 'FileName' column."); return

        filenames_to_process = df['FileName'].unique()
        logging.info(f"Found {len(filenames_to_process)} unique videos listed.")
        processed_count = 0; failed_count = 0; skipped_count = 0

        for video_name_no_ext in tqdm(filenames_to_process, desc="Preprocessing Echonet Videos"):
            video_filename = f"{video_name_no_ext}.avi"
            video_path = os.path.join(video_dir, video_filename)
            output_subdir = os.path.join(processed_dir, video_name_no_ext)

            if not os.path.exists(video_path): logging.warning(f"Video file not found: {video_path}. Skipping."); failed_count += 1; continue

            os.makedirs(output_subdir, exist_ok=True)
            existing_files = os.listdir(output_subdir) if os.path.exists(output_subdir) else []

            if existing_files:
                 current_files_count = len(existing_files)
                 target_frame_count = num_frames_extract if num_frames_extract else 1
                 if current_files_count >= target_frame_count:
                     logging.debug(f"Frames for {video_filename} seem complete ({current_files_count} >= {target_frame_count}). Skipping.")
                     skipped_count += 1; continue
                 else:
                      logging.warning(f"Output dir {output_subdir} exists but incomplete ({current_files_count} < {target_frame_count}). Re-processing.")

            if extract_frames(video_path, output_subdir, target_size_cv2, num_frames_extract, frame_rate):
                 processed_count += 1
            else:
                 failed_count += 1

                 if os.path.exists(output_subdir):
                      try:
                           shutil.rmtree(output_subdir)
                           logging.warning(f"Removed empty/failed output directory: {output_subdir}")
                      except Exception as e_rm:
                           logging.error(f"Failed to remove output directory {output_subdir} after processing failure: {e_rm}")


        logging.info(f"--- Echonet Preprocessing Finished ---")
        logging.info(f"Successfully processed/verified: {processed_count + skipped_count} videos.")
        logging.info(f"Newly processed this run: {processed_count} videos.")
        logging.info(f"Skipped (already processed): {skipped_count} videos.")
        if failed_count > 0: logging.warning(f"Failed/Not Found: {failed_count} videos.")

    except ImportError: logging.error("Failed reading Excel. Install 'openpyxl': pip install openpyxl")
    except FileNotFoundError as e: logging.error(f"File not found error during processing: {e}")
    except Exception as e: logging.error(f"Error during Echonet preprocessing loop: {e}", exc_info=True)