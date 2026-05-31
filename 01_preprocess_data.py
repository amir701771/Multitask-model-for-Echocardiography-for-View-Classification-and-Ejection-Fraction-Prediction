
import logging
import argparse
import os
import sys
from src.utils.helpers import load_config, setup_logging


from src.data_handling.preprocess_echonet import preprocess_videos as preprocess_echonet_videos

try:
    from src.data_handling.preprocess_hmcqu import preprocess_data as preprocess_hmcqu_data
except ImportError:
    logging.warning("preprocess_hmcqu.py or preprocess_data function not found. HMC-QU processing will fail.")
    def preprocess_hmcqu_data(config):
        raise NotImplementedError("HMC-QU preprocessing function is not implemented in src/data_handling/preprocess_hmcqu.py")

def main():
    parser = argparse.ArgumentParser(description="Run preprocessing for specified datasets.")

    parser.add_argument("--dataset", type=str, required=True, choices=['echonet', 'hmcqu', 'all'],
                        help="Which dataset to preprocess ('echonet', 'hmcqu', or 'all').")
    parser.add_argument("--config_echo", type=str, default="config/config_echonet_multitask.yaml",
                        help="Path to Echonet config (used for paths/settings).")

    parser.add_argument("--config_hmcqu", type=str, default="config/config_hmcqu_preprocessing.yaml",
                        help="Path to HMC-QU preprocessing config file.")
    args = parser.parse_args()

    log_dir = "results/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'preprocess_{args.dataset}.log')
    setup_logging(log_file=log_file, level=logging.INFO)

    logging.info(f"--- Starting Preprocessing for: {args.dataset.upper()} ---")

    config_echo = None
    config_hmcqu = None
    try:

        if args.dataset in ['echonet', 'all']:
            config_echo = load_config(args.config_echo)
            logging.info(f"Loaded Echonet config: {args.config_echo}")
        if args.dataset in ['hmcqu', 'all']:
            config_hmcqu = load_config(args.config_hmcqu)
            logging.info(f"Loaded HMC-QU config: {args.config_hmcqu}")
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}", exc_info=True)
        sys.exit(1)

    datasets_processed_successfully = []

    if args.dataset in ['echonet', 'all'] and config_echo:
        logging.info("--- Running Echonet Preprocessing ---")
        try:
            preprocess_echonet_videos(config_echo)
            logging.info("--- Echonet Preprocessing Finished ---")
            datasets_processed_successfully.append("Echonet")
        except Exception as e:
            logging.error(f"Echonet preprocessing failed: {e}", exc_info=True)
            if args.dataset == 'all':
                 logging.error("Stopping script due to Echonet failure while processing 'all'.")
                 sys.exit(1)

    if args.dataset in ['hmcqu', 'all'] and config_hmcqu:
        logging.info("--- Running HMC-QU Preprocessing ---")
        try:

            preprocess_hmcqu_data(config_hmcqu)
            logging.info("--- HMC-QU Preprocessing Finished ---")
            datasets_processed_successfully.append("HMC-QU")
        except NotImplementedError as nie:
             logging.error(f"HMC-QU preprocessing cannot proceed: {nie}")
             if args.dataset == 'all': sys.exit(1)
        except Exception as e:
            logging.error(f"HMC-QU preprocessing failed: {e}", exc_info=True)
            if args.dataset == 'all':
                 logging.error("Stopping script due to HMC-QU failure while processing 'all'.")
                 sys.exit(1)

    logging.info(f"--- Preprocessing Script Finished. Successfully processed: {', '.join(datasets_processed_successfully) if datasets_processed_successfully else 'None'} ---")

if __name__ == "__main__":

    main()