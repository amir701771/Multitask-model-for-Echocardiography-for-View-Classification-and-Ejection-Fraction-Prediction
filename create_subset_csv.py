# create_subset_csv.py
import pandas as pd
import numpy as np
import os
import argparse # Added argparse for flexibility

# --- Configuration via Arguments ---
parser = argparse.ArgumentParser(description="Create a random subset of a CSV file.")
parser.add_argument('--input_csv', type=str, required=True, help='Path to the full input CSV file.')
parser.add_argument('--output_csv', type=str, required=True, help='Path to save the new subset CSV file.')
parser.add_argument('--subset_size', type=int, required=True, help='Desired number of samples in the subset.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling.')
args = parser.parse_args()

# Use paths from arguments
full_csv_path = args.input_csv
subset_size = args.subset_size
subset_csv_path = args.output_csv
random_seed = args.seed
# --- End Configuration ---

print(f"Reading full dataset from: {full_csv_path}")
if not os.path.exists(full_csv_path):
    print(f"ERROR: Full CSV file not found at {full_csv_path}")
    exit()

try:
    # Read the full CSV
    df_full = pd.read_csv(full_csv_path)
    print(f"Full dataset size: {len(df_full)} samples.")

    actual_subset_size = min(subset_size, len(df_full)) # Don't sample more than available

    if actual_subset_size == len(df_full):
        print(f"Subset size ({subset_size}) is >= full dataset size. Copying the full file instead.")
        df_subset = df_full
    else:
        print(f"Sampling {actual_subset_size} random samples (seed={random_seed})...")
        # Sample randomly without replacement
        df_subset = df_full.sample(n=actual_subset_size, random_state=random_seed, replace=False)
        print(f"Subset created with {len(df_subset)} samples.")

    # Ensure the output directory exists
    output_dir = os.path.dirname(subset_csv_path)
    if output_dir: # Check if path includes a directory
         os.makedirs(output_dir, exist_ok=True)

    # Save the subset to the new CSV file
    print(f"Saving subset to: {subset_csv_path}")
    df_subset.to_csv(subset_csv_path, index=False)
    print("Subset CSV file created successfully.")

except FileNotFoundError: # Catch specific error just in case os.path.exists missed something racey
    print(f"ERROR: File not found during processing: {full_csv_path}")
except Exception as e:
    print(f"An error occurred: {e}")