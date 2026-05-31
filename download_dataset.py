import os
import sys
import zipfile
import shutil

DATASET_URL = "https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a"
TARGET_DIR = os.path.join("data", "ECHONET-Dynamic")
REQUIRED_FILES = [
    "VolumeTracings.csv",
    "FileList.csv"
]

def print_header():
    print("=" * 70)
    print("      EchoNet-Dynamic Dataset Download & Preparation Wizard")
    print("=" * 70)
    print("\nIn compliance with Stanford AIMI dataset usage policy, you must")
    print("request access and download the dataset files directly from AIMI.\n")
    print(f"Official Dataset Link:\n{DATASET_URL}\n")
    print("=" * 70 + "\n")

def check_existing():
    if not os.path.exists(TARGET_DIR):
        return False
    
    # Check for VolumeTracings.csv and the Videos folder
    csv_exists = any(os.path.exists(os.path.join(TARGET_DIR, req)) for req in REQUIRED_FILES)
    videos_exist = os.path.isdir(os.path.join(TARGET_DIR, "Videos"))
    
    if csv_exists and videos_exist:
        return True
    return False

def verify_structure():
    print(f"\nVerifying files in '{TARGET_DIR}':")
    
    missing = []
    # Check for at least one expected metadata CSV
    found_csv = False
    for req in REQUIRED_FILES:
        path = os.path.join(TARGET_DIR, req)
        if os.path.exists(path):
            print(f"  [✓] Found {req}")
            found_csv = True
        else:
            missing.append(req)
            
    if not found_csv:
        print(f"  [✗] Missing metadata CSV files (either FileList.csv or VolumeTracings.csv)")
        
    videos_dir = os.path.join(TARGET_DIR, "Videos")
    if os.path.isdir(videos_dir):
        video_count = len([f for f in os.listdir(videos_dir) if f.lower().endswith(('.avi', '.mp4'))])
        print(f"  [✓] Found Videos directory with {video_count} video(s)")
        if video_count == 0:
            print("      WARNING: Videos directory is empty!")
    else:
        print(f"  [✗] Missing Videos/ directory")
        missing.append("Videos/")
        
    if not found_csv or not os.path.isdir(videos_dir):
        print("\nDataset configuration is INCOMPLETE.")
        return False
    
    print("\nDataset configuration is COMPLETE and ready for preprocessing/training.")
    return True

def extract_zip(zip_path):
    print(f"\nExtracting '{zip_path}' to '{TARGET_DIR}'...")
    try:
        os.makedirs(TARGET_DIR, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # We list the archive contents to see if there's a nested root folder
            namelist = zip_ref.namelist()
            root_dirs = set(name.split('/')[0] for name in namelist if '/' in name)
            
            # If the zip contains a root folder like 'VolumeTracings.csv' nested inside 'EchoNet-Dynamic/'
            has_nested_root = len(root_dirs) == 1 and list(root_dirs)[0].lower() in ['echonet-dynamic', 'echonet']
            
            if has_nested_root:
                nested_root = list(root_dirs)[0]
                print(f"Detected nested root directory '{nested_root}' in zip. Extracting contents directly...")
                for member in zip_ref.infolist():
                    if member.filename.startswith(nested_root + '/'):
                        # Remove the nested root prefix
                        old_path = member.filename
                        new_path = old_path.split('/', 1)[1]
                        if not new_path:  # Skip the directory itself
                            continue
                        
                        target_path = os.path.join(TARGET_DIR, new_path)
                        if member.is_dir():
                            os.makedirs(target_path, exist_ok=True)
                        else:
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                                shutil.copyfileobj(source, target)
            else:
                zip_ref.extractall(TARGET_DIR)
                
        print("Extraction complete!")
        return True
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False

def main():
    print_header()
    
    if check_existing():
        print(f"Dataset already found in '{TARGET_DIR}'!")
        if verify_structure():
            sys.exit(0)
            
    print("Steps to prepare the dataset:")
    print(f"1. Open your browser and register at:\n   {DATASET_URL}")
    print("2. Once approved, download the dataset zip archive.")
    print("3. You can either place the extracted contents directly into:")
    print(f"   {os.path.abspath(TARGET_DIR)}")
    print("   OR you can provide the path to your downloaded zip file below to extract it automatically.")
    
    choice = input("\nDo you want to extract a downloaded ZIP file now? (y/n): ").strip().lower()
    if choice == 'y':
        zip_path = input("Enter the path to the downloaded dataset ZIP file: ").strip().strip('"').strip("'")
        if not os.path.exists(zip_path):
            print(f"File not found: {zip_path}")
            sys.exit(1)
            
        if extract_zip(zip_path):
            verify_structure()
    else:
        print(f"\nPlease manually place the dataset files in '{TARGET_DIR}/' and re-run this script to verify.")
        
if __name__ == "__main__":
    main()
