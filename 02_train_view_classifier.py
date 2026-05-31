
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import argparse
import os
import sys
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import freeze_support
import importlib

# Add src to path just in case
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.utils.helpers import load_config, setup_logging, set_seed, save_checkpoint
from src.models.cnn_view_classifier import ViewClassifierCNN
# We don't import train_model from trainer.py to avoid arguments mismatch and logic issues.

def main():
    parser = argparse.ArgumentParser(description="Train View Classifier (Reconstructed).")
    parser.add_argument("--config", type=str, default="config/quick_train_echonet.yaml", help="Path to config.")
    args = parser.parse_args()

    # --- Setup ---
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    
    # Setup Output
    output_dir = config['training']['output_dir']
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    setup_logging(log_file=os.path.join(output_dir, 'logs', 'train_view_classifier.log'), level=logging.INFO)
    logging.info(f"--- Starting Training (Self-Contained Loop) ---")
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    
    # --- Data Loading ---
    data_cfg = config['data']
    dataset_module_name = data_cfg['dataset_module']
    dataset_class_name = data_cfg['dataset_class']
    
    try:
        module = importlib.import_module(dataset_module_name)
        DatasetClass = getattr(module, dataset_class_name)
    except Exception as e:
        logging.error(f"Failed to load dataset class {dataset_class_name} from {dataset_module_name}: {e}")
        sys.exit(1)
        
    logging.info(f"Loading datasets...")
    train_dataset = DatasetClass(config=data_cfg, split='train')
    val_dataset = DatasetClass(config=data_cfg, split='val')

    # Limit samples if requested in config (for quick testing)
    if 'limit_samples' in data_cfg:
        limit = data_cfg['limit_samples']
        logging.info(f"Limiting training samples to {limit} as requested in config.")
        if hasattr(train_dataset, 'df') and not train_dataset.df.empty:
             train_dataset.df = train_dataset.df.iloc[:limit]
        if hasattr(val_dataset, 'df') and not val_dataset.df.empty:
             # Also limit val to keep it proportional or just small
             val_limit = max(int(limit * 0.2), 10)
             val_dataset.df = val_dataset.df.iloc[:val_limit]
             logging.info(f"Limiting validation samples to {val_limit}.")
    
    # Calculate Weights if needed (mimicking original script)
    labels = []
    # Quick label extraction for weighting (might be slow for huge datasets but OK for this task)
    # We can skip if not strictly critical, but let's try to be correct.
    # Actually, let's assume balanced or use what we saw in logs: [1.02, 0.98] approx equal.
    # To save time in reconstruction, we'll skip pre-calculation unless needed.
    # Let's use simple CrossEntropy for now.
    
    batch_size = config['training'].get('batch_size', 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # --- Model ---
    model_cfg = config['model']
    backbone = model_cfg.get('backbone', 'resnet18')
    num_views = data_cfg.get('num_views', 2)
    
    model = ViewClassifierCNN(
        backbone_name=backbone,
        pretrained=model_cfg.get('pretrained', True),
        num_view_classes=num_views,
        dropout_rate=model_cfg.get('dropout_rate', 0.3)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # --- Training Loop ---
    num_epochs = config['training']['epochs']
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for batch_idx, (inputs, targets, ef_dummy) in enumerate(loop):
            # inputs shape: [B, C, T, H, W] (from EchoNetViewDataset)
            # targets shape: [B]
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Handle Video -> 2D CNN (ResNet18)
            # If backbone is 2D, we expect [N, 3, H, W].
            # Input is [B, 3, T, H, W].
            # Permute to [B, T, 3, H, W] -> Reshape -> [B*T, 3, H, W]
            
            b, c, t, h, w = inputs.shape
            
            # Reshape for 2D backbone
            x = inputs.permute(0, 2, 1, 3, 4).contiguous() # [B, T, C, H, W]
            x = x.view(b * t, c, h, w) # [B*T, C, H, W]
            
            optimizer.zero_grad()
            logits = model(x) # [B*T, NumClasses]
            
            # Aggregate back to Video level (Average Logits)
            logits = logits.view(b, t, -1) # [B, T, NumClasses]
            avg_logits = logits.mean(dim=1) # [B, NumClasses]
            
            loss = criterion(avg_logits, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * b
            
            _, predicted = torch.max(avg_logits, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        train_loss = running_loss / total if total > 0 else 0
        train_acc = correct / total if total > 0 else 0
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets, ef_dummy in tqdm(val_loader, desc="[Val]", leave=False):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                b, c, t, h, w = inputs.shape
                x = inputs.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
                
                logits = model(x)
                logits = logits.view(b, t, -1)
                avg_logits = logits.mean(dim=1)
                
                loss = criterion(avg_logits, targets)
                val_loss += loss.item() * b
                
                _, predicted = torch.max(avg_logits, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
        final_val_loss = val_loss / val_total if val_total > 0 else 0
        final_val_acc = val_correct / val_total if val_total > 0 else 0
        
        logging.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {final_val_loss:.4f} Acc: {final_val_acc:.4f}")
        
        # Save Best
        if final_val_acc > best_acc:
            best_acc = final_val_acc
            save_path = os.path.join(output_dir, 'checkpoints', 'best_model.pth.tar')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, save_path)
            logging.info(f"Saved New Best Model to {save_path}")

    # Save Last
    last_path = os.path.join(output_dir, 'checkpoints', 'last_checkpoint.pth.tar')
    torch.save({
        'epoch': num_epochs,
        'state_dict': model.state_dict(),
    }, last_path)
    
    logging.info("Training Finished.")

if __name__ == "__main__":
    freeze_support()
    main()
