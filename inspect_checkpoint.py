import torch
import os

checkpoint_path = 'results/view_classifier_echonet/checkpoints/best_model.pth.tar'
if not os.path.exists(checkpoint_path):
    print(f"Propblem: {checkpoint_path} does not exist.")
else:
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        print("Keys in checkpoint:")
        print(list(state_dict.keys())[:20]) # Print first 20 keys
        
        has_ef = any('ef_regressor' in k for k in state_dict.keys())
        print(f"\nHas EF regressor keys? {has_ef}")
        
        if has_ef:
            print("Found EF keys!")
            for k in state_dict.keys():
                if 'ef_regressor' in k:
                    print(k)
    except Exception as e:
        print(f"Error loading: {e}")
