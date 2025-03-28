import torch
import numpy as np
import matplotlib.pyplot as plt
from cnn_models import FullFeaturesResNet
from data_model import SepctraDataset, collate_fn
from torch.utils.data import DataLoader
import os
import glob

# Load the model
model_path = 'model/cnn_saved_models/2025-03-24_13-30-03_model.pth'
model = FullFeaturesResNet(NUM_CLASSES=3)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load a subset of data
base_dir = 'data/full_zwarning'
file_paths = glob.glob(os.path.join(base_dir, '*/*.pkl'))[:200] 
dataset = SepctraDataset(file_paths)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)

def compute_feature_importance(model, dataloader):
    feature_importance = None
    num_samples = 0
    
    for batch in dataloader:
        features, _ = batch
    
        features.requires_grad = True
        
        outputs = model(features)
        outputs.sum().backward()
        
        if feature_importance is None:
            feature_importance = torch.abs(features.grad).sum(dim=0).cpu().numpy()
        else:
            feature_importance += torch.abs(features.grad).sum(dim=0).cpu().numpy()
        
        num_samples += features.size(0)
    
    return feature_importance / num_samples

# Compute feature importance
feature_importance = compute_feature_importance(model, dataloader)

# Adjust feature names to match the number of importance values
num_importance_values = len(feature_importance[0])
feature_names = ['flux', 'loglam', 'ivar', 'model','PLATESN2', 'Z', 
                 'Z_ERR', 'ZWARNING', 'RCHI2', 'PLATEQUALITY', 'snr_uv', 
                 'snr_r', 'snr_nir', 'snr_ir']


# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(feature_names, feature_importance[0])
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
