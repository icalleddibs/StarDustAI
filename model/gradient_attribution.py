import torch
import numpy as np
import matplotlib.pyplot as plt
from cnn_experiments.cnn_models import FullFeaturesResNet
from data_model import SpectraDataset, collate_fn
from torch.utils.data import DataLoader
import os
import glob

# Load the model
model_path = 'model/cnn_experiments/cnn_models_experiment_results/2025-03-24_13-30-03_model.pth'
model = FullFeaturesResNet(NUM_CLASSES=3)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load a subset of data
base_dir = 'data/full_zwarning'
file_paths = glob.glob(os.path.join(base_dir, '*/*.pkl'))[:300] 
dataset = SpectraDataset(file_paths)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)

def compute_feature_importance(model, dataloader):
    """
    Compute the feature importance using gradient attribution method.
    This method computes the gradient of the model's output with respect to the input features.
    The absolute value of the gradient is averaged over all samples to get the feature importance.

    Inputs:
        model: model to check feature importance for
        dataloader: dataloader to load the data

    Returns:
        feature_importance: array of feature importance scores
    """
    
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

feature_names = ['flux', 'loglam', 'Flux Inverse Variance', 'model','SNR', 'Redshift', 
                 'Redshift Error', 'ZWARNING', 'Reduced Chi^2', 'PLATEQUALITY', 'UV SNR', 
                 'Red SNR', 'Red IR SNR', 'IR SNR']

final_features = []
for f in feature_names:
    final_features.append(feature_importance[0][feature_names.index(f)])

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(feature_names[2:], final_features[2:], color = 'purple')
plt.title('Feature Importance',fontsize=20)
plt.xlabel('Features',fontsize=14)
plt.ylabel('Importance Score', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
