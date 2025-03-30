import torch
import torchinfo
import itertools
from cnn_experiments.cnn_models import EarlyStopping, FocalLoss, DilatedFullFeaturesCNN, FullFeaturesResNet, FullFeaturesCNNMoreLayers
from cnn_training import train, evaluate, save_model
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from data_model import SpectraDataset, collate_fn
import glob
import os
import subprocess
import random

# Get repo root and dataset
repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
base_dir = os.path.join(repo_root, "data/full_zwarning")
file_paths = glob.glob(os.path.join(base_dir, "*/*.pkl"))

if not file_paths:
    raise ValueError("No FITS files found in 'data/full_zwarning/'")

random.shuffle(file_paths)
dataset = SpectraDataset(file_paths)

# Split dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Hyperparameter search space
param_space = {
    "learning_rate": [0.005, 0.001],
    "dropout": [0.4, 0.5],
    "weight_decay": [0.0001, 0.001],
    "dilation": [2, 3, 4]
}

batch_size = 256
full_trials, dil_trials = 30, 30
best_params_full, best_params_dil = None, None
best_acc_full, best_acc_dil = 0,0

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

### Full Features ResNet
for _ in range(full_trials):
    # Randomly sample parameters
    params = {key: random.choice(values) for key, values in param_space.items()}
    lr, dropout, wd = (params["learning_rate"], params["dropout"], params["weight_decay"])
    print(f"\nTesting: Model= FullFeaturesCNN, lr={lr}, dropout={dropout}, weight_decay={wd}")
    
    # Model, loss, optimizer
    model = FullFeaturesResNet(NUM_CLASSES=3, dropout_rate=dropout)
    model.train()
    print(torchinfo.summary(model))
    criterion = FocalLoss(alpha=[0.2, 0.3, 0.5], gamma=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

    # Train and evaluate
    training_time = train(model, criterion, optimizer, scheduler, NUM_EPOCHS=1) 
    val_acc = evaluate(model, val_loader, ['STAR', 'GALAXY', 'QSO'], type="Validation")

    # Track best model
    if val_acc > best_acc_full:
        best_acc_full = val_acc
        best_params_full = params

print(f"\nBest Hyperparameters Full ResNet: {best_params_full}, Validation Accuracy: {best_acc_full:.2f}%")

"""
### Dilated Full Features CNN
for _ in range(full_trials):
    # Randomly sample parameters
    params = {key: random.choice(values) for key, values in param_space.items()}
    lr, dropout, wd, dilation = (params["learning_rate"], params["dropout"], params["weight_decay"], params["dilation"])
    print(f"\nTesting: Model= DilatedFullFeaturesCNN, lr={lr}, dropout={dropout}, weight_decay={wd}, dilation={dilation}")
    
    # Model, loss, optimizer
    model = DilatedFullFeaturesCNN(NUM_CLASSES=3, dropout_rate=dropout, dilation = dilation)
    model.train()
    print(torchinfo.summary(model))
    criterion = FocalLoss(alpha=[0.2, 0.3, 0.5], gamma=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Train and evaluate
    training_time = train(model, criterion, optimizer, NUM_EPOCHS=3) 
    val_acc = evaluate(model, val_loader, ['STAR', 'GALAXY', 'QSO'], type="Validation")

    # Track best model
    if val_acc > best_acc_dil:
        best_acc_dil = val_acc
        best_params_dil = params

print(f"\nBest Hyperparameters Dilated: {best_params_dil}, Validation Accuracy: {best_acc_dil:.2f}%")
"""