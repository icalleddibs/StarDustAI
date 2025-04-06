import os
import glob
import subprocess
import random
import itertools

import torchinfo
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from cnn_experiments.cnn_models import EarlyStopping, FocalLoss, AllFeaturesCNN, DilatedFullFeaturesCNN, FullFeaturesResNet, FullFeaturesCNN
from cnn_training import train, evaluate, save_model
from data_model import SpectraDataset, collate_fn

# Select model to run
model_name = "FullFeaturesResNet"

# Get repo root and dataset
repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
base_dir = os.path.join(repo_root, "data/full_zwarning")
file_paths = glob.glob(os.path.join(base_dir, "*/*.pkl"))

if not file_paths:
    raise ValueError("No PKL files found in 'data/full_zwarning/'")

random.shuffle(file_paths)
dataset = SpectraDataset(file_paths)

# Split dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Hyperparameter search space
param_space = {
    "learning_rate": [0.005, 0.001, 0.01, 0.0001, 0.0005],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "weight_decay": [0.0001, 0.001],
    "dilation": [2, 3, 4]
}

batch_size = 256
full_trials = 30
best_params = None
best_acc = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

### Generalized to all models
for _ in range(full_trials):
    # Randomly sample parameters
    params = {key: random.choice(values) for key, values in param_space.items()}
    lr, dropout, wd, dil = params["learning_rate"], params["dropout"], params["weight_decay"], params["dilation"]
    print(f"\nTesting: Model= {model_name}, lr={lr}, dropout={dropout}, weight_decay={wd}")
    
    if model_name == "FullFeaturesResNet":
        model = FullFeaturesResNet(NUM_CLASSES=3, dropout_rate=dropout)
    elif model_name == "DilatedFullFeaturesCNN":
        model = DilatedFullFeaturesCNN(NUM_CLASSES=3, dropout_rate=dropout, dilation=dilation)
    elif model_name == "FullFeaturesCNN":
        model = FullFeaturesCNN(NUM_CLASSES=3, dropout_rate=dropout)
    elif model_name == "AllFeaturesCNN":
        model = AllFeaturesCNN(NUM_CLASSES=3, dropout_rate=dropout)
    else:
        raise print("Unsupported model: {model_name}")
    
    model.train()
    print(torchinfo.summary(model))
    criterion = FocalLoss(alpha=[0.2, 0.3, 0.5], gamma=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

    # Train and evaluate
    training_time = train(model, criterion, optimizer, scheduler, NUM_EPOCHS=1) 
    val_acc = evaluate(model, val_loader, ['STAR', 'GALAXY', 'QSO'], type="Validation")

    # Track best model
    if val_acc > best_acc:
        best_acc = val_acc
        best_params = params

print(f"\nBest Hyperparameters for {model_name}: {best_params}, Validation Accuracy: {best_acc:.2f}%")