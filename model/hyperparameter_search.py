import torch
import itertools
from cnn_model import SimpleFluxCNN, train, evaluate
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from data_model import SepctraDataset, collate_fn
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
dataset = SepctraDataset(file_paths)

# Split dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Hyperparameter search space
param_grid = {
    "learning_rate": [0.001, 0.0005],
    "batch_size": [64, 128]
}

best_acc = 0
best_params = None

# Iterate through hyperparameter combinations
for params in itertools.product(*param_grid.values()):
    lr, batch_size = params
    print(f"\nTesting: learning_rate={lr}, batch_size={batch_size}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model, loss, optimizer
    model = SimpleFluxCNN(NUM_CLASSES=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train and evaluate
    train(model, criterion, optimizer, NUM_EPOCHS=3)
    val_acc = evaluate(model, val_loader, ['STAR', 'GALAXY', 'QSO'])

    # Track best model
    if val_acc > best_acc:
        best_acc = val_acc
        best_params = {"learning_rate": lr, "batch_size": batch_size}

print(f"\nBest Hyperparameters: {best_params}, Validation Accuracy: {best_acc:.2f}%")
