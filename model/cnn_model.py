import glob
import random 
import os 
import subprocess
from tqdm import tqdm 

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchinfo
from torch.utils.data import DataLoader, random_split
from data_model import FitsDataset, collate_fn

# Scientific Python 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import plotly as px 
from sklearn.model_selection import train_test_split

# List all FITS files
# Get the repo root (assumes script is inside STARDUSTAI/)
repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
base_dir = os.path.join(repo_root, "data/full")
file_paths = glob.glob(os.path.join(base_dir, "*/*.fits"))

# If no FITS files are found, raise an error
if not file_paths:
    raise ValueError("No FITS files found in 'data/full/'")

# Shuffle the file paths
random.shuffle(file_paths)

# Load the dataset
dataset = FitsDataset(file_paths)

# Training params 
batch_size = 32
num_classes = 3
num_subclasses = 44
num_epochs = 1
learning_rate = 0.001

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))  
val_size = int(0.15 * len(dataset))    
test_size = len(dataset) - train_size - val_size  
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders with the updated collate function
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

#train_labels = [label for _, label, _ in train_loader.dataset]
#train_labels = torch.cat(train_labels, dim=0)
#print("Train class labels shape:", train_labels.shape)  # (num_samples, 3)

# train_subclass_labels = [label for _, _, label in train_loader.dataset]
# train_subclass_labels = torch.cat(train_subclass_labels, dim=0)
# print("Train subclass labels shape:", train_subclass_labels.shape)  # (num_samples, 44)

## test single file then use print stuff in the class itself 
# ------
# # # Create a DataLoader for batch processing
# dataloader = DataLoader(FitsDataset(file_paths), batch_size=1, shuffle=True)
# train_loader = DataLoader(FitsDataset(file_paths), batch_size=1, shuffle=True)
# first_batch = next(iter(dataloader))




for i, batch in tqdm(enumerate(train_loader)):
    features, class_labels, subclass_labels = batch
    print(features.shape, class_labels.shape, features.names)
    if i == 10: 
        break 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model): 
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            features, class_labels, _ = batch
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)  # Get predicted class indices
            class_indices = torch.argmax(class_labels, dim=1)  # Convert one-hot labels to class indices
            total += class_labels.size(0)  # Get batch size
            correct += (predicted == class_indices).sum().item()  # Compare indices
            
    test_accuracy = 100 * correct / total
    return test_accuracy


def train(model, criterion, optimizer, num_epochs):
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        model.train()
        running_loss = 0.0
        total_predictions = 0
        correct_predictions = 0
        for i, batch in enumerate(train_loader):
            features, class_labels, _ = batch
            optimizer.zero_grad()
            outputs = model(features)
            class_indices = torch.argmax(class_labels, dim=1)
            loss = criterion(outputs, class_indices)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_predictions += class_labels.size(0)
            correct_predictions += (predicted == class_indices).sum().item()

            pbar.set_description(f"Epoch {epoch + 1} | Batch {i} | Loss: {running_loss / (i + 1):.5f} | Accuracy: {100 * correct_predictions / total_predictions:.2f}%")
            if i % 10 == 9:
                #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
            if i == 100: 
                break
        # test_accuracy = evaluate(model)
        # print('Accuracy on validation set: %.2f' % test_accuracy)

# Model definition
class SimpleFluxCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleFluxCNN, self).__init__()
        # Use a simple two-layer 1D CNN architecture.
        # Since we are only using the flux column, the input channel is 1.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Adaptive pooling to handle variable-length sequences (max_rows) per batch.
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Final fully connected layer to output logits for each class.
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # x has shape: (batch_size, max_rows, num_features)
        # Extract the flux column (first column)
        flux = x[:, :, 0]  # shape: (batch_size, max_rows)
        # Add a channel dimension so that the input becomes (batch_size, 1, max_rows)
        flux = flux.unsqueeze(1)
        # Pass through the convolutional layers with ReLU activation.
        out = F.relu(self.conv1(flux))
        out = F.relu(self.conv2(out))
        # Apply adaptive average pooling to get a fixed-size output regardless of input length.
        out = self.pool(out)  # shape: (batch_size, 32, 1)
        out = out.squeeze(2)  # shape: (batch_size, 32)
        # Compute the class logits.
        logits = self.fc(out)  # shape: (batch_size, num_classes)
        return logits

model = SimpleFluxCNN(num_classes)
model.train()
print(torchinfo.summary(model))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model, criterion, optimizer, num_epochs) 

test_accuracy = evaluate(model)
print('Accuracy on validation set: %.2f' % test_accuracy)


