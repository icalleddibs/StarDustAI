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
from data_model import SepctraDataset, collate_fn

# Scientific Python 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import plotly as px 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# List all FITS files
# Get the repo root (assumes script is inside STARDUSTAI/)
repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
base_dir = os.path.join(repo_root, "data/full_zwarning")
file_paths = glob.glob(os.path.join(base_dir, "*/*.pkl"))

# If no FITS files are found, raise an error
if not file_paths:
    raise ValueError("No FITS files found in 'data/full_zwarning/'")

# Shuffle the file paths
random.shuffle(file_paths)

# Load the dataset
dataset = SepctraDataset(file_paths)

# Training params 
batch_size = 64
num_classes = 3
num_subclasses = 44
num_epochs = 1
learning_rate = 0.001
patience = 5

class_names = ['STAR', 'GALAXY', 'QSO']

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))  
val_size = int(0.15 * len(dataset))    
test_size = len(dataset) - train_size - val_size  
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders with the updated collate function
train_loader = DataLoader(train_dataset, batch_size=batch_size,collate_fn=collate_fn ,shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size,collate_fn=collate_fn, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn,shuffle=False)

# Print classification report to show key metrics
def evaluate_metrics(model, dataloader, class_names):
    print("evaluating metrics...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            features, class_labels = batch
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            class_indices = torch.argmax(class_labels, dim=1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(class_indices.cpu().numpy())

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=1)
    print(report)

# Calculate validation accuracy and plot confusion matrix
def evaluate(model): 
    print("calculating validation accuracy...")
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            features, class_labels = batch
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)  # Get predicted class indices
            class_indices = torch.argmax(class_labels, dim=1)  # Convert one-hot labels to class indices
            total += class_labels.size(0)  # Get batch size
            correct += (predicted == class_indices).sum().item()  # Compare indices
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(class_indices.cpu().numpy())

    validation_accuracy = 100 * correct / total

    print("Plotting confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    # Plot the confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    return validation_accuracy

# Train the model
def train(model, criterion, optimizer, num_epochs):
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        model.train()
        running_loss = 0.0
        total_predictions = 0
        correct_predictions = 0
        for i, batch in enumerate(train_loader):
            features, class_labels = batch
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
                running_loss = 0.0

# Early stopping class, not currently used. 
class EarlyStopping:
    def __init__(self, patience=5, verbose=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

early_stopping = EarlyStopping(patience=patience, verbose=True)


# Model definition
class SimpleFluxCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleFluxCNN, self).__init__()
        # Since we are only using the flux column, the input channel is 1.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
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


### Training and evaluation 
model = SimpleFluxCNN(num_classes)
model.train() 
print(torchinfo.summary(model))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model, criterion, optimizer, num_epochs) 

val_accuracy = evaluate(model)
print('Accuracy on validation set: %.2f' % val_accuracy)

evaluate_metrics(model, test_loader, class_names)
