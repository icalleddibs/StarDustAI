import glob
import random 
import os 
import subprocess
from tqdm import tqdm 

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
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
num_epochs = 10
learning_rate = 0.001

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))  
val_size = int(0.15 * len(dataset))    
test_size = len(dataset) - train_size - val_size  
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders with the updated collate function
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

#train_labels = [label for _, label, _ in train_loader.dataset]
#train_labels = torch.cat(train_labels, dim=0)
#print("Train class labels shape:", train_labels.shape)  # (num_samples, 3)

# train_subclass_labels = [label for _, _, label in train_loader.dataset]
# train_subclass_labels = torch.cat(train_subclass_labels, dim=0)
# print("Train subclass labels shape:", train_subclass_labels.shape)  # (num_samples, 44)
train_labels = []

## test single file then use print stuff in the class itself 
# ------
# # # Create a DataLoader for batch processing
# dataloader = DataLoader(FitsDataset(file_paths), batch_size=1, shuffle=True)
# train_loader = DataLoader(FitsDataset(file_paths), batch_size=1, shuffle=True)
# first_batch = next(iter(dataloader))

# # Iterate through batches
# for batch in dataloader:
#     print(batch.shape)  # Print batch shape
# ------
# Example usage


for i, batch in tqdm(enumerate(train_loader)):
    features, class_labels, subclass_labels = batch
    train_labels.append(class_labels)
    if i == 1: 
        break 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model): 
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            features, class_labels, subclass_labels = batch
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += class_labels.size
            correct += (predicted == class_labels).sum().item()
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
            features, class_labels, subclass_labels = batch
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, class_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_predictions += class_labels.size(0)
            correct_predictions += (predicted == class_labels).sum().item()
            pbar.set_description(f"Epoch {epoch + 1} | Loss: {running_loss / (i + 1):.5f} | Accuracy: {100 * correct_predictions / total_predictions:.2f}%")
            # if i % 10 == 9:
            #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            #     running_loss = 0.0
        # test_accuracy = evaluate(model)
        # print('Accuracy on validation set: %.2f' % test_accuracy)

# Model definition
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.layer1 = None

    def forward(self, x):
        out = None
        return out



model = CNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model, criterion, optimizer, num_epochs) 
test_accuracy = evaluate(model)
print('Accuracy on validation set: %.2f' % test_accuracy)


# def plot_class_distribution(loader, class_categories):
#     class_counts = torch.zeros(len(class_categories))

#     for batch in loader:
#         _, class_labels, _  = batch
#         class_labels = class_labels.argmax(dim=1)
#         class_counts += torch.bincount(class_labels, minlength=len(class_categories))

#     plt.bar(class_categories, class_counts)
#     plt.xlabel("Class")
#     plt.ylabel("Count")
#     plt.title("Class Distribution")
#     plt.show()