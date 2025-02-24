from torch.utils.data import DataLoader
from data_model import FitsDataset, collate_fn
from torch.utils.data import Dataset, DataLoader, random_split
from astropy.table import Table, hstack
from astropy.utils.metadata import MergeConflictWarning
import glob
import torch
import random 
import os 
import subprocess
import numpy as np
import warnings
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
import pandas as pd 
import matplotlib.pyplot as plt
import plotly as px 


warnings.simplefilter('ignore', MergeConflictWarning)
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

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))  
val_size = int(0.15 * len(dataset))    
test_size = len(dataset) - train_size - val_size  

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders with the updated collate function
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

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
    if i == 10: 
        break
    #print("Features batch shape:", features.shape)  # (batch_size, max_rows, num_features)
    #print("Class labels batch shape:", class_labels.shape)  # (batch_size, 3)
    #print("Subclass labels batch shape:", subclass_labels.shape)  # (batch_size, 44)
    


def plot_class_distribution(loader, class_categories):
    class_counts = torch.zeros(len(class_categories))

    for batch in loader:
        _, class_labels, _  = batch
        class_labels = class_labels.argmax(dim=1)
        class_counts += torch.bincount(class_labels, minlength=len(class_categories))

    plt.bar(class_categories, class_counts)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.show()