import glob
import random 
import os 
import subprocess
from tqdm import tqdm 
from datetime import datetime

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
BATCH_SIZE = 64
NUM_CLASSES = 3
num_subclasses = 44
NUM_EPOCHS = 1
learning_rate = 0.001
patience = 5

class_names = ['STAR', 'GALAXY', 'QSO']

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))  
val_size = int(0.15 * len(dataset))    
test_size = len(dataset) - train_size - val_size  
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders with the updated collate function
# Create DataLoaders with the updated collate function
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    collate_fn=collate_fn, shuffle=True, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    collate_fn=collate_fn, shuffle=False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    collate_fn=collate_fn, shuffle=False
)

def evaluate_metrics(model, dataloader, class_names):
    """
    Evaluate and print classification metrics.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model.
    dataloader : DataLoader
        DataLoader for evaluation data.
    class_names : list of str
        List of class names.
    """
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
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=1
    )
    print(report)

def evaluate(model): 
    """
    Calculate validation accuracy and plot confusion matrix.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model.

    Returns
    -------
    float
        Validation accuracy in percentage.
    """
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
            _, predicted = torch.max(outputs, 1) 
            class_indices = torch.argmax(class_labels, dim=1)  
            total += class_labels.size(0)  
            correct += (predicted == class_indices).sum().item() 
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(class_indices.cpu().numpy())

    validation_accuracy = 100 * correct / total

    print("Plotting confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)

    # Reorder matrix to match ['GALAXY', 'QSO', 'STAR']
    reorder = [1, 2, 0]  # indices for ['GALAXY', 'QSO', 'STAR']
    cm = cm[reorder][:, reorder]
    new_class_names = ['GALAXY', 'QSO', 'STAR']

    # Plot the confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=new_class_names, yticklabels=new_class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    return validation_accuracy

def train(model, criterion, optimizer, NUM_EPOCHS):
    """
    Train the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer.
    num_epochs : int
        Number of epochs to train for.
    """
    pbar = tqdm(range(NUM_EPOCHS))
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in pbar:
        model.train()
        running_loss = 0.0
        total_predictions = 0
        correct_predictions = 0

        # Training loop
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

            # Update progress bar
            pbar.set_description(
                f"Epoch {epoch + 1} | Batch {i + 1} "
                f"| Loss: {running_loss / (i + 1):.5f} "
                f"| Accuracy: {100 * correct_predictions / total_predictions:.2f}%"
            )

        # Calculate validation loss after each epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, class_labels in val_loader:
                outputs = model(features)
                class_indices = torch.argmax(class_labels, dim=1)
                loss = criterion(outputs, class_indices)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break


def save_model(model):
    """
    Save the model to a file.

    Parameters
    ----------
    model : torch.nn.Module
        The model to save.
    model_path : str
        Path to save the model to.
    """
    # save and log 
    save_dir = 'cnn_saved_models'
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_path = os.path.join(save_dir, f'{timestamp}_model.pth')
    torch.save(model.state_dict(), model_path)
    #log hyper params 
    hyperparams = {
        'learning_rate': learning_rate,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'patience': patience,
        'num_classes': NUM_CLASSES,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'validation_accuracy': val_accuracy
    }

    hyperparams_path = os.path.join(save_dir, f'{timestamp}_hyperparams.txt')
    with open(hyperparams_path, 'w') as f:
        f.write("Hyperparameters and Results:\n")
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")

    print(f"Model saved to {model_path}")
    print(f"Hyperparameters saved to {hyperparams_path}")


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """
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


class SimpleFluxCNN(nn.Module):
    """
    Simple CNN model for flux-based classification.
    """

    def __init__(self, NUM_CLASSES=3):
        super(SimpleFluxCNN, self).__init__()
        # Since we are only using the flux column, the input channel is 1.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, NUM_CLASSES)

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
model = SimpleFluxCNN(NUM_CLASSES)
model.train() 
print(torchinfo.summary(model))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model, criterion, optimizer, NUM_EPOCHS) 

val_accuracy = evaluate(model)
print('Accuracy on validation set: %.2f' % val_accuracy)

evaluate_metrics(model, test_loader, class_names)

save_model(model)





# load the model 
# model2= SimpleFluxCNN(num_classes)
# model2.load_state_dict(torch.load('cnn_saved_models/model.pth'))

# model2.eval()
# val_accuracy = evaluate(model2)


