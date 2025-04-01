import glob
import random 
import os 
import sys
import subprocess
from tqdm import tqdm 
from datetime import datetime
import time

# Append model to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchinfo
from torch.utils.data import DataLoader, random_split
import torch.nn.utils as utils
from data_model import SpectraDataset, collate_fn
from cnn_models import SimpleFluxCNN, AllFeaturesCNN, FullFeaturesCNN, DilatedFullFeaturesCNN, FullFeaturesResNet, FullFeaturesCNNMoreLayers, EarlyStopping, FocalLoss

# Scientific Python 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import plotly as px 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# set seed for reproducibility 
random.seed(42)

# # Get the repo root (assumes script is inside STARDUSTAI/)
repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
base_dir = os.path.join(repo_root, "data/full_zwarning")
file_paths = glob.glob(os.path.join(base_dir, "*/*.pkl"))

# If no files are found, raise an error
if not file_paths:
    raise ValueError("No data files found in 'data/full_zwarning/'")

# Shuffle the file paths
random.shuffle(file_paths)

# Load the dataset
dataset = SpectraDataset(file_paths)

# Training params 
BATCH_SIZE = 32
NUM_CLASSES = 3
NUM_EPOCHS = 3
learning_rate = 0.001
patience = 2
dropout = 0.0
weight_decay = 0.01
dilation = 2

class_names = ['STAR', 'GALAXY', 'QSO']

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))  
val_size = int(0.15 * len(dataset))    
test_size = len(dataset) - train_size - val_size  
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

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




def evaluate(model, dataloader, class_names, type="Test"):
    """
    Evaluate model performance, calculate validation accuracy, plot confusion matrix,
    and print classification metrics.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model.
    dataloader : DataLoader
        DataLoader for evaluation data.
    class_names : list of str
        List of class names.

    Returns
    -------
    float
        Validation accuracy in percentage.
    """
    print("Evaluating model...")
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            features, class_labels = batch
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            class_indices = torch.argmax(class_labels, dim=1)
            total += class_labels.size(0)
            correct += (predicted == class_indices).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(class_indices.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f"\n{type} Accuracy: {accuracy:.2f}%")
    
    # Generate classification report
    print("\nClassification Report For :", type, " Data")
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=1
    )
    print(report)

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    reorder = [1, 2, 0]  
    cm = cm[reorder][:, reorder]
    new_class_names = ['GALAXY', 'QSO', 'STAR']
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 

    print("Plotting confusion matrix...")
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_percentage, 
        annot=True, 
        fmt='.1%', 
        cmap="Blues", 
        xticklabels=new_class_names, 
        yticklabels=new_class_names
    )

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix For ' + type + ' Data')
    plt.show()

    return accuracy

def train(model, criterion, optimizer,  NUM_EPOCHS=NUM_EPOCHS):
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
    start_time = time.time()
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
            utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        scheduler.step(val_loss)
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
    
    end_time = time.time()  
    training_time = end_time - start_time
    return training_time

def save_model(model, loss_fcn= "CrossEntropyLoss"):
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
    save_dir = 'cnn_models_experiment_results'
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_path = os.path.join(save_dir, f'{timestamp}_model.pth')
    torch.save(model.state_dict(), model_path)
    #log hyper params 
    hyperparams = {
        'model': model.__class__.__name__,
        'learning_rate': learning_rate,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'patience': patience,
        'dropout': dropout,
        'weight_decay': weight_decay,
        'num_classes': NUM_CLASSES,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'validation_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'training time': training_time, 
        'loss function': loss_fcn
    }

    hyperparams_path = os.path.join(save_dir, f'{timestamp}_hyperparams.txt')
    with open(hyperparams_path, 'w') as f:
        f.write("Hyperparameters and Results:\n")
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")

    print(f"Model saved to {model_path}")
    print(f"Hyperparameters saved to {hyperparams_path}")


# early_stopping = EarlyStopping(patience=patience, verbose=True)

# ### Training and evaluation 
model = SimpleFluxCNN(NUM_CLASSES, dropout_rate=dropout)
# model = AllFeaturesCNN(NUM_CLASSES, dropout_rate=dropout)
#model = FullFeaturesCNN(NUM_CLASSES, dropout_rate=dropout)
#model  = DilatedFullFeaturesCNN(NUM_CLASSES, dropout_rate=dropout, dilation=dilation)
#model = FullFeaturesCNNMoreLayers(NUM_CLASSES, dropout_rate=dropout)
#model = FullFeaturesResNet(NUM_CLASSES, dropout_rate=dropout)
model.train() 
model.to("cuda" if torch.cuda.is_available() else "cpu")
print(torchinfo.summary(model))
#criterion = nn.CrossEntropyLoss()
criterion = FocalLoss(alpha=[0.2, 0.3, 0.5], gamma=0.5)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)
training_time = train(model, criterion, optimizer, NUM_EPOCHS) 
val_accuracy = evaluate(model, val_loader, class_names, "Validation")
test_accuracy = evaluate(model, test_loader, class_names, "Test")
save_model(model, loss_fcn="FocalLoss")

