import glob
import random 
import os 
import subprocess
from tqdm import tqdm 
from datetime import datetime
import time

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchinfo
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn.utils as utils
from data_model import SpectraDataset, collate_fn
from cnn_model import FullFeaturesResNet, EarlyStopping, FocalLoss

# Scientific Python 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold


# set seed for reproducibility 
random.seed(42)

# Get the repo root (assumes script is inside STARDUSTAI/)
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
dropout = 0.4
weight_decay = 0.0001
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

def evaluate(model, dataloader, class_names =['STAR', 'GALAXY', 'QSO'], type="Test"):
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
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
    save_dir = 'experiment_results/cnn_saved_models'
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


def k_fold_cross_validation(dataset, k=5):
    """
    Perform k-fold cross-validation on the dataset.
    
    Parameters:
    - dataset: The entire dataset.
    - k: Number of folds.
    
    Returns:
    - Average validation accuracy across folds.
    """
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold+1}/{k}")

        # Create train and validation datasets for this fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        # Initialize the model for each fold
        model = FullFeaturesResNet(NUM_CLASSES, dropout_rate=dropout)
        model.train()
        criterion = FocalLoss(alpha=[0.2, 0.3, 0.5], gamma=0.5)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Train model on this fold
        training_time = train(model, criterion, optimizer, 2)

        # Evaluate on validation set
        val_accuracy = evaluate(model, val_loader, class_names, f"Validation (Fold {fold+1})")
        fold_accuracies.append(val_accuracy)

    # Compute average validation accuracy
    avg_val_accuracy = np.mean(fold_accuracies)
    print(f"\nAverage Validation Accuracy Across {k} Folds: {avg_val_accuracy:.2f}%")
    return avg_val_accuracy


### Training and evaluation 
if __name__ == "__main__":
    # Initialize, train, and evaluate the model
    model = FullFeaturesResNet(NUM_CLASSES, dropout_rate=dropout)
    model.train() 
    print(torchinfo.summary(model))
    early_stopping = EarlyStopping(patience=patience)
    criterion = FocalLoss(alpha=[0.2, 0.3, 0.5], gamma=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
    training_time = train(model, criterion, optimizer, NUM_EPOCHS) 
    val_accuracy = evaluate(model, val_loader, class_names, "Validation")
    test_accuracy = evaluate(model, test_loader, class_names, "Test")
    save_model(model, loss_fcn="FocalLoss")

    # Perform k-fold cross-validation
    k = 5  # Number of folds
    avg_accuracy = k_fold_cross_validation(dataset, k)
