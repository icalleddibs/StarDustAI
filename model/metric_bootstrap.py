import random 
from tqdm import tqdm 

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from data_model import collate_fn
from cnn_model import FullFeaturesResNet
from cnn_training import test_dataset

# Scientific Python 
import numpy as np
import pandas as pd 
from sklearn.metrics import classification_report


# set seed for reproducibility
random.seed(42)

BATCH_SIZE = 32
NUM_CLASSES = 3

def bootstrap_classification_report(model, test_dataset, B=100):
    """
    Perform bootstrapping to estimate uncertainty in classification metrics.

    Parameters:
    - model: Trained model.
    - test_dataset: Dataset for evaluation.
    - B: Number of bootstrap resamples.

    Returns:
    - Mean and confidence intervals for accuracy, precision, recall, and F1-score.
    """
    boot_accuracies = []
    boot_precisions = []
    boot_recalls = []
    boot_f1s = []

    test_indices = np.arange(len(test_dataset))

    for i in range(B):
        print(f"\nBootstrap Iteration {i+1}/{B}")

        # Sample test dataset with replacement
        boot_indices = np.random.choice(test_indices, size=len(test_dataset), replace=True)
        boot_subset = Subset(test_dataset, boot_indices)
        boot_loader = DataLoader(boot_subset, batch_size=32, collate_fn=collate_fn, shuffle=False)

        # Get predictions
        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, labels in tqdm(boot_loader):
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                class_indices = torch.argmax(labels, dim=1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(class_indices.cpu().numpy())

        # Compute classification report
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=1)

        # Store metrics
        boot_accuracies.append(report["accuracy"] * 100)
        boot_precisions.append(report["weighted avg"]["precision"] * 100)
        boot_recalls.append(report["weighted avg"]["recall"] * 100)
        boot_f1s.append(report["weighted avg"]["f1-score"] * 100)

    # Compute mean and confidence intervals
    def compute_ci(values):
        mean = np.mean(values)
        std = np.std(values)
        ci_lower, ci_upper = np.percentile(values, [2.5, 97.5])  # 95% CI
        return mean, std, ci_lower, ci_upper

    metrics = {
        "Accuracy": compute_ci(boot_accuracies),
        "Precision": compute_ci(boot_precisions),
        "Recall": compute_ci(boot_recalls),
        "F1-Score": compute_ci(boot_f1s)
    }

    print("\nBootstrapped Classification Report:")
    for metric, (mean, std, ci_lower, ci_upper) in metrics.items():
        print(f"{metric}: {mean:.2f} ± {std:.2f} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})")

    return metrics



if __name__ == "__main__":
    model = FullFeaturesResNet()
    model.load_state_dict(torch.load("cnn_experiments/cnn_models_experiment_results/2025-03-24_13-30-03_model.pth"))
    model.eval()
    test_dataset = test_dataset

    # Perform bootstrapping
    metrics = bootstrap_classification_report(model, test_dataset, B=100)

    # Save the metrics to a CSV file
    df = pd.DataFrame(metrics).T
    df.columns = ['Mean', 'Std', 'CI Lower', 'CI Upper']
    df.to_csv('bootstrapped_metrics.csv')

