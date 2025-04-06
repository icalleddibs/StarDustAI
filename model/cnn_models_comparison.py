import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from tqdm import tqdm
from glob import glob
from cnn_training import test_loader, evaluate
from cnn_experiments.cnn_models import SimpleFluxCNN, FullFeaturesCNN, DilatedFullFeaturesCNN, FullFeaturesResNet, FullFeaturesCNNMoreLayers
from sklearn.metrics import classification_report
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


model_files = ['model/cnn_experiments/cnn_models_experiment_results/2025-03-30_17-54-59_model.pth', 
               'model/cnn_experiments/cnn_models_experiment_results/2025-03-20_10-24-47_model.pth',
               'model/cnn_experiments/cnn_models_experiment_results/2025-03-20_00-38-54_model.pth',
               'model/cnn_experiments/cnn_models_experiment_results/2025-03-24_13-07-38_model.pth',
               'model/cnn_experiments/cnn_models_experiment_results/2025-03-24_13-30-03_model.pth']  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
    "SimpleFluxCNN": SimpleFluxCNN,
    "FullFeaturesCNN": FullFeaturesCNN,
    "DilatedFullFeaturesCNN": DilatedFullFeaturesCNN,
    "FullFeaturesCNNMoreLayers": FullFeaturesCNNMoreLayers,
    "FullFeaturesResNet": FullFeaturesResNet}

restnet = FullFeaturesResNet()
restnet.load_state_dict(torch.load(model_files[4]))
evaluate(restnet, test_loader, type="Test")
model_reports = {}

for i, model_path in enumerate(model_files):
    # Extract model name from filename
    model_name = list(models.keys())[i] 
    model = models[model_name]()  
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Get predictions & true labels
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            features, class_labels = batch
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            class_indices = torch.argmax(class_labels, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(class_indices.cpu().numpy())
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, output_dict=True)
    model_reports[model_name] = report

# # Define metric thresholds
thresholds = {"Accuracy": 0.93, "F1-Score": 0.94, "Recall": 0.94, "Precision": 0.93}

# #Extract relevant metrics into a DataFrame
data = []
for model_name, report in model_reports.items():
    accuracy = report["accuracy"]
    f1_score = report["weighted avg"]["f1-score"]
    recall = report["weighted avg"]["recall"]
    precision = report["weighted avg"]["precision"]
    
    # Count successes
    success_count = sum([
        accuracy >= thresholds["Accuracy"],
        f1_score >= thresholds["F1-Score"],
        recall >= thresholds["Recall"],
        precision >= thresholds["Precision"]
    ])
    
    data.append([model_name, accuracy, f1_score, recall, precision, success_count])

# Column Names
columns = [
    "Model",
    f"Accuracy (>{int(thresholds['Accuracy']*100)}%)",
    f"F1-Score (>{int(thresholds['F1-Score']*100)}%)",
    f"Recall (>{int(thresholds['Recall']*100)}%)",
    f"Precision (>{int(thresholds['Precision']*100)}%)",
    "# Successes (/4)"
]

df = pd.DataFrame(data, columns=columns).set_index("Model")
df_numeric = df.astype(float)
df = df.iloc[:, :-1]  
df_heatmap = df_numeric.iloc[:, :-1]
colormaps = ["Purples", "Blues", "Greens", "Oranges", "Reds"]
norms = [PowerNorm(gamma=2.5, vmin=df[col].min() - 0.05, vmax=df[col].max() + 0.05) for col in df.columns]

# Heatmap 
fig, ax = plt.subplots(figsize=(20, 6))
for j, col in enumerate(df.columns):
    cmap = plt.get_cmap(colormaps[j])  
    norm = norms[j] 
    for i, model in enumerate(df.index):
        value = df.at[model, col]  
        color = cmap(norm(value)) 
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))  
        ax.text(j + 0.5, i + 0.5, f"{value:.2f}", ha='center', va='center', fontsize=12, color='black', weight='bold')

ax.set_yticks(np.arange(len(df.index)) + 0.5)
ax.set_yticklabels(df.index, fontsize=12)
ax.set_xlim(0, len(df.columns))
ax.set_ylim(0, len(df.index))
ax.invert_yaxis()  
ax.set_frame_on(False)
ax.set_xlabel("")
ax.set_xticklabels([])
ax.set_xticks([])

for j, col_name in enumerate(df_heatmap.columns):
    ax.text(j + 0.5, -0.3, col_name, ha="center", va="center", fontsize=12, fontweight="bold")

ax.text(len(df_heatmap.columns) + 0.5, -0.3, "# Successes (/4)", ha="center", va="center", fontsize=12, fontweight="bold")

for i, success_count in enumerate(df_numeric["# Successes (/4)"]):
    ax.text(len(df_heatmap.columns) + 0.3, i + 0.5, f"{int(success_count)}",
            va='center', ha='left', fontsize=12, color='black', weight='bold')

plt.title("Model Performance Table", fontsize=16, weight="bold", pad=50)
plt.show()

# Data
data = {
    "Metric": ["Accuracy",  "F1-Score","Recall", "Precision"],
    "Mean": [95.1978,  95.1994, 95.1978, 95.2170],
    "Std": [0.4961, 0.4946,  0.4961,  0.4918,],
    "CI Lower": [94.2736,  94.2739, 94.2736, 94.2957],
    "CI Upper": [96.2102,   96.2097,  96.2102, 96.2164]
}
df = pd.DataFrame(data)

# K-fold average accuracy
k_fold_avg_accuracy = 94.3

colors = {
    "Accuracy": "#c39bd3",  
    "F1-Score": "#90caf9", 
    "Recall": "#a5d6a7",    
    "Precision": "#ffcc80"
}

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(
    df["Metric"], df["Mean"],
    color=[colors[m] for m in df["Metric"]],
    alpha=0.9, yerr=[df["Mean"] - df["CI Lower"], df["CI Upper"] - df["Mean"]],
    capsize=10, linewidth=2, edgecolor="black", error_kw={"elinewidth": 2, "capthick": 2}
)

ax.set_ylim(90, 97)

# Labels and Formatting
ax.set_ylabel("Mean (%)", fontsize=15, weight="bold")
ax.set_xticklabels(df["Metric"], fontsize=18, weight="bold", rotation=30, ha="right")
ax.set_title("FullFeaturesResNet Metrics with Confidence Intervals", fontsize=16, weight="bold")
for bar, mean in zip(bars, df["Mean"]):
    ax.text(bar.get_x() + bar.get_width()/4, mean + 0.05, f"{mean:.2f}%", ha="center", fontsize=15, weight="bold")

# Plot K-Fold Average Accuracy on Accuracy Bar
accuracy_idx = df[df["Metric"] == "Accuracy"].index[0]
ax.scatter(accuracy_idx, k_fold_avg_accuracy, color="purple", s=100, zorder=3, label="5-Fold Avg (94.3%)")
ax.text(accuracy_idx + 0.2, k_fold_avg_accuracy - 0.05, "94.3%", ha="center", fontsize=15, weight="bold", color="purple")
ax.legend(loc="upper right", fontsize=12)

plt.tight_layout()
plt.show()