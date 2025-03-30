import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm
from glob import glob
from cnn_training import test_loader
from cnn_experiments.cnn_models import SimpleFluxCNN, AllFeaturesCNN, FullFeaturesCNN, DilatedFullFeaturesCNN, FullFeaturesResNet, FullFeaturesCNNMoreLayers
from sklearn.metrics import classification_report


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

# Dictionary to store reports
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

#save model_reports to csv
model_reports_df = pd.DataFrame(model_reports).T
model_reports_df.to_csv("model_reports.csv", index=True)

# Define metric thresholds
thresholds = {"Accuracy": 0.93, "F1-Score": 0.94, "Recall": 0.94, "Precision": 0.93}

#Extract relevant metrics into a DataFrame
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

data.append(["XGBoost", 0.86, 0.92, 0.92, 0.92, 4])


# Create DataFrame
columns = [
    "Model",
    f"Accuracy ({int(thresholds['Accuracy']*100)}%)",
    f"F1-Score ({int(thresholds['F1-Score']*100)}%)",
    f"Recall ({int(thresholds['Recall']*100)}%)",
    f"Precision ({int(thresholds['Precision']*100)}%)",
    "# Successes"
]
df = pd.DataFrame(data, columns=columns)

# Convert to numeric for heatmap
df_numeric = df.set_index("Model").astype(float)

# Separate "# Successes" column
df_heatmap = df_numeric.iloc[:, :-1]

# Plot the heatmap as a table
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(df_heatmap, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", cbar=True, ax=ax, cbar_kws={"shrink": 0.5, "orientation": "horizontal"})


ax.set_xlabel("")
ax.set_xticklabels([])
ax.set_yticklabels(df_heatmap.index, rotation=0, ha="right", fontsize=12)

# Add column headers above the heatmap
for j, col_name in enumerate(df_heatmap.columns):
    ax.text(j + 0.5, -0.3, col_name, ha="center", va="center", fontsize=12, fontweight="bold")

# Add success column header
ax.text(len(df_heatmap.columns) + 0.5, -0.3, "# Successes", ha="center", va="center", fontsize=12, fontweight="bold")

for i, success_count in enumerate(df_numeric["# Successes"]):
    ax.text(len(df_heatmap.columns) + 0.3, i + 0.5, f"{int(success_count)}/4",
            va='center', ha='left', fontsize=12, color='black', weight='bold')

plt.title("Model Performance Table", fontsize=14, weight="bold", pad=40)
plt.tight_layout()
plt.show()
