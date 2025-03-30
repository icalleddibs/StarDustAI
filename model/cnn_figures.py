import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
from glob import glob
from cnn_training import test_loader
from cnn_experiments.cnn_models import SimpleFluxCNN, AllFeaturesCNN, FullFeaturesCNN, DilatedFullFeaturesCNN, FullFeaturesResNet, FullFeaturesCNNMoreLayers
from sklearn.metrics import classification_report


model_files = ['path/to/model1.pth', 
               'path/to/model2.pth',
               'path/to/model2.pth',
               'path/to/model2.pth',
               'path/to/model2.pth']  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
    "SimpleFluxCNN": SimpleFluxCNN,
    "FullFeaturesCNN": FullFeaturesCNN,
    "DilatedFullFeaturesCNN": DilatedFullFeaturesCNN,
    "FullFeaturesResNet": FullFeaturesResNet,
    "FullFeaturesCNNMoreLayers": FullFeaturesCNNMoreLayers}

# Dictionary to store reports
model_reports = {}

# for i, model_path in enumerate(model_files):
#     # Load model
#     model = models[list(models.keys())[i]]()
#     model.load_state_dict(torch.load(model_path))
#     model.to(device)
#     model.eval()  # Set to evaluation mode
    
#     # Extract model name from filename
#     model_name = models.keys()[i]

#     # Get predictions & true labels
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             preds = torch.argmax(outputs, dim=1)
            
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
    
#     # Generate classification report
#     report = classification_report(all_labels, all_preds, output_dict=True)
#     model_reports[model_name] = report


# Define metric thresholds
thresholds = {"Accuracy": 0.92, "F1-Score": 0.92, "Recall": 0.92, "Precision": 0.92}

# Extract relevant metrics into a DataFrame
# data = []
# for model_name, report in model_reports.items():
#     accuracy = report["accuracy"]
#     f1_score = report["weighted avg"]["f1-score"]
#     recall = report["weighted avg"]["recall"]
#     precision = report["weighted avg"]["precision"]
    
#     # Count successes
#     success_count = sum([
#         accuracy >= thresholds["Accuracy"],
#         f1_score >= thresholds["F1-Score"],
#         recall >= thresholds["Recall"],
#         precision >= thresholds["Precision"]
#     ])
    
#     data.append([model_name, accuracy, f1_score, recall, precision, success_count])
data = [["A", 0.86, 0.92, 0.92, 0.92, 4], 
        ["A",0.92, 0.92, 0.92, 0.92, 4], 
        ["A",0.92, 0.92, 0.92, 0.92, 4], 
        ["A",0.92, 0.92, 0.92, 0.92, 4], 
        ["A",0.92, 0.92, 0.92, 0.92, 4]]

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
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df_heatmap, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", cbar=False, ax=ax)

ax.set_xlabel("")
ax.set_xticklabels([])
ax.set_yticklabels(df_heatmap.index, rotation=0, ha="right", fontsize=12)

# Add column headers above the heatmap
for j, col_name in enumerate(df_heatmap.columns):
    ax.text(j + 0.5, -0.3, col_name, ha="center", va="center", fontsize=12, fontweight="bold")

# Add success column header
ax.text(len(df_heatmap.columns) + 0.5, -0.3, "# Successes", ha="center", va="center", fontsize=12, fontweight="bold")

# Add "# Successes" column as text
for i, success_count in enumerate(df_numeric["# Successes"]):
    ax.text(len(df_heatmap.columns) + 0.3, i + 0.5, f"{int(success_count)}/4",
            va='center', ha='left', fontsize=12, color='black', weight='bold')

plt.title("Model Performance Table", fontsize=14, weight="bold", pad=40)

plt.show()
