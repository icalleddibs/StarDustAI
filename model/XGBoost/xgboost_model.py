import glob
import os 
import time 
import pickle 
from collections import defaultdict

import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder

'''
This script demonstrates how to train an XGBoost model on a dataset of pickle files.
It loads the data from the pickle files, preprocesses it to the correct size, and trains an XGBoost model.
The model is evaluated on a test set, with the classification report and feature importance being displayed.
'''

# Personal File Loading of pickle files from SSD
BASE_DIR = "E:/StarDustAI/data/full_zwarning/full_zwarning"
FILE_PATHS = glob.glob(os.path.join(BASE_DIR, '**/*.pkl'), recursive=True)

if not FILE_PATHS:
    raise ValueError("No pickle files found in the specified directory")

# Initialize lists to store features and labels
X, y = [], []
MAX_LENGTH = 4615  # For padding/truncating

def pad_or_truncate(array, max_length=MAX_LENGTH):
    ''' Pad or truncate an array to the specified length for XGB model '''
    array_len = len(array)
    if array_len > max_length:
        return array[:max_length]                                            # Truncate if too long
    elif array_len < max_length:
        return np.pad(array, (0, max_length - array_len), mode='constant')   # Pad if too short
    return array                                                             # No change if equal to max size

# Load all pickle files
for pickle_file in tqdm(FILE_PATHS, desc="Loading Pickle Files", unit="file"):
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            
            # Get the features and labels by truncating/padding to the same size
            features = np.column_stack([
                pad_or_truncate(data["flux"], MAX_LENGTH),
                pad_or_truncate(data["loglam"], MAX_LENGTH),
                pad_or_truncate(data["ivar"], MAX_LENGTH),
                pad_or_truncate(data["model"], MAX_LENGTH),
                pad_or_truncate(data["Z"], MAX_LENGTH),
                pad_or_truncate(data["Z_ERR"], MAX_LENGTH),
                pad_or_truncate(data["SN_MEDIAN_UV"], MAX_LENGTH),
                pad_or_truncate(data["SN_MEDIAN_R"], MAX_LENGTH),
                pad_or_truncate(data["SN_MEDIAN_NIR"], MAX_LENGTH),
                pad_or_truncate(data["SN_MEDIAN_IR"], MAX_LENGTH),
                pad_or_truncate(data["ZWARNING"], MAX_LENGTH),
                pad_or_truncate(data["RCHI2"], MAX_LENGTH),
            ])

            X.append(features)
            y.append(data["CLASS"][0])
    
    # To handle the occasional incorrectly formatted pickle files
    except Exception as e:
        print(f"Error loading {pickle_file}: {e}")

X = np.array(X, dtype=float)  
y = np.array(y)  

X_flattened = X.reshape(X.shape[0], -1)

# Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_flattened, y_encoded, test_size=0.15, random_state=42
    )

# Create DMatrix specific to XGB for training and validation
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dval = xgb.DMatrix(data=X_val, label=y_val)

# Set XGBoost parameters: editable to tune model
params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_encoded)),
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 0.8,
    'seed': 42
}

# Train the model
num_rounds = 100

# Custom callback to update tqdm progress bar
class TQDMProgressBar(xgb.callback.TrainingCallback):
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Training", unit="round")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False 

    def after_training(self, model):
        self.pbar.close()
        return model

progress_bar = TQDMProgressBar(total=num_rounds)
start_time = time.time()

bst = xgb.train(
    params, 
    dtrain, 
    num_rounds, 
    evals=[(dval, 'validation')],
    early_stopping_rounds=10,
    callbacks=[progress_bar]
)

# Record end time and calculate total time taken
print(f"Training time: {time.time() - start_time:.2f} seconds")

# Predictions on the test set
y_pred = bst.predict(dval)
y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
y_pred_decoded = label_encoder.inverse_transform(y_pred_labels)

# Main Classification Metrics
accuracy = accuracy_score(y_val, y_pred_labels)
print(f"Validation Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_val, y_pred_labels, target_names=label_encoder.classes_))

# Compute confusion matrix
cm = confusion_matrix(y_val, y_pred_labels)

# Convert to percentage for formatting
cm_percentage = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# Display confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_percentage, 
    annot=True, 
    fmt=".1%",
    cmap="Blues", 
    xticklabels=label_encoder.classes_, 
    yticklabels=label_encoder.classes_
)
plt.title("Confusion MatrixFor Test Data")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

''' The XGB Feature Importance did not work to our expectations, thus only included for completeness. 
The following code was utilized to produce "feat-_.png" figures in /xgb_figures '''

# # Feature importance
# booster = bst  # bst is already a booster object
# correct_feature_names = ["flux", "loglam", "ivar", "model",
#                          "z", "z_err", 
#                          "sn_median_uv", "sn_median_r", "sn_median_nir", "sn_median_ir",
#                          "zwarning", "rchi2"]

# feature_importance = booster.get_score(importance_type='weight')
# print("Feature Importance Keys from XGBoost:", feature_importance.keys())
# mapped_importance = defaultdict(int)  # Default to 0 for missing features

# for f, imp in feature_importance.items():
#     if f.startswith("f"):  
#         try:
#             idx = int(f[1:])  # Get the feature index and find the correct feature name
#             if 0 <= idx < len(correct_feature_names): 
#                 mapped_importance[correct_feature_names[idx]] = imp
#             else:
#                 print(f"Warning: Feature index {idx} out of range, skipping...")
#         except ValueError:
#             print(f"Warning: Unexpected feature format '{f}' ignored.")

# sorted_importance = sorted(mapped_importance.items(), key=lambda x: x[1], reverse=True)  # Sort by importance
# feature_names, importance_values = zip(*sorted_importance)

# print("\nFeature Importance Values:")
# for name, importance in sorted_importance:
#     print(f"{name}: {importance}")

# plt.figure(figsize=(10, 6))
# plt.barh(feature_names, importance_values, color='skyblue')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.title('Feature Importance (Corrected)')
# plt.gca().invert_yaxis() 
# plt.grid(axis='x', linestyle='--', alpha=0.7) 
# plt.show()
