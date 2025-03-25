'''
Tried to change the way the features were loaded for extraction but it wasn't working
O/W same code as `xgb_train.py`
'''


import glob
import os 
import pickle 
import numpy as np
from tqdm import tqdm 
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set up paths and other variables
BASE_DIR = "E:/StarDustAI/data/full_zwarning/full_zwarning"
FILE_PATHS = glob.glob(os.path.join(BASE_DIR, '**/*.pkl'), recursive=True)

if not FILE_PATHS:
    raise ValueError("No pickle files found in the specified directory")

# Define features
FEATURES = ["flux", "loglam", "ivar", "model", 
            "Z", "Z_ERR", "SN_MEDIAN_UV", "SN_MEDIAN_R", 
            "SN_MEDIAN_NIR", "SN_MEDIAN_IR", "ZWARNING", "RCHI2"]
MAX_LENGTH = 4615  # For padding/truncating

def pad_or_truncate(array, max_length=MAX_LENGTH):
    ''' Pad or truncate an array to the specified length for XGB model '''
    array_len = len(array)
    if array_len > max_length:
        return array[:max_length]                                            # Truncate if too long
    elif array_len < max_length:
        return np.pad(array, (0, max_length - array_len), mode='constant')   # Pad if too short
    return array                                                             # No change if max size

# Initialize lists to store features and labels
X, y = [], []

# Load all pickle files and structure the data correctly
for pickle_file in tqdm(FILE_PATHS, desc="Loading Pickle Files", unit="file"):
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            
            # Stack features for one file into a row
            features = np.concatenate([
                pad_or_truncate(np.array(data[feature]), MAX_LENGTH).reshape(-1, 1)  # Convert Series to numpy array before reshape
                for feature in FEATURES
            ], axis=1)  # Combine the features horizontally into one row
                        
            # Add the features (one row per file)
            X.append(features.flatten())  # Flatten to a 1D vector for each file
            y.append(data["CLASS"][0])    # Get the class label
            
    except Exception as e:
        print(f"Error loading {pickle_file}: {e}")

# Convert to numpy arrays
X = np.array(X, dtype=float)
y = np.array(y)

# Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.15, random_state=42
)

# Create DMatrix specific to XGB for training and validation
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dval = xgb.DMatrix(data=X_val, label=y_val)

# Set XGBoost parameters
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
bst = xgb.train(
    params, 
    dtrain, 
    num_rounds, 
    evals=[(dval, 'validation')],
    early_stopping_rounds=10
)

# Feature importance
booster = bst  # bst is already a booster object
feature_importance = booster.get_score(importance_type='weight')

# Map feature importance back to feature names
mapped_importance = {FEATURES[int(f[1:])]: imp for f, imp in feature_importance.items()}

# Sort by importance
sorted_importance = sorted(mapped_importance.items(), key=lambda x: x[1], reverse=True)

# Print feature importance
print("Feature Importance Values:")
for name, importance in sorted_importance:
    print(f"{name}: {importance}")

# Plot feature importance
import matplotlib.pyplot as plt

feature_names, importance_values = zip(*sorted_importance)

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importance_values, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (Corrected)')
plt.gca().invert_yaxis() 
plt.grid(axis='x', linestyle='--', alpha=0.7) 
plt.show()
