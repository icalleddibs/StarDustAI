import glob
import os 
from tqdm import tqdm 
import matplotlib.pyplot as plt
import time  

# XGBoost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Scientific Python 
from astropy.table import Table
from astropy.io import fits
import numpy as np

# repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
# base_dir = os.path.join(repo_root, "data/full/")
# file_paths = glob.glob(os.path.join(base_dir, "*/*.fits"))

base_dir = "E:/StarDustAI/full"
file_paths = glob.glob(os.path.join(base_dir, "*/*.fits"))

# If no FITS files are found, raise an error
if not file_paths:
    raise ValueError("No FITS files found in 'data/full/'")
# Currently only using 10000 plate numbers

# Initialize lists to store features and labels
X = []
y = []

i = 0
max_length = 4615 # For the padding

# Loop through each file and extract data
for file_path in file_paths:
    
    # check for missing
    with fits.open(file_path) as hdul:
        if len(hdul) <= 1:  # If HDU 2 doesn't exist, skip this file
            print(f"Skipping {file_path}: HDU 1 not found.")
            continue
        
    with fits.open(file_path) as hdul:
        if len(hdul) <= 2:  # If HDU 2 doesn't exist, skip this file
            print(f"Skipping {file_path}: HDU 2 not found.")
            continue
        
    # ERRORS:
    # 10227/spec-10227-58224-0419.fits - HDU 2 not found
    
    # Read FITS file
    dat1 = Table.read(file_path, format='fits', hdu=1)  # HDU 1
    dat2 = Table.read(file_path, format='fits', hdu=2)  # HDU 2

    # Extract relevant columns from HDU 1
    flux = dat1['flux'].data
    loglam = dat1['loglam'].data
    ivar = dat1['ivar'].data
    model = dat1['model'].data

    # Extract relevant columns from HDU 2
    platequality = dat2['PLATEQUALITY'].data
    platesn2 = dat2['PLATESN2'].data
    plate = dat2['PLATE'].data
    tile = dat2['TILE'].data
    mjd = dat2['MJD'].data
    fiberid = dat2['FIBERID'].data
    class_label = dat2['CLASS'].data
    subclass = dat2['SUBCLASS'].data
    z = dat2['Z'].data
    z_err = dat2['Z_ERR'].data
    sn_median = dat2['SN_MEDIAN'].data
    sn_median_all = dat2['SN_MEDIAN_ALL'].data
    zwarning = dat2['ZWARNING'].data
    rchi2 = dat2['RCHI2'].data

    # Reshape SN_MEDIAN into individual filter columns
    sn_median_uv = sn_median[:, 0]  # Ultraviolet
    sn_median_g = sn_median[:, 1]   # Green
    sn_median_r = sn_median[:, 2]   # Red
    sn_median_nir = sn_median[:, 3] # Near-Infrared
    sn_median_ir = sn_median[:, 4]  # Infrared
    
    def pad_or_truncate(array, max_length):
        if len(array) > max_length:
            return array[:max_length]  # Truncate if too long
        return np.pad(array, (0, max_length - len(array)), mode='constant')  # Pad if too short

    # Apply the function to all arrays
    flux = pad_or_truncate(flux, max_length)
    loglam = pad_or_truncate(loglam, max_length)
    ivar = pad_or_truncate(ivar, max_length)
    model = pad_or_truncate(model, max_length)
    platequality = pad_or_truncate(platequality, max_length)
    platesn2 = pad_or_truncate(platesn2, max_length)
    plate = pad_or_truncate(plate, max_length)
    tile = pad_or_truncate(tile, max_length)
    mjd = pad_or_truncate(mjd, max_length)
    fiberid = pad_or_truncate(fiberid, max_length)
    z = pad_or_truncate(z, max_length)
    z_err = pad_or_truncate(z_err, max_length)
    sn_median_uv = pad_or_truncate(sn_median_uv, max_length)
    sn_median_g = pad_or_truncate(sn_median_g, max_length)
    sn_median_r = pad_or_truncate(sn_median_r, max_length)
    sn_median_nir = pad_or_truncate(sn_median_nir, max_length)
    sn_median_ir = pad_or_truncate(sn_median_ir, max_length)
    zwarning = pad_or_truncate(zwarning, max_length)
    rchi2 = pad_or_truncate(rchi2, max_length)

    # Combine all features into a single array
    # TOOK OUT PLATE QUALITY, platesn2, plate, tile, mjd, fiberid
    features = np.column_stack([
        flux, loglam, ivar, model,  
        z, z_err, 
        sn_median_uv, sn_median_g, sn_median_r, sn_median_nir, sn_median_ir,
        zwarning, rchi2
    ])

    # Append features and labels to the lists
    X.append(features)
    #print(features)
    y.append(class_label)  # Assuming class_label is the target
    #print(class_label)
    
    #i = i + 1
    #print("APPENDED ", i)
    
# Check the shape of each sample because I was having issues
shapes = [sample.shape for sample in X]
print(f"Unique shapes: {set(shapes)}")

# Convert to a single numpy array
X = np.array(X, dtype=float)
y = np.array(y)

# Flatten the features
X_flattened = X.reshape(X.shape[0], -1)

# Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_flattened, y_encoded, test_size=0.2, random_state=42)

# Create DMatrix for training and validation
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
    'lambda': 1,
    'seed': 42
}

# Train the model, can modify
num_rounds = 50

start_time = time.time()

# Custom callback to update tqdm progress bar
class TQDMProgressBar(xgb.callback.TrainingCallback):
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Training", unit="round")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False  # Return False to continue training

    def after_training(self, model):
        self.pbar.close()
        return model

# Initialize the progress bar
progress_bar = TQDMProgressBar(total=num_rounds)

bst = xgb.train(
    params, 
    dtrain, 
    num_rounds, 
    evals=[(dval, 'validation')],  # Evaluate on the validation set
    early_stopping_rounds=10,      # Stop if validation performance doesn't improve for 10 rounds
    callbacks=[progress_bar]       # Update the progress bar
)


# Record end time and calculate total time taken
end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")


# Evaluate the model
y_pred = bst.predict(dval)
y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

y_pred_decoded = label_encoder.inverse_transform(y_pred_labels)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred_labels)
print(f"Validation Accuracy: {accuracy:.4f}")

# Print classification report
print("Classification Report:")
print(classification_report(y_val, y_pred_labels, target_names=label_encoder.classes_))

# Generate confusion matrix
cm = confusion_matrix(y_val, y_pred_labels)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)

# Show the plot
plt.show()


# PROBLEMS ------------------------------------------------------------------------------
# Traceback (most recent call last):
#   File "c:\Users\diba\StarDustAI\model\XGBoost\xgboost_model.py", line 229, in <module>
#     booster = bst.get_booster()
#               ^^^^^^^^^^^^^^^
# AttributeError: 'Booster' object has no attribute 'get_booster'


# Get the booster and plot feature importance
booster = bst.get_booster()

# Get the feature importance
feature_importance = booster.get_score(importance_type='weight')

# Sort feature importance
sorted_feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

# Extract sorted feature names and importance values
features_names = [name for name, _ in sorted_feature_importance]
importance_values = [importance for _, importance in sorted_feature_importance]

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(features_names, importance_values, color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()