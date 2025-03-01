import glob
import os 
import subprocess
from tqdm import tqdm 
import matplotlib.pyplot as plt

# XGBoost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Scientific Python 
from astropy.table import Table
import numpy as np

repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
base_dir = os.path.join(repo_root, "data/full/")
file_paths = glob.glob(os.path.join(base_dir, "*/*.fits"))

# If no FITS files are found, raise an error
if not file_paths:
    raise ValueError("No FITS files found in 'data/full/'")

# Initialize lists to store features and labels
X = []
y = []

i = 0
max_length = 4615

# Loop through each file and extract data
for file_path in file_paths:
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
    
    # PADDING
    # Ensure all columns have the same length
    flux = np.pad(flux, (0, max_length - len(flux)), mode='constant')
    loglam = np.pad(loglam, (0, max_length - len(loglam)), mode='constant')
    ivar = np.pad(ivar, (0, max_length - len(ivar)), mode='constant')
    model = np.pad(model, (0, max_length - len(model)), mode='constant')
    platequality = np.pad(platequality, (0, max_length - len(platequality)), mode='constant')
    platesn2 = np.pad(platesn2, (0, max_length - len(platesn2)), mode='constant')
    plate = np.pad(plate, (0, max_length - len(plate)), mode='constant')
    tile = np.pad(tile, (0, max_length - len(tile)), mode='constant')
    mjd = np.pad(mjd, (0, max_length - len(mjd)), mode='constant')
    fiberid = np.pad(fiberid, (0, max_length - len(fiberid)), mode='constant')
    z = np.pad(z, (0, max_length - len(z)), mode='constant')
    z_err = np.pad(z_err, (0, max_length - len(z_err)), mode='constant')
    sn_median_uv = np.pad(sn_median_uv, (0, max_length - len(sn_median_uv)), mode='constant')
    sn_median_g = np.pad(sn_median_g, (0, max_length - len(sn_median_g)), mode='constant')
    sn_median_r = np.pad(sn_median_r, (0, max_length - len(sn_median_r)), mode='constant')
    sn_median_nir = np.pad(sn_median_nir, (0, max_length - len(sn_median_nir)), mode='constant')
    sn_median_ir = np.pad(sn_median_ir, (0, max_length - len(sn_median_ir)), mode='constant')
    zwarning = np.pad(zwarning, (0, max_length - len(zwarning)), mode='constant')
    rchi2 = np.pad(rchi2, (0, max_length - len(rchi2)), mode='constant')

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
    
    i = i + 1
    print("APPENDED ", i)
    
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

# Train the model
num_rounds = 100

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