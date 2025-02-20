from torch.utils.data import Dataset, DataLoader, random_split
from astropy.table import Table, hstack
from astropy.utils.metadata import MergeConflictWarning
import glob
import torch
import random 
import os 
import subprocess
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import pandas as pd 


warnings.simplefilter('ignore', MergeConflictWarning)
# List all FITS files
# Get the repo root (assumes script is inside STARDUSTAI/)
repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
base_dir = os.path.join(repo_root, "data/full")
file_paths = glob.glob(os.path.join(base_dir, "*/*.fits"))

# If no FITS files are found, raise an error
if not file_paths:
    raise ValueError("No FITS files found in 'data/full/'")

# Shuffle the file paths
random.shuffle(file_paths)


# Custom PyTorch dataset for lazy loading
class FitsDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths  # Store file paths
        self.class_categories = ['STAR', 'GALAXY', 'QSO']
        self.subclass_categories = ['nan', 'Starforming', 'Starburst', 'AGN', 'O', 'OB', 'B6', 'B9', 'A0', 'A0p', 'F2', 'F5', 'F9', 'G0', 'G2', 'G5', 'K1', 'K3', 'K5', 'K7', 'M0V', 'M2V', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L5.5', 'L9', 'T2', 'Carbon', 'Carbon_lines', 'CarbonWD', 'CV', 'BROADLINE']
        self.plate_quality_tags = {'bad': 0,  'marginal': 1, 'good': 2, 'nan': np.nan}

    def __len__(self):
        return len(self.file_paths)  # Total number of files

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # Read FITS file as Astropy Table
        dat1 = Table.read(file_path, format='fits', hdu=1)
        dat1 = dat1['flux', 'loglam', 'ivar', 'model']
        dat2 = Table.read(file_path, format='fits', hdu=2)
        dat2 = dat2['PLATEQUALITY', 'PLATESN2', 'PLATE', 'TILE', 'MJD', 'FIBERID',  'CLASS', "SUBCLASS", 'Z', 'Z_ERR', 'SN_MEDIAN', 'SN_MEDIAN_ALL', 'ZWARNING' , 'RCHI2']
        data = hstack([dat1, dat2])  # Merge HDUs
        sn_median_values = np.vstack(data['SN_MEDIAN'])  # Shape: (4590, 5)

        # Add new columns for each filter
        data['SN_MEDIAN_UV'] = sn_median_values[:, 0]  # Ultraviolet
        data['SN_MEDIAN_G'] = sn_median_values[:, 1]   # Green
        data['SN_MEDIAN_R'] = sn_median_values[:, 2]   # Red
        data['SN_MEDIAN_NIR'] = sn_median_values[:, 3] # Near-Infrared
        data['SN_MEDIAN_IR'] = sn_median_values[:, 4]  # Infrared

        # Remove the original SN_MEDIAN column if needed
        data.remove_column('SN_MEDIAN')

        # Convert Astropy Table to Pandas DataFrame
        df = data.to_pandas()

        # Map PLATEQUALITY to numerical values and fill NaNs with same value
        df['PLATEQUALITY'] = df['PLATEQUALITY'].astype(str).map(self.plate_quality_tags)
        first_value = df['PLATEQUALITY'].iloc[0]
        df.fillna(value = {'PLATEQUALITY': first_value}, inplace=True)
       

        # one hot encode class
        df['CLASS'] = df['CLASS'].astype(str)
        class_label = df['CLASS'].values[0] 
        class_one_hot = np.zeros(len(self.class_categories))
        class_one_hot[self.class_categories.index(class_label)] = 1

        # one hot encode subclass
        df['SUBCLASS'] = df['SUBCLASS'].astype(str)
        subclass_label = df['SUBCLASS'].values[0]
        if subclass_label not in self.subclass_categories:
            subclass_label = 'nan'
        subclass_one_hot = np.zeros(len(self.subclass_categories))
        subclass_one_hot[self.subclass_categories.index(subclass_label)] = 1

        # Drop class and subclass columns and keep everything else 
        features = df.drop(columns=['CLASS', 'SUBCLASS'])
        features = features.fillna(0) 
        features = features.astype(np.float32) 

        features_tensor = torch.tensor(features.values, dtype=torch.float32)
        class_label_tensor = torch.tensor(class_one_hot, dtype=torch.long)
        subclass_label_tensor = torch.tensor(subclass_one_hot, dtype=torch.long)

        return features_tensor, class_label_tensor, subclass_label_tensor  



## test single file then use print stuff in the class itself 
# ------
# # # Create a DataLoader for batch processing
# dataloader = DataLoader(FitsDataset(file_paths), batch_size=1, shuffle=True)
# train_loader = DataLoader(FitsDataset(file_paths), batch_size=1, shuffle=True)
# first_batch = next(iter(dataloader))

# # Iterate through batches
# for batch in dataloader:
#     print(batch.shape)  # Print batch shape
# ------

#Custom collate function for padding rows
# Custom collate function for padding rows
def collate_fn(batch):
    features, class_labels, subclass_labels = zip(*batch)

    # Find the maximum number of rows in the batch
    max_rows = max(f.size(0) for f in features)

    # Pad each feature tensor to the maximum number of rows
    padded_features = []
    for f in features:
        padding = torch.zeros((max_rows - f.size(0), f.size(1)), dtype=f.dtype)
        padded_f = torch.cat([f, padding], dim=0)
        padded_features.append(padded_f)

    # Convert class and subclass labels to tensors
    class_labels_tensor = torch.stack([torch.tensor(label, dtype=torch.float32) for label in class_labels])
    subclass_labels_tensor = torch.stack([torch.tensor(label, dtype=torch.float32) for label in subclass_labels])

    return torch.stack(padded_features), class_labels_tensor, subclass_labels_tensor


dataset = FitsDataset(file_paths)

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))  
val_size = int(0.15 * len(dataset))    
test_size = len(dataset) - train_size - val_size  

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders with the updated collate function
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Example usage
for batch in train_loader:
    features, class_labels, subclass_labels = batch
    print("Features batch shape:", features.shape)  # (batch_size, max_rows, num_features)
    print("Class labels batch shape:", class_labels.shape)  # (batch_size, 3)
    print("Subclass labels batch shape:", subclass_labels.shape)  # (batch_size, 44)
    break