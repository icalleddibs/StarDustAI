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

    def __len__(self):
        return len(self.file_paths)  # Total number of files

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # Read FITS file as Astropy Table
        dat1 = Table.read(file_path, format='fits', hdu=1)
        dat1 = dat1['flux', 'loglam', 'ivar', 'model']
        dat2 = Table.read(file_path, format='fits', hdu=2)
        dat2 = dat2['PLATEQUALITY', 'PLATESN2', 'PLATE', 'TILE', 'MJD', 'FIBERID', 'OBJTYPE', 'CLASS', "SUBCLASS", 'Z', 'Z_ERR', 'SN_MEDIAN', 'SN_MEDIAN_ALL', 'ZWARNING' , 'RCHI2']
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

        # Convert string columns to numerical categories
        for col in ['PLATEQUALITY', 'OBJTYPE', 'CLASS', 'SUBCLASS']:  # Columns that are strings
            df[col] = df[col].astype('category').cat.codes  # Convert to numerical
       
        # Print the data types and shapes of each column
        #for col in data.colnames:
        #   print(f"Column: {col}, Data Type: {data[col].dtype},  Shape: {data[col].shape}")
        
      
        #print("Data types in DataFrame before tensor conversion:")
        df = df.fillna(0) #IS THIS OK???? 
        df = df.astype(np.float32) # IS THIS OK???
        # print(df.dtypes)

        # print(df.head())
        
        # Convert Pandas DataFrame to PyTorch tensor
        data_tensor = torch.tensor(df.values, dtype=torch.float32)
        #print(data_tensor.shape)
        return data_tensor  # Return a single FITS file as a tensor

# # # Create a DataLoader for batch processing
# dataloader = DataLoader(FitsDataset(file_paths), batch_size=1, shuffle=True)
# train_loader = DataLoader(FitsDataset(file_paths), batch_size=1, shuffle=True)
# first_batch = next(iter(dataloader))

# # Iterate through batches
# for batch in dataloader:
#     print(batch.shape)  # Print batch shape


# Custom collate function for padding rows
def collate_fn(batch):
    # Find the maximum number of rows in the batch
    max_rows = max(tensor.size(0) for tensor in batch)

    # Pad each tensor to the maximum number of rows
    padded_batch = []
    for tensor in batch:
        # Create a new tensor filled with zeros for padding
        padding = torch.zeros((max_rows - tensor.size(0), tensor.size(1)), dtype=tensor.dtype)
        # Concatenate the original tensor with the padding
        padded_tensor = torch.cat([tensor, padding], dim=0)
        padded_batch.append(padded_tensor)

    return torch.stack(padded_batch)


dataset = FitsDataset(file_paths)

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))  # 70% for training
val_size = int(0.15 * len(dataset))    # 15% for validation
test_size = len(dataset) - train_size - val_size  # Remaining 15% for testing

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each dataset with the custom collate function
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Example usage
for batch in train_loader:
    print("Training batch shape:", batch.shape)
    break  # Just to show the first batch

for batch in val_loader:
    print("Validation batch shape:", batch.shape)
    break  # Just to show the first batch

for batch in test_loader:
    print("Testing batch shape:", batch.shape)
    break  # Just to show the first batch
