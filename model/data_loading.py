from torch.utils.data import Dataset, DataLoader
from astropy.table import Table, hstack
import glob
import torch

# List all FITS files
file_paths = glob.glob("data/full/10227/*.fits")

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
        dat2 = Table.read(file_path, format='fits', hdu=2)
        combined_table = hstack([dat1, dat2])  # Merge HDUs

        # Convert to NumPy and then to Tensor
        data = torch.tensor(combined_table.to_pandas().values, dtype=torch.float32)
        
        return data  # Return a single FITS file as a tensor

# Create a DataLoader for batch processing
dataloader = DataLoader(FitsDataset(file_paths), batch_size=2, shuffle=True, num_workers=4)

# Iterate through batches
for batch in dataloader:
    print(batch.shape)  # Print batch shape