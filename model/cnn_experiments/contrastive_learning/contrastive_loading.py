from torch.utils.data import Dataset
import torch
import numpy as np
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class SpectraDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.class_categories = ['STAR', 'GALAXY', 'QSO']
        self.plate_quality_tags = {'bad': 0, 'marginal': 1, 'good': 2, 'nan': np.nan}

        self.scaler = MinMaxScaler()  # Normalization

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with open(file_path, 'rb') as f:
            df = pkl.load(f)

        # Handle missing plate quality
        plate_quality = df['PLATEQUALITY'][0]
        plate_quality = self.plate_quality_tags.get(plate_quality, np.nan)
        df['PLATEQUALITY'] = np.array([plate_quality] * df.shape[0])

        # Convert class to integer
        class_label = self.class_categories.index(df['CLASS'][0])  # Use index instead of one-hot

        # Drop categorical columns
        df.drop(['CLASS', 'SUBCLASS'], axis=1, inplace=True)

        # Normalize numerical values
        df = df.fillna(0)  # Replace NaNs with 0
        df[['flux', 'loglam']] = self.scaler.fit_transform(df[['flux', 'loglam']])

        # Ensure all columns are numerical
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)  # Convert non-numeric values to NaN, then replace with 0

        # Convert to tensor
        features_tensor = torch.tensor(df.values, dtype=torch.float32)
        class_label_tensor = torch.tensor(class_label, dtype=torch.long)
        
        return features_tensor, class_label_tensor

def collate_fn(batch):
    features, labels = zip(*batch)
    return list(features), torch.tensor(labels, dtype=torch.long)
