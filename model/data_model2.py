from torch.utils.data import Dataset, DataLoader, random_split
from astropy.table import Table, hstack
from astropy.io import fits
import pandas as pd
from astropy.utils.metadata import MergeConflictWarning
import torch
import numpy as np
import pickle as pkl
import warnings
warnings.simplefilter('ignore', MergeConflictWarning)


class SepctraDataset(Dataset): 

    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.class_categories = ['STAR', 'GALAXY', 'QSO']
        self.subclass_categories = ['nan', 'Starforming', 'Starburst', 'AGN', 'O', 'OB', 'B6', 'B9', 'A0', 'A0p', 'F2', 'F5', 'F9', 'G0', 'G2', 'G5', 'K1', 'K3', 'K5', 'K7', 'M0V', 'M2V', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L5.5', 'L9', 'T2', 'Carbon', 'Carbon_lines', 'CarbonWD', 'CV', 'BROADLINE']
        self.plate_quality_tags = {'bad': 0,  'marginal': 1, 'good': 2, 'nan': np.nan}
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]  # Pick the file based on the given index
        with open(file_path, 'rb') as f:
            df = pkl.load(f)
            class_label = df['CLASS'][0]
            plate_quality = df['PLATEQUALITY'][0]
            if plate_quality not in self.plate_quality_tags:
                plate_quality = 'nan'
            plate_quality = self.plate_quality_tags[plate_quality]
            df['PLATEQUALITY'] = np.array([plate_quality] * df.shape[0])
            
            # One-hot encode the class label
            class_one_hot = np.zeros(len(self.class_categories))
            class_one_hot[self.class_categories.index(class_label)] = 1
            
            # Remove categorical columns
            df.drop(['CLASS', 'SUBCLASS'], axis=1, inplace=True)
            
            class_label_tensor = torch.tensor(class_one_hot, dtype=torch.long)
            
            # Fill NaNs and convert to tensor
            features = df.fillna(0).astype(np.float32)
            features_tensor = torch.tensor(features.values, dtype=torch.float32)
            
            return features_tensor, class_label_tensor

#Custom collate function for padding rows
def collate_fn(batch):
    features, class_labels = zip(*batch)

    # Find the maximum number of rows in the batch
    max_rows = max(f.size(0) for f in features)

    # Pad each feature tensor to the maximum number of rows
    padded_features = []
    for f in features:
        padding = torch.zeros((max_rows - f.size(0), f.size(1)), dtype=f.dtype)
        padded_f = torch.cat([f, padding], dim=0)
        padded_features.append(padded_f)

    # # Convert class and subclass labels to tensors
    class_labels_tensor = torch.stack([label.clone().detach().to(torch.float32) for label in class_labels])
    #subclass_labels_tensor = torch.stack([label.clone().detach().to(torch.float32) for label in subclass_labels])

    return torch.stack(padded_features), class_labels_tensor


    