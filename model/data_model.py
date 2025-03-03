from torch.utils.data import Dataset, DataLoader, random_split
from astropy.table import Table, hstack
from astropy.io import fits
import pandas as pd
from astropy.utils.metadata import MergeConflictWarning
import torch
import numpy as np

import warnings
warnings.simplefilter('ignore', MergeConflictWarning)

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
        
        with fits.open(file_path) as hdul:
            dat1 = hdul[1].data            
            dat1 = np.array([dat1['flux'], dat1['loglam'], dat1['ivar'], dat1['model']]).T
            dat2 = hdul[2].data
            
            class_label = dat2['CLASS'][0]
            subclass_label = dat2['SUBCLASS'][0]

            sn_median_values = np.vstack(dat2['SN_MEDIAN'])  # Shape: (4590, 5)
            sn_median_values = np.pad(sn_median_values, ((0, dat1.shape[0]-1), (0, 0)), 'constant', constant_values=0)
            
            plate_quality = dat2['PLATEQUALITY'][0]
            
            # map it to a numerical value 
            if plate_quality not in self.plate_quality_tags:
                plate_quality = 'nan'
            plate_quality = self.plate_quality_tags[plate_quality]
            
            dat2 = np.array([dat2['PLATESN2'],  
                 dat2['Z'], dat2['Z_ERR'], 
                 dat2['ZWARNING'], dat2['RCHI2']]).T
            dat2 = np.pad(dat2, ((0, dat1.shape[0]-1), (0, 0)), 'constant', constant_values=0)
            data = np.hstack([dat1, dat2])  # Merge HDUs
           

            df = pd.DataFrame(data, columns=['flux', 'loglam', 'ivar', 'model', 'PLATESN2', 'Z', 'Z_ERR', 'ZWARNING', 'RCHI2'])
            
            df['SN_MEDIAN_UV'] = sn_median_values[:, 0]
            df['SN_MEDIAN_R'] = sn_median_values[:, 2]
            df['SN_MEDIAN_NIR'] = sn_median_values[:, 3]
            df['SN_MEDIAN_IR'] = sn_median_values[:, 4]
            df['PLATEQUALITY'] = np.array([plate_quality]*dat1.shape[0])

            class_one_hot = np.zeros(len(self.class_categories))
            class_one_hot[self.class_categories.index(class_label)] = 1

            if subclass_label not in self.subclass_categories:
                subclass_label = 'nan'
            
            subclass_one_hot = np.zeros(len(self.subclass_categories))
            subclass_one_hot[self.subclass_categories.index(subclass_label)] = 1
            
            features = df
            features = features.fillna(0)
            features = features.astype(np.float32)
            features_tensor = torch.tensor(features.values, dtype=torch.float32)
            class_label_tensor = torch.tensor(class_one_hot, dtype=torch.long)
            subclass_label_tensor = torch.tensor(subclass_one_hot, dtype=torch.long)
            return features_tensor, class_label_tensor, subclass_label_tensor

#Custom collate function for padding rows
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

    # # Convert class and subclass labels to tensors
    class_labels_tensor = torch.stack([label.clone().detach().to(torch.float32) for label in class_labels])
    subclass_labels_tensor = torch.stack([label.clone().detach().to(torch.float32) for label in subclass_labels])

    return torch.stack(padded_features), class_labels_tensor, subclass_labels_tensor

