import glob
import subprocess
import os
from astropy.io import fits
import pandas as pd
from astropy.utils.metadata import MergeConflictWarning
import numpy as np
import pypickle as pkl
from tqdm import tqdm

import warnings
warnings.simplefilter('ignore', MergeConflictWarning)

# List all FITS files
# Get the repo root (assumes script is inside STARDUSTAI/)
repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
base_dir = os.path.join(repo_root, "data/full")
file_paths = glob.glob(os.path.join(base_dir, "*/*.fits"))

# If no FITS files are found, raise an error
if not file_paths:
    raise ValueError("No FITS files found in 'data/full/'")


def process_valid_fits(hdul):
    """
    Process the HDU list from a FITS file and return a DataFrame.
    """
    # Extract data from HDUs
    dat1 = hdul[1].data
    dat1 = np.array([dat1['flux'], dat1['loglam'], dat1['ivar'], dat1['model']]).T
    dat2 = hdul[2].data
    sn_median_values = np.vstack(dat2['SN_MEDIAN'])  # Shape: (4590, 5)
    sn_median_values = np.pad(sn_median_values, ((0, dat1.shape[0]-1), (0, 0)), 'constant', constant_values=0)
    
    dat2 = np.array([dat2['CLASS'], dat2['SUBCLASS'], dat2['PLATESN2'],  
                 dat2['Z'], dat2['Z_ERR'], 
                 dat2['ZWARNING'], dat2['RCHI2'], dat2['PLATEQUALITY']]).T
    dat2 = np.pad(dat2, ((0, dat1.shape[0]-1), (0, 0)), 'constant', constant_values=0)
    data = np.hstack([dat1, dat2])  # Merge HDUs
    df = pd.DataFrame(data, columns=['flux', 'loglam', 'ivar', 'model', 'CLASS', 'SUBCLASS', 'PLATESN2', 'Z', 'Z_ERR', 'ZWARNING', 'RCHI2', 'PLATEQUALITY'])
    df['SN_MEDIAN_UV'] = sn_median_values[:, 0]
    df['SN_MEDIAN_R'] = sn_median_values[:, 2]
    df['SN_MEDIAN_NIR'] = sn_median_values[:, 3]
    df['SN_MEDIAN_IR'] = sn_median_values[:, 4]
    return df

# go through filepaths, open fits files, save the fits file with zwarning = 0 or 16 and save it in a new folder as a pkl file 
for file_path in tqdm(file_paths, mininterval=3):
    with fits.open(file_path) as hdul:
        dat2 = hdul[2].data
        zwarning = dat2['ZWARNING'][0]
        if zwarning == 0 or zwarning == 16:
            new_file_path = file_path.replace('full', 'full_zwarning')
            new_file_path = new_file_path.replace('fits', 'pkl')
            df = process_valid_fits(hdul)
            os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
            pkl.save(new_file_path, process_valid_fits(hdul), verbose=0)


