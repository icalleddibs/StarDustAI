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
# Get the repository root (assumes script is inside STARDUSTAI/)
repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
base_dir = os.path.join(repo_root, "data/full")
file_paths = glob.glob(os.path.join(base_dir, "*/*.fits"))

# Raise an error if no FITS files are found
if not file_paths:
    raise ValueError("No FITS files found in 'data/full/'")


def process_valid_fits(hdul):
    """
    Process valid FITS data into a pandas DataFrame.

    This function extracts and processes data from the second and third HDUs
    in a FITS file, reshaping and padding it as necessary.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        Opened FITS file HDU list.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with extracted and padded FITS data.
    """
    dat1 = hdul[1].data
    dat1 = np.array(
        [dat1['flux'], dat1['loglam'], dat1['ivar'], dat1['model']]
    ).T    
    dat2 = hdul[2].data
    sn_median_values = np.vstack(dat2['SN_MEDIAN']) 
    sn_median_values = np.pad(
        sn_median_values,
        ((0, dat1.shape[0] - 1), (0, 0)),
        'constant',
        constant_values=0
    )    
    dat2 = np.array(
        [
            dat2['CLASS'], dat2['SUBCLASS'], dat2['PLATESN2'],
            dat2['Z'], dat2['Z_ERR'],
            dat2['ZWARNING'], dat2['RCHI2'], dat2['PLATEQUALITY']
        ]
    ).T
    dat2 = np.pad(
        dat2,
        ((0, dat1.shape[0] - 1), (0, 0)),
        'constant',
        constant_values=0
    )
    # Merge HDUs
    data = np.hstack([dat1, dat2])  
    df = pd.DataFrame(
        data,
        columns=[
            'flux', 'loglam', 'ivar', 'model', 'CLASS', 'SUBCLASS',
            'PLATESN2', 'Z', 'Z_ERR', 'ZWARNING', 'RCHI2', 'PLATEQUALITY'
        ]
    )
    df['SN_MEDIAN_UV'] = sn_median_values[:, 0]
    df['SN_MEDIAN_R'] = sn_median_values[:, 2]
    df['SN_MEDIAN_NIR'] = sn_median_values[:, 3]
    df['SN_MEDIAN_IR'] = sn_median_values[:, 4]
    return df

# Process FITS files and save data with ZWARNING = 0 or 16 as a .pkl file
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

