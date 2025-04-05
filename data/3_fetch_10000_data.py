import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

# Base URL
BASE_URL = "https://dr17.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_2/spectra/full/10000/"
SAVE_DIR = "sdss_fits_files"
NUM_THREADS = 10 

# Create directory if not exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Get list of FITS files
response = requests.get(BASE_URL)
soup = BeautifulSoup(response.text, "html.parser")
fits_files = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.fits')]

# Download function
def download_file(filename):
    
    file_url = urljoin(BASE_URL, filename)
    save_path = os.path.join(SAVE_DIR, filename)

    try:
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"Failed to download {file}: {e}")

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    executor.map(download_file, fits_files)

print("Download complete!")
