# StarDustAI

Repository for the course project of **ECE324: Machine Intelligence, Software, and Neural Networks**, a third year Engineering Science course at the University of Toronto taken in 2025 as part of the Machine Intelligence major.

## Overview
The goal of StarDustAI is to utilize SDSS (eBOSS) data to correctly classify galaxies based on spectral and redshift data using a CNN.


## Fetching Data
The data is fetched from the SDSS database using the `2_fetch_data.py` script. The script fetches the data requested in '1_generate_file_names.py' and saves each file as a `.fits` file in the `data/full` directory. To access the data used in the project directly, please refer to this following link: [Dataset](https://utoronto-my.sharepoint.com/:u:/g/personal/sarvnaz_ale_mail_utoronto_ca/EdfmeoDkF6BFlp5z9fwOt2oBw71Qc-u0pft_NT2IOoSc7Q?e=uWI8a1)

### Current Progress
- Started data exploration
    - Understanding eBOSS experiments and the science behind spectroscopy measurements
    - Selecting variables to utilize
- Started data preprocessing

## Disclaimer
Projects may differ between instructional years and professors. This code is provided as-is and may contain minor errors or bugs. It is intended for educational and experimental purposes only. Please do not take the results for granted and use it at your own risk. 
