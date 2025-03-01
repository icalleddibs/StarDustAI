# StarDustAI

Repository for the course project of **ECE324: Machine Intelligence, Software, and Neural Networks**, a third year Engineering Science course at the University of Toronto taken in 2025 as part of the Machine Intelligence major.

## Overview
The goal of StarDustAI is to utilize SDSS (eBOSS) data to correctly classify galaxies based on spectral and redshift data using a CNN.

## Fetching Data
The data is fetched from the SDSS database using the `2_fetch_data.py` script. The script fetches the data requested in '1_generate_file_names.py' and saves each file as a `.fits` file in the `data/full` directory. To access the data used in the project directly, please refer to this following link: [Dataset](https://utoronto-my.sharepoint.com/:u:/g/personal/sarvnaz_ale_mail_utoronto_ca/EdfmeoDkF6BFlp5z9fwOt2oBw71Qc-u0pft_NT2IOoSc7Q?e=uWI8a1)

## Current Progress
- Started data exploration
    - Understanding eBOSS experiments and the science behind spectroscopy measurements
    - Selecting variables to utilize (complete)
    - Exploring balances of different classes and data quality flags to determine data to use in our model
- Started data preprocessing
    - Removing unwanted variables
    - Creating suitable representation of data
        - Currently grouped into tensors
- Initial test of data
    - [XGBoost](https://xgboost.readthedocs.io/en/stable/): a powerful model often used as a baseline for machine learning models.
        - Running our data through an XGBoost model will set a baseline for reasonable results, as XGBoost performs reliably.
            - The classification metrics will set our expectations as we develop StarDustAI.
            - Knowing which features are most important for classification will shape our approach.
        - If we find that the results are bad (after hyperparameter tuning), we will reasses our dataset.
            - The hyperparameter required adjustments made will also inform our approach.
        - Gives us a comparison for our report
    - Simple CNN: one of the best methods for spectral analysis
        - Will use these findings to support the (baseline) CNN of StarDustAI

## Disclaimer
Projects may differ between instructional years and professors. This code is provided as-is and may contain minor errors or bugs. It is intended for educational and experimental purposes only. Please do not take the results for granted and use it at your own risk. 
