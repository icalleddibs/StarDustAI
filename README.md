# StarDustAI

Repository for the course project of **ECE324: Machine Intelligence, Software, and Neural Networks**, a third year Engineering Science course at the University of Toronto taken in 2025 as part of the Machine Intelligence major.

## Overview
The goal of StarDustAI is to utilize SDSS (eBOSS) data to correctly classify galaxies based on spectral and redshift data using machine learning techniques. A secondary goal of the project was to assess feature importance and interpretability of the models used, which would lead to a better understanding of the underlying physics of galaxies, and insight into how to further simplify the necessary parameters for classification models to learn from. The overall project was designed to increase the time-efficiency of astronomers by reducing the time spent manually classifying galaxies, quasars, and stars, as well as to verify known relationships between spectral data and galaxy classification.

The project is divided into two main components: data preprocessing and model development. The data preprocessing component focuses on cleaning and preparing the SDSS data for analysis, while the model development component involves building and evaluating multiple machine learning models to classify galaxies based on their spectral features, as well as computing feature importance to verify known relationships between spectral data and galaxy classification.

The final model used for classification is a custom ResNet architecture called **FullFeaturesResNet** which was trained on the preprocessed SDSS data. The model was evaluated using various metrics, including accuracy, precision, recall, and F1-score. The results of the model evaluation and additional experiments for comparison are documented in the `report_figures` folder. The model was able to acheive **95% accuracy** on the test set, demonstrating its effectiveness in classifying galaxies based on their spectral features.

## Project Structure
The repository is organized into the following directories and files:
- `data/`: Contains the raw and preprocessed data used for training and evaluating the models.
  - `full/`: Empty folder for the raw data files in `.fits` format. The data is fetched from the SDSS database using the `2_fetch_data.py` script.
  - `full_zwarning/`: Contains the preprocessed data files in `.pkl` format. The preprocessed data is used for training and evaluating the models.
  - `hashes/` : Contains the hases of the fits files used for retrieving the data from the SDSS database.
- `model/`: Contains the code and experiments related to the machine learning models used in the project.
  - `cnn_experiments/`: Contains saved model checkpoints and training logs for previous model iterations.
  - `XGBoost/`: Contains code and experiment results related to the XGBoost-based baseline model.
- `report_figures/`: Contains the figures and results generated during the project, including model evaluation metrics and feature importance analysis.

## Data
The data used in this project is sourced from the Sloan Digital Sky Survey (SDSS) database, specifically the eBOSS dataset. The SDSS database provides a wealth of astronomical data, including spectral and redshift information for galaxies, quasars, and stars. The data is stored in `.fits` format, which is a standard format for astronomical data.

To select which data to fetch, we used the `1_generate_file_names.py` script. This script generates a list of file names based on the hash file provided by the SDSS database which are stored in `hashes/`. The list of filenames is saved in `speclist.txt`. The data is fetched from the SDSS database using the `2_fetch_data.py` script. The script fetches the data requested in '1_generate_file_names.py' and saves each file as a `.fits` file in the `data/full` directory. 

The fetched data is then preprocessed using the `4_preprocess.py` script. This script cleans and prepares the data for analysis, including removing unnecessary columns, handling missing values, and converting the data into a suitable format for machine learning models. The preprocessed data is saved in `.pkl` format in the `data/full_zwarning` directory.


To access the fits file data used in the project directly, please refer to this following link: [Dataset](https://utoronto-my.sharepoint.com/:u:/g/personal/sarvnaz_ale_mail_utoronto_ca/EdfmeoDkF6BFlp5z9fwOt2oBw71Qc-u0pft_NT2IOoSc7Q?e=uWI8a1)

## Models
Folder for the Model elements of StarDustAI. Including details on baseline models, our models, and interpretability.

#### `model/cnn_experiments/`
Contains saved model checkpoints and training logs for previous model iterations. Used for result tracking and reproducibility. Also includes an attempt at applying contrastive learning on the dataset to extract feature importance scores between objects of different classes.

#### `model/XGBoost/`
Code and experiment results related to XGBoost-based baseline model. Includes training scripts, evaluation figures, and fully documented results in `model/XGBoost/XGB_RESULTS.md`.

- `cnn_model.py`: Defines FullFeaturesResNet, the final model architecture used for classification.
- `cnn_models_comparison.py`: Trains and evaluates multiple CNN architectures for side-by-side comparison across accuracy, precision, recall, and F1-score.
- `cnn_training.py`: Handles model training routines, model evaluation routines, logging, and saving.
- `data_model.py`: Custom PyTorch `Dataset` for loading preprocessed SDSS spectra data and `collate_fn` to support processing.
- `gradient_attribution.py`: Computes feature importance using gradient-based attribution.
- `hyperparameter_search.py`: Performs random search over learning rates, dropout, weight decay, and dilation across models with high potential.
- `metric_bootstrap.py`: Evaluates model robustness by bootstrapping evaluation metrics resulting in means and confidence intervals for each.

## Getting Started
To get started with the project, follow these steps:
1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the `2_fetch_data.py` script to fetch the raw data from the SDSS database and save it in the `data/full` directory.
4. Preprocess the data using the `data_preprocessing.py` script to generate the preprocessed data files in the `data/full_zwarning` directory.
5. Train the models using the `cnn_training.py` script, which will automatically load the preprocessed data and train the models.
6. Evaluate the models using the `cnn_training.py` script, which will generate evaluation metrics and save them in the `report_figures` directory.
7. Explore the results and figures in the `report_figures` directory to analyze the model performance and feature importance.


## Disclaimer
Projects may differ between instructional years and professors. This code is provided as-is and may contain minor errors or bugs. It is intended for educational and experimental purposes only. Please do not take the results for granted and use it at your own risk. 
