# StarDustAI

Repository for the course project of **ECE324: Machine Intelligence, Software, and Neural Networks**, a third year Engineering Science course at the University of Toronto taken in 2025 as part of the Machine Intelligence major.

## Overview
The goal of StarDustAI is to utilize SDSS (eBOSS) data to correctly classify galaxies based on spectral and redshift data using machine learning techniques. A secondary goal of the project was to assess feature importance and interpretability of the models used, which would lead to a better understanding of the underlying physics of galaxies, and insight into how to further simplify the necessary parameters for classification models to learn from. The overall project was designed to increase the time-efficiency of astronomers by reducing the time spent manually classifying galaxies, quasars, and stars, as well as to verify known relationships between spectral data and galaxy classification.

The project is divided into two main components: data preprocessing and model development. The data preprocessing component focuses on cleaning and preparing the SDSS data for analysis, while the model development component involves building and evaluating multiple machine learning models to classify galaxies based on their spectral features, as well as computing feature importance to verify known relationships between spectral data and galaxy classification.

The final model used for classification is a custom ResNet architecture called **FullFeaturesResNet** which was trained on the preprocessed SDSS data. The model was evaluated using various metrics, including accuracy, precision, recall, and F1-score. The results of the model evaluation and additional experiments for comparison are documented in the `report_figures` folder. The model was able to acheive **95% accuracy** on the test set, demonstrating its effectiveness in classifying galaxies based on their spectral features.

## Data
The data is fetched from the SDSS database using the `2_fetch_data.py` script. The script fetches the data requested in '1_generate_file_names.py' and saves each file as a `.fits` file in the `data/full` directory. To access the data used in the project directly, please refer to this following link: [Dataset](https://utoronto-my.sharepoint.com/:u:/g/personal/sarvnaz_ale_mail_utoronto_ca/EdfmeoDkF6BFlp5z9fwOt2oBw71Qc-u0pft_NT2IOoSc7Q?e=uWI8a1)

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


## Disclaimer
Projects may differ between instructional years and professors. This code is provided as-is and may contain minor errors or bugs. It is intended for educational and experimental purposes only. Please do not take the results for granted and use it at your own risk. 
