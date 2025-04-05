# Model README
Folder for the Model elements of StarDustAI. Including details on baseline models, our models, and interpretability.

#### `model/cnn_experiments/`
Contains saved model checkpoints and training logs for previous model iterations. Used for result tracking and reproducibility. Also includes attempts at contrastive learning.

#### `model/XGBoost/`
Code and experiment results related to XGBoost-based baseline model. Includes training scripts, evaluation, and results.

- `cnn_model.py`: Defines FullFeaturesResNet, the final model architecture used for classification.
- `cnn_models_comparison.py`: Trains and evaluates multiple CNN architectures for side-by-side comparison across accuracy, precision, recall, and F1-score.
- `cnn_training.py`: Handles model training routines, model evaluation routines, logging, and saving.
- `data_model.py`: Custom PyTorch `Dataset` for loading preprocessed SDSS spectra data and `collate_fn` to support processing.
- `gradient_attribution.py`: Computes feature importance using gradient-based attribution.
- `hyperparameter_search.py`: Performs random search over learning rates, dropout, weight decay, and dilation across models with high potential.
- `metric_bootstrap.py`: Evaluates model robustness by bootstrapping evaluation metrics resulting in means and confidence intervals for each.


