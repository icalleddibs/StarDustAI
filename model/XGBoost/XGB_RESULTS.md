# XGB Training Runs

## Run 1
- 100 boosts
- 950 files from 10000 plate
- params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_encoded)),
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
    }
- results:
Validation Accuracy: 0.9053
Classification Report:
              precision    recall  f1-score   support

   b'GALAXY'       0.86      0.93      0.89        80
      b'QSO'       0.93      0.84      0.88        77
     b'STAR'       0.97      1.00      0.99        33

    accuracy                           0.91       190
   macro avg       0.92      0.92      0.92       190
weighted avg       0.91      0.91      0.90       190

## Run 2
- 100 boosts
- 2.5k files from 10000 plate
- params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_encoded)),
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1,
    'seed': 42
    }
- results:
Validation Accuracy: 0.9028
Classification Report:
              precision    recall  f1-score   support

   b'GALAXY'       0.88      0.88      0.88       102
      b'QSO'       0.89      0.93      0.91       107
     b'STAR'       1.00      0.89      0.94        38

    accuracy                           0.90       247
   macro avg       0.92      0.90      0.91       247
weighted avg       0.90      0.90      0.90       247

## Run 3
- 100 boosts
- 16217 files from full dataset (except missing 10000)
- params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_encoded)),
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1,
    'seed': 42
    }
- results:
Validation Accuracy: 0.9627
Classification Report:
              precision    recall  f1-score   support

   b'GALAXY'       0.94      0.95      0.95      1119
      b'QSO'       0.97      0.96      0.96      1649
     b'STAR'       0.99      1.00      1.00       476

    accuracy                           0.96      3244
   macro avg       0.97      0.97      0.97      3244
weighted avg       0.96      0.96      0.96      3244

## Run 4
- 50 boosts
- full dataset
- params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_encoded)),
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1,
    'seed': 42
    }
- results:
Training time: 532.45 seconds
Validation Accuracy: 0.9575
Classification Report:
              precision    recall  f1-score   support

   b'GALAXY'       0.94      0.95      0.94      1417
      b'QSO'       0.97      0.95      0.96      2002
     b'STAR'       0.98      1.00      0.99       578

    accuracy                           0.96      3997
   macro avg       0.96      0.97      0.96      3997
weighted avg       0.96      0.96      0.96      3997

## Run 5
Official run used to represent the baseline model
- 100 boosts
- full dataset
- params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_encoded)),
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 0.8,
    'seed': 42
    }
- results:
- Training time: 1042.08 seconds
Validation Accuracy: 0.9557
Classification Report:
              precision    recall  f1-score   support

   b'GALAXY'       0.93      0.94      0.94      1417
      b'QSO'       0.97      0.95      0.96      2002
     b'STAR'       0.98      1.00      0.99       578
    accuracy                           0.96      3997
   macro avg       0.96      0.96      0.96      3997
weighted avg       0.96      0.96      0.96      3997
- **attempted feature importance but the XGB library didn't seem to work. will run again to test and use the new data loader.**

## Run 6
Using the cleaned pkl files
- 100 boosts
- full dataset
- params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_encoded)),
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 0.8,
    'seed': 42
    }
- results:
- Training time: 721.92 seconds
Validation Accuracy: 0.9873
Classification Report:
              precision    recall  f1-score   support

      GALAXY       0.98      0.98      0.98       950
         QSO       0.99      0.99      0.99      1574
        STAR       1.00      1.00      1.00       472

    accuracy                           0.99      2996
   macro avg       0.99      0.99      0.99      2996
weighted avg       0.99      0.99      0.99      2996
- **attempted feature importance but the XGB library didn't seem to work. will run again to test and use the new data loader.**

## Run 7
Using the cleaned pkl files and fixed feature importance
- 100 boosts
- full dataset
- params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_encoded)),
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 0.8,
    'seed': 42
    }
- results:
- Training time: 730.92 seconds
Validation Accuracy: 0.9873
Classification Report:
              precision    recall  f1-score   support

      GALAXY       0.98      0.98      0.98       950
         QSO       0.99      0.99      0.99      1574
        STAR       1.00      1.00      1.00       472 

    accuracy                           0.99      2996
   macro avg       0.99      0.99      0.99      2996
weighted avg       0.99      0.99      0.99      2996
- Feature Importance Values:
    z: 872.0
    z_err: 779.0
    rchi2: 135.0
    model: 34.0
    sn_median_ir: 13.0
    sn_median_r: 9.0
    sn_median_uv: 7.0
    flux: 4.0
    ivar: 4.0
    loglam: 3.0
    sn_median_nir: 3.0

## Run 8
Trying to update feature importance by concatenating all features into a row
- 100 boosts
- full dataset
- params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_encoded)),
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 0.8,
    'seed': 42
    }
- results:
Training time: 887.90 seconds
Validation Accuracy: 0.9875
Classification Report:
              precision    recall  f1-score   support

      GALAXY       0.98      0.98      0.98       729
         QSO       0.99      0.99      0.99      1172
        STAR       1.00      1.00      1.00       346

    accuracy                           0.99      2247
   macro avg       0.99      0.99      0.99      2247
weighted avg       0.99      0.99      0.99      2247
- Feature Importance Values:
    z: 910.0
    z_err: 788.0
    rchi2: 173.0
    model: 39.0
    sn_median_uv: 14.0
    sn_median_nir: 12.0
    sn_median_r: 11.0
    sn_median_ir: 6.0
    loglam: 5.0
    ivar: 5.0
    flux: 2.0

## Run 9
Checking consistency of parameters
- 100 boosts
- full dataset
- params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_encoded)),
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 0.8,
    'seed': 42
    }
- results:
Training time: 766.77 seconds
Validation Accuracy: 0.9875
Classification Report:
              precision    recall  f1-score   support

      GALAXY       0.98      0.98      0.98       729
         QSO       0.99      0.99      0.99      1172
        STAR       1.00      1.00      1.00       346

    accuracy                           0.99      2247
   macro avg       0.99      0.99      0.99      2247
weighted avg       0.99      0.99      0.99      2247


## Run 10
Final confusion matrix
- 100 boosts
- full dataset
- params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_encoded)),
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 0.8,
    'seed': 42
    }
- results:
Training time: 957.51 seconds
Validation Accuracy: 0.9875
Classification Report:
              precision    recall  f1-score   support

      GALAXY       0.98      0.98      0.98       729
         QSO       0.99      0.99      0.99      1172
        STAR       1.00      1.00      1.00       346

    accuracy                           0.99      2247
   macro avg       0.99      0.99      0.99      2247
weighted avg       0.99      0.99      0.99      2247