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
