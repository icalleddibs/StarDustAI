import xgboost as xgb
import pandas as pd
import sklearn as sk
from sk.metrics import mean_squared_error, accuracy_score
from sk.model_selection import train_test_split
from sk.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

'''
XGBoost Model Pipeline
Goals: Determine baseline evaluation metric results, get feature importance, get hyperparameters
'''

# Import data (tbd)
df = pd.read('name')

# Set the X and Y values 
# Y (to predict) = class
# X (feature data) = remaining data, use df.drop(columns=['name'])

# Our data has already been through OneHotEncoding, particularly for the classes
'''One Hot Encoding of Classes
0: 
1:
2:
'''

# Prepare data by splitting into train, test sete
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Select hyperparameters (to tune)
# XGB is a decision-tree model, so hyperparameters are based on those variables
params = {
  'need to find them': 'value',
  'etc': 0
}

# Training
boost = xgb.train(params, x_train, num_boost_round=100)

# Get the predictions on testing data
preds = boost.predict(x_test)

# Evaluation
# XGB uses Root Mean Squared Error (RMSE) as loss for regression
# Need to check what it is for classification - MSE?

# If we want, we can run hyperparameter search to get best values, or just do it by hand

# Obtain metrics
'''
Metrics:
- Accuracy
- Confusion Matrix
- F1 / AUROC?
- R
'''

# Check feature importance so we can use this as knowledge for our model
xbg.plot_importance(boost)
plot.show()

# End of Pipeline


