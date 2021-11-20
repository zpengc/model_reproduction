import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import time

"""
xgboost只支持libsvm和csv的文件形式
"""
train_npz = np.load(r'C:\Users\zpengc\Desktop\ct_train.npz')
test_npz = np.load(r'C:\Users\zpengc\Desktop\ct_test.npz')

# parameters for tree booster
param = {
    'learning_rate': 0.1,  # xgb's eta
    'n_estimators': 1024,  # Number of gradient boosted trees/number of boosting rounds.
    'max_depth': 7,  # Maximum tree depth for base learners
    'reg_lambda': 0.0  # l2 regularization
}

# https://www.pythontutorial.net/python-basics/python-kwargs/
# Scikit-Learn Wrapper interface for XGBoost.
model = xgb.XGBRegressor(objective='reg:squarederror', verbosity=2, seed=0, **param)

st = time.time()
model.fit(train_npz['features'], train_npz['labels'])
print("time for models fit: {} seconds".format(time.time()-st))

mse = mean_squared_error(test_npz['labels'], model.predict(test_npz['features']))
print(np.sqrt(mse))
