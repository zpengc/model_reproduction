import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_svmlight_file
import time

"""
xgboost只支持libsvm和csv的文件形式
"""
train_x, train_y = load_svmlight_file(r'C:\Users\zpengc\Desktop\higgs_train.txt')
test_x, test_y = load_svmlight_file(r'C:\Users\zpengc\Desktop\higgs_test.txt')

# parameters for tree booster
param = {
    'learning_rate': 0.05,  # xgb's eta
    'n_estimators': 800,  # Number of gradient boosted trees/number of boosting rounds.
    'max_depth': 7,  # Maximum tree depth for base learners
    'reg_lambda': 0.02  # l2 regularization
}

# https://www.pythontutorial.net/python-basics/python-kwargs/
# Scikit-Learn Wrapper interface for XGBoost.
model = xgb.XGBRegressor(objective='binary:logistic', tree_method='hist', verbosity=2, random_state=0, **param)

st = time.time()
model.fit(train_x, train_y)
print("time for models fit: {} seconds".format(time.time()-st))

auc = roc_auc_score(test_y, model.predict(test_x))
print(auc)
