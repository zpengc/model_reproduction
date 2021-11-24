import numpy as np
import pandas as pd

"""
音乐年份预测数据集一共515344条记录，91个列，第一列是label，后面90列是特征值
train: first 463,715 examples
test: last 51,629 examples
"""
file = pd.read_csv(r'C:\Users\zpengc\Desktop\YearPredictionMSD.txt')
print(file.head(5))

num_train = 463715

X_train, y_train = file.iloc[0:num_train, 1:].values, file.iloc[0:num_train, 0].values
X_test, y_test = file.iloc[num_train:, 1:].values, file.iloc[num_train:, 0].values
print(X_train.shape, y_train.shape)  # (463715, 90) (463715,)
print(X_test.shape, y_test.shape)   # (51629, 90) (51629,)

# Saving train and test into npz file
np.savez(r'C:\Users\zpengc\Desktop\music_train.npz', features=X_train, labels=y_train)
np.savez(r'C:\Users\zpengc\Desktop\music_test.npz', features=X_test, labels=y_test)