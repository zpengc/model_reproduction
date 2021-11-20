import pandas as pd
import numpy as np

"""
CT slice localization数据集一共53500条记录，一共386列，第一列是病人id，最后一列是label，中间384列是特征值
"""
file = pd.read_csv(r"C:\Users\zpengc\Desktop\slice_localization_data.csv")
print(file.head(10))

# 根据病人id来划分训练集和测试集，保证同一个病人信息不同时出现在训练集和测试集中
u_pid = np.unique(file['patientId'])
print(u_pid, len(u_pid))

# 80%训练集，20%测试集
test_idx = np.arange(1, int(len(u_pid)/5) + 1)*5 - 1
print(test_idx)

test = np.nonzero(np.isin(file['patientId'], test_idx))[0]  # np.nonzero()返回二维元组
train = np.nonzero(~np.isin(file['patientId'], test_idx))[0]
print(len(test), len(train))

# https://stackoverflow.com/questions/31593201/how-are-iloc-and-loc-different
X_train,  y_train = file.iloc[train, 1:385].values, file.iloc[train, 385].values
X_test, y_test = file.iloc[test, 1:385].values, file.iloc[test, 385].values
print(X_train.shape, X_test.shape)

np.savez(r'C:\Users\zpengc\Desktop\ct_train.npz', features=X_train, labels=y_train)
np.savez(r'C:\Users\zpengc\Desktop\ct_test.npz', features=X_test, labels=y_test)
