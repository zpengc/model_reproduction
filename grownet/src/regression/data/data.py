import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file


# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#loading-a-dataset
class LibSVMData(Dataset):

    # The __init__ function is run once when instantiating the Dataset object
    def __init__(self, root, dim, normalization, pos=1, neg=-1, out_pos=1, out_neg=-1):
        # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html
        self.feat, self.label = load_svmlight_file(root)

        # Compressed Sparse Row matrix
        self.feat = csr_matrix((self.feat.data, self.feat.indices, self.feat.indptr), shape=(len(self.label), dim))
        self.feat = self.feat.toarray().astype(np.float32)

        self.label = self.label.astype(np.float32)
        idx_pos = self.label == pos
        idx_neg = self.label == neg
        self.label[idx_pos] = out_pos
        self.label[idx_neg] = out_neg

    # The __getitem__ function loads and returns a sample from the dataset at the given index idx
    def __getitem__(self, index):
        arr = self.feat[index, :]
        return arr, self.label[index]

    # The __len__ function returns the number of samples in our dataset
    def __len__(self):
        return len(self.label)


# Map-style datasets
# https://pytorch.org/docs/stable/data.html#map-style-datasets
class LibSVMRegData(Dataset):

    def __init__(self, train_file_path, dim, normalization):
        data = np.load(train_file_path)  # 加载npz文件
        self.feat, self.label = data['features'], data['labels']
        del data
        self.feat = self.feat.astype(np.float32)
        self.label = self.label.astype(np.float32)
        # print(self.feat.shape[1])

    def __getitem__(self, index):
        return self.feat[index, :], self.label[index]

    def __len__(self):
        return len(self.label)


