import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class SWATSeqLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag    # train, val, test
        self.step = step    # step size
        self.win_size = win_size    # window size
        print(f"Loading SWaT {flag} data, window size is {win_size}, step is {step}")
        self.scaler = StandardScaler()
        train_data, test_data, labels = None, None, None
        if flag == 'train' or flag == 'val':
            train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
            train_data = train_data.values[:, :-1]  # 没有异常标签
            train_data = self.scaler.fit_transform(train_data)
            print("SWaT train data shape:", train_data.shape)
        elif flag == 'test':
            test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
            labels = test_data.values[:, -1:]
            test_data = test_data.values[:, :-1]
            test_data = self.scaler.fit_transform(test_data)
            print("SWaT test data shape:", test_data.shape)


        if flag == 'val':
            self.val = train_data[(int)(len(train_data) * 0.8):]
            del train_data
            train_data = None
        self.train = train_data
        self.test = test_data

        self.test_labels = labels

    def get_test_labels(self):
        return self.test_labels

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step 

    def __getitem__(self, index):
        index = index * self.step
        data, label = [], []
        if self.flag == "train":    
            data = self.train[index:index + self.win_size]
            label = self.train[index + self.win_size]   # 下一个数据点
        elif (self.flag == 'val'):  
            data = self.val[index:index + self.win_size]
            label = self.val[index + self.win_size]   # 下一个数据点
        elif (self.flag == 'test'):
            data = self.test[index:index + self.win_size]
            label = self.test_labels[index + self.win_size]  # 下一个数据点
        return np.float32(data), np.float32(label)
        

class SeqDataLoader(Dataset):
    def __init__(self, dataset_path="./dataset/SWaT/swat_train2.csv", win_size=64, step=1, name="SWaT Train"):
        self.step = step    # step size
        self.win_size = win_size    # window size
        print(f"Loading {name} data, window size is {win_size}, step is {step}")
        self.scaler = StandardScaler()

        _data = pd.read_csv(dataset_path)
        labels = _data.values[:, -1:]
        _data = _data.values[:, :-1]  # 没有异常标签
        _data = self.scaler.fit_transform(_data)
        _data = np.float32(_data)
        labels = np.int32(labels)
        print(f"{name} data shape:", _data.shape)

        self.data = _data
        self.labels = labels

    def get_test_labels(self):
        return self.labels

    def __len__(self):
        # 保留最后一个时间步不提供
        return (self.data.shape[0] - self.win_size) // self.step

    # 会根据__len__判断index的范围
    def __getitem__(self, index):
        index = index * self.step
        return self.data[index:index + self.win_size], self.data[index + self.win_size]
    



if __name__ == '__main__':

    dataset = SeqDataLoader("./dataset/wadi/WADI_attackdataLABLE_imputed.csv", name="WADI Test")
    print(len(dataset))
    labels = dataset.get_test_labels()
    print("Number of classes:", len(np.unique(labels)))
    print("Labels:", np.unique(labels))
    for i in range(3):
        data, label = dataset[i]
        print(data.shape, label.shape)
        print(data, label)
        print()

