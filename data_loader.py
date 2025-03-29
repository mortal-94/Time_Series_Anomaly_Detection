import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


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

    dataset = SeqDataLoader("./dataset/wadi/train.csv", name="WADI Test")
    print(len(dataset))
    labels = dataset.get_test_labels()
    print("Number of classes:", len(np.unique(labels)))
    print("Labels:", np.unique(labels))
    for i in range(3):
        data, label = dataset[i]
        print(data.shape, label.shape)
        print(data, label)
        print()

