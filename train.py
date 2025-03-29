import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from time import time
import random
import json
import argparse

from data_loader import SeqDataLoader
from models.my_model import SE_CNN_LSTM
from models.lstm import LSTMPredictor
from models.lstm_ae import LSTMAutoencoder

def set_global_seed(seed=42):
    # 设置 Python 的随机种子
    random.seed(seed)
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    # 设置 PyTorch 的 CPU 随机种子
    torch.manual_seed(seed)
    # 设置 PyTorch 的 CUDA 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 禁用 CUDA 卷积操作的非确定性算法
        torch.backends.cudnn.deterministic = True
        # 禁用 CUDA 卷积操作的自动寻找最优算法
        torch.backends.cudnn.benchmark = False

set_global_seed()

argparser = argparse.ArgumentParser(description='Anomaly detection')
# argparser.add_argument('--config', type=str, default='./run_wadi.json', help='Path to the config file')
argparser.add_argument('--config', type=str, default='./run_swat.json', help='Path to the config file')
args = argparser.parse_args()
config_path = args.config
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
with open(config_path, 'r') as f:
    config = json.load(f)

dataset_args = config["dataset_args"]
model_args = config["model_args"]
training_args = config["training_args"]

AE_MODEL = training_args["AE_MODEL"]   # 是否是自编码器模型

# 检测是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

"""数据加载"""
dataset_name = dataset_args["dataset"]
train_data = SeqDataLoader(dataset_path=dataset_args["train_data_path"], win_size=dataset_args["winsize"], step=dataset_args["step"], name=f"{dataset_name} Train")
print(f'Train data length: {len(train_data)}')
print('Data sample shape:', train_data[0][0].shape)

"""模型创建"""
data_dim = train_data[0][0].shape[-1]
model_list = [SE_CNN_LSTM, LSTMPredictor, LSTMAutoencoder]
model = model_list[model_args["model"]](model_args)  # 选择模型

model.to(device)  # 将模型移动到GPU上

filenameWithoutExt = f'{model.ModelName}_{dataset_args["dataset"]}_{int(time())}'

"""模型训练"""
learning_rate = training_args["lr"]
num_epochs = training_args["epochs"]
batch_size = training_args["batch_size"]
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# 训练循环，显示进度条
losses = []
model.train()
for epoch in tqdm(range(num_epochs)):
    for i, (x, x_1) in enumerate(data_loader):
        optimizer.zero_grad()
        x = x.to(device)  # 将数据移动到GPU上
        x_1 = x_1.to(device)  # 将数据移动到GPU上
        y = model(x)
        if AE_MODEL:
            loss = criterion(y[:,-1,:], x[:,-1,:])  # 自编码器的目标是重构，用窗口内的最后一步比较误差
        else:
            loss = criterion(y, x_1)  # 下一步的数据作为预测目标
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if i % 500 == 0:
            print(f'Epoch {epoch}, batch {i}, loss: {loss.item()}')
# 保存模型
torch.save(model.state_dict(), f"./checkpoints/{filenameWithoutExt}.pth")
config["training_args"]["model_dir"] = f"./checkpoints/{filenameWithoutExt}.pth"
print(f"Model saved to ./checkpoints/{filenameWithoutExt}.pth")

model.eval()
losses4train = []
with torch.no_grad():
    for x, x_1 in tqdm(data_loader):
        x = x.to(device)
        x_1 = x_1.to(device)
        y = model(x)
        if AE_MODEL:
            loss = criterion(y[:,-1,:], x[:,-1,:])     # 自编码器的目标是重构，用窗口内的最后一步比较误差
        else:
            loss = criterion(y, x_1)     # 下一步的数据作为预测目标

        losses4train.append(loss.item())

# 绘制误差数值分布，以便选择阈值
plt.figure(figsize=(10, 6))
plt.hist(losses4train, bins=100, color='blue', alpha=0.7)
plt.title('Loss Distribution')
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.grid()
plt.savefig(f"./checkpoints/{filenameWithoutExt}_loss_distribution.png")
plt.show()


"""结果保存"""
# 保存超参数结果到json文件
with open(f'./checkpoints/{filenameWithoutExt}.json', 'w') as f:
    json.dump({'args': config}, f, indent=4)
    print(f"Config saved to ./checkpoints/{filenameWithoutExt}.json")

