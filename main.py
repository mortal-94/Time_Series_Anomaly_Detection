import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from time import time
import random
import json

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

config_path = './run_wadi.json'
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
train_data = SeqDataLoader(dataset_path=dataset_args["train_data_path"], win_size=dataset_args["winsize"], step=dataset_args["step"], name=f"{dataset_args["dataset"]} Train")
test_data = SeqDataLoader(dataset_path=dataset_args["test_data_path"], win_size=dataset_args["winsize"], step=1, name=f"{dataset_args["dataset"]} Test")
y_true = test_data.get_test_labels()
print(f'Train data length: {len(train_data)}; Test data length {len(test_data)}')
print('Data sample shape:', train_data[0][0].shape)

"""模型创建"""
data_dim = train_data[0][0].shape[-1]
model = SE_CNN_LSTM(model_args)
# model = LSTMPredictor(input_size=data_dim, hidden_size=args.hidden, num_layers=args.layers)
# model = LSTMAutoencoder(input_size=data_dim, hidden_size=args.hidden, num_layers=args.layers)
# AE_MODEL = True

model.to(device)  # 将模型移动到GPU上

filenameWithoutExt = f'{model.ModelName}_{dataset_args["dataset"]}_{int(time())}'

"""模型训练"""
if training_args["model_dir"] is not None:
    # 如果指定了模型路径，则加载模型参数
    model.load_state_dict(torch.load(training_args["model_dir"]))
    print(f"Model loaded from {training_args["model_dir"]}")
else:
    # 设置超参数
    learning_rate = training_args["lr"]
    num_epochs = training_args["epochs"]
    batch_size = training_args["batch_size"]
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
    print(f"mean loss {np.mean(losses)}")
    # 保存模型
    torch.save(model.state_dict(), f"./checkpoints/{filenameWithoutExt}.pth")


"""模型评估"""
test_losses = []
criterion = nn.MSELoss(reduction='none')
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=False)  # 评估所用batch_size不影响
with torch.no_grad():
    for x, x_1 in tqdm(test_data_loader):
        x = x.to(device)
        x_1 = x_1.to(device)
        y = model(x)
        if AE_MODEL:
            loss = criterion(y[:,-1,:], x[:,-1,:]).mean(dim=1)     # 自编码器的目标是重构，用窗口内的最后一步比较误差
        else:
            loss = criterion(y, x_1).mean(dim=1)     # 下一步的数据作为预测目标

        test_losses += loss.tolist()

text = []
for i in [80, 85, 90, 95, 99]:
    threshold = np.percentile(test_losses, i)
    y_pred = (test_losses > threshold).astype(int)
    if AE_MODEL:    
        y_pred = np.concatenate([np.zeros(dataset_args["winsize"]-1), y_pred]) # 自编码器的预测是窗口内的最后一步
        y_pred = np.concatenate([y_pred, np.zeros(1)])     # 由于数据集构造是面向预测的，最后一步不在训练集里面
    else:
        y_pred = np.concatenate([np.zeros(dataset_args["winsize"]), y_pred])  # 预测下一步
    text.append(f'Anomaly detected: {i}%, threshold: {threshold}\n' + classification_report(y_true, y_pred, target_names=['Normal', 'Attack']) + '\n')
    print(text[-1])        
    # print(f'Anomaly detected: {i}%, threshold: {threshold}')
    # print(confusion_matrix(y_true, y_pred))   # 混淆矩阵
    # print(classification_report(y_true, y_pred, target_names=['Normal', 'Attack']))

"""结果保存"""
# 保存参数、评估结果到json文件
with open(f'./checkpoints/{filenameWithoutExt}.json', 'w') as f:
    json.dump({'args': config, 'classification_report': text}, f, indent=4)

