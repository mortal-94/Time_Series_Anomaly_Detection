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
import argparse

from data_loader import SeqDataLoader
from models.Res_SE_CNN_LSTM import Res_SE_CNN_LSTM
from models.LSTMPredictor import LSTMPredictor
from models.LSTMAutoencoder import LSTMAutoencoder



argparser = argparse.ArgumentParser(description='Anomaly detection')
# argparser.add_argument('--config', type=str, default='./run_wadi.json', help='Path to the config file')
argparser.add_argument('--config', type=str, default='./checkpoints/MyModel_swat_1743221612.json', help='Path to the config file')
args = argparser.parse_args()
config_path = args.config
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
with open(config_path, 'r') as f:
    config = json.load(f)["args"]
    print("load config file:", config_path)

if config["training_args"]["model_dir"] is None:
    raise ValueError("Model directory is not specified in the config file.")
if config["training_args"]["threshold"] is None:
    raise ValueError("Threshold is not specified in the config file.")


dataset_args = config["dataset_args"]
model_args = config["model_args"]
training_args = config["training_args"]

AE_MODEL = training_args["AE_MODEL"]   # 是否是自编码器模型

# 检测是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

"""数据加载"""
dataset_name = dataset_args["dataset"]
test_data = SeqDataLoader(dataset_path=dataset_args["test_data_path"], win_size=dataset_args["winsize"], step=1, name=f"{dataset_name} Test")
y_true = test_data.get_test_labels()
print(f'Test data length {len(test_data)}')
print(f'Test set Normal : {np.sum(y_true == 0)}; Attack : {np.sum(y_true == 1)}')
print('Data sample shape:', test_data[0][0].shape)

"""模型创建"""
data_dim = test_data[0][0].shape[-1]
model_list = [Res_SE_CNN_LSTM, LSTMPredictor, LSTMAutoencoder]
model = model_list[model_args["model"]](model_args)  # 选择模型
model.load_state_dict(torch.load(training_args["model_dir"]))
print(f"Model loaded from ", training_args["model_dir"])
model.to(device)  # 将模型移动到GPU上


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
thresholds = np.percentile(test_losses, [80, 85, 90, 95, 99]).tolist()
thresholds += [config["training_args"]["threshold"]]  

for threshold in thresholds:
    y_pred = (np.array(test_losses) > threshold).astype(int)
    if AE_MODEL:    
        y_pred = np.concatenate([np.zeros(dataset_args["winsize"]-1), y_pred]) # 自编码器的预测是窗口内的最后一步
        y_pred = np.concatenate([y_pred, np.zeros(1)])     # 由于数据集构造是面向预测的，最后一步不在训练集里面
    else:
        y_pred = np.concatenate([np.zeros(dataset_args["winsize"]), y_pred])  # 预测下一步
    text.append(f'threshold: {threshold}\n' + classification_report(y_true, y_pred, target_names=['Normal', 'Attack']) + '\n')
    print(text[-1])

"""结果保存"""
# 保存评估结果到json文件
with open(config_path, 'w') as f:
    json.dump({'args': config, 'classification_report': text}, f, indent=4)

