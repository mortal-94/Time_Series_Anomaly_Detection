- `models/`: 简单写了LSTM预测和LSTM-AE，以及我的方法 Res_SE_CNN_LSTM
- `data_check.ipynb`: 数据简单查看，对 **WADI** 数据集就行处理，去掉全为0的列，缺失值用列均值填充
- `data_loader.py`: 简单写了个序列数据加载器
- `train.py`: 读配置、训练模型和保存、图示每个epoch的loss、最终epoch的误差分布、保存配置。观察误差分布手动确定阈值 `config["training_args"]["threshold"]` 。（确认阈值方法较low）
- `predict.py`: 同上，绘制误差以及真实异常点，分类报告

> 其他 ipynb 都是一些简单实验