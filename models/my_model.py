import torch.nn as nn
import torch
import json

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 输入形状: (batch_size, num_channels, seq_len)
        avg_out = self.avg_pool(x).squeeze(-1)  # (batch_size, num_channels)
        channel_weights = self.fc(avg_out).unsqueeze(-1)  # (batch_size, num_channels, 1)
        return x * channel_weights  # 广播乘法


class SE_CNN_LSTM(nn.Module):
    ModelName = "MyModel"
    def __init__(self, args):
        super(SE_CNN_LSTM, self).__init__()
        self.args = args
        self.cnn1 = nn.Sequential(
            nn.Conv1d(args["input_dim"], args["cnn1"], kernel_size=args["k1"], padding=args["pad1"]),
            nn.BatchNorm1d(args["cnn1"]),
            nn.ReLU(),
            ChannelAttention(args["cnn1"] , args["SEratio1"]),  # 第一个通道注意力
            nn.MaxPool1d(args["maxpool1"]),
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv1d(args["input_dim"], args["cnn1"], kernel_size=1, padding=0),
            nn.BatchNorm1d(args["cnn1"]),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(args["cnn1"], args["cnn2"], kernel_size=args["k2"], padding=args["pad2"]),
            nn.BatchNorm1d(args["cnn2"]),
            nn.ReLU(),
            ChannelAttention(args["cnn2"], args["SEratio2"]),  # 第二个通道注意力
            nn.MaxPool1d(args["maxpool2"])
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv1d(args["cnn1"], args["cnn2"], kernel_size=1, padding=0),
            nn.BatchNorm1d(args["cnn2"]),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(args["cnn2"], args["hidden_dim"], batch_first=True, num_layers=args["lstm_layers"])
        self.fc = nn.Linear(args["hidden_dim"], args["input_dim"])

    def forward(self, x):
        # x形状: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch, channels or input_dim, seq_len)
        cnn_out = self.cnn1(x) # (batch, new_channels, new_seq_len)
        shortcut = self.shortcut1(x) 
        shortcut = nn.AdaptiveAvgPool1d(cnn_out.shape[2])(shortcut)
        cnn_out1 = cnn_out + shortcut

        cnn_out = self.cnn2(cnn_out1)
        shortcut = self.shortcut2(cnn_out1)
        shortcut = nn.AdaptiveAvgPool1d(cnn_out.shape[2])(shortcut)
        cnn_out = cnn_out + shortcut

        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, new_seq_len, 128)
        lstm_out, _ = self.lstm(cnn_out)
        output = self.fc(lstm_out[:, -1, :])
        return output

    @staticmethod
    def test():
        """
        查看各个过程的数据维度的变化
        """
        x = torch.randn(64, 100, 51) # (batch_size, seq_len, input_dim)
        print("Input shape:", x.shape)
        with open("./run_swat.json", "r") as f:
            config = json.load(f)

        model = SE_CNN_LSTM(config["model_args"])

        x = x.permute(0, 2, 1)  # (batch, channels or input_dim, seq_len)
        cnn_out1 = model.cnn1(x) # (batch, new_channels, new_seq_len)
        print("cnn1 shape:", cnn_out1.shape)
        shortcut = model.shortcut1(x)
        print("shortcut1 shape:", shortcut.shape)
        shortcut = nn.AdaptiveAvgPool1d(cnn_out1.shape[2])(shortcut)
        print("shortcut1 after shape:", shortcut.shape)
        cnn_out1 = cnn_out1 + shortcut
        print("cnn_out shape:", cnn_out.shape)

        cnn_out = model.cnn2(cnn_out)
        shortcut = model.shortcut2(cnn_out1)
        shortcut = nn.AdaptiveAvgPool1d(cnn_out.shape[2])(shortcut)
        cnn_out = cnn_out + shortcut

        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, new_seq_len, 128)
        lstm_out, _ = model.lstm(cnn_out)
        output = model.fc(lstm_out[:, -1, :])



if __name__ == "__main__":
    SE_CNN_LSTM.test()