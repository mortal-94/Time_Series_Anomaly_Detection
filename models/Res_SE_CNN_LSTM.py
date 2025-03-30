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

class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, reduction_ratio):
        super(CNN_Block, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.ca = ChannelAttention(out_channels, reduction_ratio)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.ca(x)
        return x

class Shortcut_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Shortcut_Block, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Res_SE_CNN_LSTM(nn.Module):
    ModelName = "Res_SE_CNN_LSTM"
    def __init__(self, args):
        super(Res_SE_CNN_LSTM, self).__init__()
        self.args = args
        self.se0 = ChannelAttention(args["input_dim"], 4)
        self.cnn1 = CNN_Block(args["input_dim"], args["cnn1"], args["k1"], args["SEratio1"])
        self.shortcut1 = Shortcut_Block(args["input_dim"], args["cnn1"])
        self.cnn2 = CNN_Block(args["cnn1"], args["cnn2"], args["k2"], args["SEratio2"])
        self.shortcut2 = Shortcut_Block(args["cnn1"], args["cnn2"])

        self.lstm = nn.LSTM(args["cnn2"], args["hidden_dim"], batch_first=True, num_layers=args["lstm_layers"])

        self.shortcut3 = Shortcut_Block(args["cnn2"], 1)
        self.fc = nn.Linear(args["hidden_dim"], args["input_dim"])

    def forward(self, x):
        # x形状: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch, channels or input_dim, seq_len)
        cnn_out = self.cnn1(self.se0(x)) # (batch, new_channels, new_seq_len)
        shortcut = self.shortcut1(x) 
        cnn_out1 = cnn_out + shortcut

        cnn_out = self.cnn2(cnn_out1)
        shortcut = self.shortcut2(cnn_out1)
        cnn_out = cnn_out + shortcut

        shortcut = self.shortcut3(cnn_out)  # (batch, 1, new_seq_len)
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, new_seq_len, 128)
        lstm_out, _ = self.lstm(cnn_out)  # (batch, new_seq_len, hidden_dim)
        shortcut = nn.AdaptiveAvgPool1d(lstm_out.shape[2])(shortcut)    # (batch, 1, hidden_dim)
        last_lstm = lstm_out[:, -1, :] + shortcut.squeeze(1)  # (batch, hidden_dim)
        output = self.fc(last_lstm)
        return output

    @staticmethod
    def test():
        """
        查看各个过程的数据维度的变化
        """
        x = torch.randn(64, 30, 51) # (batch_size, seq_len, input_dim)
        print("Input shape:", x.shape)
        with open("./run_swat.json", "r") as f:
            config = json.load(f)

        model = Res_SE_CNN_LSTM(config["model_args"])

        y = model(x)
        print("Output shape:", y.shape)



if __name__ == "__main__":
    Res_SE_CNN_LSTM.test()