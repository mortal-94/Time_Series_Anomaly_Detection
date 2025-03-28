import torch.nn as nn
import torch

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
        self.cnn = nn.Sequential(
            nn.Conv1d(args["input_dim"], args["cnn1"], kernel_size=args["k1"], padding=args["pad1"]),
            nn.ReLU(),
            ChannelAttention(args["cnn1"] , args["SEratio1"]),  # 第一个通道注意力
            nn.MaxPool1d(args["maxpool1"]),
            
            nn.Conv1d(args["cnn1"], args["cnn2"], kernel_size=args["k2"], padding=args["pad2"]),
            nn.ReLU(),
            ChannelAttention(args["cnn2"], args["SEratio1"]),  # 第二个通道注意力
            nn.MaxPool1d(args["maxpool1"])
        )
        self.lstm = nn.LSTM(args["cnn2"], args["hidden_dim"], batch_first=True, num_layers=args["lstm_layers"])
        self.fc = nn.Linear(args["hidden_dim"], args["input_dim"])

    def forward(self, x):
        # x形状: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, new_seq_len, 128)
        lstm_out, _ = self.lstm(cnn_out)
        output = self.fc(lstm_out[:, -1, :])
        return output

    @staticmethod
    def test():
        """
        查看各个过程的数据维度的变化
        """
        x = torch.randn(32, 64, 51) # (batch_size, seq_len, input_dim)
        print("Input shape:", x.shape)
        model = SE_CNN_LSTM(51, 64, 51)

        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        print("(Input) After permute:", x.shape)
        cnn_out = model.cnn(x)
        print("CNN output shape:", cnn_out.shape)
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, new_seq_len, 128)
        print("(CNN Output) After permute:", cnn_out.shape)
        lstm_out, _ = model.lstm(cnn_out)
        print("LSTM output shape:", lstm_out.shape) # (batch, new_seq_len, hidden_dim)
        output = model.fc(lstm_out[:, -1, :])
        print("Output shape:", output.shape)    # (batch, output_dim)


if __name__ == "__main__":
    SE_CNN_LSTM.test()