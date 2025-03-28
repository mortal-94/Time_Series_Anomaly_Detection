import torch.nn as nn
import torch

class CNN_LSTM(nn.Module):
    ModelName = "MyModel"
    def __init__(self, input_dim, hidden_dim, output_dim, lstm_layers=1):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True, num_layers=lstm_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

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
        model = CNN_LSTM(51, 64, 51)

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
    CNN_LSTM.test()