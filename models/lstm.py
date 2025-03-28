import torch.nn as nn
import torch

class LSTMPredictor(nn.Module):
    ModelName = "LSTMPredictor"
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM层
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 预测器
        self.liner = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        # hidden: [num_layers, batch_size, hidden_size]
        output = self.liner(hidden[-1])

        return output


if __name__ == '__main__':
    x = torch.randn(32, 64, 51)  # (batch_size, seq_len, input_dim)
    print("Input shape:", x.shape)
    model = LSTMPredictor(51, 64, 1)

    output = model(x)
    print("Output shape:", output.shape)
    print(output)
    print("Success!")