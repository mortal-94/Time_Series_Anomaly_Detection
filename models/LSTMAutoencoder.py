import torch.nn as nn
import torch

class LSTMAutoencoder(nn.Module):
    ModelName = "LSTMAutoencoder"
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 编码器
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 桥接映射
        self.bridge = nn.Linear(hidden_size, input_size)

        # 解码器
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        # 编码器
        encoder_output, (hidden, cell) = self.encoder(x)
        # encoder_output: [batch_size, seq_len, hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        # cell: [num_layers, batch_size, hidden_size]
        # 把hidden、cell映射成 decoder 的初始状态
        hidden = self.bridge(hidden)
        cell = self.bridge(cell)
        output, _ = self.decoder(encoder_output, (hidden, cell))
        return output

if __name__ == '__main__':
    x = torch.randn(32, 64, 51)  # (batch_size, seq_len, input_dim)
    print("Input shape:", x.shape)
    model = LSTMAutoencoder(51, 64, 2)

    output = model(x)
    print("Output shape:", output.shape)
