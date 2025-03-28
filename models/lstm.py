import torch.nn as nn
import torch

class LSTMPredictor(nn.Module):
    ModelName = "LSTMPredictor"
    def __init__(self, args):
        super(LSTMPredictor, self).__init__()
        self.args = args

        # LSTM层
        self.encoder = nn.LSTM(args["input_dim"], args["hidden_dim"], args["lstm_layers"], batch_first=True)
        # 预测器
        self.liner = nn.Linear(args["hidden_dim"], args["input_dim"])

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        # hidden: [num_layers, batch_size, hidden_size]
        output = self.liner(hidden[-1])

        return output


# if __name__ == '__main__':
    # x = torch.randn(32, 64, 51)  # (batch_size, seq_len, input_dim)
    # print("Input shape:", x.shape)
    # model = LSTMPredictor(51, 64, 1)

    # output = model(x)
    # print("Output shape:", output.shape)
    # print(output)
    # print("Success!")