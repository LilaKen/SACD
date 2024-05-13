import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_classes=5):
        super(LSTM, self).__init__()

        # Assuming input is of shape (batch_size, sequence_length, input_size)
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           batch_first=True)  # This assumes that the zeroth dimension is batch

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),  # RNN's hidden size to an intermediate dense layer
            nn.LeakyReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 128).to(x.device)  # Initial hidden state
        c0 = torch.zeros(1, x.size(0), 128).to(x.device)  # Initial cell state

        out, _ = self.rnn(x, (h0, c0))

        # Consider the last step's hidden state for classification (many-to-one approach)
        out = out[:, -1, :]

        out = self.classifier(out)
        return out
