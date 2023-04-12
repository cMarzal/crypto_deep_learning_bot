import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_dim=6, hidden_size=256, num_layers=2, output_dim=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True)
        self.bn_0 = nn.BatchNorm1d(self.hidden_size)
        self.fc_1 = nn.Linear(int(self.hidden_size), 6)
        self.softmax = nn.Softmax(1)

    def forward(self, padded_sequence, input_lengths, hidden=None):
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_lengths)
        out, (hn) = self.gru(packed, hidden)
        hidden_output = hn[-1]
        outputs = self.bn_0(hidden_output)
        outputs = self.softmax(self.fc_1(outputs))
        return outputs


class Model(nn.Module):
    def __init__(self, input_size=6, hidden_size=256, n_layers=2, dropout=0.01):
        super(Model, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, n_layers,
                            dropout=(0 if n_layers == 1 else dropout), bidirectional=False)
        self.bn_0 = nn.BatchNorm1d(self.hidden_size)
        self.fc_1 = nn.Linear(int(self.hidden_size), 5)
        self.softmax = nn.Softmax(1)

    def forward(self, padded_sequence, input_lengths, hidden=None):
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_lengths)
        outputs, (hn, cn) = self.lstm(packed, hidden)
        hidden_output = hn[-1]
        outputs = self.bn_0(hidden_output)
        outputs = self.softmax(self.fc_1(outputs))
        return outputs
