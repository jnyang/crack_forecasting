import torch.nn as nn
from geotorchai.models.grid import ConvLSTM

class GeoTorchConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers):
        super().__init__()
        self.lstm = ConvLSTM(input_dim=input_size, 
                             hidden_dim=hidden_dim, 
                             num_layers=num_layers)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        return lstm_out