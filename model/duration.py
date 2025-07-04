import torch
import torch.nn as nn

class StochasticDurationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(config['model']['hidden_channels'], config['model']['hidden_channels'], 
                     kernel_size=5, padding=2),
            nn.Conv1d(config['model']['hidden_channels'], config['model']['hidden_channels'], 
                     kernel_size=5, padding=2)
        ])
        self.proj = nn.Conv1d(config['model']['hidden_channels'], 1, 1)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x).tanh()
        log_dur = self.proj(x)
        return log_dur
