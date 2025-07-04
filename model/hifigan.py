import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d

class HiFiGANDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_conv = Conv1d(
            config['model']['hidden_channels'],
            config['model']['decoder_channels'],
            kernel_size=7,
            padding=3
        )
        
        self.upsample = nn.ModuleList()
        for i, (u, k) in enumerate(zip(
            config['model']['upsample_rates'],
            config['model']['upsample_kernel_sizes']
        )):
            self.upsample.append(
                ConvTranspose1d(
                    config['model']['decoder_channels'] // (2 ** i),
                    config['model']['decoder_channels'] // (2 ** (i + 1)),
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2
                )
            )
        
        self.post_conv = Conv1d(
            config['model']['decoder_channels'] // (2 ** len(config['model']['upsample_rates'])),
            1,
            kernel_size=7,
            padding=3
        )

    def forward(self, x):
        x = self.pre_conv(x)
        for layer in self.upsample:
            x = layer(x).relu()
        x = self.post_conv(x).tanh()
        return x
