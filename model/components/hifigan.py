"""
HiFi-GAN Decoder for VITS model.

This module implements the HiFi-GAN vocoder component which
converts latent representations into high-fidelity audio waveforms.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

Conv1d = lambda *args, **kwargs: spectral_norm(nn.Conv1d(*args, **kwargs))

class ResBlock(nn.Module):
    """
    Residual block for the HiFi-GAN decoder.
    
    Args:
        channels (int): Number of channels
        kernel_size (int): Kernel size for the convolutions
        dilation (int): Dilation factor
    """
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = Conv1d(channels, channels, kernel_size, 
                          dilation=dilation, padding=dilation*(kernel_size-1)//2)
        self.conv2 = Conv1d(channels, channels, kernel_size, 
                          dilation=1, padding=(kernel_size-1)//2)
        
    def forward(self, x):
        """Forward pass adding a residual connection"""
        residual = x
        x = F.leaky_relu(x, 0.1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.conv2(x)
        return x + residual

class HiFiGANDecoder(nn.Module):
    """
    HiFi-GAN decoder that converts latent representations to waveforms.
    
    This implementation is based on the HiFi-GAN vocoder architecture but
    adapted for the VITS model as a decoder.
    
    Args:
        config (dict): Configuration dictionary with model parameters
    """
    def __init__(self, config):
        super().__init__()
        hidden_channels = config['model']['hidden_channels']
        decoder_channels = config['model'].get('decoder_channels', 512)
        
        # Initial projection from latent space
        self.pre = Conv1d(hidden_channels, decoder_channels, 7, padding=3)
        
        # Multi-scale upsampling layers
        self.upsample_rates = config['model'].get('upsample_rates', [8, 8, 2, 2])
        self.upsample_kernel_sizes = config['model'].get('upsample_kernel_sizes', [16, 16, 4, 4])
        
        self.upsamples = nn.ModuleList()
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            self.upsamples.append(
                nn.ConvTranspose1d(
                    decoder_channels // (2**i), 
                    decoder_channels // (2**(i+1)), 
                    k, stride=u, padding=(k-u)//2
                )
            )
        
        # Residual blocks for each scale
        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsample_rates)):
            ch = decoder_channels // (2**(i+1))
            self.resblocks.append(
                nn.ModuleList([
                    ResBlock(ch, 3, 1),
                    ResBlock(ch, 3, 3),
                    ResBlock(ch, 3, 5)
                ])
            )
            
        # Final output projection to waveform
        self.post = Conv1d(decoder_channels // (2**(len(self.upsample_rates))), 1, 7, padding=3)
        
    def forward(self, x):
        """
        Forward pass of the HiFi-GAN decoder.
        
        Args:
            x (Tensor): Input latent representation [B, C, T]
                
        Returns:
            Tensor: Generated audio waveform [B, 1, T']
        """
        x = self.pre(x)
        
        # Apply upsampling layers and residual blocks
        for i in range(len(self.upsample_rates)):
            x = F.leaky_relu(x, 0.1)
            x = self.upsamples[i](x)
            
            # Apply residual blocks at this scale
            for res in self.resblocks[i]:
                x = res(x)
                
        # Final processing
        x = F.leaky_relu(x, 0.1)
        x = self.post(x)
        x = torch.tanh(x)  # Output in range [-1, 1]
        
        return x 