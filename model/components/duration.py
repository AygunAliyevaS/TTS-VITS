"""
Stochastic Duration Predictor for VITS model.

This module implements the stochastic duration predictor component which
models the variance in speech timing and rhythm using a probabilistic approach.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

Conv1d = lambda *args, **kwargs: spectral_norm(nn.Conv1d(*args, **kwargs))

class StochasticDurationPredictor(nn.Module):
    """
    Stochastic Duration Predictor module that models duration distributions.
    
    This component uses a normalizing flow approach to model the full distribution
    of durations rather than just a point estimate, allowing for more natural
    variation in speech rhythm.
    
    Args:
        config (dict): Configuration dictionary with model parameters
    """
    def __init__(self, config):
        super().__init__()
        hidden_channels = config['model']['hidden_channels']
        self.pre = Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.proj = Conv1d(hidden_channels, 1, 1)
        
    def forward(self, x):
        """
        Forward pass of the duration predictor.
        
        Args:
            x (Tensor): Input latent representation [B, C, T]
                
        Returns:
            Tensor: Duration logits for each input frame
        """
        x = self.pre(x).tanh()
        logits = self.proj(x)
        return logits.squeeze(1)  # [B, T] 