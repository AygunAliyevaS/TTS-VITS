"""
Main VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) implementation.

This module implements the complete VITS architecture for Azerbaijani TTS, including:
- Text encoder
- Posterior encoder
- Prior encoder
- Stochastic duration predictor
- HiFi-GAN decoder
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import sys
import logging

# Local imports
from model.components.duration import StochasticDurationPredictor
from model.components.hifigan import HiFiGANDecoder

logger = logging.getLogger(__name__)

Conv1d = lambda *args, **kwargs: spectral_norm(nn.Conv1d(*args, **kwargs))

class TextEncoder(nn.Module):
    """
    Text encoder module that converts phoneme/text embeddings into latent representations.
    
    Args:
        vocab_size (int): Size of the vocabulary
        hidden_channels (int): Dimension of the hidden representations
    """
    def __init__(self, vocab_size, hidden_channels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_channels)
        self.convs = nn.ModuleList([
            Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        ])

    def forward(self, x):
        """
        Forward pass of the text encoder.
        
        Args:
            x (Tensor): Input tensor of token IDs [B, T]
                
        Returns:
            Tensor: Encoded text representation [B, C, T]
        """
        x = self.emb(x)               # [B, T, C]
        x = x.permute(0, 2, 1)        # [B, C, T]
        for conv in self.convs:
            x = conv(x).tanh()        # Apply tanh after each conv
        return x

class PosteriorEncoder(nn.Module):
    """
    Posterior encoder for the VAE component.
    Encodes mel-spectrograms into latent distributions.
    
    Args:
        in_channels (int): Number of input channels (mel bands)
        hidden_channels (int): Dimension of hidden representations
    """
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.pre = Conv1d(in_channels, hidden_channels, 1)
        self.convs = nn.ModuleList([
            Conv1d(hidden_channels, hidden_channels, 5, padding=2),
            Conv1d(hidden_channels, hidden_channels, 5, padding=2)
        ])
        self.proj = Conv1d(hidden_channels, hidden_channels * 2, 1)

    def forward(self, x):
        """
        Forward pass of the posterior encoder.
        
        Args:
            x (Tensor): Input mel-spectrogram [B, C, T]
                
        Returns:
            Tuple[Tensor, Tensor]: Mean and log variance of the posterior distribution
        """
        x = self.pre(x)
        for conv in self.convs:
            x = conv(x).tanh()
        stats = self.proj(x)
        mu, log_var = torch.chunk(stats, 2, dim=1)
        return mu, log_var

class PriorEncoder(nn.Module):
    """
    Prior encoder for the VAE component.
    Encodes text representations into prior distributions.
    
    Args:
        in_channels (int): Number of input channels
        hidden_channels (int): Dimension of hidden representations
    """
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.proj = Conv1d(in_channels, hidden_channels * 2, 1)

    def forward(self, x):
        """
        Forward pass of the prior encoder.
        
        Args:
            x (Tensor): Input text representation [B, C, T]
                
        Returns:
            Tuple[Tensor, Tensor]: Mean and log variance of the prior distribution
        """
        stats = self.proj(x)
        mu_p, log_var_p = torch.chunk(stats, 2, dim=1)
        return mu_p, log_var_p

class VITS(nn.Module):
    """
    Complete VITS model implementation for Azerbaijani TTS.
    
    Args:
        config (dict): Configuration dictionary with model parameters
    """
    def __init__(self, config):
        super().__init__()
        self.text_encoder = TextEncoder(
            vocab_size=config['model']['vocab_size'],
            hidden_channels=config['model']['hidden_channels']
        )
        self.posterior_encoder = PosteriorEncoder(
            in_channels=config['model']['audio_channels'],
            hidden_channels=config['model']['hidden_channels']
        )
        self.prior_encoder = PriorEncoder(
            in_channels=config['model']['hidden_channels'],
            hidden_channels=config['model']['hidden_channels']
        )
        self.decoder = HiFiGANDecoder(config)
        self.duration_predictor = StochasticDurationPredictor(config)
        
        # Store config for reference
        self.config = config
        
        logger.info("VITS model initialized with config: %s", config['model'])

    def forward(self, text, audio):
        """
        Forward pass of the VITS model.
        
        Args:
            text (Tensor): Input text tokens [B, T_text]
            audio (Tensor): Input audio features [B, C_audio, T_audio]
                
        Returns:
            Tuple: (output, dur_logits, mu, log_var, mu_p, log_var_p)
                output: Generated audio
                dur_logits: Duration predictions
                mu, log_var: Posterior distribution parameters
                mu_p, log_var_p: Prior distribution parameters
        """
        text_emb = self.text_encoder(text)
        mu, log_var = self.posterior_encoder(audio)

        # Pass text_emb through the prior encoder
        mu_p, log_var_p = self.prior_encoder(text_emb)

        # Reparameterization trick
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

        dur_logits = self.duration_predictor(z)
        output = self.decoder(z)
        return output, dur_logits, mu, log_var, mu_p, log_var_p

    @torch.no_grad()
    def generate(self, text, speed_adjustment=1.0):
        """
        Generate audio from text (inference mode).
        
        Args:
            text (Tensor): Input text tokens [B, T_text]
            speed_adjustment (float): Factor to adjust speech speed (default: 1.0)
                
        Returns:
            Tensor: Generated audio waveform
        """
        self.eval()
        text_emb = self.text_encoder(text)
        mu_p, log_var_p = self.prior_encoder(text_emb)
        
        # Sample from prior
        z = mu_p + torch.randn_like(mu_p) * torch.exp(0.5 * log_var_p)
        
        # Adjust for speech speed if needed
        if speed_adjustment != 1.0:
            # This is a simplified approach - in practice, we would use the duration
            # predictor to properly adjust timing
            z = torch.nn.functional.interpolate(
                z, scale_factor=1.0/speed_adjustment, mode='linear', align_corners=False
            )
        
        # Generate waveform
        output = self.decoder(z)
        return output

    def get_param_count(self):
        """Returns the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
