import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np
import sys
# Add model directory to path if not already added
if './model' not in sys.path:
    sys.path.append('./model')

from hifigan import HiFiGANDecoder
from duration import StochasticDurationPredictor
import logging
import traceback

logger = logging.getLogger(__name__)
Conv1d = lambda *args, **kwargs: spectral_norm(nn.Conv1d(*args, **kwargs))

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_channels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_channels)
        self.convs = nn.ModuleList([
            Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        ])

    def forward(self, x):
        x = self.emb(x)
        x = x.permute(0, 2, 1)  # [B, C, T]
        for conv in self.convs:
            x = conv(x).tanh()
        return x

class PosteriorEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.pre = Conv1d(in_channels, hidden_channels, 1)
        self.convs = nn.ModuleList([
            Conv1d(hidden_channels, hidden_channels, 5, padding=2),
            Conv1d(hidden_channels, hidden_channels, 5, padding=2)
        ])
        self.proj = Conv1d(hidden_channels, hidden_channels * 2, 1)

    def forward(self, x):
        x = self.pre(x)
        for conv in self.convs:
            x = conv(x).tanh()
        stats = self.proj(x)
        mu, log_var = torch.chunk(stats, 2, dim=1)
        return mu, log_var

class PriorEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.proj = Conv1d(in_channels, hidden_channels * 2, 1)

    def forward(self, x):
        stats = self.proj(x)
        mu_p, log_var_p = torch.chunk(stats, 2, dim=1)
        return mu_p, log_var_p

class VITS(nn.Module):
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
            in_channels=config['model']['hidden_channels'], # Input is from text_encoder output
            hidden_channels=config['model']['hidden_channels']
        )
        self.decoder = HiFiGANDecoder(config)
        self.duration_predictor = StochasticDurationPredictor(config)

    def forward(self, text, audio):
        text_emb = self.text_encoder(text)
        mu, log_var = self.posterior_encoder(audio)

        # Pass text_emb through the prior encoder
        mu_p, log_var_p = self.prior_encoder(text_emb)

        # Reparameterization trick
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

        dur_logits = self.duration_predictor(z)
        output = self.decoder(z)
        return output, dur_logits, mu, log_var, mu_p, log_var_p # Return prior stats as well

    @torch.no_grad()
    def generate(self, text):
        self.eval()
        text_emb = self.text_encoder(text)
        mu_p, log_var_p = self.prior_encoder(text_emb)
        # Sample from prior
        z = mu_p + torch.randn_like(mu_p) * torch.exp(0.5 * log_var_p)
        # Generate waveform
        output = self.decoder(z)
        return output

import torch
import logging
import numpy as np
from model.vits import VITS
import json
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_config(config_path='configs/base_vits.json'):
    """Load model configuration"""
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        sys.exit(1)

def test_synthesis():
    """Test the VITS synthesis"""
    # Load config
    config = load_config()
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    model = VITS(config).to(device)
    model.eval()
    
    # Create dummy input (batch_size=1, sequence_length=10)
    dummy_input = torch.randint(0, 100, (1, 10)).to(device)
    
    # Test synthesis
    logging.info("Starting synthesis test...")
    with torch.no_grad():
        try:
            audio = model.synthesize(dummy_input)
            if audio is not None:
                logging.info(f"Success! Generated audio shape: {audio.shape}")
                logging.info(f"Audio stats - min: {audio.min():.4f}, max: {audio.max():.4f}, mean: {audio.mean():.4f}")
            else:
                logging.error("Synthesis returned None")
        except Exception as e:
            logging.error(f"Error during synthesis: {e}", exc_info=True)

if __name__ == "__main__":
    test_synthesis()
     