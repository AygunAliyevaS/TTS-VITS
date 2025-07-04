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