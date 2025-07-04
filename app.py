"""
Gradio web interface for VITS Azerbaijani TTS.

This module provides a web interface for the VITS TTS model using Gradio,
allowing users to input text and generate speech.
"""

import torch
import gradio as gr
import os
import sys
import logging
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from model.vits import VITS
from data.text.text_processor import TextProcessor
from utils.common import load_config, setup_logger

# Set up logger
setup_logger("logs/app.log")
logger = logging.getLogger(__name__)

class TTS_App:
    """
    VITS TTS web application.
    
    This class encapsulates the TTS application functionality, including model loading,
    text processing, and audio generation.
    """
    def __init__(self, config_path='config/base_vits.json', checkpoint_path='checkpoints/best_model.pth'):
        """Initialize the TTS application."""
        logger.info("Initializing TTS application")
        
        # Configure device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize model
        self.model = self._load_model(checkpoint_path)
        
        # Initialize text processor
        self.text_processor = TextProcessor(self.config)
        
        logger.info("TTS application initialized successfully")
    
    def _load_model(self, checkpoint_path):
        """Load the VITS model from checkpoint."""
        try:
            model = VITS(self.config).to(self.device)
            
            if os.path.exists(checkpoint_path):
                logger.info(f"Loading model checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                    
                logger.info("Model checkpoint loaded successfully")
            else:
                logger.warning(f"Checkpoint not found at {checkpoint_path}, using untrained model")
                
            model.eval()
            return model
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def generate_speech(self, text, speaker_id=0, speed=1.0, seed=None):
        """
        Generate speech from text.
        
        Args:
            text (str): Input text to synthesize
            speaker_id (int): Speaker ID for multi-speaker models
            speed (float): Speed adjustment factor (0.5 to 2.0)
            seed (int): Random seed for deterministic generation
                
        Returns:
            tuple: (sampling_rate, audio_array)
        """
        if not text or not text.strip():
            logger.warning("Empty input text")
            return (self.config['data']['sampling_rate'], torch.zeros(1000).numpy())
        
        logger.info(f"Generating speech for text: '{text}', speaker_id: {speaker_id}, speed: {speed}")
        
        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        try:
            # Process text
            start_time = time.time()
            encoded_text = self.text_processor.encode_text(text)
            logger.debug(f"Text encoded in {time.time() - start_time:.3f} seconds")
            
            # Add batch dimension
            encoded_text = encoded_text.unsqueeze(0).to(self.device)
            
            # Generate audio
            with torch.no_grad():
                start_time = time.time()
                audio = self.model.generate(encoded_text, speed_adjustment=speed)
                logger.debug(f"Audio generated in {time.time() - start_time:.3f} seconds")
            
            # Convert to numpy for Gradio
            audio_np = audio.cpu().numpy().squeeze()
            
            # Apply normalization
            if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))
            
            return (self.config['data']['sampling_rate'], audio_np)
            
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return (self.config['data']['sampling_rate'], torch.zeros(1000).numpy())

def create_gradio_interface():
    """Create the Gradio interface for the TTS application."""
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run VITS TTS web interface")
    parser.add_argument('--config', type=str, default='config/base_vits.json',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--share', action='store_true',
                       help='Create a public link for the interface')
    args = parser.parse_args()
    
    try:
        # Initialize TTS application
        tts_app = TTS_App(args.config, args.checkpoint)
        
        # Example Azerbaijani texts
        examples = [
            ["Salam dünya, bu bir səs sintezi nümunəsidir."],
            ["Azərbaycan dili gözəl bir dildir."],
            ["Bu sistem Azərbaycan dili üçün hazırlanmışdır."]
        ]
        
        # Create Gradio interface
        interface = gr.Interface(
            fn=tts_app.generate_speech,
            inputs=[
                gr.Textbox(
                    label="Text",
                    placeholder="Azərbaycanca mətn daxil edin...",
                    lines=3
                ),
                gr.Number(
                    label="Speaker ID",
                    value=0,
                    precision=0
                ),
                gr.Slider(
                    label="Speed",
                    minimum=0.5,
                    maximum=2.0,
                    step=0.1,
                    value=1.0
                ),
                gr.Number(
                    label="Random Seed (optional)",
                    value=None
                )
            ],
            outputs=gr.Audio(
                label="Generated Speech"
            ),
            title="VITS Azerbaijani Text-to-Speech",
            description=(
                "Enter text in Azerbaijani to generate speech. "
                "The model supports Azerbaijani-specific characters (Əə, Ğğ, Iı, İi, Öö, Üü, Çç, Şş)."
            ),
            examples=examples,
            article=(
                "This demo showcases a VITS (Variational Inference with Adversarial Learning for End-to-End Text-to-Speech) "
                "model trained for the Azerbaijani language. The model combines a conditional VAE with "
                "adversarial training and a stochastic duration predictor to generate high-quality speech."
            )
        )
        
        return interface, args.share
        
    except Exception as e:
        logger.error(f"Error creating interface: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Fallback to a simple interface explaining the error
        return gr.Interface(
            fn=lambda x: "Error initializing TTS model. Please check logs.",
            inputs=gr.Textbox(),
            outputs=gr.Textbox(),
            title="VITS TTS Error",
            description=f"Error initializing TTS model: {str(e)}"
        ), False

if __name__ == "__main__":
    # Create and launch interface
    interface, share = create_gradio_interface()
    interface.launch(debug=True, share=share)