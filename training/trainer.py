"""
Trainer module for VITS TTS.

This module provides the trainer class for training the VITS model
with appropriate loss functions and optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging
from tqdm import tqdm
import torchaudio
from torch.utils.tensorboard import SummaryWriter  # TensorBoard

logger = logging.getLogger(__name__)

class VITSTrainer:
    """
    Trainer for the VITS TTS model.
    
    This class handles the training loop, loss calculation, optimization,
    and checkpoint management for the VITS model.
    
    Args:
        model (nn.Module): VITS model instance
        config (dict): Configuration dictionary
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader, optional): Validation data loader
        device (str): Device to run training on
    """
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.learning_rate = config['train']['learning_rate']
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999)
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.999 # Adjust based on config if needed
        )
        
        # Setup mel-spectrogram transform for reconstruction loss
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config['data']['sampling_rate'],
            n_fft=config['data']['filter_length'],
            win_length=config['data']['win_length'],
            hop_length=config['data']['hop_length'],
            n_mels=config['model']['audio_channels']
        ).to(device)
        
        # Initialize training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Gradient accumulation steps
        self.grad_accum_steps = config['train'].get('grad_accum_steps', 1)

        # ------------------------------------------------------------------
        # TensorBoard writer (logdir defaults to "runs")
        # ------------------------------------------------------------------
        log_dir = self.config.get('log_dir', 'runs')
        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir)

        logger.info(f"Initialized VITS Trainer with config: {config['train']}")
        
    def train(self, epochs: int) -> None:
        """
        Train the model for the specified number of epochs.
        
        Args:
            epochs (int): Number of epochs to train
        """
        start_time = time.time()
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss = self._train_epoch()
            
            # Validate
            val_loss = None
            if self.val_loader is not None:
                val_loss = self._validate()
                
                # Check if this is the best model so far
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(f"best_model.pth")
                    logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            
            # Step the scheduler
            self.scheduler.step()
            
            # Log progress
            elapsed_time = time.time() - start_time
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
            
            log_msg = (f"Epoch {epoch + 1}/{self.current_epoch + epochs}, "
                      f"Train Loss: {train_loss:.4f}")
            
            if val_loss is not None:
                log_msg += f", Val Loss: {val_loss:.4f}"
                
            log_msg += f", Time: {elapsed_str}"
            
            logger.info(log_msg)
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['train'].get('checkpoint_interval', 5) == 0:
                self._save_checkpoint(f"checkpoint_{epoch + 1}.pth")
        
        # Save final model
        self._save_checkpoint("final_model.pth")
        logger.info("Training complete!")

        # Close TensorBoard writer
        self.tb_writer.close()
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            float: Average loss for this epoch
        """
        self.model.train()
        epoch_loss = 0.0
        
        # Use tqdm for progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        # Zero gradients at the beginning when using gradient accumulation
        self.optimizer.zero_grad()
        
        for i, batch in enumerate(pbar):
            # Move batch to device
            text_padded = batch['text_padded'].to(self.device)
            text_lengths = batch['text_lengths'].to(self.device)
            audio_padded = batch['audio_padded'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)
            
            # Forward pass
            output, dur_logits, mu, log_var, mu_p, log_var_p = self.model(
                text_padded, audio_padded
            )
            
            # Compute mel-spectrogram from model output waveform
            waveform = output.squeeze(1)  # [B, T]
            
            # Handle potential NaNs
            if torch.isnan(waveform).any():
                logger.warning(f"NaNs detected in output waveform at step {self.current_step}")
                waveform = torch.nan_to_num(waveform)
                
            # Compute mel-spectrogram from model output
            mel_output = self.mel_transform(waveform)
            
            # Align time dimension with target mel
            min_len = min(mel_output.size(-1), audio_padded.size(-1))
            mel_output = mel_output[..., :min_len]
            audio_target = audio_padded[..., :min_len]
            
            # Calculate losses
            recon_loss = F.l1_loss(mel_output, audio_target)
            
            # KL divergence loss
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Stochastic duration predictor loss (simplified for now)
            dur_loss = F.mse_loss(dur_logits, torch.randn_like(dur_logits))
            
            # Combine losses with weights
            recon_weight = self.config['train'].get('recon_weight', 1.0)
            kl_weight = self.config['train'].get('kl_weight', 0.1)
            dur_weight = self.config['train'].get('dur_weight', 0.1)
            
            total_loss = (
                recon_weight * recon_loss +
                kl_weight * kl_loss +
                dur_weight * dur_loss
            )
            
            # Scale loss for gradient accumulation
            total_loss = total_loss / self.grad_accum_steps
            
            # Backward pass
            total_loss.backward()
            
            # Update only after accumulation steps
            if (i + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update progress bar
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")
            
            # Log detailed loss components every few steps
            log_interval = self.config['train'].get('log_interval', 50)
            if i % log_interval == 0:
                logger.info(
                    f"Step {self.current_step}, Loss: {total_loss.item():.4f}, "
                    f"Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}, "
                    f"Dur: {dur_loss.item():.4f}"
                )
            
            epoch_loss += total_loss.item() * self.grad_accum_steps

            # TensorBoard – record per-step training loss *before* it is
            # multiplied back by grad_accum_steps so the numbers match the
            # console output.
            self.tb_writer.add_scalar('loss/train', total_loss.item() * self.grad_accum_steps, self.current_step)
            self.current_step += 1
        
        # Compute average loss
        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss
    
    def _validate(self) -> float:
        """
        Validate the model on the validation set.
        
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                text_padded = batch['text_padded'].to(self.device)
                audio_padded = batch['audio_padded'].to(self.device)
                
                # Forward pass
                output, dur_logits, mu, log_var, mu_p, log_var_p = self.model(
                    text_padded, audio_padded
                )
                
                # Compute mel-spectrogram from model output waveform
                waveform = output.squeeze(1)  # [B, T]
                mel_output = self.mel_transform(waveform)
                
                # Align time dimension with target mel
                min_len = min(mel_output.size(-1), audio_padded.size(-1))
                mel_output = mel_output[..., :min_len]
                audio_target = audio_padded[..., :min_len]
                
                # Calculate losses
                recon_loss = F.l1_loss(mel_output, audio_target)
                kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                dur_loss = F.mse_loss(dur_logits, torch.randn_like(dur_logits))
                
                # Combine losses with weights
                recon_weight = self.config['train'].get('recon_weight', 1.0)
                kl_weight = self.config['train'].get('kl_weight', 0.1)
                dur_weight = self.config['train'].get('dur_weight', 0.1)
                
                total_loss = (
                    recon_weight * recon_loss +
                    kl_weight * kl_loss +
                    dur_weight * dur_loss
                )
                
                val_loss += total_loss.item()
        
        # Compute average loss
        avg_val_loss = val_loss / len(self.val_loader)

        # TensorBoard – validation loss once per epoch
        self.tb_writer.add_scalar('loss/val', avg_val_loss, self.current_epoch)
        return avg_val_loss
    
    def _save_checkpoint(self, filename: str) -> None:
        """
        Save a checkpoint of the model and training state.
        
        Args:
            filename (str): Name of the checkpoint file
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'step': self.current_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = True) -> None:
        """
        Load a checkpoint to resume training or for inference.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
            resume_training (bool): Whether to also load optimizer state
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found at {checkpoint_path}")
            return
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if resume_training:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.current_step = checkpoint['step']
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            logger.info(f"Resumed training from epoch {self.current_epoch + 1}, step {self.current_step}")
        else:
            logger.info("Loaded model weights for inference")
            
    def generate_samples(self, texts: List[str], text_processor) -> Tuple[List[torch.Tensor], int]:
        """
        Generate audio samples for a list of texts.
        
        Args:
            texts (List[str]): List of input texts
            text_processor: Text processor for encoding text
                
        Returns:
            Tuple[List[torch.Tensor], int]: List of audio waveforms and sampling rate
        """
        self.model.eval()
        
        # Process texts
        text_tensors = []
        for text in texts:
            encoded = text_processor.encode_text(text)
            text_tensors.append(encoded.unsqueeze(0).to(self.device))
            
        # Generate audio
        waveforms = []
        with torch.no_grad():
            for text_tensor in text_tensors:
                try:
                    waveform = self.model.generate(text_tensor)
                    waveforms.append(waveform.cpu().squeeze(1))
                except Exception as e:
                    logger.error(f"Error generating audio: {e}")
                    # Return an empty tensor on error
                    waveforms.append(torch.zeros(1, 1000))
        
        return waveforms, self.config['data']['sampling_rate'] 