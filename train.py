#!/usr/bin/env python3
"""
Training script for VITS Azerbaijani TTS.

This script trains the VITS model using the provided configuration.
"""

import torch
import argparse
import os
import sys
import logging
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from model.vits import VITS
from data.processing import AudioProcessor
from data.dataset import VITSDataset
from data.text.text_processor import TextProcessor
from training.trainer import VITSTrainer
from utils.common import setup_logger, load_config, prepare_device

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train VITS TTS model")
    parser.add_argument(
        '--config', type=str, default='config/base_vits.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory to save checkpoints and logs'
    )
    parser.add_argument(
        '--log_file', type=str, default=None,
        help='Path to log file'
    )
    parser.add_argument(
        '--seed', type=int, default=1234,
        help='Random seed'
    )
    parser.add_argument(
        '--n_gpus', type=int, default=1,
        help='Number of GPUs to use'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file
    if log_file is None and args.output_dir is not None:
        log_file = os.path.join(args.output_dir, 'train.log')
    
    setup_logger(log_file, level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting VITS training with config: {args.config}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Update config with command line args
    if args.output_dir is not None:
        config['checkpoint_dir'] = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare device
    device = prepare_device(args.n_gpus)
    logger.info(f"Using device: {device}")
    
    # Initialize model
    logger.info("Initializing VITS model")
    model = VITS(config)
    
    # Initialize processors
    logger.info("Initializing processors")
    audio_processor = AudioProcessor(config)
    text_processor = TextProcessor(config)
    
    # Initialize datasets
    logger.info("Initializing datasets")
    train_dataset = VITSDataset(
        config=config,
        filelist_path=config['data']['training_files'],
        audio_processor=audio_processor,
        text_processor=text_processor,
        is_validation=False
    )
    
    val_dataset = None
    if config['data'].get('validation_files'):
        val_dataset = VITSDataset(
            config=config,
            filelist_path=config['data']['validation_files'],
            audio_processor=audio_processor,
            text_processor=text_processor,
            is_validation=True
        )
    
    # Initialize data loaders
    logger.info("Initializing data loaders")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train'].get('num_workers', 4),
        collate_fn=VITSDataset.collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['train'].get('val_batch_size', 16),
            shuffle=False,
            num_workers=config['train'].get('num_workers', 4),
            collate_fn=VITSDataset.collate_fn,
            pin_memory=True
        )
    
    # Initialize trainer
    logger.info("Initializing trainer")
    trainer = VITSTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Load checkpoint if provided
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Train model
    logger.info("Starting training")
    trainer.train(config['train'].get('max_epochs', 1000))
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
