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
    
    # Initialize TextProcessor first so we can compute vocabulary size dynamically
    logger.info("Initializing TextProcessor to compute vocabulary size")
    text_processor = TextProcessor(config)
    config['model']['vocab_size'] = len(text_processor.chars)
    logger.info(f"Set vocab_size to {config['model']['vocab_size']}")

    # Now initialize model
    logger.info("Initializing VITS model")
    model = VITS(config)

    # Initialize AudioProcessor (depends on config but not on model)
    logger.info("Initializing AudioProcessor")
    audio_processor = AudioProcessor(config)
    
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
    # Avoid creating an empty DataLoader when the dataset has fewer items than the batch
    batch_size = config['train']['batch_size']
    drop_last = True if len(train_dataset) >= batch_size else False

    # If the dataset is very small (<= workers), PyTorch may emit 'Bad file
    # descriptor' or 'semaphore released too many times' errors coming from the
    # internal QueueFeederThread.  Mitigate by clamping the number of workers
    # to at most len(dataset) - 1 (and never below 0).

    requested_workers = config['train'].get('num_workers', 4)
    # Workers for the training set (at least 0, at most len(dataset)-1)
    num_workers_train = max(0, min(requested_workers, max(len(train_dataset) - 1, 0)))
    # Workers for the validation set (handle None / tiny val set separately)
    if val_dataset is not None:
        num_workers_val = max(0, min(requested_workers, max(len(val_dataset) - 1, 0)))
    else:
        num_workers_val = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers_train,
        collate_fn=VITSDataset.collate_fn,
        pin_memory=True,
        drop_last=drop_last
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['train'].get('val_batch_size', 16),
            shuffle=False,
            num_workers=num_workers_val,
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
