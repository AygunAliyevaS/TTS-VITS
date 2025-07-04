"""
Common utilities for the VITS TTS system.

This module provides common utility functions used throughout the codebase.
"""

import torch
import numpy as np
import logging
import json
import os
import yaml
import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

def setup_logger(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Set up the logger with console and file handlers.
    
    Args:
        log_file (Optional[str]): Path to log file (if None, no file logging)
        level (int): Logging level (default: INFO)
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add console handler to root logger
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is provided
    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    logger.info(f"Logger initialized with level {level}")

def load_config(config_path: str) -> Dict:
    """
    Load configuration from a JSON or YAML file.
    
    Args:
        config_path (str): Path to config file
            
    Returns:
        Dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading config from {config_path}")
    
    # Determine file type from extension
    ext = os.path.splitext(config_path)[1].lower()
    
    try:
        if ext in ['.json']:
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif ext in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            logger.error(f"Unsupported config file type: {ext}")
            raise ValueError(f"Unsupported config file type: {ext}")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise
    
    return config

def save_config(config: Dict, config_path: str) -> None:
    """
    Save configuration to a JSON or YAML file.
    
    Args:
        config (Dict): Configuration dictionary
        config_path (str): Path to save config file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    # Determine file type from extension
    ext = os.path.splitext(config_path)[1].lower()
    
    try:
        if ext in ['.json']:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        elif ext in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            logger.error(f"Unsupported config file type: {ext}")
            raise ValueError(f"Unsupported config file type: {ext}")
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        raise
    
    logger.info(f"Config saved to {config_path}")

def load_audio_files(file_list: str, max_files: Optional[int] = None) -> Dict[str, Dict]:
    """
    Load audio files from a file list.
    
    Args:
        file_list (str): Path to file list
        max_files (Optional[int]): Maximum number of files to load
            
    Returns:
        Dict[str, Dict]: Dictionary of audio info keyed by file ID
    """
    audio_files = {}
    
    with open(file_list, 'r') as f:
        lines = f.readlines()
        
    # Limit number of files if specified
    if max_files is not None:
        lines = lines[:max_files]
    
    for i, line in enumerate(lines):
        parts = line.strip().split('|')
        if len(parts) < 2:
            logger.warning(f"Invalid line in file list: {line}")
            continue
        
        audio_path = parts[0]
        text = parts[1]
        
        # Get speaker ID if available
        speaker_id = None
        if len(parts) > 2 and parts[2]:
            try:
                speaker_id = int(parts[2])
            except ValueError:
                logger.warning(f"Invalid speaker ID: {parts[2]}")
        
        # Create unique ID for this audio file
        file_id = f"file_{i:05d}"
        
        audio_files[file_id] = {
            'path': audio_path,
            'text': text,
            'speaker_id': speaker_id
        }
    
    logger.info(f"Loaded {len(audio_files)} audio files from {file_list}")
    return audio_files

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.
    
    Args:
        tensor (torch.Tensor): PyTorch tensor
            
    Returns:
        np.ndarray: NumPy array
    """
    return tensor.detach().cpu().numpy()

def prepare_device(n_gpus: int) -> Union[torch.device, list]:
    """
    Prepare device(s) for training.
    
    Args:
        n_gpus (int): Number of GPUs to use
            
    Returns:
        Union[torch.device, list]: Device or list of devices
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        return torch.device('cpu')
    
    # Check if requested number of GPUs is available
    n_gpus_available = torch.cuda.device_count()
    if n_gpus_available < n_gpus:
        logger.warning(
            f"Requested {n_gpus} GPUs but only {n_gpus_available} available. "
            f"Using {n_gpus_available} GPUs."
        )
        n_gpus = n_gpus_available
    
    # Return single device if n_gpus = 1
    if n_gpus == 1:
        return torch.device('cuda:0')
    
    # Return list of devices if n_gpus > 1
    return [torch.device(f'cuda:{i}') for i in range(n_gpus)]

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (torch.nn.Module): PyTorch model
            
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 