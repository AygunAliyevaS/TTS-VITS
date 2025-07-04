"""
Dataset implementations for VITS TTS.

This module provides dataset classes for loading and processing text and audio data
for training and evaluation of the VITS TTS model.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

from data.processing import AudioProcessor
from data.text.text_processor import TextProcessor

logger = logging.getLogger(__name__)

class VITSDataset(Dataset):
    """
    Dataset for VITS TTS training.
    
    This dataset loads text and corresponding audio files from a filelist,
    processes them according to the configuration, and returns batches suitable
    for VITS training.
    
    Args:
        config (dict): Configuration dictionary
        filelist_path (str): Path to filelist containing audio-text pairs
        audio_processor (AudioProcessor): Audio processor instance
        text_processor (TextProcessor): Text processor instance
        is_validation (bool): Whether this is a validation dataset
    """
    def __init__(
        self, 
        config: Dict, 
        filelist_path: str, 
        audio_processor: AudioProcessor,
        text_processor: TextProcessor,
        is_validation: bool = False
    ):
        self.config = config
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.is_validation = is_validation
        
        # Load and parse filelist
        self.items = self._parse_filelist(filelist_path)
        logger.info(f"Loaded {len(self.items)} items from {filelist_path}")
        
        # Set max/min audio length based on whether this is training or validation
        self.max_audio_length = float('inf') if is_validation else config.get('max_audio_length', 8.0)
        self.min_audio_length = 0.0 if is_validation else config.get('min_audio_length', 1.0)
        
    def _parse_filelist(self, filelist_path: str) -> List[Dict]:
        """Parse filelist containing audio-text pairs."""
        items = []
        base_dir = os.path.dirname(filelist_path)
        
        with open(filelist_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 2:
                    logger.warning(f"Skipping invalid line in filelist: {line}")
                    continue
                    
                audio_path, text = parts[0], parts[1]
                
                # Handle relative paths
                if not os.path.isabs(audio_path):
                    audio_path = os.path.join(base_dir, audio_path)
                
                # Check if file exists
                if not os.path.exists(audio_path):
                    logger.warning(f"Audio file does not exist, skipping: {audio_path}")
                    continue
                    
                # Add optional speaker ID if available
                speaker_id = None
                if len(parts) > 2 and parts[2]:
                    try:
                        speaker_id = int(parts[2])
                    except ValueError:
                        logger.warning(f"Invalid speaker ID, using default: {parts[2]}")
                
                items.append({
                    'audio_path': audio_path,
                    'text': text,
                    'speaker_id': speaker_id
                })
                
        return items
    
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Item index
                
        Returns:
            dict: Item with processed text and audio
        """
        item = self.items[idx]
        
        # Process audio
        audio_data = self.audio_processor.process_audio_file(
            item['audio_path'], 
            apply_vad=not self.is_validation
        )
        
        # Check audio length constraints
        audio_length_sec = audio_data['waveform'].shape[1] / audio_data['sampling_rate']
        if audio_length_sec > self.max_audio_length or audio_length_sec < self.min_audio_length:
            # For validation, we still return the item but log a warning
            if self.is_validation:
                logger.warning(
                    f"Validation audio length {audio_length_sec:.2f}s outside range "
                    f"[{self.min_audio_length}, {self.max_audio_length}]: {item['audio_path']}"
                )
            # For training, we replace with a random valid item
            else:
                logger.info(
                    f"Audio length {audio_length_sec:.2f}s outside range "
                    f"[{self.min_audio_length}, {self.max_audio_length}], replacing: {item['audio_path']}"
                )
                return self.__getitem__(np.random.randint(0, len(self)))
        
        # Process text
        text_encoded = self.text_processor.encode_text(item['text'])
        
        # Prepare return dictionary
        result = {
            'text': text_encoded,
            'text_raw': item['text'],
            'audio': audio_data['mel_spectrogram'],
            'waveform': audio_data['waveform'].squeeze(0),  # Remove channel dim
            'audio_path': item['audio_path'],
            'audio_length': audio_data['waveform'].shape[1]
        }
        
        # Add speaker ID if available
        if item['speaker_id'] is not None:
            result['speaker_id'] = item['speaker_id']
            
        return result
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate function for DataLoader.
        
        Pads sequences to the same length and creates a batch.
        
        Args:
            batch (List[Dict]): List of individual items
                
        Returns:
            Dict: Batched data with padded sequences
        """
        # Get max lengths
        max_text_len = max(item['text'].shape[0] for item in batch)
        max_audio_len = max(item['audio'].shape[1] for item in batch)
        max_waveform_len = max(item['waveform'].shape[0] for item in batch)
        
        # Initialize tensors
        text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        text_lengths = torch.zeros(len(batch), dtype=torch.long)
        
        audio_padded = torch.zeros(
            len(batch), 
            batch[0]['audio'].shape[0],  # n_mel_channels
            max_audio_len
        )
        audio_lengths = torch.zeros(len(batch), dtype=torch.long)
        
        waveform_padded = torch.zeros(len(batch), max_waveform_len)
        
        # Store original values for reference
        ids = []
        text_raw = []
        audio_paths = []
        
        # Collect speaker IDs if available
        has_speaker_ids = 'speaker_id' in batch[0]
        if has_speaker_ids:
            speaker_ids = torch.zeros(len(batch), dtype=torch.long)
        
        for i, item in enumerate(batch):
            # Text
            text = item['text']
            text_padded[i, :text.shape[0]] = text
            text_lengths[i] = text.shape[0]
            
            # Audio (mel spectrogram)
            audio = item['audio']
            audio_padded[i, :, :audio.shape[1]] = audio
            audio_lengths[i] = audio.shape[1]
            
            # Waveform
            waveform = item['waveform']
            waveform_padded[i, :waveform.shape[0]] = waveform
            
            # Reference info
            ids.append(i)
            text_raw.append(item['text_raw'])
            audio_paths.append(item['audio_path'])
            
            # Speaker ID if available
            if has_speaker_ids:
                speaker_ids[i] = item['speaker_id']
        
        # Create output batch
        output = {
            'text_padded': text_padded,
            'text_lengths': text_lengths,
            'audio_padded': audio_padded,
            'audio_lengths': audio_lengths,
            'waveform_padded': waveform_padded,
            'ids': ids,
            'text_raw': text_raw,
            'audio_paths': audio_paths
        }
        
        # Add speaker IDs if available
        if has_speaker_ids:
            output['speaker_ids'] = speaker_ids
            
        return output 