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
        
        # Length constraints – training can optionally restrict to a
        # configurable window of seconds.  The values may live either at the
        # top level of the JSON (legacy) or under the "data" subsection (new
        # configs).  A value of null/None disables that particular limit.

        cfg_lengths = config.get('data', {}) if isinstance(config.get('data'), dict) else {}

        def _get(key, default):
            # Look under data[key] first, then fall back to top-level.
            return cfg_lengths.get(key, config.get(key, default))

        if is_validation:
            self.max_audio_length = float('inf')
            self.min_audio_length = 0.0
        else:
            max_len = _get('max_audio_length', float('inf'))
            self.max_audio_length = float('inf') if max_len in (None, 'null') else max_len

            min_len = _get('min_audio_length', 0.0)
            self.min_audio_length = 0.0 if min_len in (None, 'null') else min_len
        
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
                    # First try relative to the file-list directory (historical behaviour)
                    candidate_path = os.path.join(base_dir, audio_path)

                    # If that file does not exist, fall back to interpreting the path
                    # as relative to the current working directory/project root. This
                    # is convenient when filelists are stored in a sub-folder but the
                    # paths they contain are written relative to the repository root,
                    # e.g. "datasets/raw/filename.wav". (Colab & many training scripts
                    # generate such lists.)
                    if os.path.exists(candidate_path):
                        audio_path = candidate_path
                    else:
                        audio_path = os.path.abspath(audio_path)
                
                # Check if file exists
                if not os.path.exists(audio_path):
                    logger.warning(f"Audio file does not exist, skipping: {audio_path}")
                    continue
                    
                # Optional IDs
                speaker_id = None
                emotion_id = None
                language_id = None
                if len(parts) > 2 and parts[2]:
                    try:
                        speaker_id = int(parts[2])
                    except ValueError:
                        logger.warning(f"Invalid speaker ID, using default: {parts[2]}")
                if len(parts) > 3 and parts[3]:
                    try:
                        emotion_id = int(parts[3])
                    except ValueError:
                        logger.warning(f"Invalid emotion ID, using default: {parts[3]}")
                if len(parts) > 4 and parts[4]:
                    try:
                        language_id = int(parts[4])
                    except ValueError:
                        logger.warning(f"Invalid language ID, using default: {parts[4]}")
                speaker_id = None
                if len(parts) > 2 and parts[2]:
                    try:
                        speaker_id = int(parts[2])
                    except ValueError:
                        logger.warning(f"Invalid speaker ID, using default: {parts[2]}")
                
                items.append({
                    'audio_path': audio_path,
                    'text': text,
                    'speaker_id': speaker_id,
                    'emotion_id': emotion_id,
                    'language_id': language_id
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
        
        # ------------------------------------------------------------------
        # Prepare reference mel for prosody (simple heuristic – first N frames)
        # ------------------------------------------------------------------
        prosody_frames = self.config['model'].get('prosody_ref_frames', 80)  # ~0.5 s if hop 256 at 48 kHz
        ref_mel = audio_data['mel_spectrogram'][..., :prosody_frames]

        # Prepare return dictionary
        result = {
            'text': text_encoded,
            'text_raw': item['text'],
            'audio': audio_data['mel_spectrogram'],
            'waveform': audio_data['waveform'].squeeze(0),  # Remove channel dim
            'audio_path': item['audio_path'],
            'audio_length': audio_data['waveform'].shape[1],
            'ref_mel': ref_mel
        }
        
        # Add speaker ID if available
        if item['speaker_id'] is not None:
            result['speaker_id'] = item['speaker_id']
            
        return result
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Create a padded batch from a list of dataset items.

        Each ``item`` contains at least the keys ``text``, ``audio`` (mel), and ``waveform``.
        Optional keys: ``speaker_id``, ``emotion_id``, ``language_id``, ``ref_mel``.
        """
        # ------------------------------------------------------------------
        # Prepare dynamic dimensions
        # ------------------------------------------------------------------
        max_text_len = max(item['text'].shape[0] for item in batch)
        n_mel_channels = batch[0]['audio'].shape[0]
        max_mel_len = max(item['audio'].shape[1] for item in batch)
        max_wav_len = max(item['waveform'].shape[0] for item in batch)

        # Reference‐mel lengths (may be shorter than full mel)
        has_ref = all('ref_mel' in item for item in batch)
        max_ref_len = max(item['ref_mel'].shape[1] for item in batch) if has_ref else 0

        B = len(batch)

        # ------------------------------------------------------------------
        # Allocate padded tensors
        # ------------------------------------------------------------------
        text_padded = torch.zeros(B, max_text_len, dtype=torch.long)
        text_lengths = torch.zeros(B, dtype=torch.long)

        mel_padded = torch.zeros(B, n_mel_channels, max_mel_len)
        mel_lengths = torch.zeros(B, dtype=torch.long)

        wav_padded = torch.zeros(B, max_wav_len)

        if has_ref:
            ref_padded = torch.zeros(B, n_mel_channels, max_ref_len)
            ref_lengths = torch.zeros(B, dtype=torch.long)
        else:
            ref_padded = None
            ref_lengths = None

        # Optional ID tensors
        has_spk = all('speaker_id' in item for item in batch)
        has_emo = all('emotion_id' in item for item in batch)
        has_lang = all('language_id' in item for item in batch)
        if has_spk:
            speaker_ids = torch.zeros(B, dtype=torch.long)
        if has_emo:
            emotion_ids = torch.zeros(B, dtype=torch.long)
        if has_lang:
            language_ids = torch.zeros(B, dtype=torch.long)

        ids, text_raw, audio_paths = [], [], []

        # ------------------------------------------------------------------
        # Fill tensors
        # ------------------------------------------------------------------
        for i, item in enumerate(batch):
            # text
            t = item['text']
            text_padded[i, : t.shape[0]] = t
            text_lengths[i] = t.shape[0]

            # mel
            m = item['audio']
            mel_padded[i, :, : m.shape[1]] = m
            mel_lengths[i] = m.shape[1]

            # waveform
            w = item['waveform']
            wav_padded[i, : w.shape[0]] = w

            # reference mel
            if has_ref:
                r = item['ref_mel']
                ref_padded[i, :, : r.shape[1]] = r
                ref_lengths[i] = r.shape[1]

            # IDs
            if has_spk:
                speaker_ids[i] = item['speaker_id']
            if has_emo:
                emotion_ids[i] = item['emotion_id']
            if has_lang:
                language_ids[i] = item['language_id']

            # meta
            ids.append(i)
            text_raw.append(item['text_raw'])
            audio_paths.append(item['audio_path'])

        batch_out = {
            'text_padded': text_padded,
            'text_lengths': text_lengths,
            'audio_padded': mel_padded,
            'audio_lengths': mel_lengths,
            'waveform_padded': wav_padded,
            'ids': ids,
            'text_raw': text_raw,
            'audio_paths': audio_paths,
        }
        if has_ref:
            batch_out['ref_mels'] = ref_padded
            batch_out['ref_lengths'] = ref_lengths
        if has_spk:
            batch_out['speaker_ids'] = speaker_ids
        if has_emo:
            batch_out['emotion_ids'] = emotion_ids
        if has_lang:
            batch_out['language_ids'] = language_ids

        # return the batched dictionary
        return batch_out 