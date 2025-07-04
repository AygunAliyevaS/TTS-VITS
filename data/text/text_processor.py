"""
Text processing utilities for Azerbaijani TTS.

This module provides functionality for processing and encoding Azerbaijani text
for use with the VITS TTS model, including phoneme conversion.
"""

import torch
import re
from typing import Dict, List, Optional, Union
import logging
import unicodedata

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Text processor for Azerbaijani TTS.
    
    This class handles all text preprocessing required for training and inference,
    including normalization, cleaning, and conversion to phoneme or token IDs.
    
    Args:
        config (dict): Configuration dictionary
    """
    def __init__(self, config: Dict):
        self.config = config
        
        # Create character mapping
        # Define Azerbaijani character set (uppercase, lowercase, digits, and special chars)
        # This includes the specific Azerbaijani characters: Əə, Ğğ, Iı, İi, Öö, Üü, Çç, Şş
        self.chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                         "ƏəĞğIıİiÖöÜüÇçŞş0123456789 .,!?'-")
        
        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"  # beginning of sequence
        self.eos_token = "<eos>"  # end of sequence
        
        # Add special tokens to character list
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.chars = special_tokens + self.chars
        
        # Create char-to-id and id-to-char mappings
        self.char_to_id = {c: i for i, c in enumerate(self.chars)}
        self.id_to_char = {i: c for i, c in enumerate(self.chars)}
        
        # Set indices of special tokens
        self.pad_id = self.char_to_id[self.pad_token]
        self.unk_id = self.char_to_id[self.unk_token]
        self.bos_id = self.char_to_id[self.bos_token]
        self.eos_id = self.char_to_id[self.eos_token]
        
        # Load phoneme converter if available
        self.use_phonemes = config.get('use_phonemes', False)
        if self.use_phonemes:
            try:
                from text.az_symbols import AzerbaijaniPhonemes
                self.phoneme_converter = AzerbaijaniPhonemes()
                logger.info("Loaded Azerbaijani phoneme converter")
            except ImportError:
                logger.warning("Could not import AzerbaijaniPhonemes, falling back to character encoding")
                self.use_phonemes = False
        
        logger.info(f"Initialized TextProcessor with {len(self.chars)} characters")
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize Azerbaijani text.
        
        Args:
            text (str): Input text
                
        Returns:
            str: Normalized text
        """
        # Convert to lowercase if specified in config
        if self.config.get('lowercase', True):
            text = text.lower()
            
        # Normalize Unicode characters (NFD -> NFC)
        text = unicodedata.normalize('NFC', text)
        
        # Replace common punctuation variations
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r'['']', "'", text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def text_to_sequence(self, text: str) -> List[int]:
        """
        Convert text to a sequence of token IDs.
        
        Args:
            text (str): Input text
                
        Returns:
            List[int]: Sequence of token IDs
        """
        # First normalize the text
        text = self.normalize_text(text)
        
        # Convert to sequence of IDs
        sequence = []
        
        # Add beginning of sequence token if configured
        if self.config.get('use_bos_eos', True):
            sequence.append(self.bos_id)
        
        # Convert each character to its ID
        for char in text:
            if char in self.char_to_id:
                sequence.append(self.char_to_id[char])
            else:
                sequence.append(self.unk_id)
                logger.warning(f"Unknown character in text: {char}")
        
        # Add end of sequence token if configured
        if self.config.get('use_bos_eos', True):
            sequence.append(self.eos_id)
            
        return sequence
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """
        Convert a sequence of token IDs back to text.
        
        Args:
            sequence (List[int]): Sequence of token IDs
                
        Returns:
            str: Reconstructed text
        """
        # Convert IDs back to characters
        chars = []
        for idx in sequence:
            if idx in self.id_to_char:
                # Skip special tokens
                if idx not in [self.pad_id, self.bos_id, self.eos_id]:
                    chars.append(self.id_to_char[idx])
            else:
                logger.warning(f"Unknown ID in sequence: {idx}")
                
        return ''.join(chars)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text into a tensor of token IDs.
        
        Args:
            text (str): Input text
                
        Returns:
            torch.Tensor: Tensor of token IDs
        """
        if self.use_phonemes:
            # Convert text to phonemes first
            phonemes = self.phoneme_converter.text_to_phonemes(text)
            sequence = self.text_to_sequence(phonemes)
        else:
            # Use direct character encoding
            sequence = self.text_to_sequence(text)
            
        return torch.LongTensor(sequence)
    
    def decode_ids(self, ids: Union[torch.Tensor, List[int]]) -> str:
        """
        Decode a sequence of IDs back to text.
        
        Args:
            ids (Union[torch.Tensor, List[int]]): Sequence of token IDs
                
        Returns:
            str: Decoded text
        """
        # Convert tensor to list if necessary
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        return self.sequence_to_text(ids) 