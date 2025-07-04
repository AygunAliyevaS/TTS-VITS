"""
Azerbaijani phoneme processing for TTS.

This module provides functionality for converting Azerbaijani text to phoneme
sequences suitable for TTS training and inference.
"""

import torch
import re
import logging
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

class AzerbaijaniPhonemes:
    """
    Azerbaijani phoneme processor.
    
    This class handles conversion of Azerbaijani text to phoneme sequences
    for more accurate pronunciation modeling.
    
    Args:
        device (str): Device to run phoneme conversion on ('cuda' or 'cpu')
        use_xlsr (bool): Whether to use the XLSR model for phoneme extraction
    """
    def __init__(self, device: str = 'cuda', use_xlsr: bool = False):
        self.device = device
        self.use_xlsr = use_xlsr
        
        # Initialize phoneme set
        self.phonemes = [
            'a', 'æ', 'b', 'tʃ', 'd', 'e', 'f', 'g', 'ɣ', 'h', 'x',
            'i', 'dʒ', 'k', 'l', 'm', 'n', 'o', 'œ', 'p', 'q', 'ɾ',
            's', 'ʃ', 't', 'u', 'y', 'v', 'j', 'z', 'ʒ', 'ʔ'
        ]
        
        # Create mappings from phonemes to IDs and back
        self.phoneme_to_id = {p: i for i, p in enumerate(self.phonemes)}
        self.id_to_phoneme = {i: p for i, p in enumerate(self.phonemes)}
        
        # Special tokens
        self.silence_token = "<sil>"
        self.phoneme_to_id[self.silence_token] = len(self.phonemes)
        self.id_to_phoneme[len(self.phonemes)] = self.silence_token
        
        # Initialize XLSR model if requested
        if self.use_xlsr:
            try:
                self.xlsr = torch.hub.load('facebookresearch/fairseq', 'xlsr_300m').to(device)
                self.xlsr.eval()
                logger.info("Initialized XLSR model for phoneme extraction")
            except Exception as e:
                logger.error(f"Failed to initialize XLSR model: {e}")
                self.use_xlsr = False
        
        # Initialize grapheme-to-phoneme mapping for Azerbaijani
        self.g2p_map = self._initialize_g2p_map()
        
    def _initialize_g2p_map(self) -> dict:
        """
        Initialize the grapheme-to-phoneme mapping for Azerbaijani.
        
        Returns:
            dict: Mapping from graphemes to phonemes
        """
        # This is a simplified G2P mapping for Azerbaijani
        # A more complete implementation would use a trained G2P model
        g2p = {
            'a': 'a',
            'b': 'b',
            'c': 'dʒ',
            'ç': 'tʃ',
            'd': 'd',
            'e': 'e',
            'ə': 'æ',
            'f': 'f',
            'g': 'g',
            'ğ': 'ɣ',
            'h': 'h',
            'x': 'x',
            'ı': 'i',  # close back unrounded vowel
            'i': 'i',  # close front unrounded vowel
            'j': 'ʒ',
            'k': 'k',
            'q': 'q',
            'l': 'l',
            'm': 'm',
            'n': 'n',
            'o': 'o',
            'ö': 'œ',
            'p': 'p',
            'r': 'ɾ',
            's': 's',
            'ş': 'ʃ',
            't': 't',
            'u': 'u',
            'ü': 'y',
            'v': 'v',
            'y': 'j',
            'z': 'z',
        }
        return g2p
    
    def text_to_phonemes(self, text: str) -> str:
        """
        Convert Azerbaijani text to phoneme sequence.
        
        Args:
            text (str): Input Azerbaijani text
                
        Returns:
            str: Phoneme sequence as a string
        """
        # If XLSR model is available, use it for phoneme extraction
        if self.use_xlsr and hasattr(self, 'xlsr'):
            return self._extract_phonemes_with_xlsr(text)
        
        # Otherwise, use the rule-based G2P mapping
        return self._apply_g2p_rules(text)
    
    def _apply_g2p_rules(self, text: str) -> str:
        """
        Apply rule-based grapheme-to-phoneme conversion.
        
        Args:
            text (str): Input text
                
        Returns:
            str: Phoneme sequence
        """
        # Convert text to lowercase
        text = text.lower()
        
        # Split into words for better context handling
        words = text.split()
        phonemes = []
        
        for word in words:
            word_phonemes = []
            i = 0
            while i < len(word):
                # Check for digraphs first
                if i < len(word) - 1 and word[i:i+2] in self.g2p_map:
                    word_phonemes.append(self.g2p_map[word[i:i+2]])
                    i += 2
                # Then check for single characters
                elif word[i] in self.g2p_map:
                    word_phonemes.append(self.g2p_map[word[i]])
                    i += 1
                # Skip unknown characters
                else:
                    i += 1
            
            # Add word boundary
            if word_phonemes:
                phonemes.append(''.join(word_phonemes))
        
        # Join words with silence token
        return f" {self.silence_token} ".join(phonemes)
    
    def _extract_phonemes_with_xlsr(self, text: str) -> str:
        """
        Extract phonemes using the XLSR model.
        
        Args:
            text (str): Input text
                
        Returns:
            str: Phoneme sequence
        """
        # This is a placeholder for the XLSR-based phoneme extraction
        # In a real implementation, this would use the XLSR model to predict phonemes
        logger.warning("XLSR-based phoneme extraction not fully implemented, using rule-based approach")
        return self._apply_g2p_rules(text)
    
    def get_phoneme_ids(self, phoneme_text: str) -> List[int]:
        """
        Convert phoneme text to a sequence of phoneme IDs.
        
        Args:
            phoneme_text (str): Phoneme text
                
        Returns:
            List[int]: Sequence of phoneme IDs
        """
        ids = []
        i = 0
        while i < len(phoneme_text):
            # Try multi-character phonemes first (max 2 chars in our set)
            if i < len(phoneme_text) - 1 and phoneme_text[i:i+2] in self.phoneme_to_id:
                ids.append(self.phoneme_to_id[phoneme_text[i:i+2]])
                i += 2
            # Then try single-character phonemes
            elif phoneme_text[i] in self.phoneme_to_id:
                ids.append(self.phoneme_to_id[phoneme_text[i]])
                i += 1
            # Skip unknown characters (like spaces)
            else:
                i += 1
                
        return ids
    
    def phoneme_ids_to_text(self, ids: List[int]) -> str:
        """
        Convert a sequence of phoneme IDs back to text.
        
        Args:
            ids (List[int]): Sequence of phoneme IDs
                
        Returns:
            str: Phoneme text
        """
        phonemes = []
        for idx in ids:
            if idx in self.id_to_phoneme:
                phonemes.append(self.id_to_phoneme[idx])
                
        return ''.join(phonemes) 