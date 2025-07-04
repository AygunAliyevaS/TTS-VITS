"""
Audio processing utilities for VITS TTS.

This module provides functionality for loading, processing and transforming audio
data for use with the VITS TTS model, including mel-spectrogram extraction.
"""

import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import webrtcvad
    _has_vad = True
except ImportError:
    webrtcvad = None
    _has_vad = False
    logger.warning("webrtcvad not installed, VAD functionality will be unavailable")

class AudioProcessor:
    """
    Audio processor for VITS TTS.
    
    This class handles all audio preprocessing required for training and inference,
    including loading, resampling, normalization, and mel-spectrogram extraction.
    
    Args:
        config (dict): Configuration dictionary with audio parameters
    """
    def __init__(self, config):
        self.config = config
        self.sampling_rate = config['data']['sampling_rate']
        self.filter_length = config['data']['filter_length']
        self.hop_length = config['data']['hop_length']
        self.win_length = config['data']['win_length']
        self.n_mel_channels = config['model']['audio_channels']
        
        # Initialize mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.filter_length,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mel_channels,
            power=1.0  # Use amplitude spectrogram for better inversion
        )
        
        # Initialize VAD if available
        if _has_vad:
            self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (most aggressive)
        
        logger.info(f"AudioProcessor initialized: sr={self.sampling_rate}, "
                    f"n_mel={self.n_mel_channels}, hop={self.hop_length}")
    
    def load_audio(self, path, normalize=True):
        """
        Load audio from file.
        
        Args:
            path (str): Path to audio file
            normalize (bool): Whether to normalize audio to [-1, 1]
                
        Returns:
            Tensor: Audio waveform
            int: Sample rate
        """
        try:
            # Try using torchaudio first (faster)
            waveform, sr = torchaudio.load(path)
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
        except Exception as e:
            # Fallback to librosa if torchaudio fails
            logger.warning(f"torchaudio load failed, falling back to librosa: {e}")
            waveform, sr = librosa.load(path, sr=None, mono=True)
            waveform = torch.FloatTensor(waveform).unsqueeze(0)
            
        # Resample if needed
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
            
        # Normalize if requested
        if normalize:
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            
        return waveform, self.sampling_rate
    
    def apply_vad(self, waveform, frame_duration_ms=30):
        """
        Apply Voice Activity Detection to trim silence.
        
        Args:
            waveform (Tensor): Audio waveform [1, T]
            frame_duration_ms (int): Frame size for VAD in milliseconds
                
        Returns:
            Tensor: Trimmed audio waveform [1, T']
        """
        if not _has_vad:
            logger.warning("VAD requested but webrtcvad not installed")
            return waveform
            
        # Convert to numpy for VAD processing
        audio_np = waveform.numpy().flatten()
        
        # Calculate parameters
        frame_size = int(self.sampling_rate * frame_duration_ms / 1000)
        frames = int(len(audio_np) / frame_size)
        
        # Process frames
        voiced_frames = []
        for i in range(frames):
            frame = audio_np[i*frame_size:(i+1)*frame_size]
            # Convert to 16-bit PCM
            frame_pcm = (frame * 32768).astype(np.int16).tobytes()
            if self.vad.is_speech(frame_pcm, self.sampling_rate):
                voiced_frames.append(frame)
                
        # Reconstruct audio
        if voiced_frames:
            audio_np = np.concatenate(voiced_frames)
            return torch.FloatTensor(audio_np).unsqueeze(0)
        else:
            # If no voiced frames found, return original
            return waveform
    
    def extract_mel_spectrogram(self, waveform):
        """
        Extract mel-spectrogram from audio waveform.
        
        Args:
            waveform (Tensor): Audio waveform [1, T]
                
        Returns:
            Tensor: Mel-spectrogram [n_mel_channels, T']
        """
        # Ensure waveform is on the same device as the mel transform
        device = next(self.mel_transform.parameters()).device
        waveform = waveform.to(device)
        
        # Extract mel-spectrogram
        mel = self.mel_transform(waveform)
        
        # Convert to log mel-spectrogram
        mel = torch.log(torch.clamp(mel, min=1e-5))
        
        return mel.squeeze(0)  # Remove batch dimension
    
    def process_audio_file(self, file_path, apply_vad=True):
        """
        Process audio file for model input.
        
        Args:
            file_path (str): Path to audio file
            apply_vad (bool): Whether to apply VAD
                
        Returns:
            dict: Processed audio data with waveform and mel-spectrogram
        """
        waveform, sr = self.load_audio(file_path)
        
        if apply_vad and _has_vad:
            waveform = self.apply_vad(waveform)
            
        mel_spectrogram = self.extract_mel_spectrogram(waveform)
        
        return {
            'waveform': waveform,
            'mel_spectrogram': mel_spectrogram,
            'sampling_rate': sr,
            'file_path': file_path
        } 