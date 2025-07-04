import librosa
try:
    import webrtcvad
    _has_vad = True
except ImportError:
    webrtcvad = None
    _has_vad = False
import numpy as np


def preprocess_audio(path, target_sr=16000):
    # Load and resample
    y, sr = librosa.load(path, sr=target_sr)
    
    # Optional VAD (skip if webrtcvad not available)
    if _has_vad:
        # implement simple frame-based VAD if desired in future
        pass
    
    return y  # Return raw audio for embedding
