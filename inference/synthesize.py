#!/usr/bin/env python3
"""
Simple inference script for Azerbaijani VITS + HiFi-GAN.

Example usage:
    python inference/synthesize.py --config config/base_vits.json \
           --checkpoint checkpoints/vits_latest.ckpt \
           --text "Salam dünya!" --out wav_outputs/sample.wav

It loads the saved checkpoint, converts the input text to phonemes (if
`use_phonemes` is true), generates a waveform and writes it to disk.
"""
from __future__ import annotations

import argparse
import os
import sys
import logging
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + os.sep + '..')

from utils.common import load_config, prepare_device
from data.text.text_processor import TextProcessor
from model.vits import VITS

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("synthesize")

def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    """Load model weights (state_dict) from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt, strict=False)
    logger.info("Loaded checkpoint %s", ckpt_path)


def synthesize_one(text: str, model: VITS, text_processor: TextProcessor, device: torch.device, *, speaker=None, emotion=None, language=None, ref_audio=None, speed=1.0, pitch=0):
    """Generate waveform for a single sentence."""
    model.eval()
    with torch.no_grad():
        ids = text_processor.encode_text(text).unsqueeze(0).to(device)  # [1, T]
        # Optional language ID
        lang_id = language

        # Optional reference mel
        ref_mel = None
        if ref_audio is not None:
            wav_ref, sr = torchaudio.load(ref_audio)
            if sr != model.config['data']['sampling_rate']:
                wav_ref = torchaudio.functional.resample(wav_ref, sr, model.config['data']['sampling_rate'])
            from data.processing import AudioProcessor
            ap = AudioProcessor(model.config)
            mel = ap.waveform_to_mel(wav_ref.squeeze(0))  # [n_mel, T]
            ref_mel = mel.unsqueeze(0).to(device)

        wav = model.generate(
            ids,
            speed_adjustment=speed,
            speaker_id=speaker,
            emotion_id=emotion,
            lang_id=lang_id,
            ref_mel=ref_mel,
            pitch_shift=pitch
        )[0].cpu()
        return wav


def main():
    p = argparse.ArgumentParser(description="Run inference with VITS on Azerbaijani text")
    p.add_argument('--config', required=True, help='Path to config JSON')
    p.add_argument('--checkpoint', required=True, help='Path to model checkpoint (*.ckpt)')
    p.add_argument('--text', required=True, help='Text to synthesize')
    p.add_argument('--speaker', type=int, default=None, help='Speaker ID (if multi-speaker)')
    p.add_argument('--emotion', type=int, default=None, help='Emotion ID (if emotion embedding)')
    p.add_argument('--language', type=int, default=None, help='Language ID (if multi-lingual)')
    p.add_argument('--ref_audio', default=None, help='Optional reference audio to mimic prosody/style')
    p.add_argument('--speed', type=float, default=1.0, help='Speed adjustment factor')
    p.add_argument('--pitch', type=int, default=0, help='Pitch shift in semitones')
    p.add_argument('--out', required=False, default='output.wav', help='Output wav file')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = prepare_device(1 if 'cuda' in args.device else 0)

    # Load configuration
    cfg = load_config(args.config)

    # Build text processor first to obtain vocab size
    tp = TextProcessor(cfg)
    cfg['model']['vocab_size'] = len(tp.chars)

    # Build model and load weights
    model = VITS(cfg).to(device)
    load_checkpoint(model, args.checkpoint, device)

    # Synthesize
    logger.info("Synthesizing: %s", args.text)
    wav = synthesize_one(
        args.text, model, tp, device,
        speaker=args.speaker, emotion=args.emotion,
        language=args.language, ref_audio=args.ref_audio,
        speed=args.speed, pitch=args.pitch
    )

    # Save
    torchaudio.save(args.out, wav.unsqueeze(0), cfg['data']['sampling_rate'])
    logger.info("Saved output to %s", args.out)


if __name__ == '__main__':
    main()
