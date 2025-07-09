#!/usr/bin/env python3
"""
Filelist Preparation Utility for VITS

This script generates train/val filelists for VITS training by:
1. Finding all WAV files in a directory
2. Splitting them into train/validation sets
3. Optionally pairing them with transcriptions
"""

import argparse
from pathlib import Path
import random
import logging
from typing import Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_transcriptions(trans_file: Path) -> Dict[str, str]:
    """Load transcriptions from a `path|text` file.
    
    This helper is tolerant of unescaped double-quotes or additional pipe
    characters inside the sentence – it only splits on the *first* pipe.
    """
    transcriptions: Dict[str, str] = {}

    if not trans_file.exists():
        logger.warning(f"Transcription file {trans_file} not found. Using dummy text.")
        return transcriptions

    with open(trans_file, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            if not line:
                continue  # skip empty lines
            if '|' not in line:
                logger.warning("Skipping line without '|': %s", line[:60])
                continue
            path_str, text = line.split('|', 1)  # split only on the *first* pipe
            filename = Path(path_str.strip()).stem  # stem -> key like '0003'
            transcriptions[filename] = text.strip()

    logger.info(f"Loaded {len(transcriptions)} transcriptions from {trans_file}")
    return transcriptions

def main():
    parser = argparse.ArgumentParser(description='Generate VITS training filelists')
    parser.add_argument('--wavs', type=str, required=True, 
                      help='Path to directory containing WAV files')
    parser.add_argument('--output', type=str, required=True,
                      help='Output directory for filelists')
    parser.add_argument('--transcriptions', type=str, default=None,
                      help='Optional CSV file with transcriptions (filename,text)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                      help='Proportion of files for validation (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    wavs_dir = Path(args.wavs)
    output_dir = Path(args.output)
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load transcriptions if provided
    transcriptions = {}
    if args.transcriptions:
        transcriptions = load_transcriptions(Path(args.transcriptions))

    # Get all WAV files
    wav_files = list(wavs_dir.glob('*.wav'))
    if not wav_files:
        logger.error(f"No WAV files found in {wavs_dir}")
        return
        
    logger.info(f"Found {len(wav_files)} WAV files in {wavs_dir}")
    random.shuffle(wav_files)
    
    # Split train/val
    split_idx = int(len(wav_files) * (1 - args.val_ratio))
    train_files = wav_files[:split_idx]
    val_files = wav_files[split_idx:]
    
    logger.info(f"Split dataset: {len(train_files)} training files, {len(val_files)} validation files")

    # Write filelists
    train_path = output_dir / 'train.txt'
    val_path = output_dir / 'val.txt'
    
    with open(train_path, 'w', encoding='utf-8') as f_train:
        for wav_path in train_files:
            filename = wav_path.stem
            text = transcriptions.get(filename, "dummy_text")
            f_train.write(f"{wav_path}|{text}\n")
            
    with open(val_path, 'w', encoding='utf-8') as f_val:
        for wav_path in val_files:
            filename = wav_path.stem
            text = transcriptions.get(filename, "dummy_text")
            f_val.write(f"{wav_path}|{text}\n")
            
    logger.info(f"Filelists created at {train_path} and {val_path}")

if __name__ == '__main__':
    main() 