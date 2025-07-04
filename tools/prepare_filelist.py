
import argparse
from pathlib import Path
import random

def main():
    parser = argparse.ArgumentParser(description='Generate VITS training filelists')
    parser.add_argument('--wavs', type=Path, required=True, 
                      help='Path to directory containing WAV files')
    parser.add_argument('--output', type=Path, required=True,
                      help='Output directory for filelists')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                      help='Proportion of files for validation (default: 0.1)')
    args = parser.parse_args()

    # Create output directory if needed
    args.output.mkdir(parents=True, exist_ok=True)

    # Get all WAV files
    wav_files = list(args.wavs.glob('*.wav'))
    random.shuffle(wav_files)
    
    # Split train/val
    split_idx = int(len(wav_files) * (1 - args.val_ratio))
    train_files = wav_files[:split_idx]
    val_files = wav_files[split_idx:]

    # Write filelists
    with (args.output/'train.txt').open('w') as f_train, (args.output/'val.txt').open('w') as f_val:
        
        for wav_path in train_files:
            f_train.write(f'{wav_path.absolute()}|dummy_phrases')
            
        for wav_path in val_files:
            f_val.write(f'{wav_path.absolute()}|dummy_phrases')

if __name__ == '__main__':
    main()
