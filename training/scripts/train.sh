#!/bin/bash
# VITS Training Script for Azerbaijani TTS

# Set up environment if needed
# source venv/bin/activate

# Run training with recommended parameters
python train.py \
  --config config/base_vits.json \
  --use_xlsr \
  --speaker_embed_dim 512 \
  --prosody_control \
  --batch_size 16 \
  --lr 2e-4 \
  --epochs 1000 \
  --checkpoint_dir checkpoints \
  --log_dir logs 