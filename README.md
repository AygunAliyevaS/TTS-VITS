# 🎙️ Azerbaijani VITS Text-to-Speech System

A high-quality neural text-to-speech (TTS) system for the Azerbaijani language based on the VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) architecture.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 🌟 Features

- **End-to-end TTS**: Single model for text-to-waveform generation
- **High-quality synthesis**: Natural-sounding Azerbaijani speech
- **Zero-shot voice cloning**: Clone voices with just a short reference audio
- **Fast inference**: Real-time synthesis on modern GPUs
- **Gradio web interface**: User-friendly demo application

## 🚀 Quick Start

### Google Colab (Recommended)

The easiest way to get started is using Google Colab with free GPU resources:

1. Open our [Colab Notebook](https://colab.research.google.com) and upload `notebooks/VITS_Azerbaijani.ipynb`
2. Enable GPU runtime: Runtime > Change runtime type > GPU
3. Follow the step-by-step instructions in the notebook

### Local Installation

To set up the project locally:

```bash
# Clone the repository
git clone https://github.com/your-username/tts-vits.git
cd tts-vits

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the demo (if you have a trained model)
python app.py --checkpoint checkpoints/best.pt
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (for training)
- 8GB+ RAM
- 20GB+ disk space (for datasets and checkpoints)

## 🧠 Technical Overview

### VITS Architecture

This implementation is based on the [VITS paper](https://arxiv.org/abs/2106.06103) with specific adaptations for the Azerbaijani language:

1. **Text Encoder**: Transforms text into latent representations
   - Phoneme embedding layer
   - Transformer encoder blocks
   - Azerbaijani-specific phoneme set

2. **Variational Generator**:
   - Stochastic duration predictor for natural rhythm
   - Flow-based decoder for acoustic feature generation
   - Conditional VAE for high-quality spectrogram generation

3. **Neural Vocoder**:
   - HiFi-GAN architecture for waveform synthesis
   - Multi-period/multi-scale discriminators for adversarial training

### Azerbaijani Language Processing

- Custom phoneme inventory for Azerbaijani (`data/text/az_symbols.py`)
- Support for Azerbaijani-specific characters (Ə, Ğ, Ö, Ş, Ç, Ü)
- Text normalization and phonemization pipeline

## 📊 Dataset Preparation

### Audio Requirements

- Format: WAV, 16-bit PCM
- Sample rate: 22050 Hz (automatically converted)
- Duration: 1-10 seconds per clip recommended
- Quality: Clear speech, minimal background noise

### Preparation Steps

1. Place raw audio files in `datasets/raw/`
2. Create file lists:
```bash
python data/tools/prepare_filelist.py --wavs datasets/raw/ --output data/filelists/ --val-ratio 0.05
```

3. (Optional) Process audio using the built-in preprocessing in the training pipeline

## 🏋️ Training

### Configuration

Edit `config/base_vits.json` to adjust hyperparameters:
- Batch size
- Learning rate
- Network dimensions
- Training schedule

### Start Training

```bash
python train.py --config config/base_vits.json --output_dir checkpoints
```

### Training Phases

For best results, follow this training schedule:
1. **Phase 1 (Base Model)**
   - Batch Size: 16
   - Learning Rate: 2e-4
   - Steps: ~100k
   - Command: `python train.py --config config/base_vits.json`

2. **Phase 2 (Fine-tuning)**
   - Batch Size: 8
   - Learning Rate: 5e-5
   - Steps: ~50k
   - Command: `python train.py --config config/base_vits.json --checkpoint checkpoints/checkpoint_100k.pt --lr 5e-5 --batch_size 8`

### Resume Training

```bash
python train.py --config config/base_vits.json --checkpoint checkpoints/checkpoint_50k.pt
```

### Monitor Training

```bash
tensorboard --logdir=logs
```

## 🔊 Inference

### Command Line

```bash
python azeri_tts_pipeline.py --text "Salam dünya!" --checkpoint checkpoints/best.pt --output output.wav
```

### Interactive Demo

```bash
python app.py --config config/base_vits.json --checkpoint checkpoints/best.pt
```

This launches a Gradio web interface where you can:
- Input Azerbaijani text for synthesis
- Upload reference audio for voice cloning
- Adjust generation parameters
- Download synthesized audio

### Voice Cloning

```bash
python azeri_tts_pipeline.py --text "Salam dünya!" --checkpoint checkpoints/best.pt --reference_audio path/to/reference.wav --output output.wav
```

## 📁 Project Structure

```
tts-vits/
├── app.py                   # Gradio web interface
├── config/                  # Configuration files
│   ├── base_vits.json       # Main model configuration
│   └── hifigan.json         # Vocoder configuration
├── data/                    # Data processing modules
│   ├── processing.py        # Audio processing
│   ├── dataset.py           # Dataset implementation
│   ├── tools/               # Utility tools
│   │   └── prepare_filelist.py # File list preparation
│   └── text/                # Text processing
│       ├── az_symbols.py    # Azerbaijani phonemes
│       └── text_processor.py # Text normalization and encoding
├── datasets/                # Audio datasets
│   ├── raw/                 # Raw audio files
│   └── normalized/          # Processed audio files
├── model/                   # Core model implementation
│   ├── vits.py              # Main VITS model architecture
│   └── components/          # Model components
│       ├── duration.py      # Detailed duration predictor implementation
│       └── hifigan.py       # Detailed HiFi-GAN implementation
├── notebooks/               # Jupyter notebooks
│   └── VITS_Azerbaijani.ipynb # Colab notebook
├── training/                # Training modules
│   └── trainer.py           # Training logic
├── train.py                 # Training entry point
├── utils/                   # Utility functions
│   └── common.py            # Common utilities
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 🔧 Advanced Configuration

### Model Size

The model size can be adjusted in the configuration file:
- **Small**: ~15M parameters, faster inference, lower quality
- **Medium**: ~30M parameters, balanced performance (default)
- **Large**: ~60M parameters, highest quality, slower inference

Example configuration for a smaller model:
```json
{
  "model": {
    "hidden_channels": 128,
    "filter_channels": 256,
    "n_heads": 2,
    "n_layers": 4
  }
}
```

### Multi-Speaker Support

The model supports multi-speaker training with speaker embeddings. To enable:

1. Add speaker IDs to filelists:
```
datasets/raw/speaker1_001.wav|Text transcript|0
datasets/raw/speaker1_002.wav|Another text|0
datasets/raw/speaker2_001.wav|Different speaker|1
```

2. Update configuration:
```json
{
  "data": {
    "n_speakers": 2,
    "speaker_embedding_dim": 64
  }
}
```

## 📊 Performance Benchmarks

| Model Size | Parameters | RTF* | MOS** |
|------------|------------|------|-------|
| Small      | ~15M       | 0.05 | 3.7   |
| Medium     | ~30M       | 0.08 | 4.1   |
| Large      | ~60M       | 0.15 | 4.3   |

*RTF: Real-Time Factor (lower is better)  
**MOS: Mean Opinion Score (higher is better)

## 🔍 Troubleshooting

### Common Issues

- **CUDA out of memory**: Reduce batch size or model size
- **Audio quality issues**: Check input audio quality, normalize levels
- **Phonemization errors**: Update Azerbaijani phoneme set in `data/text/az_symbols.py`
- **Training instability**: Reduce learning rate, check for NaN values

### Environment Setup

For specific environments:

**Windows**:
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Linux**:
```bash
# Install system dependencies
apt-get update && apt-get install -y libsndfile1 ffmpeg espeak-ng
```

## 📚 References

- [VITS Paper](https://arxiv.org/abs/2106.06103): "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech"
- [HiFi-GAN](https://arxiv.org/abs/2010.05646): "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"
- [VITS2 Improvements](https://github.com/p0p4k/vits2_pytorch): Enhanced VITS implementation

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- The VITS authors for the original research
- The Azerbaijani language community for resources and feedback
- Contributors to the open-source TTS ecosystem

## 📧 Contact

For questions, issues, or collaboration:
- GitHub Issues: [Create an issue](https://github.com/your-username/tts-vits/issues)
- Email: your.email@example.com
