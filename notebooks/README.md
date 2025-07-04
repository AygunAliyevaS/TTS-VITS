# VITS Azerbaijani TTS Notebooks

This directory contains Jupyter notebooks for training and using the VITS Azerbaijani TTS system.

## Google Colab Setup

The main notebook, `VITS_Azerbaijani_Colab.ipynb`, is designed to work with Google Colab's free GPU resources. To use it:

1. Upload the notebook to Google Colab:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click on "File" > "Upload notebook"
   - Select the notebook file from your computer

2. Enable GPU acceleration:
   - Go to "Runtime" > "Change runtime type"
   - Set "Hardware accelerator" to "GPU"
   - Click "Save"

3. The notebook includes all necessary steps to:
   - Install required dependencies
   - Clone the repository
   - Set up Google Drive integration for data persistence
   - Prepare training data
   - Train the model
   - Run inference
   - Create a Gradio interface for testing

## Tips for Using Colab

- **Session Limits**: Colab free tier has a usage limit of about 12 hours per session. Save your model checkpoints regularly to Google Drive.

- **Memory Management**: 
  - If you encounter out-of-memory errors, reduce the batch size and use gradient accumulation
  - Clear GPU memory with `torch.cuda.empty_cache()` when needed
  - Restart the runtime if memory issues persist

- **Data Persistence**:
  - Always mount Google Drive to save your model checkpoints
  - Consider breaking your training into smaller epochs and resuming from checkpoints
  - Use the `--checkpoint` flag when resuming training

## Running Inference Only

If you just want to run inference with a pretrained model:

1. Upload your model checkpoint to Google Drive
2. Skip to the "Inference and Testing" section of the notebook
3. Load your model and run the Gradio interface

## Using Your Own Dataset

To use your own Azerbaijani audio dataset:

1. Upload your audio files to Google Drive
2. Update the filelists to point to your data
3. Make sure your transcripts follow the format: `path/to/audio.wav|Text transcript here.` 