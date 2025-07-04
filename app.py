import torch
import gradio as gr
import sys
import os

# Attempt to import the necessary module with fallback
try:
    from speechbrain.inference.classifiers import EncoderClassifier
    print("Import successful for EncoderClassifier")
except ImportError:
    print("Failed to import speechbrain.inference.classifiers.EncoderClassifier directly.")
    print("Attempting to import speechbrain.pretrained.EncoderClassifier (deprecated)")
    try:
        from speechbrain.pretrained import EncoderClassifier
        print("Import successful for EncoderClassifier")
    except ImportError:
        print("Failed to import EncoderClassifier from speechbrain. Please ensure speechbrain is installed correctly.")
        sys.exit(1) # Exit if import fails


from model.vits import VITS # Assuming vits model is defined here
import json # Assuming you need json for model config

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model configuration (assuming base_vits.json is the correct config)
try:
    with open('configs/base_vits.json') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: configs/base_vits.json not found.")
    sys.exit(1)


# Initialize the model
model = VITS(config).to(device)

# Load the model state dictionary
# Assuming 'best_model.pth' is the saved checkpoint from training
# checkpoint_path = 'checkpoints/best_model.pth' # Original path
checkpoint_path = 'checkpoints/checkpoint_182.pth' # Trying a different checkpoint


if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded model from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model state dictionary: {e}")
        sys.exit(1)
else:
    print(f"Error: Model checkpoint not found at {checkpoint_path}")
    sys.exit(1)

# Azerbaijani alphabet (with specific letters like Ə, Ğ, Ö, Ş, Ç, Ü)
az_chars = list("AÄBCÇDEƏFGĞHIİJKQLMNOÖPRSŞTUÜVXYZaäbcçdeəfgğhiijkqlmnoöprsştuüvxyz0123456789 .,!?'-")

# Ensure we fit the vocab limit
# Pad with <pad> or clip extra chars if needed
unique_chars = az_chars[:100]  # Adjust to fit vocab_size

# Create mapping dicts
char_to_symbol = {char: idx for idx, char in enumerate(unique_chars)}
symbol_to_char = {idx: char for char, idx in char_to_symbol.items()}

def text_to_sequence(text):
    """Convert text to sequence of indices, with better error handling."""
    try:
        # Convert text to lowercase and filter out unknown characters
        text = text.lower().strip()
        sequence = [char_to_symbol.get(c, char_to_symbol.get(' ', 0)) for c in text]
        print(f"Converted text to sequence. Length: {len(sequence)}")
        return sequence
    except Exception as e:
        print(f"Error in text_to_sequence: {e}")
        return None

def inference(text, speaker_id=0):
    if not text or not text.strip():
        print("Error: Empty input text")
        return None
        
    print(f"Input text: '{text}'")
    
    sequence = text_to_sequence(text)
    if not sequence:
        print("Error: Failed to convert text to sequence")
        return None

    try:
        x = torch.LongTensor(sequence).unsqueeze(0).to(device)
        print(f"Input tensor shape: {x.shape}")
        
        audio = model.generate(x)
        
        if audio is None:
            print("Error: Model returned None")
            return None
            
        print(f"Generated audio shape: {audio.shape if hasattr(audio, 'shape') else 'N/A'}")
        print(f"Audio stats - min: {audio.min()}, max: {audio.max()}, mean: {audio.mean()}")
        
        # Ensure audio is in the correct range [-1, 1]
        if audio.max() > 1.0 or audio.min() < -1.0:
            print("Warning: Audio values out of [-1, 1] range, normalizing...")
            audio = torch.clamp(audio, -1.0, 1.0)
        
        sampling_rate = config['data']['sampling_rate']
        return (sampling_rate, audio)
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Create Gradio interface
iface = gr.Interface(
    fn=inference,
    inputs=[gr.Textbox(label="Text Input"), gr.Number(label="Speaker ID", value=0)],
    outputs=gr.Audio(label="Generated Audio"),
    title="VITS Text-to-Speech",
    description="Enter text and generate speech using a trained VITS model."
)

if __name__ == "__main__":
    iface.launch(debug=True, share=True) # Added debug=True