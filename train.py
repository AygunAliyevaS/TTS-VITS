import torch
import os
from torch.utils.data import DataLoader
from model.vits import VITS
from model.duration import StochasticDurationPredictor
import json
import torchaudio
from torchaudio.transforms import MelSpectrogram

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('configs/base_vits.json') as f:
    config = json.load(f)

# Load HiFiGAN config
with open('configs/hifigan.json') as f:
    hifigan_config = json.load(f)

# Initialize model
model = VITS(config).to(device)

# Mel-spectrogram transform for reconstruction loss
mel_transform = MelSpectrogram(
    sample_rate=config['data']['sampling_rate'],
    n_fft=config['data']['filter_length'],
    win_length=config['data']['win_length'],
    hop_length=config['data']['hop_length'],
    n_mels=config['model']['audio_channels']
).to(device)

base_lr = config['train']['learning_rate']
warmup_epochs = 5
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

# Dummy dataset
class TTSCollator:
    def __call__(self, batch):
        return {
            'text': torch.randint(0, 100, (len(batch), 50)),
            'audio': torch.randn(len(batch), 80, 200)  # 80-channel mel-spectrogram
        }

train_loader = DataLoader(
    range(100),  # Dummy dataset
    batch_size=config['train']['batch_size'],
    collate_fn=TTSCollator()
)

# Create checkpoint directory
checkpoint_dir = 'checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

best_loss = float('inf')
# Training loop
for epoch in range(config['train']['max_epochs']):
    for batch in train_loader:
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        
        output, dur_logits, mu, log_var, *_ = model(text, audio)
        
        # Compute mel-spectrogram from model output waveform
        waveform = output.squeeze(1)  # [B, T]
        mel_output = mel_transform(waveform)
        
        # Align time dimension with target mel
        min_len = min(mel_output.size(-1), audio.size(-1))
        mel_output = mel_output[..., :min_len]
        audio = audio[..., :min_len]
        
        # Calculate losses
        recon_loss = torch.nn.functional.l1_loss(mel_output, audio)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        dur_loss = torch.nn.functional.mse_loss(dur_logits, torch.randn_like(dur_logits))
        
        total_loss = recon_loss + kl_loss + dur_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        optimizer.step()
        
        # Learning rate warmup
        current_lr = base_lr * min(epoch / warmup_epochs, 1.0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        print(f'Epoch {epoch} Loss: {total_loss.item():.4f}')
    
    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': total_loss.item()
    }, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth'))
    
    # Update best model
    if total_loss.item() < best_loss:
        best_loss = total_loss.item()
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': total_loss.item()
        }, os.path.join(checkpoint_dir, 'best_model.pth'))
