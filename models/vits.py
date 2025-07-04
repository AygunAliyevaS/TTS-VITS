import torch
import torch.nn as nn

class VITS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spk_proj = nn.Linear(256, 512)
        
        # Initialize other components from config
        self.encoder = Encoder(**config['encoder'])
        self.decoder = Decoder(**config['decoder'])

    def forward(self, text, speaker_embeddings):
        spk_emb = self.spk_proj(speaker_embeddings)
        x = self.encoder(text)
        x = x + spk_emb.unsqueeze(1)
        return self.decoder(x)
