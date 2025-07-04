import torch
import torch.nn as nn

class ProsodyControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.pitch_scale = nn.Parameter(torch.ones(1))
        self.duration_scale = nn.Parameter(torch.ones(1))

    def forward(self, duration, pitch):
        duration = duration * self.duration_scale
        pitch = pitch * self.pitch_scale
        return duration, pitch
