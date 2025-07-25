"""Prosody / style reference encoder.

Based loosely on the GST-Tacotron reference encoder. It encodes a mel-spectrogram
snippet into a fixed-length style (prosody) embedding that can be injected into
the text latent to control intonation and overall speaking style.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceEncoder(nn.Module):
    """Encode a reference mel-spectrogram into a style embedding."""

    def __init__(self, n_mel_channels: int, ref_hidden: int = 128, ref_layers: int = 6, embed_dim: int = 128):
        super().__init__()
        convs = []
        in_ch = 1
        for i in range(ref_layers):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_ch,
                        ref_hidden,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=(1, 1)
                    ),
                    nn.BatchNorm2d(ref_hidden),
                    nn.ReLU(inplace=True)
                )
            )
            in_ch = ref_hidden
        self.convs = nn.Sequential(*convs)
        out_channels = n_mel_channels // (2 ** ref_layers)
        self.gru = nn.GRU(ref_hidden * out_channels, embed_dim, batch_first=True)

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        """mels: [B, n_mel, T] -> returns [B, embed_dim]"""
        x = mels.unsqueeze(1)  # [B, 1, n_mel, T]
        x = self.convs(x)  # [B, C, n_mel', T']
        B, C, F, T = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)  # [B, T, C*F]
        _, h = self.gru(x)
        return h.squeeze(0)  # [B, embed_dim]
