import torch
from torch import nn


class DualViewRefiner(nn.Module):
    """Shared acoustic encoder, learned view gating, event Transformer.

    All heads operate on candidate notes. Missing pitches outside the candidate
    union cannot be recovered. This is not an end-to-end transcription network.
    """
    def __init__(self, width=128, layers=3, dropout=.15):
        super().__init__()
        self.config = dict(width=width, layers=layers, dropout=dropout)
        self.encoder = nn.Sequential(
            nn.Conv1d(6, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(256, width))
        self.view_gate = nn.Linear(width * 2 + 24, width)
        self.context = nn.Linear(24, width)
        self.position = nn.Linear(10, width)
        layer = nn.TransformerEncoderLayer(width, 4, width * 3, dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.sequence = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.action = nn.Linear(width, 3)  # reject / true onset / true same-key reattack
        self.velocity = nn.Linear(width, 1)
        self.duration = nn.Linear(width, 1)  # log1p(key-hold seconds)

    def forward(self, patch, context, position, valid):
        b, n, _, t = patch.shape
        encoded = self.encoder(patch.reshape(b*n*2, 6, t)).reshape(b, n, 2, -1)
        mix, piano = encoded.unbind(2)
        gate = self.view_gate(torch.cat([mix, piano, context], -1)).sigmoid()
        x = gate * mix + (1-gate) * piano + self.context(context) + self.position(position)
        x = self.norm(self.sequence(x, src_key_padding_mask=~valid))
        return {'action': self.action(x), 'velocity': self.velocity(x).squeeze(-1).sigmoid(),
                'duration': nn.functional.softplus(self.duration(x).squeeze(-1))}
