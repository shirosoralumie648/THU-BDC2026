from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    'FeatureAttention',
    'PositionalEncoding',
    '_normalize_scale_windows',
    'build_temporal_encoder',
]


def _normalize_scale_windows(windows, sequence_length, default_windows):
    if windows is None:
        windows = default_windows
    if isinstance(windows, (int, float, str)):
        windows = [windows]

    normalized = []
    for w in windows:
        try:
            w_int = int(w)
        except (TypeError, ValueError):
            continue
        w_int = max(1, min(int(sequence_length), w_int))
        if w_int not in normalized:
            normalized.append(w_int)

    if not normalized:
        fallback = []
        for w in default_windows:
            w_int = max(1, min(int(sequence_length), int(w)))
            if w_int not in fallback:
                fallback.append(w_int)
        normalized = fallback if fallback else [max(1, int(sequence_length))]

    return normalized


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FeatureAttention(nn.Module):
    """特征注意力模块"""

    def __init__(self, d_model, dropout=0.1):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_weights = self.attention(x)
        attended = torch.sum(x * attention_weights, dim=1)
        return self.dropout(attended)


def build_temporal_encoder(config):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=config['d_model'],
        nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
