from __future__ import annotations

import torch.nn as nn

__all__ = [
    'build_multi_scale_fusion_gate',
    'build_short_horizon_fusion_gate',
]


def build_short_horizon_fusion_gate(d_model, dropout):
    return nn.Sequential(
        nn.LayerNorm(d_model * 2),
        nn.Linear(d_model * 2, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, 2),
    )


def build_multi_scale_fusion_gate(d_model, dropout):
    return nn.Sequential(
        nn.LayerNorm(d_model * 3),
        nn.Linear(d_model * 3, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, 3),
    )
