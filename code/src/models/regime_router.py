from __future__ import annotations

import torch.nn as nn

__all__ = ['build_market_gate', 'build_market_macro_proj']


def build_market_gate(input_dim, hidden_dim, dropout):
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout * 0.5),
        nn.Linear(hidden_dim, input_dim),
        nn.Sigmoid(),
    )


def build_market_macro_proj(market_context_dim, hidden_dim, input_dim, dropout):
    return nn.Sequential(
        nn.LayerNorm(market_context_dim),
        nn.Linear(market_context_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout * 0.5),
        nn.Linear(hidden_dim, input_dim),
    )
