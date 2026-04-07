from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ['compute_volatility_huber_loss']


def compute_volatility_huber_loss(volatility_pred, volatility_targets):
    if volatility_targets is None or volatility_pred is None:
        return None

    vol_mask = ~torch.isnan(volatility_targets)
    if not vol_mask.any():
        return None

    return F.huber_loss(
        volatility_pred[vol_mask],
        volatility_targets[vol_mask],
    )
