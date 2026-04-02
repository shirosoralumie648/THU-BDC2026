from __future__ import annotations

import numpy as np


def compute_weights(selected_scores, strategy):
    selected_scores = np.asarray(selected_scores, dtype=np.float64)
    top_k = len(selected_scores)
    if top_k <= 0:
        raise ValueError('持仓股票数量必须大于 0')

    weighting = strategy['weighting']
    if weighting == 'equal' or top_k == 1:
        return np.full(top_k, 1.0 / top_k, dtype=np.float64)
    if weighting == 'softmax':
        temperature = max(float(strategy.get('temperature', 1.0)), 1e-6)
        stable_scores = selected_scores - selected_scores.max()
        weights = np.exp(stable_scores / temperature)
        return weights / weights.sum()
    raise ValueError(f'不支持的权重方式: {weighting}')
