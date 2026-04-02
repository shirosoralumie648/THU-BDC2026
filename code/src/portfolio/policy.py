from __future__ import annotations

import numpy as np


def scores_to_portfolio(scores, stock_ids, strategy):
    top_k = min(int(strategy['top_k']), len(stock_ids), 5)
    if top_k <= 0:
        raise ValueError('持仓股票数量必须大于 0')

    order = np.argsort(scores)[::-1]
    top_indices = order[:top_k]
    selected_ids = [stock_ids[i] for i in top_indices]
    selected_scores = scores[top_indices]

    if strategy['weighting'] == 'equal' or top_k == 1:
        weights = np.full(top_k, 1.0 / top_k, dtype=np.float64)
    elif strategy['weighting'] == 'softmax':
        temperature = max(float(strategy.get('temperature', 1.0)), 1e-6)
        stable_scores = selected_scores - selected_scores.max()
        weights = np.exp(stable_scores / temperature)
        weights = weights / weights.sum()
    else:
        raise ValueError(f"不支持的权重方式: {strategy['weighting']}")

    return selected_ids, weights
