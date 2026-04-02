from __future__ import annotations

import numpy as np


def select_candidates(scores, stock_ids):
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1:
        raise ValueError('scores 必须是一维数组')
    if len(score_array) != len(stock_ids):
        raise ValueError('scores 与 stock_ids 长度不一致')

    order = np.argsort(score_array, kind='mergesort')[::-1]
    selected_ids = [stock_ids[idx] for idx in order]
    selected_scores = score_array[order]
    return selected_ids, selected_scores
