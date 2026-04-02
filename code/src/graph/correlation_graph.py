from __future__ import annotations

import numpy as np

from config import config


def build_correlation_prior_adjacency(train_data, num_stocks, runtime_config=None):
    runtime_config = runtime_config or config
    if not bool(runtime_config.get('prior_graph_use_correlation', True)):
        return np.zeros((num_stocks, num_stocks), dtype=bool)
    if 'instrument' not in train_data.columns or '日期' not in train_data.columns:
        return np.zeros((num_stocks, num_stocks), dtype=bool)

    source_col = str(runtime_config.get('prior_graph_corr_source_col', 'label_raw')).strip()
    if source_col not in train_data.columns:
        source_col = 'label' if 'label' in train_data.columns else source_col
    if source_col not in train_data.columns:
        return np.zeros((num_stocks, num_stocks), dtype=bool)

    corr_min_periods = int(runtime_config.get('prior_graph_corr_min_periods', 20))
    corr_threshold = float(runtime_config.get('prior_graph_corr_threshold', 0.2))
    corr_topk = max(0, int(runtime_config.get('prior_graph_corr_topk', 20)))

    pivot = train_data.pivot_table(
        index='日期',
        columns='instrument',
        values=source_col,
        aggfunc='mean',
    )
    if pivot.empty:
        return np.zeros((num_stocks, num_stocks), dtype=bool)

    corr = pivot.corr(min_periods=max(2, corr_min_periods))
    if corr.empty:
        return np.zeros((num_stocks, num_stocks), dtype=bool)

    valid_cols = []
    for col in corr.columns:
        try:
            idx = int(col)
        except Exception:
            continue
        if 0 <= idx < num_stocks:
            valid_cols.append(idx)
    if not valid_cols:
        return np.zeros((num_stocks, num_stocks), dtype=bool)

    corr = corr.loc[valid_cols, valid_cols]
    abs_corr = np.nan_to_num(np.abs(corr.to_numpy(dtype=np.float32)), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(abs_corr, 0.0)

    local_adj = np.zeros_like(abs_corr, dtype=bool)
    if corr_threshold > 0.0:
        local_adj |= abs_corr >= corr_threshold
    if corr_topk > 0:
        num_local = abs_corr.shape[0]
        for i in range(num_local):
            row = abs_corr[i]
            if row.size == 0:
                continue
            k = min(corr_topk, row.size)
            if k <= 0:
                continue
            topk_idx = np.argpartition(row, -k)[-k:]
            local_adj[i, topk_idx] = True

    local_adj = local_adj | local_adj.T
    np.fill_diagonal(local_adj, False)

    full_adj = np.zeros((num_stocks, num_stocks), dtype=bool)
    idx_arr = np.asarray(valid_cols, dtype=np.int64)
    full_adj[np.ix_(idx_arr, idx_arr)] = local_adj
    return full_adj
