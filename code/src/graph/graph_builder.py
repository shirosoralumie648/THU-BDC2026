from __future__ import annotations

import numpy as np

from graph.correlation_graph import build_correlation_prior_adjacency
from graph.industry_graph import build_industry_prior_adjacency


def build_prior_graph_adjacency(train_data, stockid2idx, runtime_config=None):
    num_stocks = len(stockid2idx)
    if num_stocks <= 0:
        raise ValueError('stockid2idx 为空，无法构建先验图')

    industry_adj = build_industry_prior_adjacency(stockid2idx, runtime_config)
    corr_adj = build_correlation_prior_adjacency(train_data, num_stocks, runtime_config)
    prior_adj = industry_adj | corr_adj
    np.fill_diagonal(prior_adj, True)

    total_edges = int(prior_adj.sum())
    density = total_edges / float(num_stocks * num_stocks)
    print(
        f"先验图构建完成: num_stocks={num_stocks}, "
        f"industry_edges={int(industry_adj.sum())}, "
        f"corr_edges={int(corr_adj.sum())}, "
        f"merged_edges={total_edges}, density={density:.4f}"
    )
    return prior_adj.astype(np.bool_)
