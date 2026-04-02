from __future__ import annotations

from portfolio.candidate_selector import select_candidates
from portfolio.constraints import apply_constraints
from portfolio.weighting import compute_weights


def scores_to_portfolio(scores, stock_ids, strategy):
    top_k = min(int(strategy['top_k']), len(stock_ids), 5)
    if top_k <= 0:
        raise ValueError('持仓股票数量必须大于 0')

    strategy = dict(strategy)
    strategy['top_k'] = top_k
    candidate_ids, candidate_scores = select_candidates(scores, stock_ids)
    selected_ids, selected_scores = apply_constraints(candidate_ids, candidate_scores, strategy)
    weights = compute_weights(selected_scores, strategy)
    return selected_ids, weights
