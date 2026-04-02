from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_metadata(strategy):
    metadata = strategy.get('metadata', {})
    if not isinstance(metadata, dict):
        metadata = {}
    return metadata


def _as_previous_holdings(value) -> set[str]:
    if isinstance(value, dict):
        return {str(key).strip() for key in value.keys() if str(key).strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    if isinstance(value, str) and value.strip():
        return {value.strip()}
    return set()


def _rank_with_turnover_penalty(candidate_ids: Iterable[str], candidate_scores: np.ndarray, strategy):
    metadata = _as_metadata(strategy)
    previous_holdings = _as_previous_holdings(
        metadata.get('previous_holdings', strategy.get('previous_holdings'))
    )
    penalty = max(float(strategy.get('turnover_penalty', 0.0) or 0.0), 0.0)
    if penalty <= 0.0 or not previous_holdings:
        return list(candidate_ids), np.asarray(candidate_scores, dtype=np.float64)

    adjusted = []
    for idx, (stock_id, score) in enumerate(zip(candidate_ids, candidate_scores)):
        keep_bonus = 0.0 if stock_id in previous_holdings else penalty
        adjusted.append((float(score) - keep_bonus, float(score), -idx, stock_id))

    reordered = sorted(adjusted, reverse=True)
    score_lookup = {
        stock_id: float(score)
        for stock_id, score in zip(candidate_ids, np.asarray(candidate_scores, dtype=np.float64))
    }
    ordered_ids = [stock_id for _, _, _, stock_id in reordered]
    ordered_scores = np.asarray([score_lookup[stock_id] for stock_id in ordered_ids], dtype=np.float64)
    return ordered_ids, ordered_scores


def apply_constraints(candidate_ids, candidate_scores, strategy):
    ordered_ids, ordered_scores = _rank_with_turnover_penalty(candidate_ids, candidate_scores, strategy)
    metadata = _as_metadata(strategy)
    stock_to_industry = metadata.get('stock_to_industry', strategy.get('stock_to_industry', {}))
    if not isinstance(stock_to_industry, dict):
        stock_to_industry = {}

    top_k = min(int(strategy['top_k']), len(ordered_ids), 5)
    max_per_industry = int(strategy.get('max_per_industry', 0) or 0)

    selected_ids = []
    selected_scores = []
    industry_counts = {}

    for stock_id, score in zip(ordered_ids, ordered_scores):
        industry = str(stock_to_industry.get(stock_id, '') or '').strip()
        if max_per_industry > 0 and industry:
            current_count = industry_counts.get(industry, 0)
            if current_count >= max_per_industry:
                continue
            industry_counts[industry] = current_count + 1

        selected_ids.append(stock_id)
        selected_scores.append(float(score))
        if len(selected_ids) >= top_k:
            break

    if not selected_ids:
        raise ValueError('约束后没有可用持仓股票')

    return selected_ids, np.asarray(selected_scores, dtype=np.float64)
