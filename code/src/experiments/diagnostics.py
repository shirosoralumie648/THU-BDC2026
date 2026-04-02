from __future__ import annotations

import math

from experiments.metrics import choose_best_strategy


def build_strategy_comparison(metrics, strategy_candidates):
    rows = []
    for candidate in strategy_candidates:
        metric_name = f'return_{candidate["name"]}'
        rows.append({
            'name': candidate['name'],
            'top_k': int(candidate['top_k']),
            'weighting': candidate['weighting'],
            'mean_return': float(metrics.get(metric_name, 0.0)),
            'return_std': float(metrics.get(f'{metric_name}_std', 0.0)),
            'risk_adjusted_return': float(
                metrics.get(f'{metric_name}_risk_adjusted', metrics.get(metric_name, 0.0))
            ),
        })
    return rows


def _resolve_candidate_score(metrics, candidate, runtime_config):
    selection_mode = str(runtime_config.get('strategy_selection_mode', 'risk_adjusted')).lower()
    base_metric_name = f'return_{candidate["name"]}'
    if selection_mode == 'risk_adjusted':
        return float(metrics.get(f'{base_metric_name}_risk_adjusted', metrics.get(base_metric_name, 0.0)))
    return float(metrics.get(base_metric_name, 0.0))


def build_regime_summary(fold_results, strategy_candidates, runtime_config):
    if not fold_results:
        return {
            'fold_count': 0,
            'best_candidate_win_counts': {
                candidate['name']: 0
                for candidate in strategy_candidates
            },
            'positive_return_fold_count': 0,
            'negative_return_fold_count': 0,
            'flat_return_fold_count': 0,
            'average_best_score': 0.0,
            'average_best_return': 0.0,
            'average_score_spread': 0.0,
            'dominant_strategy': '',
        }

    win_counts = {
        candidate['name']: 0
        for candidate in strategy_candidates
    }
    best_fold_rows = []
    score_spreads = []
    positive_return_fold_count = 0
    negative_return_fold_count = 0
    flat_return_fold_count = 0

    for fold_result in fold_results:
        metrics = fold_result['metrics']
        best_candidate, best_score = choose_best_strategy(
            metrics,
            strategy_candidates,
            runtime_config,
        )
        best_return = float(metrics.get(f'return_{best_candidate["name"]}', best_score))
        win_counts[best_candidate['name']] = win_counts.get(best_candidate['name'], 0) + 1

        if best_return > 0:
            positive_return_fold_count += 1
        elif best_return < 0:
            negative_return_fold_count += 1
        else:
            flat_return_fold_count += 1

        candidate_scores = sorted(
            (
                _resolve_candidate_score(metrics, candidate, runtime_config)
                for candidate in strategy_candidates
            ),
            reverse=True,
        )
        if len(candidate_scores) >= 2:
            score_spreads.append(candidate_scores[0] - candidate_scores[1])

        best_fold_rows.append({
            'name': fold_result['name'],
            'start_date': fold_result['start_date'],
            'end_date': fold_result['end_date'],
            'best_candidate': best_candidate['name'],
            'best_score': float(best_score),
            'best_return': best_return,
        })

    average_best_score = sum(row['best_score'] for row in best_fold_rows) / len(best_fold_rows)
    average_best_return = sum(row['best_return'] for row in best_fold_rows) / len(best_fold_rows)
    average_score_spread = sum(score_spreads) / len(score_spreads) if score_spreads else 0.0
    dominant_strategy = max(
        strategy_candidates,
        key=lambda candidate: win_counts.get(candidate['name'], 0),
    )['name']
    best_fold = max(best_fold_rows, key=lambda row: row['best_score'])
    worst_fold = min(best_fold_rows, key=lambda row: row['best_score'])

    summary = {
        'fold_count': int(len(fold_results)),
        'selection_mode': str(runtime_config.get('strategy_selection_mode', 'risk_adjusted')),
        'best_candidate_win_counts': win_counts,
        'positive_return_fold_count': int(positive_return_fold_count),
        'negative_return_fold_count': int(negative_return_fold_count),
        'flat_return_fold_count': int(flat_return_fold_count),
        'average_best_score': float(average_best_score),
        'average_best_return': float(average_best_return),
        'average_score_spread': float(average_score_spread),
        'dominant_strategy': dominant_strategy,
        'best_fold': best_fold,
        'worst_fold': worst_fold,
    }
    if not math.isfinite(summary['average_score_spread']):
        summary['average_score_spread'] = 0.0
    return summary


def build_fold_diagnostics(fold_results, strategy_candidates, runtime_config):
    diagnostics = []
    for fold_result in fold_results:
        best_candidate, best_score = choose_best_strategy(
            fold_result['metrics'],
            strategy_candidates,
            runtime_config,
        )
        diagnostics.append({
            'name': fold_result['name'],
            'start_date': fold_result['start_date'],
            'end_date': fold_result['end_date'],
            'num_samples': int(fold_result['num_samples']),
            'loss': float(fold_result['loss']),
            'best_candidate': dict(best_candidate),
            'best_score': float(best_score),
            'strategy_comparison': build_strategy_comparison(
                fold_result['metrics'],
                strategy_candidates,
            ),
        })
    return diagnostics
