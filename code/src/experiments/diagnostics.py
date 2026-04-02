from __future__ import annotations

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
