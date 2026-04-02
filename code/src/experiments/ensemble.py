from __future__ import annotations

import numbers

from experiments.diagnostics import build_strategy_comparison
from experiments.metrics import choose_best_strategy
from experiments.metrics import format_strategy_metric_summary


def summarize_multi_seed_runs(*, run_summaries, strategy_candidates, runtime_config):
    if not run_summaries:
        raise ValueError('run_summaries 不能为空')

    metric_buckets = {}
    for run in run_summaries:
        for key, value in run.get('metrics', {}).items():
            if isinstance(value, numbers.Real):
                metric_buckets.setdefault(key, []).append(float(value))

    aggregate_metrics = {
        key: sum(values) / len(values)
        for key, values in metric_buckets.items()
        if values
    }
    best_candidate, best_score = choose_best_strategy(
        aggregate_metrics,
        strategy_candidates,
        runtime_config,
    )

    best_run = None
    best_run_score = -float('inf')
    for run in run_summaries:
        run_candidate, run_score = choose_best_strategy(
            run.get('metrics', {}),
            strategy_candidates,
            runtime_config,
        )
        annotated = dict(run)
        annotated['best_candidate'] = dict(run_candidate)
        annotated['best_score'] = float(run_score)
        if run_score > best_run_score:
            best_run = annotated
            best_run_score = float(run_score)

    return {
        'num_runs': int(len(run_summaries)),
        'aggregate_metrics': aggregate_metrics,
        'best_candidate': dict(best_candidate),
        'best_score': float(best_score),
        'best_run': best_run,
        'strategy_summary': format_strategy_metric_summary(aggregate_metrics, strategy_candidates),
        'strategy_comparison': build_strategy_comparison(aggregate_metrics, strategy_candidates),
    }
