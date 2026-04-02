from __future__ import annotations

from experiments.diagnostics import build_fold_diagnostics
from experiments.diagnostics import build_strategy_comparison
from experiments.metrics import choose_best_strategy
from experiments.metrics import format_strategy_metric_summary


def summarize_experiment_run(
    *,
    eval_loss,
    eval_metrics,
    fold_results,
    strategy_candidates,
    runtime_config,
):
    best_candidate, best_score = choose_best_strategy(
        eval_metrics,
        strategy_candidates,
        runtime_config,
    )
    best_return = float(eval_metrics.get(f'return_{best_candidate["name"]}', best_score))
    return {
        'loss': float(eval_loss),
        'metrics': {key: float(value) for key, value in eval_metrics.items()},
        'best_candidate': dict(best_candidate),
        'best_score': float(best_score),
        'best_return': best_return,
        'strategy_summary': format_strategy_metric_summary(eval_metrics, strategy_candidates),
        'strategy_comparison': build_strategy_comparison(eval_metrics, strategy_candidates),
        'fold_diagnostics': build_fold_diagnostics(
            fold_results,
            strategy_candidates,
            runtime_config,
        ),
    }
