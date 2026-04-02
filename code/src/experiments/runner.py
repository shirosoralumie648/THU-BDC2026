from __future__ import annotations

import numbers
from datetime import datetime

from experiments.diagnostics import build_fold_diagnostics
from experiments.diagnostics import build_regime_summary
from experiments.diagnostics import build_strategy_comparison
from experiments.metrics import choose_best_strategy
from experiments.metrics import format_strategy_metric_summary


def _normalize_date(value):
    if value in (None, ''):
        return ''
    if hasattr(value, 'strftime'):
        return value.strftime('%Y-%m-%d')
    return str(value)[:10]


def _json_safe(value):
    if isinstance(value, dict):
        return {
            key: _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if hasattr(value, 'strftime'):
        return _normalize_date(value)
    return value


def serialize_validation_folds(validation_folds, runtime_config):
    default_label_horizon = int(runtime_config.get('label_horizon', 5) or 5)
    rows = []
    for fold in validation_folds:
        rows.append({
            'name': fold.get('name', ''),
            'start_date': _normalize_date(fold.get('start_date', '')),
            'end_date': _normalize_date(fold.get('end_date', '')),
            'purge_days': int(fold.get('purge_days', 0) or 0),
            'embargo_days': int(fold.get('embargo_days', 0) or 0),
            'label_horizon': int(fold.get('label_horizon', default_label_horizon) or 0),
        })
    return rows


def build_strategy_export_payload(
    *,
    run_summary,
    validation_folds,
    runtime_config,
    source,
    exported_at=None,
    exported_at_field='generated_at',
    best_epoch=None,
):
    best_candidate = run_summary['best_candidate']
    payload = {
        'name': best_candidate['name'],
        'top_k': int(best_candidate['top_k']),
        'weighting': best_candidate['weighting'],
        'temperature': float(runtime_config.get('softmax_temperature', 1.0)),
        'validation_objective': float(run_summary['best_score']),
        'validation_return': float(run_summary.get('best_return', run_summary['best_score'])),
        'validation_mode': str(runtime_config.get('validation_mode', 'rolling')).lower(),
        'strategy_selection_mode': runtime_config.get('strategy_selection_mode', 'risk_adjusted'),
        'strategy_risk_lambda': float(runtime_config.get('strategy_risk_lambda', 0.2)),
        'rank_ic_mean': float(run_summary.get('metrics', {}).get('rank_ic_mean', 0.0)),
        'rank_ic_ir': float(run_summary.get('metrics', {}).get('rank_ic_ir', 0.0)),
        'validation_folds': serialize_validation_folds(validation_folds, runtime_config),
        'validation_metrics': _json_safe(run_summary.get('metrics', {})),
        'validation_strategy_summary': run_summary.get('strategy_summary', ''),
        'validation_strategy_comparison': _json_safe(run_summary.get('strategy_comparison', [])),
        'validation_fold_diagnostics': _json_safe(run_summary.get('fold_diagnostics', [])),
        'validation_regime_summary': _json_safe(run_summary.get('regime_summary', {})),
        'source': source,
    }
    if best_epoch is not None:
        payload['best_epoch'] = int(best_epoch)
    payload[exported_at_field] = exported_at or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return payload


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
        'regime_summary': build_regime_summary(
            fold_results,
            strategy_candidates,
            runtime_config,
        ),
        'fold_diagnostics': build_fold_diagnostics(
            fold_results,
            strategy_candidates,
            runtime_config,
        ),
    }
