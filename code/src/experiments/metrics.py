from __future__ import annotations


def build_strategy_candidates(runtime_config):
    top_k_candidates = sorted({int(k) for k in runtime_config.get('prediction_top_k_candidates', [5]) if 1 <= int(k) <= 5})
    weighting_candidates = list(dict.fromkeys(runtime_config.get('prediction_weighting_candidates', ['equal'])))

    candidates = []
    for top_k in top_k_candidates:
        for weighting in weighting_candidates:
            if weighting not in {'equal', 'softmax'}:
                continue
            if weighting == 'softmax' and top_k == 1:
                continue
            candidates.append({
                'name': f'{weighting}_top{top_k}',
                'top_k': top_k,
                'weighting': weighting,
            })

    if not candidates:
        candidates = [{'name': 'equal_top5', 'top_k': 5, 'weighting': 'equal'}]
    return candidates


def choose_best_strategy(eval_metrics, strategy_candidates, runtime_config):
    selection_metric = runtime_config.get('selection_metric', 'auto')
    selection_mode = str(runtime_config.get('strategy_selection_mode', 'risk_adjusted')).lower()

    if selection_metric != 'auto':
        if selection_metric not in eval_metrics:
            raise ValueError(f'selection_metric 不在评估指标中: {selection_metric}')
        metric_value = eval_metrics.get(selection_metric, float('-inf'))
        for candidate in strategy_candidates:
            base_metric = f'return_{candidate["name"]}'
            if selection_metric == base_metric or selection_metric.startswith(f'{base_metric}_'):
                return candidate, metric_value
        raise ValueError(f'未找到 selection_metric 对应的策略: {selection_metric}')

    best_candidate = None
    best_score = -float('inf')

    for candidate in strategy_candidates:
        if selection_mode == 'risk_adjusted':
            metric_name = f'return_{candidate["name"]}_risk_adjusted'
        else:
            metric_name = f'return_{candidate["name"]}'
        metric_value = eval_metrics.get(metric_name, -float('inf'))
        if metric_value > best_score:
            best_score = metric_value
            best_candidate = candidate

    if best_candidate is None:
        raise ValueError('验证指标为空，无法选择最优持仓策略')

    return best_candidate, best_score


def format_strategy_metric_summary(metrics, strategy_candidates):
    parts = []
    for candidate in strategy_candidates:
        metric_name = f'return_{candidate["name"]}'
        metric_std_name = f'{metric_name}_std'
        metric_ra_name = f'{metric_name}_risk_adjusted'
        if metric_name in metrics:
            mean_ret = metrics[metric_name]
            std_ret = metrics.get(metric_std_name, 0.0)
            ra_ret = metrics.get(metric_ra_name, mean_ret)
            parts.append(f'{candidate["name"]}=mean:{mean_ret:.4f}|std:{std_ret:.4f}|ra:{ra_ret:.4f}')
    return ', '.join(parts)
