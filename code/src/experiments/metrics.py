from __future__ import annotations

import numpy as np
import torch

from config import config
from objectives.target_transforms import rank_normalize_tensor


def _resolve_runtime_config(runtime_config=None):
    return runtime_config or config


def _rank_ic(valid_pred, valid_true_return):
    n = valid_true_return.numel()
    if n <= 2:
        return np.nan

    pred_rank = rank_normalize_tensor(valid_pred).detach()
    true_rank = rank_normalize_tensor(valid_true_return).detach()

    pred_centered = pred_rank - pred_rank.mean()
    true_centered = true_rank - true_rank.mean()
    denom = torch.sqrt((pred_centered ** 2).sum() * (true_centered ** 2).sum()) + 1e-12
    return float((pred_centered * true_centered).sum().item() / denom.item())


def build_strategy_candidates(runtime_config=None):
    runtime_config = _resolve_runtime_config(runtime_config)
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


def build_portfolio_weights(scores, top_k, weighting='equal', temperature=1.0):
    top_k = min(int(top_k), scores.numel())
    top_scores, top_indices = torch.topk(scores, top_k)

    if weighting == 'equal' or top_k == 1:
        weights = torch.full(
            (top_k,),
            1.0 / top_k,
            dtype=top_scores.dtype,
            device=top_scores.device,
        )
    elif weighting == 'softmax':
        temperature = max(float(temperature), 1e-6)
        weights = torch.softmax(top_scores / temperature, dim=0)
    else:
        raise ValueError(f'不支持的权重方式: {weighting}')

    return top_indices, weights


def calculate_ranking_metrics(y_pred, y_true, masks, strategy_candidates=None, temperature=1.0, runtime_config=None):
    """按候选持仓策略计算验证收益率，直接服务于最终总收益目标。"""
    runtime_config = _resolve_runtime_config(runtime_config)
    batch_size = y_pred.size(0)

    if strategy_candidates is None:
        strategy_candidates = [{'name': 'equal_top5', 'top_k': 5, 'weighting': 'equal'}]

    strategy_risk_lambda = float(runtime_config.get('strategy_risk_lambda', 0.2))
    metrics_lists = {f'return_{candidate["name"]}': [] for candidate in strategy_candidates}
    max_top_k = max(candidate['top_k'] for candidate in strategy_candidates)
    oracle_return_list = []
    rank_ic_list = []

    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()

        if valid_indices.numel() < max_top_k:
            continue

        valid_pred = y_pred[i][valid_indices]
        valid_true_return = y_true[i][valid_indices]

        for candidate in strategy_candidates:
            metric_name = f'return_{candidate["name"]}'
            pred_indices, weights = build_portfolio_weights(
                valid_pred,
                top_k=candidate['top_k'],
                weighting=candidate['weighting'],
                temperature=temperature,
            )
            pred_top_returns = valid_true_return[pred_indices]
            portfolio_return = torch.sum(pred_top_returns * weights).item()
            metrics_lists[metric_name].append(portfolio_return)

        _, true_indices = torch.topk(valid_true_return, 5)
        true_top_returns = valid_true_return[true_indices]
        oracle_return_list.append(true_top_returns.mean().item())
        rank_ic_list.append(_rank_ic(valid_pred, valid_true_return))

    metrics = {}
    for name, values in metrics_lists.items():
        if values:
            mean_ret = float(np.mean(values))
            std_ret = float(np.std(values))
        else:
            mean_ret = 0.0
            std_ret = 0.0
        metrics[name] = mean_ret
        metrics[f'{name}_std'] = std_ret
        metrics[f'{name}_risk_adjusted'] = mean_ret - strategy_risk_lambda * std_ret

    metrics['oracle_top5_equal'] = np.mean(oracle_return_list) if oracle_return_list else 0.0
    valid_rank_ics = [x for x in rank_ic_list if not np.isnan(x)]
    if valid_rank_ics:
        rank_ic_mean = float(np.mean(valid_rank_ics))
        rank_ic_std = float(np.std(valid_rank_ics))
        rank_ic_ir = rank_ic_mean / (rank_ic_std + 1e-12)
    else:
        rank_ic_mean = 0.0
        rank_ic_std = 0.0
        rank_ic_ir = 0.0
    metrics['rank_ic_mean'] = rank_ic_mean
    metrics['rank_ic_std'] = rank_ic_std
    metrics['rank_ic_ir'] = rank_ic_ir

    return metrics


def choose_best_strategy(eval_metrics, strategy_candidates, runtime_config=None):
    runtime_config = _resolve_runtime_config(runtime_config)
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
