from __future__ import annotations

import torch

from config import config

__all__ = [
    'mad_clip_bounds',
    'rank_normalize_tensor',
    'transform_targets_for_loss',
]


def _runtime_config(runtime_config=None):
    return runtime_config if runtime_config is not None else config


def rank_normalize_tensor(values):
    flat_values = values.reshape(-1)
    n = flat_values.numel()
    if n <= 1:
        return torch.zeros_like(values, dtype=torch.float32)

    order = torch.argsort(flat_values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(n, device=flat_values.device, dtype=torch.float32)
    normalized = (ranks / max(n - 1, 1)) * 2.0 - 1.0
    return normalized.view_as(values)


def mad_clip_bounds(values, mad_n=5.0, mad_min_scale=1e-6):
    flat_values = values.reshape(-1)
    median = torch.median(flat_values)
    mad = torch.median(torch.abs(flat_values - median))
    robust_sigma = torch.clamp(mad * 1.4826, min=float(mad_min_scale))
    lower = median - float(mad_n) * robust_sigma
    upper = median + float(mad_n) * robust_sigma
    return lower, upper


def transform_targets_for_loss(valid_pred, valid_target, runtime_config=None):
    """
    为排序损失准备训练目标：
    1) 对极端样本做 drop / clip；
    2) 按配置执行截面标准化。
    """
    runtime = _runtime_config(runtime_config)
    mode = str(runtime.get('label_extreme_mode', 'none')).lower()
    lower_q = float(runtime.get('label_extreme_lower_quantile', 0.05))
    upper_q = float(runtime.get('label_extreme_upper_quantile', 0.95))
    lower_q = max(0.0, min(lower_q, 0.49))
    upper_q = min(1.0, max(upper_q, 0.51))
    mad_n = float(runtime.get('label_mad_clip_n', 5.0))
    mad_min_scale = float(runtime.get('label_mad_min_scale', 1e-6))

    pred = valid_pred
    target = valid_target

    if mode in {'mad_drop', 'mad_drop_clip'} and target.numel() > 5:
        lower, upper = mad_clip_bounds(target, mad_n=mad_n, mad_min_scale=mad_min_scale)
        keep_mask = (target >= lower) & (target <= upper)
        if int(keep_mask.sum().item()) >= 2:
            pred = pred[keep_mask]
            target = target[keep_mask]

    if mode in {'mad_clip', 'mad_drop_clip'} and target.numel() > 2:
        lower, upper = mad_clip_bounds(target, mad_n=mad_n, mad_min_scale=mad_min_scale)
        target = torch.clamp(target, min=lower, max=upper)

    if mode in {'drop', 'drop_clip'} and target.numel() > 5:
        lower = torch.quantile(target, lower_q)
        upper = torch.quantile(target, upper_q)
        keep_mask = (target >= lower) & (target <= upper)
        if int(keep_mask.sum().item()) >= 2:
            pred = pred[keep_mask]
            target = target[keep_mask]

    if mode in {'clip', 'drop_clip'} and target.numel() > 2:
        lower = torch.quantile(target, lower_q)
        upper = torch.quantile(target, upper_q)
        target = torch.clamp(target, min=lower, max=upper)

    if runtime.get('use_cross_sectional_label_norm', True):
        label_norm_method = str(runtime.get('label_cs_norm_method', 'zscore')).lower()
        if label_norm_method == 'zscore':
            mean = target.mean()
            std = target.std(unbiased=False)
            target = (target - mean) / (std + 1e-6)
        elif label_norm_method == 'rank':
            target = rank_normalize_tensor(target)
        else:
            raise ValueError(f'不支持的 label_cs_norm_method: {label_norm_method}')

        clip_value = runtime.get('label_cs_clip_value', None)
        if clip_value is not None:
            clip_value = float(clip_value)
            target = torch.clamp(target, min=-clip_value, max=clip_value)

    return pred, target
