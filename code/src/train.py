import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import lru_cache
from functools import partial
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from factor_store import resolve_factor_pipeline
from factor_store import save_factor_snapshot
from data_manager import collect_data_sources
from data_manager import load_market_dataset
from data_manager import load_market_dataset_from_path
from data_manager import load_train_dataset_from_build_manifest
from data_manager import load_stock_to_industry_map
from data_manager import resolve_industry_mapping_path
from data_manager import save_data_manifest
from experiments.metrics import build_strategy_candidates as build_strategy_candidates_shared
from experiments.metrics import choose_best_strategy as choose_best_strategy_shared
from experiments.metrics import format_strategy_metric_summary as format_strategy_metric_summary_shared
from experiments.runner import build_strategy_export_payload
from experiments.runner import summarize_experiment_run
from experiments.splits import build_rolling_validation_folds as build_rolling_validation_folds_shared
from features.feature_assembler import augment_feature_table
from features.feature_assembler import build_feature_table
from graph.correlation_graph import build_correlation_prior_adjacency
from graph.graph_builder import build_prior_graph_adjacency as build_prior_graph_adjacency_shared
from graph.industry_graph import build_industry_prior_adjacency
from graph.industry_graph import build_stock_industry_index as build_stock_industry_index_shared
from models.rank_model import StockTransformer
from objectives.ranking_loss import PortfolioOptimizationLoss
from objectives.target_transforms import rank_normalize_tensor as _tensor_rank_normalize
from objectives.target_transforms import transform_targets_for_loss
from utils import create_ranking_dataset_vectorized
from utils import resolve_feature_indices
import joblib
import os
import json
import multiprocessing as mp
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def _infer_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_stock_code_series(series):
    s = series.astype(str).str.strip()
    s = s.str.split('.').str[-1]
    s = s.str.replace(r'[^0-9]', '', regex=True)
    s = s.str[-6:].str.zfill(6)
    return s


def _neutralize_label_by_cross_section_mean(processed, label_col='label', date_col='日期'):
    daily_mean = processed.groupby(date_col)[label_col].transform('mean')
    out = processed.copy()
    out[label_col] = out[label_col] - daily_mean
    return out


@lru_cache(maxsize=1)
def _load_benchmark_return_series():
    bench_path = str(config.get('label_benchmark_return_path', '')).strip()
    if not bench_path:
        return None
    if not os.path.exists(bench_path):
        print(f'未找到 benchmark 文件，跳过指数中性化: {bench_path}')
        return None

    bench_df = pd.read_csv(bench_path)
    date_col = _infer_existing_column(bench_df, ['日期', 'date', 'datetime', 'trade_date'])
    if date_col is None:
        print(f'benchmark 文件缺少日期列，跳过指数中性化: {bench_path}')
        return None

    bench_df[date_col] = pd.to_datetime(bench_df[date_col], errors='coerce').dt.normalize()
    bench_df = bench_df.dropna(subset=[date_col]).copy()

    return_col = str(config.get('label_benchmark_return_col', '')).strip()
    if return_col and return_col not in bench_df.columns:
        print(f'benchmark_return_col 不存在: {return_col}，将尝试自动识别')
        return_col = ''
    if not return_col:
        return_col = _infer_existing_column(
            bench_df,
            ['benchmark_return', 'return', '收益率', '涨跌幅', 'pct_chg', 'pctChg']
        )

    if return_col is not None:
        bench_ret = pd.to_numeric(bench_df[return_col], errors='coerce')
        if bench_ret.abs().median(skipna=True) > 1.0:
            # 若输入为百分数点（如 1.23 表示 1.23%），统一缩放到小数收益率
            bench_ret = bench_ret / 100.0
    else:
        open_col = _infer_existing_column(bench_df, ['开盘', 'open'])
        if open_col is None:
            print(f'benchmark 文件缺少收益率列和开盘价列，跳过指数中性化: {bench_path}')
            return None
        bench_df = bench_df.sort_values(date_col).reset_index(drop=True)
        open_px = pd.to_numeric(bench_df[open_col], errors='coerce')
        open_t1 = open_px.shift(-1)
        open_t5 = open_px.shift(-5)
        bench_ret = (open_t5 - open_t1) / (open_t1 + 1e-12)

    bench_series = pd.Series(bench_ret.values, index=bench_df[date_col]).dropna()
    bench_series = bench_series[~bench_series.index.duplicated(keep='last')]
    if bench_series.empty:
        return None
    return bench_series.astype(np.float32)


@lru_cache(maxsize=1)
def _load_stock_industry_mapping():
    mapping_path = str(config.get('label_industry_map_path', '')).strip()
    if not mapping_path or not os.path.exists(mapping_path):
        if mapping_path:
            print(f'未找到行业映射文件，跳过行业中性化: {mapping_path}')
        return None
    mapping = load_stock_to_industry_map(
        config,
        stock_col_key='label_industry_stock_col',
        industry_col_key='label_industry_col',
        mapping_path=mapping_path,
    )
    return mapping if mapping else None


def _resolve_prior_graph_industry_path():
    return resolve_industry_mapping_path(config)


@lru_cache(maxsize=1)
def _load_prior_graph_industry_mapping():
    mapping_path = _resolve_prior_graph_industry_path()
    if not mapping_path:
        return {}
    mapping = load_stock_to_industry_map(
        config,
        stock_col_key='prior_graph_stock_col',
        industry_col_key='prior_graph_industry_col',
        mapping_path=mapping_path,
    )
    return mapping if mapping else {}


def _build_industry_prior_adjacency(stockid2idx, num_stocks):
    return build_industry_prior_adjacency(stockid2idx, config)


def _build_correlation_prior_adjacency(train_data, num_stocks):
    return build_correlation_prior_adjacency(train_data, num_stocks, config)


def build_prior_graph_adjacency(train_data, stockid2idx):
    return build_prior_graph_adjacency_shared(train_data, stockid2idx, config)


def build_stock_industry_index(stockid2idx):
    return build_stock_industry_index_shared(stockid2idx, config)


def _neutralize_label_by_benchmark(processed, label_col='label', date_col='日期'):
    bench_series = _load_benchmark_return_series()
    if bench_series is None:
        return _neutralize_label_by_cross_section_mean(processed, label_col=label_col, date_col=date_col)

    out = processed.copy()
    norm_dates = pd.to_datetime(out[date_col], errors='coerce').dt.normalize()
    bench_ret = norm_dates.map(bench_series)
    if bench_ret.isna().any():
        # 对缺失日期回退截面均值中性化，避免样本丢失
        fallback_mean = out.groupby(date_col)[label_col].transform('mean')
        bench_ret = bench_ret.fillna(fallback_mean)
    out[label_col] = out[label_col] - bench_ret.to_numpy(dtype=np.float32)
    return out


def _neutralize_label_by_industry(processed, label_col='label', date_col='日期'):
    mapping = _load_stock_industry_mapping()
    if not mapping:
        return _neutralize_label_by_cross_section_mean(processed, label_col=label_col, date_col=date_col)
    if '股票代码' not in processed.columns:
        return _neutralize_label_by_cross_section_mean(processed, label_col=label_col, date_col=date_col)

    out = processed.copy()
    out['_stock_norm'] = _normalize_stock_code_series(out['股票代码'])
    out['_industry'] = out['_stock_norm'].map(mapping)
    industry_mean = out.groupby([date_col, '_industry'])[label_col].transform('mean')
    daily_mean = out.groupby(date_col)[label_col].transform('mean')
    neutral_base = industry_mean.fillna(daily_mean)
    out[label_col] = out[label_col] - neutral_base
    out = out.drop(columns=['_stock_norm', '_industry'])
    return out


def _apply_label_market_neutralization(processed, label_col='label', date_col='日期'):
    """
    标签市场中性化：将绝对收益率转为超额收益率(alpha)。
    当前默认实现为“按日减去全市场均值收益”。
    """
    if label_col not in processed.columns:
        return processed
    if date_col not in processed.columns:
        raise ValueError(f'缺少日期列，无法做标签中性化: {date_col}')

    if not bool(config.get('use_label_market_neutralization', True)):
        return processed

    method = str(config.get('label_market_neutralization', 'cross_sectional_mean')).lower()
    out = processed.copy()
    if method == 'none':
        return out
    if method in {'cross_sectional_mean', 'cross_mean'}:
        return _neutralize_label_by_cross_section_mean(out, label_col=label_col, date_col=date_col)
    if method in {'benchmark', 'index', 'hs300'}:
        return _neutralize_label_by_benchmark(out, label_col=label_col, date_col=date_col)
    if method in {'industry', 'industry_mean', 'sw_l1'}:
        return _neutralize_label_by_industry(out, label_col=label_col, date_col=date_col)
    if method in {'benchmark_then_industry', 'index_then_industry'}:
        out = _neutralize_label_by_benchmark(out, label_col=label_col, date_col=date_col)
        out = _neutralize_label_by_industry(out, label_col=label_col, date_col=date_col)
        return out
    if method in {'industry_then_benchmark'}:
        out = _neutralize_label_by_industry(out, label_col=label_col, date_col=date_col)
        out = _neutralize_label_by_benchmark(out, label_col=label_col, date_col=date_col)
        return out

    raise ValueError(f'不支持的 label_market_neutralization: {method}')


def _apply_label_mad_clipping(processed, label_col='label', date_col='日期'):
    """
    按日 MAD 去极值，抑制异常收益样本对梯度的破坏。
    clip 区间: median ± n * 1.4826 * MAD
    """
    if label_col not in processed.columns or date_col not in processed.columns:
        return processed
    if not bool(config.get('use_label_mad_clip', True)):
        return processed

    mad_n = float(config.get('label_mad_clip_n', 5.0))
    mad_min_scale = float(config.get('label_mad_min_scale', 1e-6))
    min_group_size = int(config.get('label_mad_min_group_size', 5))
    min_group_size = max(1, min_group_size)

    out = processed.copy()
    group_size = out.groupby(date_col)[label_col].transform('size')
    apply_mask = group_size >= min_group_size
    if not bool(apply_mask.any()):
        return out

    median = out.groupby(date_col)[label_col].transform('median')
    abs_dev = (out[label_col] - median).abs()
    mad = abs_dev.groupby(out[date_col]).transform('median')
    robust_sigma = (mad * 1.4826).clip(lower=mad_min_scale)
    lower = median - mad_n * robust_sigma
    upper = median + mad_n * robust_sigma
    clipped = out[label_col].clip(lower=lower, upper=upper)
    out.loc[apply_mask, label_col] = clipped.loc[apply_mask]
    return out


def _build_future_volatility_label(
    processed,
    stock_col='股票代码',
    open_col='开盘',
    horizon=5,
):
    """
    未来实现波动率标签：
    使用 future open-to-open 收益率序列的标准差（ddof=0）。
    """
    out = processed.copy()
    horizon = max(3, int(horizon))

    grouped_open = out.groupby(stock_col)[open_col]
    open_cols = []
    for step in range(1, horizon + 1):
        col = f'open_t{step}'
        out[col] = grouped_open.shift(-step)
        open_cols.append(col)

    future_ret_cols = []
    for step in range(1, horizon):
        prev_col = f'open_t{step}'
        next_col = f'open_t{step + 1}'
        ret_col = f'future_ret_{step}_{step + 1}'
        out[ret_col] = (out[next_col] - out[prev_col]) / (out[prev_col] + 1e-12)
        future_ret_cols.append(ret_col)

    out['vol_label'] = out[future_ret_cols].std(axis=1, ddof=0)
    out = out.drop(columns=future_ret_cols)
    return out, open_cols


def _apply_volatility_label_processing(processed, label_col='vol_label', date_col='日期'):
    if label_col not in processed.columns or date_col not in processed.columns:
        return processed

    out = processed.copy()
    out[label_col] = pd.to_numeric(out[label_col], errors='coerce').clip(lower=0.0)

    if bool(config.get('use_volatility_label_log1p', True)):
        out[label_col] = np.log1p(out[label_col])

    if bool(config.get('use_volatility_label_mad_clip', True)):
        mad_n = float(config.get('volatility_label_mad_clip_n', 5.0))
        mad_min_scale = float(config.get('volatility_label_mad_min_scale', 1e-6))

        median = out.groupby(date_col)[label_col].transform('median')
        abs_dev = (out[label_col] - median).abs()
        mad = abs_dev.groupby(out[date_col]).transform('median')
        robust_sigma = (mad * 1.4826).clip(lower=mad_min_scale)
        lower = median - mad_n * robust_sigma
        upper = median + mad_n * robust_sigma
        out[label_col] = out[label_col].clip(lower=lower, upper=upper)

    if bool(config.get('use_volatility_cs_norm', True)):
        method = str(config.get('volatility_cs_norm_method', 'zscore')).lower()
        if method == 'zscore':
            means = out.groupby(date_col)[label_col].transform('mean')
            stds = out.groupby(date_col)[label_col].transform('std').replace(0.0, np.nan)
            out[label_col] = (out[label_col] - means) / (stds + 1e-6)
        elif method == 'rank':
            out[label_col] = out.groupby(date_col)[label_col].rank(pct=True) * 2.0 - 1.0
        else:
            raise ValueError(f'不支持的 volatility_cs_norm_method: {method}')

        clip_value = config.get('volatility_cs_clip_value', None)
        if clip_value is not None:
            clip_value = float(clip_value)
            out[label_col] = out[label_col].clip(lower=-clip_value, upper=clip_value)

    out[label_col] = out[label_col].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return out


def _build_label_and_clean(processed, drop_small_open=True):
    """统一构建标签并清洗无效样本。"""
    processed = processed.copy()
    return_horizon = 5
    vol_horizon = max(return_horizon, int(config.get('volatility_horizon', 5)))
    processed, open_cols = _build_future_volatility_label(
        processed,
        stock_col='股票代码',
        open_col='开盘',
        horizon=vol_horizon,
    )

    # 过滤无效开盘价，避免收益率极端爆炸
    if drop_small_open:
        processed = processed.loc[processed['open_t1'] > 1e-4].copy()

    processed.loc[:, 'label'] = (
        (processed[f'open_t{return_horizon}'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    )
    processed = processed.dropna(subset=['label', 'vol_label']).copy()
    processed.loc[:, 'label_raw'] = processed['label'].astype(np.float32)
    processed.loc[:, 'vol_label_raw'] = processed['vol_label'].astype(np.float32)

    # 标签处理关键修正：
    # 1) 市场中性化 -> 预测超额收益率(alpha)；
    # 2) MAD 去极值 -> 抑制异常波动样本对训练的干扰。
    processed = _apply_label_market_neutralization(processed, label_col='label', date_col='日期')
    processed = _apply_label_mad_clipping(processed, label_col='label', date_col='日期')
    processed = _apply_volatility_label_processing(processed, label_col='vol_label', date_col='日期')
    processed.loc[:, 'label'] = processed['label'].astype(np.float32)
    processed = processed.dropna(subset=['label', 'vol_label']).copy()
    processed = processed.drop(columns=[col for col in open_cols if col in processed.columns])
    return processed


def _preprocess_common(df, stockid2idx, desc, feature_pipeline, drop_small_open=True):
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_columns = list(feature_pipeline['active_features'])

    # 保证时序正确，避免 shift 标签错位
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(int(config.get('feature_engineer_processes', 4)), mp.cpu_count())
    feature_builder = partial(
        build_feature_table,
        feature_set=feature_pipeline['feature_set'],
        runtime_config=config,
        feature_pipeline=feature_pipeline,
    )
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_builder, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed, feature_columns = augment_feature_table(
        processed,
        feature_columns,
        runtime_config=config,
        feature_pipeline=feature_pipeline,
        date_col='日期',
        stock_col='股票代码',
        apply_feature_enhancements=False,
        apply_cross_sectional_norm=False,
    )

    # 映射股票索引，并剔除映射失败样本
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    processed, feature_columns = augment_feature_table(
        processed,
        feature_columns,
        runtime_config=config,
        feature_pipeline=None,
        date_col='日期',
        stock_col='股票代码',
        apply_factor_pipeline=False,
        apply_feature_enhancements=True,
        apply_cross_sectional_norm=True,
    )
    return processed, feature_columns


# 数据预处理函数
def preprocess_data(df, feature_pipeline, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", feature_pipeline=feature_pipeline, drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", feature_pipeline=feature_pipeline, drop_small_open=True)


def preprocess_val_data(df, feature_pipeline, stockid2idx=None):
    # 验证集与训练集保持同口径，避免 label 分布漂移
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", feature_pipeline=feature_pipeline, drop_small_open=True)


# 排序损失与目标变换已收敛到 objectives/*，训练脚本只保留编排逻辑。
def _tensor_rank_normalize(values):
    n = values.numel()
    if n <= 1:
        return torch.zeros_like(values)
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(n, device=values.device, dtype=torch.float32)
    return (ranks / max(n - 1, 1)) * 2.0 - 1.0


def _tensor_mad_bounds(values, mad_n=5.0, mad_min_scale=1e-6):
    median = torch.median(values)
    mad = torch.median(torch.abs(values - median))
    robust_sigma = torch.clamp(mad * 1.4826, min=float(mad_min_scale))
    lower = median - float(mad_n) * robust_sigma
    upper = median + float(mad_n) * robust_sigma
    return lower, upper


def transform_targets_for_loss(valid_pred, valid_target):
    """
    为损失函数准备训练目标：
    1) 极值样本 drop / clip；
    2) 截面标准化（zscore/rank）。
    """
    mode = str(config.get('label_extreme_mode', 'none')).lower()
    lower_q = float(config.get('label_extreme_lower_quantile', 0.05))
    upper_q = float(config.get('label_extreme_upper_quantile', 0.95))
    lower_q = max(0.0, min(lower_q, 0.49))
    upper_q = min(1.0, max(upper_q, 0.51))
    mad_n = float(config.get('label_mad_clip_n', 5.0))
    mad_min_scale = float(config.get('label_mad_min_scale', 1e-6))

    pred = valid_pred
    target = valid_target

    if mode in {'mad_drop', 'mad_drop_clip'} and target.numel() > 5:
        lower, upper = _tensor_mad_bounds(target, mad_n=mad_n, mad_min_scale=mad_min_scale)
        keep_mask = (target >= lower) & (target <= upper)
        if int(keep_mask.sum().item()) >= 2:
            pred = pred[keep_mask]
            target = target[keep_mask]

    if mode in {'mad_clip', 'mad_drop_clip'} and target.numel() > 2:
        lower, upper = _tensor_mad_bounds(target, mad_n=mad_n, mad_min_scale=mad_min_scale)
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

    if config.get('use_cross_sectional_label_norm', True):
        label_norm_method = str(config.get('label_cs_norm_method', 'zscore')).lower()
        if label_norm_method == 'zscore':
            mean = target.mean()
            std = target.std(unbiased=False)
            target = (target - mean) / (std + 1e-6)
        elif label_norm_method == 'rank':
            target = _tensor_rank_normalize(target)
        else:
            raise ValueError(f'不支持的 label_cs_norm_method: {label_norm_method}')

        clip_value = config.get('label_cs_clip_value', None)
        if clip_value is not None:
            clip_value = float(clip_value)
            target = torch.clamp(target, min=-clip_value, max=clip_value)

    return pred, target


def _rank_ic(valid_pred, valid_true_return):
    n = valid_true_return.numel()
    if n <= 2:
        return np.nan

    pred_rank = _tensor_rank_normalize(valid_pred).detach()
    true_rank = _tensor_rank_normalize(valid_true_return).detach()

    pred_centered = pred_rank - pred_rank.mean()
    true_centered = true_rank - true_rank.mean()
    denom = torch.sqrt((pred_centered ** 2).sum() * (true_centered ** 2).sum()) + 1e-12
    return float((pred_centered * true_centered).sum().item() / denom.item())


def build_strategy_candidates():
    return build_strategy_candidates_shared(config)


def build_portfolio_weights(scores, top_k, weighting='equal', temperature=1.0):
    top_k = min(int(top_k), scores.numel())
    top_scores, top_indices = torch.topk(scores, top_k)

    if weighting == 'equal' or top_k == 1:
        weights = torch.full(
            (top_k,),
            1.0 / top_k,
            dtype=top_scores.dtype,
            device=top_scores.device
        )
    elif weighting == 'softmax':
        temperature = max(float(temperature), 1e-6)
        weights = torch.softmax(top_scores / temperature, dim=0)
    else:
        raise ValueError(f'不支持的权重方式: {weighting}')

    return top_indices, weights


def calculate_ranking_metrics(y_pred, y_true, masks, strategy_candidates=None, temperature=1.0):
    """按候选持仓策略计算验证收益率，直接服务于最终总收益目标。"""
    batch_size = y_pred.size(0)

    if strategy_candidates is None:
        strategy_candidates = [{'name': 'equal_top5', 'top_k': 5, 'weighting': 'equal'}]

    strategy_risk_lambda = float(config.get('strategy_risk_lambda', 0.2))
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


def choose_best_strategy(eval_metrics, strategy_candidates):
    return choose_best_strategy_shared(eval_metrics, strategy_candidates, config)


def format_strategy_metric_summary(metrics, strategy_candidates):
    return format_strategy_metric_summary_shared(metrics, strategy_candidates)


def format_factor_summary(feature_pipeline):
    summary = feature_pipeline['summary']
    group_parts = [
        f'{group}={count}'
        for group, count in sorted(summary['group_counts'].items())
    ]
    return (
        f"feature_set={feature_pipeline['feature_set']}, "
        f"active={summary['active_total']}, "
        f"cross_sectional={summary.get('cross_sectional_total', 0)}, "
        f"builtin={summary['builtin_enabled']}/{summary['builtin_total']}, "
        f"builtin_overridden={summary.get('builtin_overridden', 0)}, "
        f"custom={summary['custom_enabled']}/{summary['custom_total']}, "
        f"groups=({', '.join(group_parts)})"
    )


def print_active_factors(feature_pipeline):
    grouped_specs = {}
    for spec in feature_pipeline['active_specs']:
        group = spec.get('group', 'unknown')
        label = spec['name']
        if spec.get('source') == 'custom':
            label = f'{label} [custom]'
        elif spec.get('overridden'):
            label = f'{label} [override]'
        grouped_specs.setdefault(group, []).append(label)

    print("当前启用因子明细:")
    for group, factor_names in sorted(grouped_specs.items()):
        print(f"  - {group} ({len(factor_names)}):")
        print("    " + ", ".join(factor_names))


def _build_factor_markdown(feature_pipeline):
    summary = feature_pipeline['summary']
    lines = [
        f"- feature_set: `{feature_pipeline['feature_set']}`",
        f"- factor_store: `{feature_pipeline['store_path']}`",
        f"- builtin_registry: `{feature_pipeline.get('builtin_registry_path', '')}`",
        f"- factor_fingerprint: `{feature_pipeline.get('factor_fingerprint', '')}`",
        f"- snapshot_created_at: `{feature_pipeline.get('snapshot_meta', {}).get('created_at', '')}`",
        f"- active_total: `{summary['active_total']}`",
        f"- builtin_enabled: `{summary['builtin_enabled']}/{summary['builtin_total']}`",
        f"- builtin_overridden: `{summary.get('builtin_overridden', 0)}`",
        f"- custom_enabled: `{summary['custom_enabled']}/{summary['custom_total']}`",
        f"- cross_sectional_total: `{summary.get('cross_sectional_total', 0)}`",
        f"- groups: `{json.dumps(summary['group_counts'], ensure_ascii=False)}`",
        "",
        "Active factors:",
        ", ".join(feature_pipeline['active_features']),
    ]
    if feature_pipeline['custom_specs']:
        lines.extend([
            "",
            "Custom factors:",
            json.dumps(feature_pipeline['custom_specs'], ensure_ascii=False, indent=2),
        ])
    return "\n".join(lines)


def log_factor_dashboard(writer, feature_pipeline, raw_hist_frame, scaled_hist_frame):
    if writer is None:
        return

    summary = feature_pipeline['summary']
    writer.add_text('factors/overview', _build_factor_markdown(feature_pipeline), global_step=0)
    writer.add_scalar('factors/active_total', summary['active_total'], global_step=0)
    writer.add_scalar('factors/builtin_enabled', summary['builtin_enabled'], global_step=0)
    writer.add_scalar('factors/builtin_overridden', summary.get('builtin_overridden', 0), global_step=0)
    writer.add_scalar('factors/custom_enabled', summary['custom_enabled'], global_step=0)

    for group, count in sorted(summary['group_counts'].items()):
        writer.add_scalar(f'factors/group_count/{group}', count, global_step=0)

    max_histograms = max(0, int(config.get('factor_histogram_max_features', 0)))
    if raw_hist_frame is None or scaled_hist_frame is None:
        return

    for feature_name in raw_hist_frame.columns[:max_histograms]:
        raw_values = raw_hist_frame[feature_name].to_numpy(dtype=np.float32, copy=True)
        scaled_values = scaled_hist_frame[feature_name].to_numpy(dtype=np.float32, copy=True)
        writer.add_histogram(f'factors/raw/{feature_name}', raw_values, global_step=0)
        writer.add_histogram(f'factors/scaled/{feature_name}', scaled_values, global_step=0)


def _feature_stats_frame(df, feature_columns):
    if not feature_columns:
        return pd.DataFrame(columns=['feature', 'mean', 'std', 'min', 'max', 'na_ratio'])

    feature_df = df[feature_columns]
    stats = pd.DataFrame({
        'feature': feature_columns,
        'mean': feature_df.mean(axis=0, skipna=True).values,
        'std': feature_df.std(axis=0, skipna=True).values,
        'min': feature_df.min(axis=0, skipna=True).values,
        'max': feature_df.max(axis=0, skipna=True).values,
        'na_ratio': feature_df.isna().mean(axis=0).values,
    })
    return stats


def dump_factor_artifacts(split_name, df, feature_columns, output_dir):
    if not bool(config.get('dump_factor_artifacts', True)):
        return
    if df is None or len(df) == 0:
        return

    artifact_dir = os.path.join(output_dir, 'factor_artifacts')
    os.makedirs(artifact_dir, exist_ok=True)
    max_rows = int(config.get('factor_artifact_max_rows', 100000))
    max_rows = max(0, max_rows)

    base_cols = [
        col for col in ['日期', '股票代码', 'instrument', 'label', 'label_raw', 'vol_label', 'vol_label_raw']
        if col in df.columns
    ]
    export_cols = base_cols + [col for col in feature_columns if col in df.columns]

    export_df = df[export_cols].copy()
    if max_rows > 0 and len(export_df) > max_rows:
        export_df = (
            export_df.sample(n=max_rows, random_state=42)
            .sort_values([col for col in ['日期', '股票代码'] if col in export_df.columns])
            .reset_index(drop=True)
        )

    values_path = os.path.join(artifact_dir, f'{split_name}_factor_values.csv')
    export_df.to_csv(values_path, index=False, encoding='utf-8')

    stats_path = os.path.join(artifact_dir, f'{split_name}_factor_stats.csv')
    if bool(config.get('factor_artifact_include_full_feature_stats', True)):
        stats_df = _feature_stats_frame(df, [col for col in feature_columns if col in df.columns])
        stats_df.to_csv(stats_path, index=False, encoding='utf-8')

    meta = {
        'split': split_name,
        'rows_total': int(len(df)),
        'rows_exported': int(len(export_df)),
        'feature_count': int(len(feature_columns)),
        'feature_count_present': int(sum(1 for col in feature_columns if col in df.columns)),
        'values_path': values_path,
        'stats_path': stats_path if bool(config.get('factor_artifact_include_full_feature_stats', True)) else '',
    }
    meta_path = os.path.join(artifact_dir, f'{split_name}_factor_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(
        f'已导出 {split_name} 因子结果: values={values_path}, '
        f'rows={meta["rows_exported"]}/{meta["rows_total"]}, features={meta["feature_count_present"]}'
    )

class RankingDataset(torch.utils.data.Dataset):
    """排序数据集类"""
    def __init__(self, sequences, targets, relevance_scores, stock_indices, vol_targets=None):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices
        self.vol_targets = vol_targets if vol_targets is not None else targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': torch.FloatTensor(np.array(self.sequences[idx])),  # [num_stocks, seq_len, features]
            'targets': torch.FloatTensor(np.array(self.targets[idx])),      # [num_stocks] 真实涨跌幅
            'relevance': torch.LongTensor(np.array(self.relevance_scores[idx])),  # [num_stocks] 排序标签
            'stock_indices': torch.LongTensor(np.array(self.stock_indices[idx])),  # [num_stocks] 股票索引
            'vol_targets': torch.FloatTensor(np.array(self.vol_targets[idx])),      # [num_stocks] 波动率标签
        }


class LazyRankingDataset(torch.utils.data.Dataset):
    """懒加载排序数据集，避免一次性将全部窗口序列展开到内存。"""
    def __init__(self, stock_cache, day_entries, sequence_length):
        self.stock_cache = stock_cache
        self.day_entries = day_entries
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.day_entries)
    
    def __getitem__(self, idx):
        entry = self.day_entries[idx]
        day_sequences = []
        day_targets = []
        day_vol_targets = []
        day_stock_indices = []

        for stock_idx, end_idx in entry['entries']:
            stock_data = self.stock_cache[stock_idx]
            if end_idx >= len(stock_data['labels']):
                raise IndexError(
                    f"懒加载索引越界: date={entry['date']}, stock_idx={stock_idx}, "
                    f"end_idx={end_idx}, labels_len={len(stock_data['labels'])}"
                )
            seq = stock_data['features'][end_idx - self.sequence_length + 1:end_idx + 1]
            target = stock_data['labels'][end_idx]
            vol_target = stock_data['vol_labels'][end_idx]
            day_sequences.append(seq)
            day_targets.append(target)
            day_vol_targets.append(vol_target)
            day_stock_indices.append(stock_idx)

        day_targets = np.asarray(day_targets, dtype=np.float32)
        day_vol_targets = np.asarray(day_vol_targets, dtype=np.float32)
        threshold_2pct = np.quantile(day_targets, 0.98)
        relevance = (day_targets >= threshold_2pct).astype(np.float32)

        return {
            'sequences': torch.FloatTensor(np.asarray(day_sequences, dtype=np.float32)),
            'targets': torch.FloatTensor(day_targets),
            'vol_targets': torch.FloatTensor(day_vol_targets),
            'relevance': torch.LongTensor(relevance.astype(np.int64)),
            'stock_indices': torch.LongTensor(np.asarray(day_stock_indices, dtype=np.int64)),
        }

def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    vol_targets = [item.get('vol_targets', item['targets']) for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]
    
    # 找到最大股票数量
    max_stocks = max(seq.size(0) for seq in sequences)
    
    # Padding到相同长度
    padded_sequences = []
    padded_targets = []
    padded_vol_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []
    
    for seq, tgt, vol_tgt, rel, stock_idx in zip(sequences, targets, vol_targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)
        
        # 创建padding
        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            vol_tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size, dtype=torch.long)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)
            
            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            vol_tgt = torch.cat([vol_tgt, vol_tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)
        
        # 创建mask标记有效位置
        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0
        
        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_vol_targets.append(vol_tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)
    
    return {
        'sequences': torch.stack(padded_sequences),      # [batch, max_stocks, seq_len, features]
        'targets': torch.stack(padded_targets),          # [batch, max_stocks]
        'vol_targets': torch.stack(padded_vol_targets),  # [batch, max_stocks]
        'relevance': torch.stack(padded_relevance),      # [batch, max_stocks]
        'stock_indices': torch.stack(padded_stock_indices),  # [batch, max_stocks]
        'masks': torch.stack(masks)                      # [batch, max_stocks]
    }


def build_lazy_ranking_index(data, features, sequence_length, min_window_end_date=None, max_window_end_date=None):
    """构建懒加载训练索引，仅保存按股票缓存和按日期索引，不保存完整窗口内容。"""
    print("正在创建排序数据集索引（懒加载版本）...")
    indexed = data.copy()
    indexed = indexed.rename(columns={'日期': 'datetime'})
    indexed['datetime'] = pd.to_datetime(indexed['datetime'])
    indexed = indexed.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    required_cols = ['label']
    if bool(config.get('use_multitask_volatility', False)):
        required_cols.append('vol_label')
    indexed = indexed.dropna(subset=required_cols)

    if min_window_end_date is not None:
        min_window_end_date = pd.to_datetime(min_window_end_date)
    if max_window_end_date is not None:
        max_window_end_date = pd.to_datetime(max_window_end_date)

    stock_cache = {}
    date_to_entries = {}

    grouped = indexed.groupby('instrument', sort=False)
    for stock_idx, group in tqdm(grouped, desc="Indexing stocks"):
        group = group.reset_index(drop=True)
        if len(group) < sequence_length:
            continue

        feature_values = group[features].to_numpy(dtype=np.float32, copy=True)
        labels = group['label'].to_numpy(dtype=np.float32, copy=True)
        if 'vol_label' in group.columns:
            vol_labels = group['vol_label'].to_numpy(dtype=np.float32, copy=True)
        else:
            vol_labels = labels.copy()
        dates = pd.to_datetime(group['datetime']).to_numpy()

        stock_idx = int(stock_idx)
        stock_cache[stock_idx] = {
            'features': feature_values,
            'labels': labels,
            'vol_labels': vol_labels,
        }

        for end_idx in range(sequence_length - 1, len(group)):
            end_date = pd.Timestamp(dates[end_idx]).normalize()
            if min_window_end_date is not None and end_date < min_window_end_date:
                continue
            if max_window_end_date is not None and end_date > max_window_end_date:
                continue
            date_to_entries.setdefault(end_date, []).append((stock_idx, int(end_idx)))

    day_entries = []
    for date in sorted(date_to_entries):
        entries = date_to_entries[date]
        if len(entries) < 10:
            continue
        day_entries.append({
            'date': date,
            'entries': entries,
        })

    print(f"成功创建 {len(day_entries)} 个训练索引样本")
    if day_entries:
        avg_stocks = np.mean([len(entry['entries']) for entry in day_entries])
        print(f"每个训练样本平均包含 {avg_stocks:.1f} 只股票")

    return stock_cache, day_entries

# 排序训练函数
def _compute_volatility_aux_loss(vol_pred, vol_target, stock_valid_mask):
    if vol_pred is None or vol_target is None:
        return None, None
    valid_mask = stock_valid_mask.bool()
    if valid_mask.numel() == 0 or (not bool(valid_mask.any())):
        return None, None

    pred = vol_pred[valid_mask]
    true = vol_target[valid_mask]
    if pred.numel() == 0:
        return None, None

    loss_type = str(config.get('volatility_loss_type', 'huber')).lower()
    if loss_type == 'mse':
        loss = F.mse_loss(pred, true)
    elif loss_type in {'l1', 'mae'}:
        loss = F.l1_loss(pred, true)
    elif loss_type in {'huber', 'smooth_l1'}:
        loss = F.smooth_l1_loss(pred, true)
    else:
        raise ValueError(f'不支持的 volatility_loss_type: {loss_type}')

    mae = torch.mean(torch.abs(pred - true))
    return loss, mae


def train_ranking_model(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    writer,
    strategy_candidates,
    *,
    use_amp=False,
    scaler=None,
):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0
    use_multitask_volatility = bool(config.get('use_multitask_volatility', False))
    volatility_loss_weight = float(config.get('volatility_loss_weight', 0.2))
    use_amp = bool(use_amp) and (device.type == 'cuda')
    max_train_batches = int(config.get('max_train_batches_per_epoch', 0) or 0)
    zero_grad_set_to_none = bool(config.get('train_zero_grad_set_to_none', True))
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}"), start=1):
        if max_train_batches > 0 and batch_idx > max_train_batches:
            break
        sequences = batch['sequences'].to(device)    # [batch, max_stocks, seq_len, features]
        targets = batch['targets'].to(device)        # [batch, max_stocks] 真实涨跌幅
        vol_targets = batch.get('vol_targets', None)
        if vol_targets is not None:
            vol_targets = vol_targets.to(device)
        stock_indices = batch['stock_indices'].to(device)  # [batch, max_stocks]
        masks = batch['masks'].to(device)            # [batch, max_stocks] 有效位置mask
        stock_valid_mask = masks > 0.5
        
        optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            # 模型预测
            outputs = model(
                sequences,
                stock_indices=stock_indices,
                stock_valid_mask=stock_valid_mask,
                return_aux=use_multitask_volatility,
            )  # [batch, max_stocks] 预测分数
            if use_multitask_volatility:
                outputs, vol_outputs = outputs
            else:
                vol_outputs = None
            
            # 应用mask，只考虑有效股票
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)  # 无效位置设为很小的值
            masked_targets = targets * masks
            
            # 计算损失（只对有效股票计算）
            batch_loss = None
            batch_size = sequences.size(0)
            
            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()
                
                if valid_indices.numel() == 0:
                    continue
                    
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                
                # 获取有效股票的预测值和真实收益率
                valid_pred = masked_outputs[i][valid_indices]
                valid_target = masked_targets[i][valid_indices]

                valid_pred_for_loss, valid_target_for_loss = transform_targets_for_loss(valid_pred, valid_target)
                
                if len(valid_pred_for_loss) > 1:
                    # 使用 PortfolioOptimizationLoss 直接对实际收益率进行平滑优化
                    loss = criterion(valid_pred_for_loss.unsqueeze(0), valid_target_for_loss.unsqueeze(0))
                    batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss
        
        if batch_loss is not None:
            rank_loss = batch_loss / batch_size
            total_batch_loss = rank_loss

            vol_aux_loss = None
            vol_aux_mae = None
            if use_multitask_volatility and vol_outputs is not None and vol_targets is not None:
                vol_aux_loss, vol_aux_mae = _compute_volatility_aux_loss(
                    vol_outputs,
                    vol_targets,
                    stock_valid_mask=stock_valid_mask,
                )
                if vol_aux_loss is not None:
                    total_batch_loss = total_batch_loss + volatility_loss_weight * vol_aux_loss

            use_grad_clip = bool(config.get('enable_grad_clip', True))
            max_grad_norm = float(config.get('max_grad_norm', 0.0) or 0.0)
            if use_amp:
                if scaler is None:
                    scaler = torch.amp.GradScaler('cuda', enabled=True)
                scaler.scale(total_batch_loss).backward()
                if use_grad_clip and max_grad_norm > 0.0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if writer:
                        writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch*len(dataloader)+local_step)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_batch_loss.backward()
                if use_grad_clip and max_grad_norm > 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if writer:
                        writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch*len(dataloader)+local_step)
                optimizer.step()
            
            total_loss += float(total_batch_loss.detach().item())
            
            # 计算评估指标
            with torch.no_grad():
                metrics = calculate_ranking_metrics(
                    masked_outputs,
                    masked_targets,
                    masks,
                    strategy_candidates=strategy_candidates,
                    temperature=config.get('softmax_temperature', 1.0),
                )
                metrics['rank_loss'] = float(rank_loss.item())
                if vol_aux_loss is not None and vol_aux_mae is not None:
                    metrics['vol_aux_loss'] = float(vol_aux_loss.item())
                    metrics['vol_mae'] = float(vol_aux_mae.item())
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v
            
            local_step += 1
            if writer:
                writer.add_scalar('train/loss', total_batch_loss.item(), global_step=epoch*len(dataloader)+local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step=epoch*len(dataloader)+local_step)
    
    # 计算平均指标
    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step
    
    return total_loss / local_step if local_step > 0 else 0, total_metrics

def evaluate_ranking_model(
    model,
    dataloader,
    criterion,
    device,
    writer,
    epoch,
    strategy_candidates,
    ablation_feature_indices=None,
):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0
    use_multitask_volatility = bool(config.get('use_multitask_volatility', False))
    volatility_loss_weight = float(config.get('volatility_loss_weight', 0.2))
    use_amp_eval = bool(config.get('use_amp_eval', config.get('use_amp', True))) and (device.type == 'cuda')
    max_eval_batches = int(config.get('max_eval_batches_per_fold', 0) or 0)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"), start=1):
            if max_eval_batches > 0 and batch_idx > max_eval_batches:
                break
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            vol_targets = batch.get('vol_targets', None)
            if vol_targets is not None:
                vol_targets = vol_targets.to(device)
            stock_indices = batch['stock_indices'].to(device)
            masks = batch['masks'].to(device)
            stock_valid_mask = masks > 0.5

            if ablation_feature_indices:
                sequences = sequences.clone()
                sequences[:, :, :, ablation_feature_indices] = 0
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp_eval):
                # 模型预测
                outputs = model(
                    sequences,
                    stock_indices=stock_indices,
                    stock_valid_mask=stock_valid_mask,
                    return_aux=use_multitask_volatility,
                )
                if use_multitask_volatility:
                    outputs, vol_outputs = outputs
                else:
                    vol_outputs = None
                
                # 应用mask
                masked_outputs = outputs * masks + (1 - masks) * (-1e9)
                masked_targets = targets * masks
                
                # 计算损失
                batch_loss = None
                batch_size = sequences.size(0)
                
                for i in range(batch_size):
                    mask = masks[i]
                    valid_indices = mask.nonzero().squeeze()
                    
                    if valid_indices.numel() == 0:
                        continue
                        
                    if valid_indices.dim() == 0:
                        valid_indices = valid_indices.unsqueeze(0)
                    
                    valid_pred = masked_outputs[i][valid_indices]
                    valid_true = masked_targets[i][valid_indices]

                    valid_pred_for_loss, valid_target_for_loss = transform_targets_for_loss(valid_pred, valid_true)
                    if len(valid_pred_for_loss) > 1:
                        # 使用实际收益率进行 loss 验证
                        loss = criterion(valid_pred_for_loss.unsqueeze(0), valid_target_for_loss.unsqueeze(0))
                        batch_loss = batch_loss + loss if batch_loss is not None else loss
            
            rank_loss = None
            if batch_loss is not None:
                rank_loss = batch_loss / batch_size
                total_batch_loss = rank_loss

                vol_aux_loss = None
                vol_aux_mae = None
                if use_multitask_volatility and vol_outputs is not None and vol_targets is not None:
                    vol_aux_loss, vol_aux_mae = _compute_volatility_aux_loss(
                        vol_outputs,
                        vol_targets,
                        stock_valid_mask=stock_valid_mask,
                    )
                    if vol_aux_loss is not None:
                        total_batch_loss = total_batch_loss + volatility_loss_weight * vol_aux_loss

                total_loss += total_batch_loss.item()
            else:
                vol_aux_loss = None
                vol_aux_mae = None
            
            # 计算评估指标
            metrics = calculate_ranking_metrics(
                masked_outputs,
                masked_targets,
                masks,
                strategy_candidates=strategy_candidates,
                temperature=config.get('softmax_temperature', 1.0),
            )
            if rank_loss is not None:
                metrics['rank_loss'] = float(rank_loss.item())
            if vol_aux_loss is not None and vol_aux_mae is not None:
                metrics['vol_aux_loss'] = float(vol_aux_loss.item())
                metrics['vol_mae'] = float(vol_aux_mae.item())
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v
            
            num_batches += 1
    
    # 计算平均指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches
    
    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)
    
    return avg_loss, total_metrics


def evaluate_ranking_folds(
    model,
    fold_loaders,
    criterion,
    device,
    writer,
    epoch,
    strategy_candidates,
    ablation_feature_indices=None,
):
    """在多个滚动验证折上评估，并返回折均值指标。"""
    fold_results = []
    total_loss = 0.0
    total_metrics = {}

    for fold in fold_loaders:
        fold_loss, fold_metrics = evaluate_ranking_model(
            model,
            fold['loader'],
            criterion,
            device,
            writer=None,
            epoch=epoch,
            strategy_candidates=strategy_candidates,
            ablation_feature_indices=ablation_feature_indices,
        )

        fold_result = {
            'name': fold['name'],
            'start_date': fold['start_date'],
            'end_date': fold['end_date'],
            'num_samples': fold['num_samples'],
            'loss': fold_loss,
            'metrics': fold_metrics,
        }
        fold_results.append(fold_result)

        total_loss += fold_loss
        for key, value in fold_metrics.items():
            total_metrics[key] = total_metrics.get(key, 0.0) + value

    num_folds = len(fold_results)
    avg_loss = total_loss / num_folds if num_folds > 0 else 0.0
    avg_metrics = {
        key: value / num_folds
        for key, value in total_metrics.items()
    } if num_folds > 0 else {}

    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for key, value in avg_metrics.items():
            writer.add_scalar(f'eval/{key}', value, global_step=epoch)

        for fold_result in fold_results:
            fold_prefix = f"eval_{fold_result['name']}"
            writer.add_scalar(f'{fold_prefix}/loss', fold_result['loss'], global_step=epoch)
            for key, value in fold_result['metrics'].items():
                writer.add_scalar(f'{fold_prefix}/{key}', value, global_step=epoch)

    return avg_loss, avg_metrics, fold_results


def build_factor_group_indices(feature_pipeline):
    group_indices = {}
    for feature_idx, spec in enumerate(feature_pipeline['active_specs']):
        group = spec.get('group', 'unknown')
        group_indices.setdefault(group, []).append(feature_idx)
    return group_indices


def evaluate_factor_group_ablation(
    model,
    fold_loaders,
    criterion,
    device,
    epoch,
    strategy_candidates,
    feature_pipeline,
    baseline_candidate,
):
    group_indices = build_factor_group_indices(feature_pipeline)
    ablation_results = []
    selection_mode = str(config.get('strategy_selection_mode', 'risk_adjusted')).lower()
    if selection_mode == 'risk_adjusted':
        baseline_metric_name = f'return_{baseline_candidate["name"]}_risk_adjusted'
    else:
        baseline_metric_name = f'return_{baseline_candidate["name"]}'

    for group_name, feature_indices in sorted(group_indices.items()):
        ablation_loss, ablation_metrics, _ = evaluate_ranking_folds(
            model,
            fold_loaders,
            criterion,
            device,
            writer=None,
            epoch=epoch,
            strategy_candidates=strategy_candidates,
            ablation_feature_indices=feature_indices,
        )

        ablated_return = ablation_metrics.get(baseline_metric_name, 0.0)
        ablation_results.append({
            'group': group_name,
            'num_features': len(feature_indices),
            'loss': ablation_loss,
            'return': ablated_return,
            'metrics': ablation_metrics,
        })

    return ablation_results


def log_factor_ablation(writer, epoch, baseline_return, ablation_results):
    if writer is None:
        return

    writer.add_scalar('factors/ablation/baseline_return', baseline_return, global_step=epoch)
    for result in ablation_results:
        group = result['group']
        delta = result['return'] - baseline_return
        writer.add_scalar(f'factors/ablation/{group}/return', result['return'], global_step=epoch)
        writer.add_scalar(f'factors/ablation/{group}/delta', delta, global_step=epoch)
        writer.add_scalar(f'factors/ablation/{group}/num_features', result['num_features'], global_step=epoch)


def predict_top_stocks(model, data, features, sequence_length, scaler, stockid2idx, device, top_k=5):
    """
    预测某一天涨幅前top_k的股票
    """
    model.eval()
    
    # 获取最后一天的数据作为预测基础
    latest_date = data['日期'].max()
    
    # 准备预测数据
    day_sequences = []
    day_stock_codes = []
    day_stock_indices = []
    
    for stock_code in data['股票代码'].unique():
        # 获取该股票历史sequence_length天的数据
        stock_history = data[
            (data['股票代码'] == stock_code) & 
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)
        
        if len(stock_history) == sequence_length:
            seq = stock_history[features].values
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)
            day_stock_indices.append(stockid2idx[stock_code])
    
    if len(day_sequences) == 0:
        return []
    
    # 转换为tensor
    sequences = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)  # [1, num_stocks, seq_len, features]
    stock_indices = torch.LongTensor(np.array(day_stock_indices, dtype=np.int64)).unsqueeze(0).to(device)
    stock_valid_mask = torch.ones_like(stock_indices, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        # 模型预测
        outputs = model(
            sequences,
            stock_indices=stock_indices,
            stock_valid_mask=stock_valid_mask,
        )  # [1, num_stocks]
        scores = outputs.squeeze().cpu().numpy()  # [num_stocks]
        
        # 获取排名前top_k的股票
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        top_stocks = []
        for idx in top_indices:
            top_stocks.append({
                'stock_code': day_stock_codes[idx],
                'predicted_score': scores[idx],
                'rank': len(top_stocks) + 1
            })
    
    return top_stocks

def save_predictions(top_stocks, output_path):
    """保存预测结果"""
    results = []
    for stock in top_stocks:
        results.append({
            '排名': stock['rank'],
            '股票代码': stock['stock_code'],
            '预测分数': stock['predicted_score']
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {output_path}")


def split_train_val_by_last_month(df, sequence_length):
    """按最后一个月做验证集划分，并为验证集补充序列上下文。"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=2)).normalize()

    # 验证集需要保留前 sequence_length-1 个交易日作为序列上下文，
    # 这样第一个验证样本的窗口结束日就可以落在 val_start。
    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围(最后一个月): {val_start.date()} 到 {last_date.date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    # 恢复为字符串，保持与原流程一致
    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start


def build_rolling_validation_folds(df, sequence_length):
    return build_rolling_validation_folds_shared(df, sequence_length, config)


def build_validation_fold_loaders(val_data, features, val_folds):
    """为每个滚动验证折构建独立的数据集与 DataLoader。"""
    fold_loaders = []
    total_samples = 0

    for fold in val_folds:
        sequences, targets, relevance, stock_indices, vol_targets = create_ranking_dataset_vectorized(
            val_data,
            features,
            config['sequence_length'],
            ranking_data_path=None,
            min_window_end_date=fold['start_date'].strftime('%Y-%m-%d'),
            max_window_end_date=fold['end_date'].strftime('%Y-%m-%d'),
        )

        if len(sequences) == 0:
            raise ValueError(
                f"{fold['name']} ({fold['start_date'].date()} ~ {fold['end_date'].date()}) 未生成任何验证样本"
            )

        dataset = RankingDataset(
            sequences,
            targets,
            relevance,
            stock_indices,
            vol_targets=vol_targets,
        )
        loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
        )

        fold_loaders.append({
            'name': fold['name'],
            'start_date': fold['start_date'],
            'end_date': fold['end_date'],
            'num_samples': len(sequences),
            'loader': loader,
        })
        total_samples += len(sequences)

        print(
            f"验证折 {fold['name']} 样本数: {len(sequences)} "
            f"({fold['start_date'].date()} ~ {fold['end_date'].date()})"
        )

    print(f"滚动验证总样本数: {total_samples}")
    return fold_loaders

# 主程序
def main():
    set_seed(config.get('seed', 42))
    output_dir = config['output_dir']
    os.makedirs(output_dir,exist_ok=True)
    # 保存在output_dir中保存当前的配置文件，以便复现
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    data_manifest = collect_data_sources(config, include_csv_stats=True)
    manifest_path = save_data_manifest(output_dir, data_manifest)
    print(f"已生成数据源清单: {manifest_path}")
    is_train = True
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log')) if is_train else None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if device.type == 'cuda':
        device_msg = f"当前训练设备: cuda ({torch.cuda.get_device_name(device)})"
    elif device.type == 'mps':
        device_msg = "当前训练设备: mps (Apple Silicon)"
    else:
        device_msg = "当前训练设备: cpu"
    print(device_msg)
    
    factor_pipeline = resolve_factor_pipeline(
        config['feature_num'],
        config['factor_store_path'],
        config['builtin_factor_registry_path'],
    )
    dataset_manifest_train_path, dataset_manifest_info = load_train_dataset_from_build_manifest(config, factor_pipeline)
    if dataset_manifest_info.get('enabled', False):
        print(
            f"dataset build manifest: {dataset_manifest_info.get('manifest_path', '')} "
            f"(strict={dataset_manifest_info.get('strict', False)})"
        )
        for msg in dataset_manifest_info.get('warnings', []):
            print(f"[manifest-warning] {msg}")
        for msg in dataset_manifest_info.get('errors', []):
            print(f"[manifest-error] {msg}")
        if dataset_manifest_info.get('used', False):
            print(
                "manifest 元信息: "
                f"build_id={dataset_manifest_info.get('build_id', '')}, "
                f"feature_set_version={dataset_manifest_info.get('feature_set_version', '')}, "
                f"factor_fingerprint={dataset_manifest_info.get('factor_fingerprint', '')}"
            )

    # 1. 数据加载（优先使用 build-dataset manifest 指向的数据集）
    if dataset_manifest_train_path:
        full_df, data_file = load_market_dataset_from_path(config, dataset_manifest_train_path)
        print(f"训练数据文件(manifest): {data_file}")
    else:
        full_df, data_file = load_market_dataset(config, 'train.csv')
        print(f"训练数据文件: {data_file}")

    snapshot_path = os.path.join(output_dir, 'active_factors.json')
    snapshot_info = save_factor_snapshot(factor_pipeline, snapshot_path)
    factor_pipeline['snapshot_meta'] = snapshot_info.get('snapshot', {})
    factor_pipeline['factor_fingerprint'] = snapshot_info.get(
        'factor_fingerprint',
        factor_pipeline.get('factor_fingerprint', ''),
    )
    print(
        f"已保存因子快照: {snapshot_path} | "
        f"fingerprint={factor_pipeline.get('factor_fingerprint', '')}"
    )
    print("当前因子配置:", format_factor_summary(factor_pipeline))
    print_active_factors(factor_pipeline)
    validation_mode = config.get('validation_mode', 'rolling')
    if validation_mode == 'rolling':
        train_df, val_df, val_folds = build_rolling_validation_folds_shared(full_df, config['sequence_length'], config)
    else:
        train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'])
        val_folds = [{
            'name': 'holdout',
            'start_date': pd.Timestamp(val_start).normalize(),
            'end_date': pd.to_datetime(val_df['日期']).max().normalize(),
        }]
    
    # 获取所有股票ID，建立映射
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)
    
    # 2. 特征工程与预处理
    train_data, features = preprocess_data(train_df, factor_pipeline, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, factor_pipeline, stockid2idx=stockid2idx)
    with open(os.path.join(output_dir, 'effective_features.json'), 'w', encoding='utf-8') as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    print(f'已保存训练特征清单: {os.path.join(output_dir, "effective_features.json")} | 特征数: {len(features)}')
    
    # 3. 特征缩放（默认仅保留截面标准化，不做全局 StandardScaler）
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    # 丢弃nan数据
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    dump_factor_artifacts('train', train_data, features, output_dir)
    dump_factor_artifacts('val', val_data, features, output_dir)

    histogram_features = features[:max(0, int(config.get('factor_histogram_max_features', 0)))]
    raw_train_hist_frame = train_data[histogram_features].copy() if histogram_features else None

    # 关键修正：仅按日截面标准化（已在 preprocess_* 中完成），
    # 这里明确禁用全局拟合缩放，避免时序泄露并保留日内相对强弱。
    train_data[features] = train_data[features].astype(np.float32)
    val_data[features] = val_data[features].astype(np.float32)
    joblib.dump({'type': 'identity', 'name': 'cross_sectional_only'}, scaler_path)
    print('已固定为截面标准化，禁用全局 StandardScaler。')

    scaled_train_hist_frame = train_data[histogram_features] if histogram_features else None
    log_factor_dashboard(writer, factor_pipeline, raw_train_hist_frame, scaled_train_hist_frame)

    
    # 4. 创建排序数据集
    train_stock_cache, train_day_entries = build_lazy_ranking_index(
        train_data,
        features,
        config['sequence_length'],
    )
    print(f"训练集样本数: {len(train_day_entries)}")
    val_fold_loaders = build_validation_fold_loaders(val_data, features, val_folds)

    mask_mode = str(config.get('cross_stock_mask_mode', 'similarity')).lower()
    need_prior_graph = bool(config.get('use_cross_stock_attention_mask', True)) and mask_mode in {
        'prior',
        'prior_similarity',
    }
    prior_graph_adj = None
    if need_prior_graph:
        prior_graph_adj = build_prior_graph_adjacency(train_data, stockid2idx)
        prior_graph_path = os.path.join(output_dir, 'prior_graph_adj.npy')
        np.save(prior_graph_path, prior_graph_adj.astype(np.uint8))
        print(f"已保存先验图邻接矩阵: {prior_graph_path}")

    use_industry_virtual = bool(config.get('use_industry_virtual_stock', False))
    use_industry_virtual_temporal = bool(config.get('industry_virtual_on_temporal_cross_stock', False))
    stock_industry_idx = np.full(num_stocks, -1, dtype=np.int64)
    if use_industry_virtual or use_industry_virtual_temporal:
        stock_industry_idx, industry_vocab = build_stock_industry_index(stockid2idx)
        industry_index_path = os.path.join(output_dir, 'stock_industry_idx.npy')
        industry_vocab_path = os.path.join(output_dir, 'industry_vocab.json')
        np.save(industry_index_path, stock_industry_idx.astype(np.int64))
        with open(industry_vocab_path, 'w', encoding='utf-8') as f:
            json.dump(industry_vocab, f, ensure_ascii=False, indent=2)
        print(f"已保存行业索引映射: {industry_index_path} (行业数={len(industry_vocab)})")
    
    # 5. 创建排序数据集和数据加载器
    train_dataset = LazyRankingDataset(train_stock_cache, train_day_entries, config['sequence_length'])
    del train_data
    del val_data
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # 减少worker数量避免内存问题
        pin_memory=False
    )
    
    # 6. 模型初始化
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    market_context_feature_names = config.get(
        'market_gating_context_feature_names',
        [
            'market_median_return',
            'market_total_turnover_log',
            'market_limit_up_count_log',
            'market_limit_up_ratio',
        ],
    )
    market_context_indices = resolve_feature_indices(features, market_context_feature_names)
    model.set_market_context_feature_indices(market_context_indices)
    if prior_graph_adj is not None:
        model.set_prior_graph(torch.from_numpy(prior_graph_adj))
    model.set_stock_industry_index(torch.from_numpy(stock_industry_idx))
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    strategy_candidates = build_strategy_candidates()
    print("候选持仓策略:", ", ".join(candidate['name'] for candidate in strategy_candidates))
    
    # 7. 损失函数和优化器
    criterion = PortfolioOptimizationLoss(
        temperature=float(config.get('loss_temperature', 10.0)),
        listnet_weight=float(config.get('listnet_weight', 1.0)),
        pairwise_weight=float(config.get('pairwise_weight', 1.0)),
        lambda_ndcg_weight=float(config.get('lambda_ndcg_weight', 1.0)),
        lambda_ndcg_topk=int(config.get('lambda_ndcg_topk', 50)),
        ic_weight=float(config.get('ic_weight', 0.0)),
        ic_mode=str(config.get('ic_mode', 'pearson')),
        topk_focus_weight=float(config.get('topk_focus_weight', 0.0)),
        topk_focus_k=int(config.get('topk_focus_k', 5)),
        topk_focus_gain_mode=str(config.get('topk_focus_gain_mode', 'binary')),
        topk_focus_normalize=config.get('topk_focus_normalize', True),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=float(config.get('weight_decay', 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.2, total_iters=config['num_epochs'])
    use_amp = bool(config.get('use_amp', True)) and (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"AMP 混合精度训练: {'开启' if use_amp else '关闭'}")
    
    # 8. 排序模型训练
    if is_train:
        best_score = -float('inf')
        best_epoch = -1
        early_stop_enabled = bool(config.get('early_stopping_enabled', True))
        early_stop_patience = int(config.get('early_stopping_patience', 8))
        early_stop_min_delta = float(config.get('early_stopping_min_delta', 1e-4))
        early_stop_monitor = str(config.get('early_stopping_monitor', 'rank_ic_mean'))
        early_stop_mode = str(config.get('early_stopping_mode', 'max')).lower()
        if early_stop_mode not in {'max', 'min'}:
            raise ValueError(f'early_stopping_mode 非法: {early_stop_mode}')
        best_monitor = -float('inf') if early_stop_mode == 'max' else float('inf')
        bad_epochs = 0
        
        for epoch in range(config['num_epochs']):
            print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")
            
            # 训练
            train_loss, train_metrics = train_ranking_model(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                writer,
                strategy_candidates,
                use_amp=use_amp,
                scaler=scaler,
            )
            
            print(f"Train Loss: {train_loss:.4f}")
            for k, v in train_metrics.items():
                print(f"Train {k}: {v:.4f}")
            
            # 验证
            eval_loss, eval_metrics, fold_results = evaluate_ranking_folds(
                model, val_fold_loaders, criterion, device, writer, epoch, strategy_candidates
            )
            run_summary = summarize_experiment_run(
                eval_loss=eval_loss,
                eval_metrics=eval_metrics,
                fold_results=fold_results,
                strategy_candidates=strategy_candidates,
                runtime_config=config,
            )
            
            print(f"Eval Loss: {eval_loss:.4f}")
            for k, v in eval_metrics.items():
                print(f"Eval {k}: {v:.4f}")
            print(
                "Eval 策略收益汇总: "
                + run_summary['strategy_summary']
            )
            print("Eval 策略对比:")
            for strategy_row in run_summary['strategy_comparison']:
                print(
                    f"  - {strategy_row['name']}: "
                    f"mean={strategy_row['mean_return']:.4f}, "
                    f"std={strategy_row['return_std']:.4f}, "
                    f"ra={strategy_row['risk_adjusted_return']:.4f}"
                )
            for fold_result in run_summary['fold_diagnostics']:
                fold_strategy_summary = ', '.join(
                    (
                        f"{row['name']}=mean:{row['mean_return']:.4f}"
                        f"|std:{row['return_std']:.4f}"
                        f"|ra:{row['risk_adjusted_return']:.4f}"
                    )
                    for row in fold_result['strategy_comparison']
                )
                print(
                    f"Eval {fold_result['name']} "
                    f"({fold_result['start_date'].date()} ~ {fold_result['end_date'].date()}) "
                    f"样本数: {fold_result['num_samples']} | "
                    f"Loss: {fold_result['loss']:.4f} | "
                    f"best={fold_result['best_candidate']['name']}:{fold_result['best_score']:.4f}"
                )
                print(
                    "  策略收益: "
                    + fold_strategy_summary
                )
            regime_summary = run_summary['regime_summary']
            print(
                "Eval Regime 摘要: "
                f"dominant={regime_summary['dominant_strategy']}, "
                f"positive={regime_summary['positive_return_fold_count']}, "
                f"negative={regime_summary['negative_return_fold_count']}, "
                f"flat={regime_summary['flat_return_fold_count']}"
            )
            
            # 学习率调度
            scheduler.step()
            if writer:
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step=epoch)
            

            best_candidate = run_summary['best_candidate']
            current_final_score = run_summary['best_score']
            best_candidate_return = run_summary['best_return']
            print(
                f"当前最优持仓策略: {best_candidate['name']} | "
                f"验证目标值: {current_final_score:.4f} | "
                f"策略收益均值: {best_candidate_return:.4f} | "
                f"RankIC: {eval_metrics.get('rank_ic_mean', 0.0):.4f}"
            )

            if config.get('factor_ablation_enabled', True):
                ablation_results = evaluate_factor_group_ablation(
                    model,
                    val_fold_loaders,
                    criterion,
                    device,
                    epoch,
                    strategy_candidates,
                    factor_pipeline,
                    best_candidate,
                )
                log_factor_ablation(writer, epoch, current_final_score, ablation_results)
                print("因子分组消融:")
                for result in ablation_results:
                    delta = result['return'] - current_final_score
                    print(
                        f"  - {result['group']}: "
                        f"features={result['num_features']}, "
                        f"return={result['return']:.4f}, "
                        f"delta={delta:.4f}"
                    )

            if current_final_score > best_score:
                best_score = current_final_score
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                with open(os.path.join(output_dir, 'best_strategy.json'), 'w') as f:
                    json.dump(
                        build_strategy_export_payload(
                            run_summary=run_summary,
                            validation_folds=val_folds,
                            runtime_config=config,
                            source='training_validation',
                            exported_at_field='saved_at',
                            best_epoch=best_epoch,
                        ),
                        f,
                        indent=4,
                        ensure_ascii=False,
                    )
                print(f"保存最佳模型 - objective: {best_score:.4f}")

            monitor_value = eval_metrics.get(early_stop_monitor, None)
            if monitor_value is None:
                print(f"早停监控指标缺失，跳过本轮监控: {early_stop_monitor}")
                continue

            if early_stop_mode == 'max':
                improved = monitor_value > (best_monitor + early_stop_min_delta)
            else:
                improved = monitor_value < (best_monitor - early_stop_min_delta)

            if improved:
                best_monitor = monitor_value
                bad_epochs = 0
            else:
                bad_epochs += 1

            print(
                f"早停状态: monitor={early_stop_monitor}, value={monitor_value:.6f}, "
                f"best={best_monitor:.6f}, bad_epochs={bad_epochs}/{early_stop_patience}"
            )

            if writer:
                writer.add_scalar(f'early_stop/{early_stop_monitor}', monitor_value, global_step=epoch)
                writer.add_scalar('early_stop/bad_epochs', bad_epochs, global_step=epoch)

            if early_stop_enabled and bad_epochs >= early_stop_patience:
                print(
                    f"触发早停: monitor={early_stop_monitor}, mode={early_stop_mode}, "
                    f"patience={early_stop_patience}, best={best_monitor:.6f}"
                )
                break
        print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 objective: {best_score:.4f}")
        with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
            f.write(f"Best epoch: {best_epoch}\n")
            f.write(f"Best objective: {best_score:.6f}\n")

        if writer:
            writer.close()

        return best_score

if __name__ == "__main__":
    # 多进程保护
    mp.set_start_method('spawn', force=True)
    best_score = main()
    print(f"\n########## 训练完成！最佳 objective: {best_score:.4f} ##########")
