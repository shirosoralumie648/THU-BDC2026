import multiprocessing as mp
import os
from functools import lru_cache
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import config
from data_manager import load_stock_to_industry_map
from features.feature_assembler import augment_feature_table
from features.feature_assembler import build_feature_table


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
            ['benchmark_return', 'return', '收益率', '涨跌幅', 'pct_chg', 'pctChg'],
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


def _build_future_volatility_label(processed, stock_col='股票代码', open_col='开盘', horizon=5):
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
    assert stockid2idx is not None, 'stockid2idx 不能为空'
    feature_columns = list(feature_pipeline['active_features'])

    # 保证时序正确，避免 shift 标签错位
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f'正在使用多进程进行{desc}...')
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f'{desc}输入为空，无法继续')

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


def preprocess_data(df, feature_pipeline, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(
            df,
            stockid2idx,
            desc='特征工程',
            feature_pipeline=feature_pipeline,
            drop_small_open=False,
        )
    return _preprocess_common(
        df,
        stockid2idx,
        desc='特征工程',
        feature_pipeline=feature_pipeline,
        drop_small_open=True,
    )


def preprocess_val_data(df, feature_pipeline, stockid2idx=None):
    # 验证集与训练集保持同口径，避免 label 分布漂移
    return _preprocess_common(
        df,
        stockid2idx,
        desc='验证集特征工程',
        feature_pipeline=feature_pipeline,
        drop_small_open=True,
    )


def preprocess_predict_data(df, stockid2idx, feature_pipeline, runtime_config=None):
    runtime_config = runtime_config or config
    feature_columns = list(feature_pipeline['active_features'])

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError('输入数据为空，无法预测')

    num_processes = min(int(runtime_config.get('feature_engineer_processes', 4)), mp.cpu_count())
    feature_builder = partial(
        build_feature_table,
        feature_set=feature_pipeline['feature_set'],
        runtime_config=runtime_config,
        feature_pipeline=feature_pipeline,
    )
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_builder, groups), total=len(groups), desc='预测集特征工程'))

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed, feature_columns = augment_feature_table(
        processed,
        feature_columns,
        runtime_config=runtime_config,
        feature_pipeline=feature_pipeline,
        date_col='日期',
        stock_col='股票代码',
        apply_feature_enhancements=False,
        apply_cross_sectional_norm=False,
    )
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])

    processed, feature_columns = augment_feature_table(
        processed,
        feature_columns,
        runtime_config=runtime_config,
        feature_pipeline=None,
        date_col='日期',
        stock_col='股票代码',
        apply_factor_pipeline=False,
        apply_feature_enhancements=True,
        apply_cross_sectional_norm=True,
    )

    return processed, feature_columns
