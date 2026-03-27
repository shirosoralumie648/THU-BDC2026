import pandas as pd
import numpy as np
import joblib
import os
import re
from functools import lru_cache
from tqdm import tqdm


def apply_cross_sectional_normalization(
    df,
    columns,
    date_col='日期',
    method='zscore',
    clip_value=None,
    exclude_columns=None,
):
    """
    按交易日对给定列做截面标准化，降低跨时段分布漂移影响。

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        需要做截面归一化的列。
    date_col : str
        交易日列名。
    method : str
        'zscore' 或 'rank'。
    clip_value : float | None
        仅 zscore 生效。若设置，则对归一化结果做对称截断 [-clip, clip]。
    """
    if not columns:
        return df
    if date_col not in df.columns:
        raise ValueError(f'缺少日期列，无法做截面标准化: {date_col}')

    out = df.copy()
    exclude_set = set(exclude_columns or [])
    target_cols = [col for col in columns if col in out.columns and col not in exclude_set]
    if not target_cols:
        return out

    grouped = out.groupby(date_col)[target_cols]
    if method == 'zscore':
        means = grouped.transform('mean')
        stds = grouped.transform('std').replace(0.0, np.nan)
        normalized = (out[target_cols] - means) / (stds + 1e-12)
        if clip_value is not None:
            clip_value = float(clip_value)
            normalized = normalized.clip(lower=-clip_value, upper=clip_value)
        out[target_cols] = normalized.astype(np.float32)
    elif method == 'rank':
        ranks = grouped.rank(pct=True)
        out[target_cols] = ((ranks * 2.0) - 1.0).astype(np.float32)
    else:
        raise ValueError(f'不支持的截面标准化方式: {method}')

    out[target_cols] = out[target_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def resolve_feature_indices(feature_columns, target_feature_names):
    """
    将目标特征名映射到当前输入特征索引，缺失项返回 -1。
    """
    if not target_feature_names:
        return []
    if not feature_columns:
        return [-1 for _ in target_feature_names]

    col2idx = {name: idx for idx, name in enumerate(feature_columns)}
    return [int(col2idx.get(name, -1)) for name in target_feature_names]


def _safe_feature_name(name):
    safe = re.sub(r'[^0-9A-Za-z_]+', '_', str(name))
    safe = re.sub(r'_+', '_', safe).strip('_').lower()
    if not safe:
        safe = 'f'
    if safe[0].isdigit():
        safe = f'f_{safe}'
    return safe


def _resolve_collision_safe_feature_names(source_names, prefix):
    """
    生成无冲突特征名。
    为兼容旧训练结果：冲突组中“最后一个”保留基础名，其余早期项追加 __dup{i} 后缀。
    """
    if not source_names:
        return []

    base_names = [f'{prefix}{_safe_feature_name(name)}' for name in source_names]
    remaining_counts = {}
    for base in base_names:
        remaining_counts[base] = remaining_counts.get(base, 0) + 1

    seen_counts = {}
    resolved = []
    for base in base_names:
        remaining_counts[base] -= 1
        if remaining_counts[base] == 0:
            resolved.append(base)
            continue

        duplicate_idx = seen_counts.get(base, 0) + 1
        seen_counts[base] = duplicate_idx
        resolved.append(f'{base}__dup{duplicate_idx}')
    return resolved


def _normalize_stock_code_series(series):
    s = series.astype(str).str.strip()
    s = s.str.split('.').str[-1]
    s = s.str.replace(r'[^0-9]', '', regex=True)
    s = s.str[-6:].str.zfill(6)
    return s


def _infer_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


@lru_cache(maxsize=16)
def _load_static_feature_mapping(
    mapping_path,
    stock_col,
    industry_col,
    market_cap_col,
    industry_topk,
    include_other,
):
    if not mapping_path or not os.path.exists(mapping_path):
        return None

    mapping_df = pd.read_csv(mapping_path, dtype=str)
    resolved_stock_col = stock_col if stock_col in mapping_df.columns else _infer_existing_column(
        mapping_df, ['股票代码', 'stock_id', 'code', 'ts_code']
    )
    if resolved_stock_col is None:
        return None

    resolved_industry_col = industry_col if industry_col in mapping_df.columns else _infer_existing_column(
        mapping_df, ['行业', 'industry', 'sw_l1', 'sector']
    )
    resolved_market_cap_col = market_cap_col if market_cap_col in mapping_df.columns else _infer_existing_column(
        mapping_df, ['流通市值', 'float_mv', 'market_cap', 'circulating_market_value']
    )

    mapping_df = mapping_df.copy()
    mapping_df['__stock_norm'] = _normalize_stock_code_series(mapping_df[resolved_stock_col])
    mapping_df = mapping_df.dropna(subset=['__stock_norm'])
    mapping_df = mapping_df[mapping_df['__stock_norm'] != '']
    mapping_df = mapping_df.drop_duplicates(subset=['__stock_norm'], keep='last')
    if mapping_df.empty:
        return None

    stock_to_industry = {}
    industry_buckets = []
    if resolved_industry_col is not None:
        industry_series = mapping_df[resolved_industry_col].astype(str).str.strip()
        industry_series = industry_series.replace({'': np.nan})
        if industry_series.notna().any():
            mapping_df['__industry'] = industry_series
            stock_to_industry = (
                mapping_df
                .dropna(subset=['__industry'])
                .set_index('__stock_norm')['__industry']
                .to_dict()
            )
            counts = mapping_df['__industry'].value_counts()
            topk_values = list(counts.head(max(1, int(industry_topk))).index)
            if bool(include_other):
                industry_buckets = topk_values + ['__OTHER__']
            else:
                industry_buckets = topk_values

    stock_to_log_mcap = {}
    if resolved_market_cap_col is not None:
        cap_raw = mapping_df[resolved_market_cap_col].astype(str).str.replace(',', '', regex=False)
        cap_num = pd.to_numeric(cap_raw, errors='coerce')
        cap_num = cap_num.clip(lower=0.0)
        if cap_num.notna().any():
            log_cap = np.log1p(cap_num)
            mapping_df['__log_cap'] = log_cap
            stock_to_log_mcap = (
                mapping_df
                .dropna(subset=['__log_cap'])
                .set_index('__stock_norm')['__log_cap']
                .astype(np.float32)
                .to_dict()
            )

    return {
        'stock_to_industry': stock_to_industry,
        'stock_to_log_mcap': stock_to_log_mcap,
        'industry_buckets': tuple(industry_buckets),
    }


def _attach_static_stock_features(
    df,
    config,
    stock_col='股票代码',
    date_col='日期',
):
    if not bool(config.get('use_static_stock_features', True)):
        return df, [], None

    mapping_path = str(config.get('stock_static_feature_path', '')).strip()
    if not mapping_path:
        return df, [], None

    static_data = _load_static_feature_mapping(
        mapping_path=mapping_path,
        stock_col=str(config.get('stock_static_stock_col', '股票代码')),
        industry_col=str(config.get('stock_static_industry_col', '行业')),
        market_cap_col=str(config.get('stock_static_market_cap_col', '流通市值')),
        industry_topk=int(config.get('stock_static_industry_topk', 12)),
        include_other=bool(config.get('stock_static_include_other_bucket', True)),
    )
    if static_data is None:
        return df, [], None

    out = df.copy()
    out['__stock_norm'] = _normalize_stock_code_series(out[stock_col])
    extra_features = []
    industry_bucket_col = None

    stock_to_log_mcap = static_data['stock_to_log_mcap']
    if stock_to_log_mcap:
        out['st_log_mcap'] = out['__stock_norm'].map(stock_to_log_mcap).astype(np.float32)
        out['st_log_mcap'] = out['st_log_mcap'].fillna(out['st_log_mcap'].median(skipna=True))
        out['st_log_mcap_cs_rank'] = out.groupby(date_col)['st_log_mcap'].rank(pct=True).astype(np.float32)
        extra_features.extend(['st_log_mcap', 'st_log_mcap_cs_rank'])

    stock_to_industry = static_data['stock_to_industry']
    buckets = list(static_data['industry_buckets'])
    if stock_to_industry and buckets:
        out['__industry_raw'] = out['__stock_norm'].map(stock_to_industry)
        if '__OTHER__' in buckets:
            top_set = {b for b in buckets if b != '__OTHER__'}
            out['__industry_bucket'] = out['__industry_raw'].where(
                out['__industry_raw'].isin(top_set),
                '__OTHER__',
            )
        else:
            out['__industry_bucket'] = out['__industry_raw']
        industry_bucket_col = '__industry_bucket'

        for idx, bucket in enumerate(buckets):
            col_name = f'st_ind_oh_{idx:02d}'
            out[col_name] = (out['__industry_bucket'] == bucket).astype(np.float32)
            extra_features.append(col_name)

    out.drop(columns=[c for c in ['__stock_norm', '__industry_raw'] if c in out.columns], inplace=True)
    return out, extra_features, industry_bucket_col


def _add_price_volume_distribution_features(df, stock_col='股票代码', date_col='日期'):
    if not {'开盘', '收盘', '最高', '最低'}.issubset(df.columns):
        return df, []

    out = df.copy()
    out = out.sort_values([stock_col, date_col]).reset_index(drop=True)
    grouped = out.groupby(stock_col, sort=False)
    eps = 1e-12

    open_px = out['开盘'].astype(float)
    close_px = out['收盘'].astype(float)
    high_px = out['最高'].astype(float)
    low_px = out['最低'].astype(float)
    prev_close = grouped['收盘'].shift(1).astype(float)

    out['pv_intraday_return'] = ((close_px - open_px) / (open_px + eps)).astype(np.float32)
    out['pv_intraday_range_pct'] = ((high_px - low_px) / (open_px + eps)).astype(np.float32)
    out['pv_gap_return'] = ((open_px - prev_close) / (prev_close + eps)).astype(np.float32)
    out['pv_close_location'] = ((close_px - low_px) / (high_px - low_px + eps)).astype(np.float32)

    tr1 = (high_px - low_px).abs()
    tr2 = (high_px - prev_close).abs()
    tr3 = (low_px - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out['pv_true_range_pct'] = (true_range / (prev_close + eps)).astype(np.float32)

    log_hl = np.log(high_px + eps) - np.log(low_px + eps)
    parkinson_daily = np.sqrt(np.maximum((log_hl ** 2) / (4.0 * np.log(2.0) + eps), 0.0))
    out['pv_parkinson_daily'] = parkinson_daily.astype(np.float32)
    out['pv_parkinson_5'] = grouped['pv_parkinson_daily'].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    ).astype(np.float32)
    out['pv_parkinson_20'] = grouped['pv_parkinson_daily'].transform(
        lambda s: s.rolling(20, min_periods=1).mean()
    ).astype(np.float32)
    out['pv_intraday_return_std5'] = grouped['pv_intraday_return'].transform(
        lambda s: s.rolling(5, min_periods=1).std()
    ).astype(np.float32)

    if '成交额' in out.columns:
        ret1 = grouped['收盘'].transform(lambda s: s.pct_change(fill_method=None))
        amount = out['成交额'].astype(float).abs()
        out['pv_amihud_illiq'] = (ret1.abs() / (amount + eps)).astype(np.float32)

    if '成交量' in out.columns:
        log_vol = np.log1p(out['成交量'].astype(float).clip(lower=0.0))
        vol_mean20 = log_vol.groupby(out[stock_col]).transform(lambda s: s.rolling(20, min_periods=1).mean())
        vol_std20 = log_vol.groupby(out[stock_col]).transform(lambda s: s.rolling(20, min_periods=1).std())
        out['pv_volume_z20'] = ((log_vol - vol_mean20) / (vol_std20 + eps)).astype(np.float32)

    new_cols = [
        'pv_intraday_return',
        'pv_intraday_range_pct',
        'pv_gap_return',
        'pv_close_location',
        'pv_true_range_pct',
        'pv_parkinson_daily',
        'pv_parkinson_5',
        'pv_parkinson_20',
        'pv_intraday_return_std5',
    ]
    if 'pv_amihud_illiq' in out.columns:
        new_cols.append('pv_amihud_illiq')
    if 'pv_volume_z20' in out.columns:
        new_cols.append('pv_volume_z20')
    return out, new_cols


def _add_cross_sectional_rank_features(
    df,
    config,
    date_col='日期',
    industry_bucket_col=None,
):
    out = df.copy()
    extra_features = []
    rank_sources = config.get(
        'cross_sectional_rank_source_features',
        ['rsi', 'return_1', 'return_5', 'volatility_20', 'atr_14', '换手率', '成交额', 'pv_intraday_range_pct'],
    )
    industry_sources = config.get(
        'industry_relative_source_features',
        ['rsi', 'return_1', 'return_5', 'volatility_20', 'atr_14'],
    )

    if bool(config.get('use_cross_sectional_rank_features', True)):
        present_rank_sources = [col for col in rank_sources if col in out.columns]
        rank_feature_names = _resolve_collision_safe_feature_names(
            present_rank_sources,
            prefix='cs_rank_',
        )
        for col, rank_col in zip(present_rank_sources, rank_feature_names):
            out[rank_col] = out.groupby(date_col)[col].rank(pct=True).astype(np.float32)
            extra_features.append(rank_col)

    if bool(config.get('use_industry_relative_z_features', True)) and industry_bucket_col and industry_bucket_col in out.columns:
        present_industry_sources = [col for col in industry_sources if col in out.columns]
        industry_feature_names = _resolve_collision_safe_feature_names(
            present_industry_sources,
            prefix='ind_z_',
        )
        for col, ind_col in zip(present_industry_sources, industry_feature_names):
            means = out.groupby([date_col, industry_bucket_col])[col].transform('mean')
            stds = out.groupby([date_col, industry_bucket_col])[col].transform('std').replace(0.0, np.nan)
            out[ind_col] = ((out[col] - means) / (stds + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
            extra_features.append(ind_col)

    return out, extra_features


def _add_market_sentiment_features(
    df,
    config,
    date_col='日期',
):
    """
    追加日级市场情绪特征（对同一交易日全部股票共享）：
    - 全市场中位数涨幅
    - 全市场总成交额（log1p）
    - 涨停家数（log1p）
    - 涨停占比
    """
    if not bool(config.get('use_market_sentiment_features', True)):
        return df, []

    out = df.copy()
    feature_names = []

    return_candidates = config.get('market_sentiment_return_col_candidates', ['涨跌幅', 'return_1'])
    if isinstance(return_candidates, str):
        return_candidates = [return_candidates]
    turnover_candidates = config.get('market_sentiment_turnover_col_candidates', ['成交额'])
    if isinstance(turnover_candidates, str):
        turnover_candidates = [turnover_candidates]

    return_col = next((col for col in return_candidates if col in out.columns), None)
    turnover_col = next((col for col in turnover_candidates if col in out.columns), None)
    if return_col is None:
        return out, feature_names

    ret_raw = pd.to_numeric(out[return_col], errors='coerce')
    ret_scale = 100.0 if ret_raw.abs().median(skipna=True) > 1.0 else 1.0
    out['__market_ret_scaled'] = ret_raw / ret_scale

    market_median_col = 'market_median_return'
    out[market_median_col] = (
        out.groupby(date_col)['__market_ret_scaled']
        .transform('median')
        .astype(np.float32)
    )
    feature_names.append(market_median_col)

    if turnover_col is not None:
        turnover = pd.to_numeric(out[turnover_col], errors='coerce').fillna(0.0).clip(lower=0.0)
        out['__market_turnover'] = turnover
        market_turnover_col = 'market_total_turnover_log'
        out[market_turnover_col] = np.log1p(
            out.groupby(date_col)['__market_turnover'].transform('sum')
        ).astype(np.float32)
        feature_names.append(market_turnover_col)

    limit_up_threshold = float(config.get('market_sentiment_limit_up_threshold', 0.095))
    out['__market_limit_up_flag'] = (out['__market_ret_scaled'] >= limit_up_threshold).astype(np.float32)

    limit_up_count = out.groupby(date_col)['__market_limit_up_flag'].transform('sum')
    market_limit_up_count_col = 'market_limit_up_count_log'
    out[market_limit_up_count_col] = np.log1p(limit_up_count).astype(np.float32)
    feature_names.append(market_limit_up_count_col)

    sample_count = out.groupby(date_col)['__market_limit_up_flag'].transform('count').astype(np.float32)
    market_limit_up_ratio_col = 'market_limit_up_ratio'
    out[market_limit_up_ratio_col] = (limit_up_count / (sample_count + 1e-12)).astype(np.float32)
    feature_names.append(market_limit_up_ratio_col)

    out.drop(
        columns=[
            c for c in ['__market_ret_scaled', '__market_turnover', '__market_limit_up_flag']
            if c in out.columns
        ],
        inplace=True,
    )
    return out, feature_names


def augment_engineered_features(
    df,
    feature_columns,
    config,
    date_col='日期',
    stock_col='股票代码',
):
    """
    在基础因子之上追加增强特征：
    1) 静态行业/市值特征；
    2) 量价分布风险特征；
    3) 截面 rank 与行业内 zscore 特征。
    """
    out = df.copy()
    extra_features = []
    industry_bucket_col = None

    out, static_cols, industry_bucket_col = _attach_static_stock_features(
        out,
        config=config,
        stock_col=stock_col,
        date_col=date_col,
    )
    extra_features.extend(static_cols)

    if bool(config.get('use_price_volume_distribution_features', True)):
        out, pv_cols = _add_price_volume_distribution_features(
            out,
            stock_col=stock_col,
            date_col=date_col,
        )
        extra_features.extend(pv_cols)

    out, market_cols = _add_market_sentiment_features(
        out,
        config=config,
        date_col=date_col,
    )
    extra_features.extend(market_cols)

    out, cs_cols = _add_cross_sectional_rank_features(
        out,
        config=config,
        date_col=date_col,
        industry_bucket_col=industry_bucket_col,
    )
    extra_features.extend(cs_cols)

    if industry_bucket_col and industry_bucket_col in out.columns:
        out.drop(columns=[industry_bucket_col], inplace=True)

    all_features = list(dict.fromkeys([*feature_columns, *extra_features]))
    all_features = [col for col in all_features if col in out.columns]
    if all_features:
        out[all_features] = out[all_features].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out, all_features

# 特征工程
def _rolling_linear_regression(x, y):
    x = np.vstack([np.ones(len(x)), x]).T
    beta, res, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return beta[1], res[0] if len(res) > 0 else 0.0, np.sum((y - (x @ beta))**2)
def engineer_features_158plus39(df):
    """
    计算39个技术指标特征和158个Alpha特征，并合并它们。
    """
    # 为了避免修改原始DataFrame，创建一个副本
    df_copy = df.copy()

    # 1. 计算158个Alpha特征
    df_158 = engineer_features(df_copy)
    
    # 2. 计算39个技术指标特征
    df_39 = engineer_features_39(df_copy)

    # 3. 合并两个DataFrame
    # 首先，从df_39中选取我们需要的列，避免与df_158中的原始列（如'开盘'）重复
    feature_cols_39 = [
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 
        'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 
        'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 
        'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread',
        'mom_acc', 'pv_div', 'consecutive_pos'
    ]
    
    # 确保所有列都存在于df_39中
    feature_cols_39_exist = [col for col in feature_cols_39 if col in df_39.columns]
    
    # 合并，df_158 已经包含了原始列和158个特征
    df_final = pd.concat([df_158, df_39[feature_cols_39_exist]], axis=1)

    # 4. 处理可能因为合并产生的重复列（如果两个函数生成了同名特征）
    df_final = df_final.loc[:,~df_final.columns.duplicated()]

    # 5. 统一处理inf和NaN
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.fillna(0, inplace=True)
    
    return df_final

def engineer_features_39(df):
    """
    计算39个技术指标特征。
    'stock_idx','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
    'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
    'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 
    'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  
    'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    """
    try:
        import talib
        import numpy as np
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    df = df.copy()

    # 基础变量
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)

    # 移动平均线 (SMA, EMA)
    df['sma_5'] = talib.SMA(close, timeperiod=5)
    df['sma_20'] = talib.SMA(close, timeperiod=20)
    df['ema_12'] = talib.EMA(close, timeperiod=12)
    df['ema_26'] = talib.EMA(close, timeperiod=26)
    df['ema_60'] = talib.EMA(close, timeperiod=60)

    # MACD
    macd_line, macd_signal_line, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal_line

    # RSI
    df['rsi'] = talib.RSI(close, timeperiod=14)

    # KDJ
    df['kdj_k'], df['kdj_d'] = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    # Bollinger Bands
    df['boll_mid'], df['boll_upper'], df['boll_lower'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    # 标准差 = (上轨 - 中轨) / 2
    df['boll_std'] = (df['boll_upper'] - df['boll_mid']) / 2

    # 删除临时列
    df.drop(columns=['boll_upper', 'boll_lower'], inplace=True)

    # ATR
    df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)

    # OBV (On-Balance Volume)
    df['obv'] = talib.OBV(close, volume)

    # Volume-related features
    df['volume_change'] = volume.pct_change(fill_method=None)
    df['volume_ma_5'] = talib.SMA(volume, timeperiod=5)
    df['volume_ma_20'] = talib.SMA(volume, timeperiod=20)
    df['volume_ratio'] = df['volume_ma_5'] / df['volume_ma_20']

    # Returns and Volatility
    df['return_1'] = close.pct_change(1, fill_method=None)
    df['return_5'] = close.pct_change(5, fill_method=None)
    df['return_10'] = close.pct_change(10, fill_method=None)
    df['volatility_10'] = df['return_1'].rolling(10).std()
    df['volatility_20'] = df['return_1'].rolling(20).std()

    # Spreads
    df['high_low_spread'] = high - low
    df['open_close_spread'] = open_ - close
    df['high_close_spread'] = high - close
    df['low_close_spread'] = low - close

    # 新增“高弹性/高博弈”因子
    # 动量加速因子 (Momentum Acceleration)： 过去3天的涨幅 - 过去10天的涨幅
    df['mom_acc'] = close.pct_change(3, fill_method=None) - close.pct_change(10, fill_method=None)
    
    # 量价背离极致因子： (最高价 - 开盘价) / 成交量
    df['pv_div'] = (high - open_) / (volume + 1e-12)
    
    # 连续阳线特征： 过去5天收盘价高于开盘价的天数
    df['consecutive_pos'] = (close > open_).rolling(5).sum()

    # 处理 inf 和 -inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 填充 NaN 值（注意：这可能引入偏差，根据下游任务决定是否保留）
    df.fillna(0, inplace=True)

    return df

def engineer_features(df):
    """
    使用talib加速特征计算
    """
    try:
        import talib
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    # 为了避免修改原始DataFrame，创建一个副本
    df = df.copy()

    # 基础变量
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)
    vwap = df['成交额'] / (volume + 1e-12)

    # 特征列表
    features = []
    feature_names = []

    # 1. K-line features (9 features) - 向量化操作，速度很快，无需更改
    features.extend([
        (close - open_) / (open_ + 1e-12),
        (high - low) / (open_ + 1e-12),
        (close - open_) / (high - low + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (open_ + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (high - low + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (open_ + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (high - low + 1e-12),
        (2 * close - high - low) / (open_ + 1e-12),
        (2 * close - high - low) / (high - low + 1e-12)
    ])
    feature_names.extend(['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2'])

    # 2. Price-related features (4 features) - 向量化操作，无需更改
    features.extend([
        open_ / (close + 1e-12),
        high / (close + 1e-12),
        low / (close + 1e-12),
        vwap / (close + 1e-12)
    ])
    feature_names.extend(['OPEN0', 'HIGH0', 'LOW0', 'VWAP0'])

    windows = [5, 10, 20, 30, 60]

    # 3. Price change features (5 features) - 向量化操作，无需更改
    for w in windows:
        features.append(close.shift(w) / (close + 1e-12))
        feature_names.append(f'ROC{w}')

    # 4. Moving average features (5 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.SMA(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MA{w}')

    # 5. Standard deviation features (5 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.STDDEV(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'STD{w}')

    # 6. Regression-based features (15 features) - 使用 talib 加速
    for w in windows:
        slope = talib.LINEARREG_SLOPE(close, timeperiod=w)
        features.append(slope / (close + 1e-12))
        feature_names.append(f'BETA{w}')
        
        # R-squared can be calculated as CORREL^2
        time_period_series = pd.Series(range(w), index=close.index[:w])
        rolling_corr = close.rolling(w).corr(time_period_series)
        rsquare = rolling_corr**2
        features.append(rsquare)
        feature_names.append(f'RSQR{w}')

        # Residuals
        intercept = talib.LINEARREG_INTERCEPT(close, timeperiod=w)
        predicted = slope * (w - 1) + intercept
        resi = close - predicted
        features.append(resi / (close + 1e-12))
        feature_names.append(f'RESI{w}')

    # 7. Max/Min features (10 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.MAX(high, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MAX{w}')
    for w in windows:
        features.append(talib.MIN(low, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MIN{w}')

    # 8. Quantile features (10 features) - talib 不支持，保留原实现
    for w in windows:
        features.append(close.rolling(w).quantile(0.8) / (close + 1e-12))
        feature_names.append(f'QTLU{w}')
    for w in windows:
        features.append(close.rolling(w).quantile(0.2) / (close + 1e-12))
        feature_names.append(f'QTLD{w}')

    # 9. Rank features (5 features) - talib 不支持，保留原实现
    for w in windows:
        features.append(close.rolling(w).rank(pct=True))
        feature_names.append(f'RANK{w}')

    # 10. Stochastic oscillator features (5 features) - talib.STOCH 计算的是另一指标，保留原实现
    for w in windows:
        min_low = low.rolling(w).min()
        max_high = high.rolling(w).max()
        features.append((close - min_low) / (max_high - min_low + 1e-12))
        feature_names.append(f'RSV{w}')

    # 11. Index of Max/Min features (15 features) - talib 不支持，保留原实现
    for w in windows:
        features.append(high.rolling(w).apply(np.argmax, raw=True) / w)
        feature_names.append(f'IMAX{w}')
    for w in windows:
        features.append(low.rolling(w).apply(np.argmin, raw=True) / w)
        feature_names.append(f'IMIN{w}')
    for w in windows:
        imax = high.rolling(w).apply(np.argmax, raw=True)
        imin = low.rolling(w).apply(np.argmin, raw=True)
        features.append((imax - imin) / w)
        feature_names.append(f'IMXD{w}')

    # 12. Correlation features (10 features) - 使用 talib 加速
    log_volume = np.log(volume + 1)
    for w in windows:
        features.append(talib.CORREL(close, log_volume, timeperiod=w))
        feature_names.append(f'CORR{w}')
    
    close_ret = close / close.shift(1)
    volume_ret = volume / (volume.shift(1) + 1e-12)
    log_volume_ret = np.log(volume_ret + 1)
    for w in windows:
        # talib.CORREL 需要 Series，且不能有 NaN
        corr_df = pd.concat([close_ret, log_volume_ret], axis=1).fillna(0)
        features.append(talib.CORREL(corr_df.iloc[:, 0], corr_df.iloc[:, 1], timeperiod=w))
        feature_names.append(f'CORD{w}')

    # 13. Count features (15 features) - 向量化操作，无需更改
    close_diff_pos = (close > close.shift(1))
    close_diff_neg = (close < close.shift(1))
    for w in windows:
        features.append(close_diff_pos.rolling(w).mean())
        feature_names.append(f'CNTP{w}')
    for w in windows:
        features.append(close_diff_neg.rolling(w).mean())
        feature_names.append(f'CNTN{w}')
    for w in windows:
        cntp = close_diff_pos.rolling(w).mean()
        cntn = close_diff_neg.rolling(w).mean()
        features.append(cntp - cntn)
        feature_names.append(f'CNTD{w}')

    # 14. Sum of price change features (15 features) - 向量化操作，无需更改
    close_diff_abs = (close - close.shift(1)).abs()
    close_diff_up = (close - close.shift(1)).clip(lower=0)
    close_diff_down = -(close - close.shift(1)).clip(upper=0)
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_up = close_diff_up.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f'SUMP{w}')
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_down = close_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f'SUMN{w}')
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_up = close_diff_up.rolling(w).sum()
        sum_down = close_diff_down.rolling(w).sum()
        features.append((sum_up - sum_down) / (sum_abs + 1e-12))
        feature_names.append(f'SUMD{w}')

    # 15. Volume-related features (10 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.SMA(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VMA{w}')
    for w in windows:
        features.append(talib.STDDEV(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VSTD{w}')

    # 16. Weighted volume features (5 features) - 向量化操作，无需更改
    vol_weighted_ret = (close / close.shift(1) - 1).abs() * volume
    for w in windows:
        mean_vol_w_ret = vol_weighted_ret.rolling(w).mean()
        std_vol_w_ret = vol_weighted_ret.rolling(w).std()
        features.append(std_vol_w_ret / (mean_vol_w_ret + 1e-12))
        feature_names.append(f'WVMA{w}')

    # 17. Volume change sum features (15 features) - 向量化操作，无需更改
    volume_diff_abs = (volume - volume.shift(1)).abs()
    volume_diff_up = (volume - volume.shift(1)).clip(lower=0)
    volume_diff_down = -(volume - volume.shift(1)).clip(upper=0)
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_up = volume_diff_up.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f'VSUMP{w}')
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_down = volume_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f'VSUMN{w}')
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_up = volume_diff_up.rolling(w).sum()
        sum_down = volume_diff_down.rolling(w).sum()
        features.append((sum_up - sum_down) / (sum_abs + 1e-12))
        feature_names.append(f'VSUMD{w}')

    # Combine all features into a new DataFrame
    feature_df = pd.concat(features, axis=1)
    feature_df.columns = feature_names
    
    # Merge with original df
    df = pd.concat([df, feature_df], axis=1)
    
    # 填充缺失值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df
def process_single_stock(stock_row, data, features, sequence_length, date):
    """处理单只股票的数据，返回序列、目标值和股票索引"""
    stock_code = stock_row['instrument']
    # stock_idx = stock_row['stock_idx']
    
    # 获取该股票历史sequence_length天的数据（包括当天）
    stock_history = data[
        (data['instrument'] == stock_code) & 
        (data['datetime'] <= date)
    ].sort_values('datetime').tail(sequence_length)

    if len(stock_history) == sequence_length:
        seq = stock_history[features].values
        target = stock_row['label']  # 下一天的涨跌幅
        return seq, target, stock_code
    else:
        return None, None, None

def process_single_date(date, data, features, sequence_length):
    """处理单个日期的所有股票数据"""
    try:
        # 获取当天有target的股票（即有下一天数据的股票）
        day_data = data[data['datetime'] == date]
        day_data = day_data.dropna(subset=['label'])  # 确保有target
        
        if len(day_data) < 10:  # 确保至少有10只股票
            return None
            
        # 获取当天所有股票的特征序列
        day_sequences = []
        day_targets = []
        day_stock_indices = []
        
        # 对于单个日期内的股票处理，仍使用串行方式避免过度并行化
        # 因为多进程的开销可能超过收益
        for _, stock_row in day_data.iterrows():
            seq, target, stock_idx = process_single_stock(
                stock_row, data, features, sequence_length, date
            )
            if seq is not None:
                day_sequences.append(seq)
                day_targets.append(target)
                day_stock_indices.append(stock_idx)
        
        if len(day_sequences) >= 10:  # 确保有足够的股票
            # 创建排序标签：涨跌幅越高，相关性得分越高
            day_targets = np.array(day_targets)
            # 使用涨跌幅的排序作为相关性得分（值越大排名越高）
            sorted_indices = np.argsort(day_targets)[::-1]  # 降序排列
            relevance = np.zeros_like(day_targets, dtype=np.float32)
            for rank, idx in enumerate(sorted_indices):
                relevance[idx] = len(day_targets) - rank  # 最高涨跌幅得分最高
            
            return {
                'sequences': np.array(day_sequences),
                'targets': day_targets,
                'relevance': relevance,
                'stock_indices': day_stock_indices,
                'date': date
            }
        else:
            return None
            
    except Exception as e:
        print(f"处理日期 {date} 时出错: {e}")
        return None

def create_ranking_dataset_multiprocess(data, features, sequence_length, ranking_data_path=None, max_workers=None):
    """
    输入：股票历史数据 DataFrame，特征列名列表，序列长度，排名数据保存路径，最大工作进程数
    输出：排序数据集，格式为：(sequences, targets, relevance_scores, stock_indices)
    - sequences: List of np.array, 每个元素形状为 (num_stocks, sequence_length, num_features)
    - targets: List of np.array, 每个元素形状为 (num_stocks,)
    - relevance_scores: List of np.array, 每个元素形状为 (num_stocks,)
    - stock_indices: List of List, 每个元素为对应股票的索引列表
    """
    """多进程版本的排序数据集创建函数"""
    if ranking_data_path is not None:
        # 如果指定了ranking_data_path，尝试加载已有的数据集
        if os.path.exists(ranking_data_path):
            print(f"加载已有的排序数据集: {ranking_data_path}")
            return joblib.load(ranking_data_path)
    """
    创建排序数据集，按日期组织数据，每个样本包含同一天所有股票的特征和涨跌幅排序
    使用多线程加速处理
    """
    sequences = []
    targets = []
    relevance_scores = []
    stock_indices = []
    
    print("正在创建排序数据集（多线程版本）...")
    
    # 获取所有日期，确保有足够的历史数据
    all_dates = sorted(data['datetime'].unique())
    min_date_for_sequences = all_dates[sequence_length-1]  # 确保有足够历史数据
    
    # 只使用有足够历史数据的日期
    valid_dates = [date for date in all_dates if date >= min_date_for_sequences]
    
    print(f"总日期数: {len(all_dates)}, 有效日期数: {len(valid_dates)}")
    
    # 设置最大工作进程数
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    from tqdm import tqdm
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 10)
    
    print(f"使用 {max_workers} 个进程处理数据")
    
    # 分批处理日期以避免内存问题
    processed_count = 0
        
    # 使用进程池并行处理日期批次
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 创建处理函数的偏函数
            process_func = partial(process_single_date,
                                    data=data,
                                    features=features,
                                    sequence_length=sequence_length)
            
            # 并行处理批次中的所有日期
            futures = [executor.submit(process_func, date) for date in valid_dates]
            
            # 收集结果
            for future in tqdm(futures, desc="Processing dates", total=len(valid_dates)):
                try:
                    result = future.result(timeout=60)  # 设置超时
                    if result is not None:
                        sequences.append(result['sequences'])
                        targets.append(result['targets'])
                        relevance_scores.append(result['relevance'])
                        stock_indices.append(result['stock_indices'])
                        processed_count += 1
                except Exception as e:
                    print(f"处理某个日期时出错: {e}")
                    continue
                    
    except Exception as e:
        print(f"进程池处理出错，回退到串行处理: {e}")
        # 如果多进程出错，回退到串行处理
        for date in tqdm(valid_dates, desc="串行处理"):
            result = process_single_date(date, data, features, sequence_length)
            if result is not None:
                sequences.append(result['sequences'])
                targets.append(result['targets'])
                relevance_scores.append(result['relevance'])
                stock_indices.append(result['stock_indices'])
                processed_count += 1
    
    print(f"成功创建 {len(sequences)} 个训练样本")
    if len(sequences) > 0:
        print(f"每个样本平均包含 {np.mean([len(seq) for seq in sequences]):.1f} 只股票")
    
    # 将四个数据保存下来，下次直接读取
    if ranking_data_path:
        joblib.dump((sequences, targets, relevance_scores, stock_indices), ranking_data_path)
        print(f"数据集已保存到: {ranking_data_path}")
    
    return sequences, targets, relevance_scores, stock_indices

def create_dataset(data, features, sequence_length, ranking_data_path=None):
    """保持原有接口，但内部调用新的排序数据集创建函数"""
    return create_ranking_dataset_multiprocess(data, features, sequence_length, ranking_data_path)

def create_ranking_dataset_vectorized(
    data,
    features,
    sequence_length,
    ranking_data_path=None,
    min_window_end_date=None,
    max_window_end_date=None,
):
    """
    向量化加速版本：预计算每只股票的所有滑动窗口，再按日期聚合。
    保持与原函数完全相同的输出格式。
    """
    # if ranking_data_path and os.path.exists(ranking_data_path):
    #     print(f"加载已有的排序数据集: {ranking_data_path}")
    #     return joblib.load(ranking_data_path)

    print("正在创建排序数据集（向量化加速版本）...")
    # data.rename(columns={'stock_idx': 'instrument'}, inplace=True)
    data = data.copy()
    data.rename(columns={'日期': 'datetime'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])

    # 1. 确保数据按股票和时间排序
    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    
    # 2. 确保每只股票都有 'label'（次日涨跌幅），否则无法作为 target
    has_vol_label = 'vol_label' in data.columns
    required_cols = ['label'] + (['vol_label'] if has_vol_label else [])
    data = data.dropna(subset=required_cols)
    
    # 3. 为每只股票生成所有滑动窗口
    # 仅保留满足以下条件的 end_date：
    # - 历史窗口长度满足 sequence_length
    #
    # 注意：data 在进入这里之前已经按 label 做过 dropna。
    # 也就是说，保留下来的每一行都已经满足
    # T+1 开盘买入到 T+5 开盘卖出的收益率可计算，
    # 这里不应再额外要求未来 5 个交易日，否则会重复缩短样本。
    all_windows = []  # 每个元素: (end_date, stock_code, sequence, target, vol_target)

    print("Step 1: 为每只股票生成滑动窗口...")
    grouped = data.groupby('instrument')
    
    for stock_code, group in tqdm(grouped, desc="Processing stocks"):
        if len(group) < sequence_length:
            continue
        
        # 提取特征和 label
        feature_values = group[features].values.astype(np.float32)  # (T, F)
        labels = group['label'].values.astype(np.float32)           # (T,)
        if has_vol_label:
            vol_labels = group['vol_label'].values.astype(np.float32)
        else:
            vol_labels = labels.copy()
        dates = group['datetime'].values                            # (T,)

        # 生成滑动窗口：从第 sequence_length-1 行开始（0-indexed）
        num_windows = len(group) - sequence_length + 1
        for i in range(num_windows):
            end_idx = i + sequence_length - 1

            seq = feature_values[i : i + sequence_length]   # (L, F)
            target = labels[end_idx]                        # label 对应窗口最后一天的次日涨跌幅
            vol_target = vol_labels[end_idx]                # 波动率辅助标签
            end_date = dates[end_idx]                       # 窗口结束日期（即预测日）
            all_windows.append((end_date, stock_code, seq, target, vol_target))

    # 4. 转为 DataFrame 便于按日期聚合
    print("Step 2: 按日期聚合窗口...")
    window_df = pd.DataFrame(all_windows, columns=['date', 'stock_code', 'seq', 'target', 'vol_target'])

    # 5. 按 date 分组，构建每日样本
    sequences = []
    targets = []
    vol_targets = []
    relevance_scores = []
    stock_indices = []

    print("Step 3: 构建每日样本并计算 relevance...")
    grouped_by_date = window_df.groupby('date')

    if min_window_end_date is not None:
        min_window_end_date = pd.to_datetime(min_window_end_date)
    if max_window_end_date is not None:
        max_window_end_date = pd.to_datetime(max_window_end_date)
    
    for date, group in tqdm(grouped_by_date, desc="Aggregating by date"):
        if min_window_end_date is not None and pd.to_datetime(date) < min_window_end_date:
            continue
        if max_window_end_date is not None and pd.to_datetime(date) > max_window_end_date:
            continue

        if len(group) < 10:
            continue
        
        # 提取数据
        day_seqs = np.stack(group['seq'].values)          # (N, L, F)
        day_targets = group['target'].values              # (N,)
        day_vol_targets = group['vol_target'].values      # (N,)
        day_stocks = group['stock_code'].tolist()         # [str]

        # 计算 relevance（新逻辑：是否属于全市场前 2% 的妖股）
        threshold_2pct = np.quantile(day_targets, 0.98)
        relevance = (day_targets >= threshold_2pct).astype(np.float32)

        sequences.append(day_seqs)
        targets.append(day_targets)
        vol_targets.append(day_vol_targets)
        relevance_scores.append(relevance)
        stock_indices.append(day_stocks)

    print(f"成功创建 {len(sequences)} 个训练样本")
    if len(sequences) > 0:
        avg_stocks = np.mean([len(seq) for seq in sequences])
        print(f"每个样本平均包含 {avg_stocks:.1f} 只股票")

    # 6. 保存
    # if ranking_data_path:
    #     joblib.dump((sequences, targets, relevance_scores, stock_indices), ranking_data_path)
    #     print(f"数据集已保存到: {ranking_data_path}")

    return sequences, targets, relevance_scores, stock_indices, vol_targets
