import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def infer_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_stock_code_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.split('.').str[-1]
    s = s.str.replace(r'[^0-9]', '', regex=True)
    s = s.str[-6:].str.zfill(6)
    return s


def resolve_data_root(config: Dict) -> str:
    return str(config.get('data_path', './data')).strip() or './data'


def resolve_dataset_path(config: Dict, filename: str) -> str:
    data_root = os.path.abspath(resolve_data_root(config))
    return os.path.join(data_root, filename)


def load_market_dataset(
    config: Dict,
    filename: str = 'train.csv',
    *,
    dtype: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, str]:
    dataset_path = resolve_dataset_path(config, filename)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f'未找到数据文件: {dataset_path}')
    return pd.read_csv(dataset_path, dtype=dtype), dataset_path


def resolve_industry_mapping_path(config: Dict) -> str:
    candidates = [
        str(config.get('prior_graph_industry_map_path', '')).strip(),
        str(config.get('label_industry_map_path', '')).strip(),
        str(config.get('stock_static_feature_path', '')).strip(),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ''


def load_stock_to_industry_map(
    config: Dict,
    *,
    stock_col_key: str = 'prior_graph_stock_col',
    industry_col_key: str = 'prior_graph_industry_col',
    mapping_path: Optional[str] = None,
) -> Dict[str, str]:
    source_path = str(mapping_path or '').strip()
    if not source_path:
        source_path = resolve_industry_mapping_path(config)
    if not source_path or not os.path.exists(source_path):
        return {}

    mapping_df = pd.read_csv(source_path, dtype=str)
    stock_col = str(config.get(stock_col_key, '股票代码')).strip()
    industry_col = str(config.get(industry_col_key, '行业')).strip()

    if stock_col not in mapping_df.columns:
        stock_col = infer_existing_column(mapping_df, ['股票代码', 'stock_id', 'code', 'ts_code'])
        if stock_col is None:
            return {}
    if industry_col not in mapping_df.columns:
        industry_col = infer_existing_column(mapping_df, ['行业', 'industry', 'sw_l1', 'sector'])
        if industry_col is None:
            return {}

    mapping_df = mapping_df[[stock_col, industry_col]].copy()
    mapping_df[stock_col] = normalize_stock_code_series(mapping_df[stock_col])
    mapping_df[industry_col] = mapping_df[industry_col].astype(str).str.strip()
    mapping_df = mapping_df.dropna(subset=[stock_col, industry_col])
    mapping_df = mapping_df[mapping_df[industry_col] != '']
    if mapping_df.empty:
        return {}

    return (
        mapping_df
        .drop_duplicates(subset=[stock_col], keep='last')
        .set_index(stock_col)[industry_col]
        .to_dict()
    )


def build_stock_industry_index(
    stock_ids: List[str],
    stock_to_industry: Dict[str, str],
) -> Tuple[np.ndarray, List[str], int]:
    num_stocks = len(stock_ids)
    stock_industry_idx = np.full(num_stocks, -1, dtype=np.int64)
    if num_stocks <= 0 or not stock_to_industry:
        return stock_industry_idx, [], 0

    normalized_codes = normalize_stock_code_series(pd.Series(stock_ids)).tolist()
    industries = []
    for normalized in normalized_codes:
        industry = stock_to_industry.get(normalized, None)
        if industry:
            industries.append(industry)
    if not industries:
        return stock_industry_idx, [], 0

    industry_vocab = sorted(set(industries))
    industry2idx = {industry: idx for idx, industry in enumerate(industry_vocab)}

    matched = 0
    for idx, normalized in enumerate(normalized_codes):
        industry = stock_to_industry.get(normalized, None)
        if not industry:
            continue
        stock_industry_idx[idx] = int(industry2idx[industry])
        matched += 1
    return stock_industry_idx, industry_vocab, matched


def collect_data_sources(config: Dict, *, include_csv_stats: bool = False) -> Dict:
    data_root = os.path.abspath(resolve_data_root(config))
    train_csv = resolve_dataset_path(config, 'train.csv')
    test_csv = resolve_dataset_path(config, 'test.csv')
    stock_data_csv = resolve_dataset_path(config, 'stock_data.csv')
    industry_map = resolve_industry_mapping_path(config)
    benchmark_path = str(config.get('label_benchmark_return_path', '')).strip()
    static_feature_path = str(config.get('stock_static_feature_path', '')).strip()

    return {
        'data_root': data_root,
        'train_csv': build_file_snapshot(train_csv, inspect_csv=include_csv_stats),
        'test_csv': build_file_snapshot(test_csv, inspect_csv=include_csv_stats),
        'stock_data_csv': build_file_snapshot(stock_data_csv, inspect_csv=include_csv_stats),
        'industry_mapping': build_file_snapshot(industry_map),
        'benchmark': build_file_snapshot(benchmark_path),
        'static_feature': build_file_snapshot(static_feature_path),
    }


def inspect_csv_metadata(
    path: str,
    *,
    date_col_candidates: Optional[Iterable[str]] = None,
    stock_col_candidates: Optional[Iterable[str]] = None,
) -> Dict:
    if not path or not os.path.exists(path):
        return {}

    date_candidates = list(date_col_candidates or ['日期', 'date', 'datetime', 'trade_date'])
    stock_candidates = list(stock_col_candidates or ['股票代码', 'stock_id', 'code', 'ts_code'])
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return {'error': f'csv_parse_error: {exc}'}

    info = {
        'row_count': int(len(df)),
        'column_count': int(len(df.columns)),
    }

    date_col = infer_existing_column(df, date_candidates)
    if date_col is not None:
        date_values = pd.to_datetime(df[date_col], errors='coerce')
        date_values = date_values.dropna()
        if not date_values.empty:
            info['date_column'] = date_col
            info['date_min'] = str(date_values.min().date())
            info['date_max'] = str(date_values.max().date())

    stock_col = infer_existing_column(df, stock_candidates)
    if stock_col is not None:
        stock_norm = normalize_stock_code_series(df[stock_col]).replace('', np.nan).dropna()
        if not stock_norm.empty:
            info['stock_column'] = stock_col
            info['stock_count'] = int(stock_norm.nunique())

    return info


def build_file_snapshot(
    path: str,
    *,
    inspect_csv: bool = False,
    date_col_candidates: Optional[Iterable[str]] = None,
    stock_col_candidates: Optional[Iterable[str]] = None,
) -> Dict:
    target_path = str(path or '').strip()
    if not target_path:
        return {'path': '', 'exists': False}

    abs_path = os.path.abspath(target_path)
    exists = os.path.exists(abs_path)
    snapshot = {'path': abs_path, 'exists': bool(exists)}
    if not exists:
        return snapshot

    try:
        snapshot['size_bytes'] = int(os.path.getsize(abs_path))
    except Exception:
        pass

    if inspect_csv and abs_path.lower().endswith('.csv'):
        csv_meta = inspect_csv_metadata(
            abs_path,
            date_col_candidates=date_col_candidates,
            stock_col_candidates=stock_col_candidates,
        )
        if csv_meta:
            snapshot['csv'] = csv_meta
    return snapshot


def save_data_manifest(output_dir: str, manifest: Dict, filename: str = 'data_manifest.json') -> str:
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, filename)
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return target_path
