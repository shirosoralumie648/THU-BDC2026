from __future__ import annotations

import numpy as np
import pandas as pd

from config import config
from data_manager import build_stock_industry_index as build_stock_industry_index_from_manager
from data_manager import load_stock_to_industry_map
from data_manager import resolve_industry_mapping_path


def normalize_stock_code_series(series):
    s = series.astype(str).str.strip()
    s = s.str.split('.').str[-1]
    s = s.str.replace(r'[^0-9]', '', regex=True)
    s = s.str[-6:].str.zfill(6)
    return s


def load_prior_graph_industry_mapping(runtime_config):
    mapping_path = resolve_industry_mapping_path(runtime_config)
    if not mapping_path:
        return {}
    mapping = load_stock_to_industry_map(
        runtime_config,
        stock_col_key='prior_graph_stock_col',
        industry_col_key='prior_graph_industry_col',
        mapping_path=mapping_path,
    )
    return mapping if mapping else {}


def build_industry_prior_adjacency(stockid2idx, runtime_config=None):
    runtime_config = runtime_config or config
    num_stocks = len(stockid2idx)
    if not bool(runtime_config.get('prior_graph_use_industry', True)):
        return np.zeros((num_stocks, num_stocks), dtype=bool)

    stock_to_industry = load_prior_graph_industry_mapping(runtime_config)
    if not stock_to_industry:
        return np.zeros((num_stocks, num_stocks), dtype=bool)

    industry_to_indices = {}
    stock_codes = list(stockid2idx.keys())
    normalized_codes = normalize_stock_code_series(pd.Series(stock_codes)).tolist()
    for stock_code, normalized in zip(stock_codes, normalized_codes):
        idx = stockid2idx[stock_code]
        industry = stock_to_industry.get(normalized, None)
        if not industry:
            continue
        industry_to_indices.setdefault(industry, []).append(int(idx))

    adj = np.zeros((num_stocks, num_stocks), dtype=bool)
    for indices in industry_to_indices.values():
        if len(indices) <= 1:
            continue
        idx_arr = np.asarray(indices, dtype=np.int64)
        adj[np.ix_(idx_arr, idx_arr)] = True
    return adj


def build_stock_industry_index(stockid2idx, runtime_config=None):
    runtime_config = runtime_config or config
    num_stocks = len(stockid2idx)
    stock_industry_idx = np.full(num_stocks, -1, dtype=np.int64)
    if num_stocks <= 0:
        return stock_industry_idx, []

    stock_to_industry = load_prior_graph_industry_mapping(runtime_config)
    if not stock_to_industry:
        print('未加载到行业映射，行业虚拟股将回退为关闭状态。')
        return stock_industry_idx, []

    stock_codes = list(stockid2idx.keys())
    stock_industry_idx, industry_vocab, matched = build_stock_industry_index_from_manager(
        stock_codes,
        stock_to_industry,
    )
    if not industry_vocab:
        print('行业映射为空，行业虚拟股将回退为关闭状态。')
        return np.full(num_stocks, -1, dtype=np.int64), []

    coverage = matched / float(max(1, num_stocks))
    print(
        f'行业索引构建完成: stocks={num_stocks}, matched={matched}, '
        f'coverage={coverage:.2%}, industries={len(industry_vocab)}'
    )
    return stock_industry_idx.astype(np.int64), industry_vocab
