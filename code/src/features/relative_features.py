from __future__ import annotations

from utils import _add_cross_sectional_rank_features
from utils import _attach_static_stock_features

__all__ = ['add_relative_feature_blocks']


def add_relative_feature_blocks(
    df,
    feature_columns,
    runtime_config,
    date_col='日期',
    stock_col='股票代码',
):
    out = df.copy()
    active_features = list(feature_columns or [])
    out, static_cols, industry_bucket_col = _attach_static_stock_features(
        out,
        config=runtime_config,
        stock_col=stock_col,
        date_col=date_col,
    )
    out, relative_cols = _add_cross_sectional_rank_features(
        out,
        config=runtime_config,
        date_col=date_col,
        industry_bucket_col=industry_bucket_col,
    )
    if industry_bucket_col and industry_bucket_col in out.columns:
        out = out.drop(columns=[industry_bucket_col])

    active_features = list(dict.fromkeys([*active_features, *static_cols, *relative_cols]))
    active_features = [col for col in active_features if col in out.columns]
    return out, active_features
