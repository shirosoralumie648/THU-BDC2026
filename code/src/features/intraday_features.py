from __future__ import annotations

from utils import _add_market_sentiment_features

__all__ = ['add_intraday_feature_blocks']


def add_intraday_feature_blocks(
    df,
    feature_columns,
    runtime_config,
    date_col='日期',
    stock_col='股票代码',
):
    del stock_col
    out = df.copy()
    active_features = list(feature_columns or [])
    out, intraday_cols = _add_market_sentiment_features(
        out,
        config=runtime_config,
        date_col=date_col,
    )
    active_features = list(dict.fromkeys([*active_features, *intraday_cols]))
    active_features = [col for col in active_features if col in out.columns]
    return out, active_features
