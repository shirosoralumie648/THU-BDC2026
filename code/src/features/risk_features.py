from __future__ import annotations

from utils import _add_price_volume_distribution_features
from utils import apply_cross_sectional_normalization

__all__ = ['add_risk_feature_blocks', 'normalize_feature_table']


def add_risk_feature_blocks(
    df,
    feature_columns,
    runtime_config,
    date_col='日期',
    stock_col='股票代码',
):
    out = df.copy()
    active_features = list(feature_columns or [])
    if bool(runtime_config.get('use_price_volume_distribution_features', True)):
        out, risk_cols = _add_price_volume_distribution_features(
            out,
            stock_col=stock_col,
            date_col=date_col,
        )
        active_features = list(dict.fromkeys([*active_features, *risk_cols]))
    active_features = [col for col in active_features if col in out.columns]
    return out, active_features


def normalize_feature_table(df, feature_columns, runtime_config, date_col='日期'):
    if not runtime_config.get('use_cross_sectional_feature_norm', True):
        return df
    cs_exclude = [col for col in feature_columns if col.startswith('market_')]
    return apply_cross_sectional_normalization(
        df,
        feature_columns,
        date_col=date_col,
        method=runtime_config.get('feature_cs_norm_method', 'zscore'),
        clip_value=runtime_config.get('feature_cs_clip_value', None),
        exclude_columns=cs_exclude,
    )
