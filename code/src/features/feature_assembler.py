from __future__ import annotations

from factor_store import apply_factor_expressions
from factor_store import engineer_group_features

from features.daily_features import build_daily_feature_table
from features.intraday_features import add_intraday_feature_blocks
from features.relative_features import add_relative_feature_blocks
from features.risk_features import add_risk_feature_blocks
from features.risk_features import normalize_feature_table

__all__ = ['augment_feature_table', 'build_feature_table']


def _resolve_time_series_specs(feature_pipeline):
    if not feature_pipeline:
        return []
    time_series_specs = feature_pipeline.get('time_series_specs')
    if time_series_specs is not None:
        return time_series_specs

    legacy_specs = [
        spec for spec in feature_pipeline.get('builtin_override_specs', [])
        if spec.get('overridden')
    ]
    legacy_specs.extend(feature_pipeline.get('custom_specs', []))
    return legacy_specs


def build_feature_table(df, feature_set: str, runtime_config=None, feature_pipeline=None):
    del runtime_config
    feature_set = str(feature_set or '').strip()
    if feature_pipeline:
        return engineer_group_features(
            (df, feature_set, _resolve_time_series_specs(feature_pipeline))
        )
    return build_daily_feature_table(df, feature_set)


def augment_feature_table(
    df,
    feature_columns,
    runtime_config=None,
    feature_pipeline=None,
    date_col='日期',
    stock_col='股票代码',
    apply_factor_pipeline=True,
    apply_feature_enhancements=True,
    apply_cross_sectional_norm=True,
):
    runtime = runtime_config or {}
    out = df.copy()
    active_features = list(feature_columns or [])

    cross_sectional_specs = []
    if feature_pipeline and apply_factor_pipeline:
        cross_sectional_specs = feature_pipeline.get('cross_sectional_specs', [])
    if cross_sectional_specs:
        out = out.sort_values([date_col, stock_col]).reset_index(drop=True)
        out = apply_factor_expressions(
            out,
            cross_sectional_specs,
            error_prefix='截面因子',
            date_col=date_col,
        )

    if apply_feature_enhancements and bool(runtime.get('use_feature_enhancements', True)):
        out, active_features = add_intraday_feature_blocks(
            out,
            active_features,
            runtime,
            date_col=date_col,
            stock_col=stock_col,
        )
        out, active_features = add_relative_feature_blocks(
            out,
            active_features,
            runtime,
            date_col=date_col,
            stock_col=stock_col,
        )
        out, active_features = add_risk_feature_blocks(
            out,
            active_features,
            runtime,
            date_col=date_col,
            stock_col=stock_col,
        )

    if apply_cross_sectional_norm:
        out = normalize_feature_table(
            out,
            active_features,
            runtime,
            date_col=date_col,
        )

    return out, active_features
