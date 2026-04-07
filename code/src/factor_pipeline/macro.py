from typing import Optional

import numpy as np
import pandas as pd


def _build_macro_cutoff_frame(daily_dates: pd.Series) -> pd.DataFrame:
    left = pd.DataFrame({'trade_date': pd.to_datetime(daily_dates).dropna().drop_duplicates().sort_values()})
    if left.empty:
        return left
    # 使用交易日收盘 15:00 作为可用性截止时间
    left['__cutoff_ts'] = left['trade_date'] + pd.Timedelta(hours=15)
    return left.reset_index(drop=True)


def _compute_macro_series_asof(
    macro_cutoff_frame: pd.DataFrame,
    macro_series_df: Optional[pd.DataFrame],
    *,
    max_staleness_days: Optional[int],
    fill_method: str,
) -> pd.DataFrame:
    if macro_cutoff_frame is None or macro_cutoff_frame.empty:
        return pd.DataFrame(columns=['trade_date', 'value'])
    left = macro_cutoff_frame[['trade_date', '__cutoff_ts']].copy()

    if macro_series_df is None:
        right = pd.DataFrame(columns=['available_time', 'value'])
    else:
        right = macro_series_df[['available_time', 'value']].copy()
    if right.empty:
        out = left[['trade_date']].copy()
        out['value'] = np.nan
        return out

    merged = pd.merge_asof(
        left.sort_values('__cutoff_ts'),
        right[['available_time', 'value']].sort_values('available_time'),
        left_on='__cutoff_ts',
        right_on='available_time',
        direction='backward',
    )

    stale_mask = None
    if max_staleness_days is not None and int(max_staleness_days) > 0:
        delta_days = (merged['__cutoff_ts'] - merged['available_time']).dt.total_seconds() / 86400.0
        stale_mask = delta_days > float(max_staleness_days)
        merged.loc[stale_mask, 'value'] = np.nan

    fill_method = str(fill_method or '').strip().lower()
    if fill_method in {'forward', 'ffill'}:
        merged['value'] = merged['value'].ffill()
        if stale_mask is not None:
            # 过期值可以在有效区间内参与补值，但不能穿透 staleness 边界重新“复活”。
            merged.loc[stale_mask, 'value'] = np.nan

    return merged[['trade_date', 'value']].copy()
