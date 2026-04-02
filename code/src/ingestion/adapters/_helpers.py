from __future__ import annotations

from typing import Iterable


def require_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError('pandas is required for ingestion adapters') from exc
    return pd


def as_dataframe(data):
    pd = require_pandas()
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)


def empty_frame(columns: Iterable[str]):
    pd = require_pandas()
    return pd.DataFrame(columns=list(columns))


def normalize_instrument_id(value) -> str:
    if value is None:
        return ''
    text = str(value).strip()
    if not text:
        return ''
    if '.' in text:
        text = text.split('.')[-1]
    digits = ''.join(ch for ch in text if ch.isdigit())
    if not digits:
        return ''
    return digits[-6:].zfill(6)


def to_provider_symbol(value) -> str:
    text = normalize_instrument_id(value)
    if not text:
        return str(value).strip()
    if text.startswith(('4', '8')):
        prefix = 'bj'
    elif text.startswith(('5', '6', '9')):
        prefix = 'sh'
    else:
        prefix = 'sz'
    return f'{prefix}.{text}'


def first_present(frame, names: Iterable[str]):
    for name in names:
        if name in frame.columns:
            return frame[name]
    return None


def normalize_date_text(series) -> list[str]:
    pd = require_pandas()
    if series is None:
        return []
    parsed = pd.to_datetime(series, errors='coerce')
    return parsed.dt.strftime('%Y-%m-%d').fillna('').tolist()


def normalize_timestamp_text(series) -> list[str]:
    pd = require_pandas()
    if series is None:
        return []
    parsed = pd.to_datetime(series, errors='coerce')
    return parsed.dt.strftime('%Y-%m-%d %H:%M:%S').fillna('').tolist()


def to_numeric_series(series):
    pd = require_pandas()
    if series is None:
        return pd.Series(dtype='float64')
    return pd.to_numeric(series, errors='coerce')
