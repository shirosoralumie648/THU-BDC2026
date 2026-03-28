import importlib
import os

import pandas as pd


def _resolve_column(df: pd.DataFrame, candidates) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f'macro input missing candidate columns: {list(candidates)}')


def _find_column_by_keywords(df: pd.DataFrame, *, candidates=None, keywords=None):
    candidates = list(candidates or [])
    keywords = [str(item).lower() for item in list(keywords or []) if str(item).strip()]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        name = str(col).lower()
        if keywords and all(keyword in name for keyword in keywords):
            return col
    return None


def _month_release_ts(series: pd.Series, day: int) -> pd.Series:
    period = pd.to_datetime(series, errors='coerce').dt.to_period('M') + 1
    return period.dt.to_timestamp() + pd.Timedelta(days=max(0, int(day) - 1), hours=9)


def _daily_release_ts(series: pd.Series, hour: int, minute: int = 0) -> pd.Series:
    return pd.to_datetime(series, errors='coerce').dt.normalize() + pd.Timedelta(hours=hour, minutes=minute)


def _finalize_macro_frame(
    *,
    series_id: str,
    observation_date,
    value,
    frequency: str,
    release_time,
    available_time=None,
):
    observation_series = pd.Series(pd.to_datetime(observation_date, errors='coerce'))
    row_count = int(len(observation_series))
    out = pd.DataFrame(
        {
            'series_id': [str(series_id)] * row_count,
            'observation_date': observation_series,
            'release_time': pd.Series(pd.to_datetime(release_time, errors='coerce')),
            'available_time': pd.Series(
                pd.to_datetime(available_time if available_time is not None else release_time, errors='coerce')
            ),
            'frequency': [str(frequency)] * row_count,
            'vintage': ['latest'] * row_count,
            'value': pd.Series(pd.to_numeric(value, errors='coerce')),
        }
    )
    out = out.dropna(subset=['series_id', 'observation_date', 'release_time', 'available_time', 'value']).copy()
    return out.reset_index(drop=True)


def _load_m2_yoy(ak):
    if hasattr(ak, 'macro_china_m2_yearly'):
        df = ak.macro_china_m2_yearly()
    elif hasattr(ak, 'macro_china_money_supply'):
        df = ak.macro_china_money_supply()
    else:
        raise AttributeError('akshare missing M2 series interface')
    date_col = _resolve_column(df, ['月份', '日期', 'date'])
    value_col = _find_column_by_keywords(
        df,
        candidates=['货币和准货币(M2)-同比增长', 'M2-同比增长', 'M2同比', '今值'],
        keywords=['m2', '同比'],
    )
    if value_col is None:
        raise ValueError('M2 series missing value column')
    obs = pd.to_datetime(df[date_col], errors='coerce').dt.to_period('M').dt.to_timestamp('M')
    release = _month_release_ts(obs, day=12)
    return _finalize_macro_frame(
        series_id='m2_yoy',
        observation_date=obs,
        value=df[value_col],
        frequency='monthly',
        release_time=release,
    )


def _load_cpi_yoy(ak):
    if not hasattr(ak, 'macro_china_cpi_yearly'):
        raise AttributeError('akshare missing CPI yearly interface')
    df = ak.macro_china_cpi_yearly()
    date_col = _resolve_column(df, ['月份', '日期', 'date'])
    value_col = _find_column_by_keywords(
        df,
        candidates=['全国-同比增长', '同比增长', '今值'],
        keywords=['同比'],
    )
    if value_col is None:
        raise ValueError('CPI series missing value column')
    obs = pd.to_datetime(df[date_col], errors='coerce').dt.to_period('M').dt.to_timestamp('M')
    release = _month_release_ts(obs, day=9)
    return _finalize_macro_frame(
        series_id='cpi_yoy',
        observation_date=obs,
        value=df[value_col],
        frequency='monthly',
        release_time=release,
    )


def _load_ppi_yoy(ak):
    if not hasattr(ak, 'macro_china_ppi_yearly'):
        raise AttributeError('akshare missing PPI yearly interface')
    df = ak.macro_china_ppi_yearly()
    date_col = _resolve_column(df, ['月份', '日期', 'date'])
    value_col = _find_column_by_keywords(
        df,
        candidates=['全部工业品:当月同比', '当月同比', '同比增长', '今值'],
        keywords=['同比'],
    )
    if value_col is None:
        raise ValueError('PPI series missing value column')
    obs = pd.to_datetime(df[date_col], errors='coerce').dt.to_period('M').dt.to_timestamp('M')
    release = _month_release_ts(obs, day=9)
    return _finalize_macro_frame(
        series_id='ppi_yoy',
        observation_date=obs,
        value=df[value_col],
        frequency='monthly',
        release_time=release,
    )


def _load_shibor_3m(ak):
    if not hasattr(ak, 'macro_china_shibor_all'):
        raise AttributeError('akshare missing Shibor interface')
    df = ak.macro_china_shibor_all()
    date_col = _resolve_column(df, ['日期', 'date'])
    value_col = _find_column_by_keywords(
        df,
        candidates=['3M', '3M-定价', '3M-均值', '3M-涨跌'],
        keywords=['3m'],
    )
    if value_col is None:
        raise ValueError('Shibor 3M series missing value column')
    obs = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
    release = _daily_release_ts(obs, hour=11)
    return _finalize_macro_frame(
        series_id='shibor_3m',
        observation_date=obs,
        value=df[value_col],
        frequency='daily',
        release_time=release,
    )


def _load_usdcny(ak):
    if hasattr(ak, 'macro_china_rmb'):
        df = ak.macro_china_rmb()
        date_col = _resolve_column(df, ['日期', 'date'])
        value_col = _find_column_by_keywords(
            df,
            candidates=['美元/人民币', '美元对人民币', '美元兑人民币', '今值'],
            keywords=['美元', '人民币'],
        )
    elif hasattr(ak, 'currency_boc_safe'):
        df = ak.currency_boc_safe()
        date_col = _resolve_column(df, ['日期', 'date'])
        value_col = _find_column_by_keywords(df, candidates=['美元'], keywords=['美元'])
    else:
        raise AttributeError('akshare missing RMB FX interface')
    if value_col is None:
        raise ValueError('USD/CNY series missing value column')
    values = pd.to_numeric(df[value_col], errors='coerce')
    if values.dropna().gt(50).all():
        values = values / 100.0
    obs = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
    release = _daily_release_ts(obs, hour=16, minute=30)
    return _finalize_macro_frame(
        series_id='usdcny',
        observation_date=obs,
        value=values,
        frequency='daily',
        release_time=release,
    )


def _load_csi300_pe_ttm(ak):
    if not hasattr(ak, 'stock_index_pe_lg'):
        raise AttributeError('akshare missing index PE interface')
    df = ak.stock_index_pe_lg(symbol='沪深300')
    date_col = _resolve_column(df, ['日期', 'date'])
    value_col = _find_column_by_keywords(
        df,
        candidates=['滚动市盈率', '市盈率TTM', '市盈率', '平均市盈率'],
        keywords=['市盈'],
    )
    if value_col is None:
        raise ValueError('CSI300 PE series missing value column')
    obs = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
    release = _daily_release_ts(obs, hour=15, minute=5)
    return _finalize_macro_frame(
        series_id='csi300_pe_ttm',
        observation_date=obs,
        value=df[value_col],
        frequency='daily',
        release_time=release,
    )


MACRO_SERIES_LOADERS = {
    'cpi_yoy': _load_cpi_yoy,
    'ppi_yoy': _load_ppi_yoy,
    'm2_yoy': _load_m2_yoy,
    'shibor_3m': _load_shibor_3m,
    'usdcny': _load_usdcny,
    'csi300_pe_ttm': _load_csi300_pe_ttm,
}


class AkshareMacroAdapter:
    adapter_name = 'macro_timeseries'

    def _resolve_input_path(self, request, spec) -> str:
        extra = request.extra if isinstance(request.extra, dict) else {}
        source_params = spec.request_spec or {}
        return str(extra.get('input_path', '') or source_params.get('input_path', '')).strip()

    def _load_file_input(self, input_path: str) -> pd.DataFrame:
        if not input_path or (not os.path.exists(input_path)):
            return pd.DataFrame(columns=['series_id', 'observation_date', 'release_time', 'available_time', 'frequency', 'vintage', 'value'])

        df = pd.read_csv(input_path)
        rename_map = {
            '日期': 'observation_date',
            'date': 'observation_date',
            'release_time': 'available_time',
            'value': 'value',
        }
        out = df.rename(columns=rename_map).copy()
        required = ['series_id', 'available_time', 'value']
        for col in required:
            if col not in out.columns:
                raise ValueError(f'macro input missing required column: {col}')
        if 'observation_date' not in out.columns:
            out['observation_date'] = out['available_time']
        if 'release_time' not in out.columns:
            out['release_time'] = out['available_time']
        if 'frequency' not in out.columns:
            out['frequency'] = ''
        if 'vintage' not in out.columns:
            out['vintage'] = 'latest'
        out['series_id'] = out['series_id'].astype(str).str.strip()
        out['observation_date'] = pd.to_datetime(out['observation_date'], errors='coerce')
        out['release_time'] = pd.to_datetime(out['release_time'], errors='coerce')
        out['available_time'] = pd.to_datetime(out['available_time'], errors='coerce')
        out['value'] = pd.to_numeric(out['value'], errors='coerce')
        return out.dropna(subset=['series_id', 'observation_date', 'release_time', 'available_time']).reset_index(drop=True)

    def _resolve_series_catalog(self, request, spec):
        extra = request.extra if isinstance(request.extra, dict) else {}
        if extra.get('series_catalog'):
            return [str(item).strip() for item in list(extra.get('series_catalog', [])) if str(item).strip()]
        request_spec = spec.request_spec or {}
        return [str(item).strip() for item in list(request_spec.get('series_catalog', [])) if str(item).strip()]

    def fetch(self, request, spec):
        input_path = self._resolve_input_path(request, spec)
        if input_path:
            return self._load_file_input(input_path)

        try:
            ak = importlib.import_module('akshare')
        except Exception:
            return pd.DataFrame(columns=['series_id', 'observation_date', 'release_time', 'available_time', 'frequency', 'vintage', 'value'])

        series_catalog = self._resolve_series_catalog(request, spec)
        frames = []
        errors = []
        for series_id in series_catalog:
            loader = MACRO_SERIES_LOADERS.get(series_id)
            if loader is None:
                errors.append(f'unsupported series: {series_id}')
                continue
            try:
                part = loader(ak)
            except Exception as exc:
                errors.append(f'{series_id}: {exc}')
                continue
            if not part.empty:
                frames.append(part)

        if frames:
            out = pd.concat(frames, axis=0, ignore_index=True)
            out = out.sort_values(['series_id', 'observation_date', 'release_time']).drop_duplicates(
                subset=['series_id', 'observation_date', 'release_time'],
                keep='last',
            )
            return out.reset_index(drop=True)
        if errors:
            raise RuntimeError('macro adapter live fetch failed: ' + '; '.join(errors[:5]))
        return pd.DataFrame(columns=['series_id', 'observation_date', 'release_time', 'available_time', 'frequency', 'vintage', 'value'])
