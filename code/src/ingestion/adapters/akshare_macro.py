from __future__ import annotations

from ingestion.adapters._helpers import as_dataframe
from ingestion.adapters._helpers import empty_frame
from ingestion.adapters._helpers import first_present
from ingestion.adapters._helpers import normalize_date_text
from ingestion.adapters._helpers import normalize_timestamp_text
from ingestion.adapters._helpers import require_pandas
from ingestion.adapters._helpers import to_numeric_series


CANONICAL_COLUMNS = ['series_id', 'observation_date', 'release_time', 'available_time', 'frequency', 'vintage', 'value']
FREQUENCY_BY_SERIES = {
    'cpi_yoy': 'monthly',
    'ppi_yoy': 'monthly',
    'm2_yoy': 'monthly',
    'shibor_3m': 'daily',
    'usdcny': 'daily',
    'csi300_pe_ttm': 'daily',
}
VALUE_COLUMN_CANDIDATES = {
    'cpi_yoy': ['全国-同比增长', 'value'],
    'ppi_yoy': ['当月同比增长', 'value'],
    'm2_yoy': ['货币和准货币(M2)-同比增长', 'value'],
    'shibor_3m': ['利率', '3月', 'value'],
    'usdcny': ['美元', '美元/人民币', 'value'],
    'csi300_pe_ttm': ['滚动市盈率', '等权滚动市盈率', 'ttmPe', 'value'],
}
OBSERVATION_DATE_CANDIDATES = ['observation_date', '日期', 'date', '月份', '季度', '报告日']
RELEASE_TIME_CANDIDATES = ['release_time', '发布时间', 'TIME', 'time']
AVAILABLE_TIME_CANDIDATES = ['available_time', '可用时间']
VINTAGE_CANDIDATES = ['vintage']


class _AkshareMacroClient:
    def fetch_series(self, series_id: str, *, start: str, end: str, include_revisions: bool):
        try:
            import akshare as ak
        except ModuleNotFoundError as exc:
            raise RuntimeError('akshare is required for macro ingestion adapter') from exc

        if series_id == 'cpi_yoy':
            return ak.macro_china_cpi()
        if series_id == 'ppi_yoy':
            return ak.macro_china_ppi()
        if series_id == 'm2_yoy':
            return ak.macro_china_money_supply()
        if series_id == 'shibor_3m':
            return ak.rate_interbank(
                market='上海银行同业拆借市场',
                symbol='Shibor人民币',
                indicator='3月',
            )
        if series_id == 'usdcny':
            return ak.currency_boc_safe()
        if series_id == 'csi300_pe_ttm':
            return ak.stock_index_pe_lg(symbol='沪深300')
        raise KeyError(f'unsupported macro series: {series_id}')


class AkshareMacroAdapter:
    adapter_name = 'macro_timeseries'

    def __init__(self, client=None):
        self.client = client or _AkshareMacroClient()

    def _resolve_series_catalog(self, request, spec):
        series = []
        if isinstance(request.extra, dict):
            catalog = request.extra.get('series_catalog')
            if isinstance(catalog, list):
                series.extend(catalog)
            if request.extra.get('series_id'):
                series.append(request.extra['series_id'])
        configured = spec.request_spec.get('series_catalog', [])
        if isinstance(configured, list):
            series.extend(configured)
        return list(dict.fromkeys([str(item).strip() for item in series if str(item).strip()]))

    def _select_value_column(self, frame, *, series_id: str):
        pd = require_pandas()
        named = first_present(frame, VALUE_COLUMN_CANDIDATES.get(series_id, ['value']))
        if named is not None:
            return named
        for column in frame.columns[::-1]:
            numeric = pd.to_numeric(frame[column], errors='coerce')
            if numeric.notna().any():
                return frame[column]
        return None

    def _normalize_frame(self, frame, *, request, series_id: str):
        pd = require_pandas()
        frame = as_dataframe(frame)
        if frame.empty:
            return empty_frame(CANONICAL_COLUMNS)

        observation_source = first_present(frame, OBSERVATION_DATE_CANDIDATES)
        if observation_source is None:
            return empty_frame(CANONICAL_COLUMNS)

        observation_dates = normalize_date_text(observation_source)
        release_source = first_present(frame, RELEASE_TIME_CANDIDATES)
        if release_source is None:
            release_source = observation_source
        release_times = normalize_timestamp_text(release_source)
        available_source = first_present(frame, AVAILABLE_TIME_CANDIDATES)
        if available_source is None:
            available_source = release_source
        available_times = normalize_timestamp_text(available_source)
        value_source = self._select_value_column(frame, series_id=series_id)
        vintage_source = first_present(frame, VINTAGE_CANDIDATES)

        output = pd.DataFrame()
        output['series_id'] = [series_id] * len(frame)
        output['observation_date'] = observation_dates
        output['release_time'] = release_times
        output['available_time'] = available_times
        output['frequency'] = [FREQUENCY_BY_SERIES.get(series_id, 'daily')] * len(frame)
        if vintage_source is None:
            output['vintage'] = ['latest'] * len(frame)
        else:
            output['vintage'] = vintage_source.fillna('latest').astype(str)
        output['value'] = to_numeric_series(value_source)
        output = output[CANONICAL_COLUMNS]
        output = output[output['observation_date'] != ''].reset_index(drop=True)
        if output.empty:
            return output

        observation_ts = pd.to_datetime(output['observation_date'], errors='coerce')
        start_ts = pd.to_datetime(request.start, errors='coerce')
        end_ts = pd.to_datetime(request.end, errors='coerce')
        mask = observation_ts.notna()
        if not pd.isna(start_ts):
            mask &= observation_ts >= start_ts
        if not pd.isna(end_ts):
            mask &= observation_ts <= end_ts
        return output.loc[mask].reset_index(drop=True)

    def fetch(self, request, spec):
        pd = require_pandas()
        include_revisions = bool(spec.request_spec.get('include_revisions', False))
        frames = []
        for series_id in self._resolve_series_catalog(request, spec):
            raw = self.client.fetch_series(
                series_id,
                start=request.start,
                end=request.end,
                include_revisions=include_revisions,
            )
            normalized = self._normalize_frame(raw, request=request, series_id=series_id)
            if not normalized.empty:
                frames.append(normalized)
        if not frames:
            return empty_frame(CANONICAL_COLUMNS)
        result = pd.concat(frames, ignore_index=True)
        return result.sort_values(['series_id', 'observation_date', 'release_time']).reset_index(drop=True)
