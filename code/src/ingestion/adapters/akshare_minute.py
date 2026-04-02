from __future__ import annotations

from ingestion.adapters._helpers import as_dataframe
from ingestion.adapters._helpers import empty_frame
from ingestion.adapters._helpers import first_present
from ingestion.adapters._helpers import normalize_date_text
from ingestion.adapters._helpers import normalize_instrument_id
from ingestion.adapters._helpers import normalize_timestamp_text
from ingestion.adapters._helpers import require_pandas
from ingestion.adapters._helpers import to_numeric_series


CANONICAL_COLUMNS = ['instrument_id', 'ts', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount']


class _AkshareMinuteClient:
    def fetch_minute(self, symbol: str, *, start_date: str, end_date: str, period: str, adjust: str):
        try:
            import akshare as ak
        except ModuleNotFoundError as exc:
            raise RuntimeError('akshare is required for minute ingestion adapter') from exc

        return ak.stock_zh_a_hist_min_em(
            symbol=normalize_instrument_id(symbol) or symbol,
            start_date=f'{start_date} 09:30:00',
            end_date=f'{end_date} 15:00:00',
            period=period,
            adjust=adjust,
        )


class AkshareMinuteAdapter:
    adapter_name = 'market_minute_bar'

    def __init__(self, client=None):
        self.client = client or _AkshareMinuteClient()

    def _resolve_symbols(self, request, spec):
        symbols = []
        if isinstance(request.extra, dict):
            extra_symbols = request.extra.get('symbols')
            if isinstance(extra_symbols, list):
                symbols.extend(extra_symbols)
            if request.extra.get('symbol'):
                symbols.append(request.extra['symbol'])
        if request.universe:
            symbols.extend([item.strip() for item in str(request.universe).split(',') if item.strip()])
        configured = spec.request_spec.get('symbols', [])
        if isinstance(configured, list):
            symbols.extend(configured)
        normalized = [normalize_instrument_id(value) for value in symbols if normalize_instrument_id(value)]
        if normalized:
            return list(dict.fromkeys(normalized))
        return ['000001']

    def _normalize_frame(self, frame, *, symbol: str):
        pd = require_pandas()
        frame = as_dataframe(frame)
        if frame.empty:
            return empty_frame(CANONICAL_COLUMNS)

        output = pd.DataFrame()
        output['instrument_id'] = [symbol] * len(frame)
        ts_source = first_present(frame, ['ts', 'datetime', '时间'])
        output['ts'] = normalize_timestamp_text(ts_source)
        output['trade_date'] = normalize_date_text(ts_source)
        output['open'] = to_numeric_series(first_present(frame, ['open', '开盘']))
        output['high'] = to_numeric_series(first_present(frame, ['high', '最高']))
        output['low'] = to_numeric_series(first_present(frame, ['low', '最低']))
        output['close'] = to_numeric_series(first_present(frame, ['close', '收盘']))
        output['volume'] = to_numeric_series(first_present(frame, ['volume', '成交量']))
        output['amount'] = to_numeric_series(first_present(frame, ['amount', '成交额']))
        output = output[CANONICAL_COLUMNS]
        output = output[output['ts'] != ''].reset_index(drop=True)
        return output

    def fetch(self, request, spec):
        pd = require_pandas()
        period = '1'
        adjust = str(request.adjustment or '')
        frames = []
        for symbol in self._resolve_symbols(request, spec):
            raw = self.client.fetch_minute(
                symbol,
                start_date=request.start,
                end_date=request.end,
                period=period,
                adjust=adjust,
            )
            normalized = self._normalize_frame(raw, symbol=symbol)
            if not normalized.empty:
                frames.append(normalized)
        if not frames:
            return empty_frame(CANONICAL_COLUMNS)
        result = pd.concat(frames, ignore_index=True)
        return result.sort_values(['ts', 'instrument_id']).reset_index(drop=True)
