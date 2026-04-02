from __future__ import annotations

import io
from contextlib import redirect_stdout

from ingestion.adapters._helpers import as_dataframe
from ingestion.adapters._helpers import empty_frame
from ingestion.adapters._helpers import normalize_date_text
from ingestion.adapters._helpers import normalize_instrument_id
from ingestion.adapters._helpers import require_pandas
from ingestion.adapters._helpers import to_numeric_series
from ingestion.adapters._helpers import to_provider_symbol


CANONICAL_COLUMNS = [
    'instrument_id',
    'trade_date',
    'open',
    'high',
    'low',
    'close',
    'preclose',
    'volume',
    'amount',
    'turnover',
    'pct_chg',
    'trade_status',
]


class _BaoStockClient:
    def _quiet_call(self, func, *args, **kwargs):
        with redirect_stdout(io.StringIO()):
            return func(*args, **kwargs)

    def fetch_hs300_symbols(self, as_of_date: str):
        try:
            import baostock as bs
        except ModuleNotFoundError as exc:
            raise RuntimeError('baostock is required for daily ingestion adapter') from exc

        login_result = self._quiet_call(bs.login)
        if getattr(login_result, 'error_code', '0') != '0':
            raise RuntimeError(getattr(login_result, 'error_msg', '') or 'baostock login failed')
        try:
            result = bs.query_hs300_stocks(as_of_date or '')
            if getattr(result, 'error_code', '0') != '0':
                raise RuntimeError(getattr(result, 'error_msg', '') or 'baostock hs300 query failed')
            frame = result.get_data()
            if 'code' not in frame.columns:
                return []
            return [str(value) for value in frame['code'].dropna().tolist()]
        finally:
            self._quiet_call(bs.logout)

    def fetch_daily(self, symbol: str, *, start_date: str, end_date: str, frequency: str, adjustflag: str):
        try:
            import baostock as bs
        except ModuleNotFoundError as exc:
            raise RuntimeError('baostock is required for daily ingestion adapter') from exc

        login_result = self._quiet_call(bs.login)
        if getattr(login_result, 'error_code', '0') != '0':
            raise RuntimeError(getattr(login_result, 'error_msg', '') or 'baostock login failed')
        try:
            result = bs.query_history_k_data_plus(
                symbol,
                'date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,tradestatus',
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                adjustflag=adjustflag,
            )
            if getattr(result, 'error_code', '0') != '0':
                raise RuntimeError(getattr(result, 'error_msg', '') or f'baostock daily query failed for {symbol}')
            return result.get_data()
        finally:
            self._quiet_call(bs.logout)


class BaoStockDailyAdapter:
    adapter_name = 'baostock_daily'

    def __init__(self, client=None):
        self.client = client or _BaoStockClient()

    def _resolve_symbols(self, request, spec):
        explicit = []
        if isinstance(request.extra, dict):
            symbols = request.extra.get('symbols')
            if isinstance(symbols, list):
                explicit.extend(symbols)
            symbol = request.extra.get('symbol')
            if symbol:
                explicit.append(symbol)
        if request.universe:
            explicit.extend([item.strip() for item in str(request.universe).split(',') if item.strip()])
        configured = spec.request_spec.get('symbols', [])
        if isinstance(configured, list):
            explicit.extend(configured)
        if explicit:
            seen = set()
            ordered = []
            for value in explicit:
                symbol = to_provider_symbol(value)
                if symbol not in seen:
                    seen.add(symbol)
                    ordered.append(symbol)
            return ordered

        universe_ref = str(spec.request_spec.get('universe_ref', '') or '')
        if universe_ref == 'universe_membership.hs300':
            symbols = list(self.client.fetch_hs300_symbols(request.end))
            if symbols:
                return symbols
        return [to_provider_symbol('000001')]

    def _normalize_frame(self, frame):
        pd = require_pandas()
        frame = as_dataframe(frame)
        if frame.empty:
            return empty_frame(CANONICAL_COLUMNS)

        output = pd.DataFrame()
        output['instrument_id'] = frame.get('code', pd.Series(dtype='object')).map(normalize_instrument_id)
        output['trade_date'] = normalize_date_text(frame.get('date', pd.Series(dtype='object')))
        output['open'] = to_numeric_series(frame.get('open'))
        output['high'] = to_numeric_series(frame.get('high'))
        output['low'] = to_numeric_series(frame.get('low'))
        output['close'] = to_numeric_series(frame.get('close'))
        output['preclose'] = to_numeric_series(frame.get('preclose'))
        output['volume'] = to_numeric_series(frame.get('volume'))
        output['amount'] = to_numeric_series(frame.get('amount'))
        output['turnover'] = to_numeric_series(frame.get('turnover', frame.get('turn')))
        output['pct_chg'] = to_numeric_series(frame.get('pct_chg', frame.get('pctChg')))
        output['trade_status'] = pd.to_numeric(frame.get('trade_status', frame.get('tradestatus')), errors='coerce').astype('Int64')
        output = output[CANONICAL_COLUMNS]
        output = output[output['instrument_id'] != ''].reset_index(drop=True)
        return output

    def fetch(self, request, spec):
        pd = require_pandas()
        frames = []
        frequency = 'd'
        adjustflag = str(request.adjustment or '1')
        for symbol in self._resolve_symbols(request, spec):
            raw = self.client.fetch_daily(
                symbol,
                start_date=request.start,
                end_date=request.end,
                frequency=frequency,
                adjustflag=adjustflag,
            )
            normalized = self._normalize_frame(raw)
            if not normalized.empty:
                frames.append(normalized)
        if not frames:
            return empty_frame(CANONICAL_COLUMNS)
        result = pd.concat(frames, ignore_index=True)
        return result.sort_values(['trade_date', 'instrument_id']).reset_index(drop=True)
