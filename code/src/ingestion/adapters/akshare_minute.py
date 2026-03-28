import importlib
import os
from typing import List

import baostock as bs
import pandas as pd


def _normalize_code(value: str) -> str:
    text = ''.join(ch for ch in str(value or '').strip() if ch.isdigit())
    return text[-6:].zfill(6)


def _resolve_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f'minute input missing candidate columns: {candidates}')


class AkshareMinuteBarAdapter:
    adapter_name = 'market_minute_bar'

    def _resolve_input_path(self, request, spec) -> str:
        extra = request.extra if isinstance(request.extra, dict) else {}
        source_params = spec.request_spec or {}
        input_path = str(extra.get('input_path', '') or source_params.get('input_path', '')).strip()
        return input_path

    def _load_file_input(self, input_path: str) -> pd.DataFrame:
        if not input_path or (not os.path.exists(input_path)):
            return pd.DataFrame(columns=['instrument_id', 'ts', 'open', 'high', 'low', 'close', 'volume', 'amount'])

        df = pd.read_csv(input_path)
        rename_map = {
            '股票代码': 'instrument_id',
            'code': 'instrument_id',
            'datetime': 'ts',
            '时间': 'ts',
            '成交量': 'volume',
            '成交额': 'amount',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
        }
        out = df.rename(columns=rename_map).copy()
        required = ['instrument_id', 'ts', 'close']
        for col in required:
            if col not in out.columns:
                raise ValueError(f'minute input missing required column: {col}')
        out['instrument_id'] = out['instrument_id'].map(_normalize_code)
        out['ts'] = pd.to_datetime(out['ts'], errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors='coerce')
        return out.dropna(subset=['instrument_id', 'ts', 'close']).reset_index(drop=True)

    def _resolve_symbols(self, request, spec) -> List[str]:
        extra = request.extra if isinstance(request.extra, dict) else {}
        if extra.get('symbols'):
            return [_normalize_code(item) for item in list(extra.get('symbols', [])) if str(item).strip()]

        request_spec = spec.request_spec or {}
        universe_ref = str(extra.get('universe_ref', '') or request_spec.get('universe_ref', '')).strip().lower()
        if 'hs300' not in universe_ref:
            return []

        index_date = str(extra.get('index_date', '') or request.end or '').strip()
        rs = bs.query_hs300_stocks(date=index_date)
        rows = []
        while rs.error_code == '0' and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return []
        df = pd.DataFrame(rows, columns=rs.fields)
        code_col = 'code' if 'code' in df.columns else df.columns[0]
        return [_normalize_code(value) for value in df[code_col].tolist()]

    def _load_live_symbol(self, ak, symbol: str, request, spec) -> pd.DataFrame:
        extra = request.extra if isinstance(request.extra, dict) else {}
        request_spec = spec.request_spec or {}
        period = str(extra.get('period', '') or request_spec.get('period', '') or '1').strip()
        adjust = str(extra.get('adjust', '') or request_spec.get('adjust', '') or '').strip()
        start_date = str(request.start).strip()
        end_date = str(request.end).strip()
        if len(start_date) <= 10:
            start_date = f'{start_date} 09:30:00'
        if len(end_date) <= 10:
            end_date = f'{end_date} 15:00:00'

        live_df = ak.stock_zh_a_hist_min_em(
            symbol=str(symbol),
            start_date=start_date,
            end_date=end_date,
            period=period,
            adjust=adjust,
        )
        if live_df is None or live_df.empty:
            return pd.DataFrame(columns=['instrument_id', 'ts', 'open', 'high', 'low', 'close', 'volume', 'amount'])

        ts_col = _resolve_column(live_df, ['时间', 'datetime', '日期时间', 'date'])
        close_col = _resolve_column(live_df, ['收盘', 'close', '最新价', 'price'])
        open_col = next((col for col in ['开盘', 'open'] if col in live_df.columns), '')
        high_col = next((col for col in ['最高', 'high'] if col in live_df.columns), '')
        low_col = next((col for col in ['最低', 'low'] if col in live_df.columns), '')
        volume_col = next((col for col in ['成交量', 'volume', 'vol'] if col in live_df.columns), '')
        amount_col = next((col for col in ['成交额', 'amount', 'turnover'] if col in live_df.columns), '')
        code_col = next((col for col in ['代码', '股票代码', 'symbol', 'code'] if col in live_df.columns), '')

        out = pd.DataFrame(index=live_df.index)
        out['instrument_id'] = live_df[code_col].map(_normalize_code) if code_col else _normalize_code(symbol)
        out['ts'] = pd.to_datetime(live_df[ts_col], errors='coerce')
        out['close'] = pd.to_numeric(live_df[close_col], errors='coerce')
        out['open'] = pd.to_numeric(live_df[open_col], errors='coerce') if open_col else out['close']
        out['high'] = pd.to_numeric(live_df[high_col], errors='coerce') if high_col else out['close']
        out['low'] = pd.to_numeric(live_df[low_col], errors='coerce') if low_col else out['close']
        out['volume'] = pd.to_numeric(live_df[volume_col], errors='coerce') if volume_col else pd.NA
        out['amount'] = pd.to_numeric(live_df[amount_col], errors='coerce') if amount_col else pd.NA
        return out.dropna(subset=['instrument_id', 'ts', 'close']).reset_index(drop=True)

    def fetch(self, request, spec):
        input_path = self._resolve_input_path(request, spec)
        if input_path:
            return self._load_file_input(input_path)

        try:
            ak = importlib.import_module('akshare')
        except Exception:
            return pd.DataFrame(columns=['instrument_id', 'ts', 'open', 'high', 'low', 'close', 'volume', 'amount'])

        symbols = self._resolve_symbols(request, spec)
        frames = []
        errors = []
        for symbol in symbols:
            try:
                part = self._load_live_symbol(ak, symbol, request, spec)
            except Exception as exc:
                errors.append(f'{symbol}: {exc}')
                continue
            if not part.empty:
                frames.append(part)

        if frames:
            out = pd.concat(frames, axis=0, ignore_index=True)
            out = out.sort_values(['instrument_id', 'ts']).drop_duplicates(
                subset=['instrument_id', 'ts'],
                keep='last',
            )
            return out.reset_index(drop=True)
        if errors:
            raise RuntimeError('minute adapter live fetch failed: ' + '; '.join(errors[:5]))
        return pd.DataFrame(columns=['instrument_id', 'ts', 'open', 'high', 'low', 'close', 'volume', 'amount'])
