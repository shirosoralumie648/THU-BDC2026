from typing import List

import baostock as bs
import pandas as pd


def _normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip().lower()
    if text.startswith(('sh.', 'sz.')):
        return text
    digits = ''.join(ch for ch in text if ch.isdigit())[-6:].zfill(6)
    if digits.startswith(('5', '6', '9')):
        return f'sh.{digits}'
    return f'sz.{digits}'


class BaostockDailyAdapter:
    adapter_name = 'baostock_daily'

    def _resolve_symbols(self, request) -> List[str]:
        extra_symbols = request.extra.get('symbols', []) if isinstance(request.extra, dict) else []
        if extra_symbols:
            return [_normalize_symbol(symbol) for symbol in extra_symbols if str(symbol).strip()]

        index_date = ''
        if isinstance(request.extra, dict):
            index_date = str(request.extra.get('index_date', '') or '').strip()
        rs = bs.query_hs300_stocks(date=index_date or str(request.end or '').strip())
        rows = []
        while rs.error_code == '0' and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return []
        df = pd.DataFrame(rows, columns=rs.fields)
        code_col = 'code' if 'code' in df.columns else df.columns[0]
        return [_normalize_symbol(symbol) for symbol in df[code_col].tolist()]

    def fetch(self, request, spec):
        fields = 'date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,tradestatus'
        lg = bs.login()
        if lg.error_code != '0':
            raise RuntimeError(f'baostock login failed: {lg.error_msg}')
        try:
            symbols = self._resolve_symbols(request)
            frames = []
            for symbol in symbols:
                rs = bs.query_history_k_data_plus(
                    symbol,
                    fields,
                    start_date=request.start,
                    end_date=request.end,
                    frequency='d',
                    adjustflag=request.adjustment or '1',
                )
                if rs.error_code != '0':
                    raise RuntimeError(f'baostock query failed for {symbol}: {rs.error_msg}')
                rows = []
                while rs.error_code == '0' and rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    continue
                part = pd.DataFrame(rows, columns=rs.fields).rename(columns={'code': 'instrument_id', 'date': 'trade_date'})
                frames.append(part)
            if not frames:
                return pd.DataFrame(
                    columns=['instrument_id', 'trade_date', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg', 'tradestatus']
                )

            out = pd.concat(frames, ignore_index=True)
            out['instrument_id'] = out['instrument_id'].astype(str).str.replace(r'[^0-9]', '', regex=True).str[-6:].str.zfill(6)
            out['trade_date'] = pd.to_datetime(out['trade_date'], errors='coerce').dt.strftime('%Y-%m-%d')
            numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg', 'tradestatus']
            for col in numeric_cols:
                if col in out.columns:
                    out[col] = pd.to_numeric(out[col], errors='coerce')
            return out.dropna(subset=['instrument_id', 'trade_date']).reset_index(drop=True)
        finally:
            bs.logout()
