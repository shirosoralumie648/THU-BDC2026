import os

import pandas as pd


class AkshareMinuteBarAdapter:
    adapter_name = 'market_minute_bar'

    def fetch(self, request, spec):
        source_params = spec.request_spec or {}
        input_path = str(source_params.get('input_path', '')).strip()
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
        out['instrument_id'] = out['instrument_id'].astype(str).str.replace(r'[^0-9]', '', regex=True).str[-6:].str.zfill(6)
        out['ts'] = pd.to_datetime(out['ts'], errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors='coerce')
        return out.dropna(subset=['instrument_id', 'ts', 'close']).reset_index(drop=True)
