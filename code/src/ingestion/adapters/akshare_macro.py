import os

import pandas as pd


class AkshareMacroAdapter:
    adapter_name = 'macro_timeseries'

    def fetch(self, request, spec):
        source_params = spec.request_spec or {}
        input_path = str(source_params.get('input_path', '')).strip()
        if not input_path or (not os.path.exists(input_path)):
            return pd.DataFrame(columns=['series_id', 'observation_date', 'available_time', 'value'])

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
        out['series_id'] = out['series_id'].astype(str).str.strip()
        out['observation_date'] = pd.to_datetime(out['observation_date'], errors='coerce')
        out['available_time'] = pd.to_datetime(out['available_time'], errors='coerce')
        out['value'] = pd.to_numeric(out['value'], errors='coerce')
        return out.dropna(subset=['series_id', 'observation_date', 'available_time']).reset_index(drop=True)
