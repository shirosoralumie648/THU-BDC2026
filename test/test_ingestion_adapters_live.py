import os
import sys
import types
import unittest
from unittest import mock

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ingestion.adapters.akshare_macro import AkshareMacroAdapter
from ingestion.adapters.akshare_minute import AkshareMinuteBarAdapter
from ingestion.models import DatasetSpec
from ingestion.models import IngestionRequest


class IngestionAdapterLiveTests(unittest.TestCase):
    def test_minute_adapter_uses_live_provider_when_available(self):
        fake_ak = types.SimpleNamespace(
            stock_zh_a_hist_min_em=lambda **kwargs: pd.DataFrame(
                [
                    {
                        '时间': '2024-01-02 09:30:00',
                        '开盘': 10.0,
                        '收盘': 10.1,
                        '最高': 10.2,
                        '最低': 9.9,
                        '成交量': 1000,
                        '成交额': 10100,
                    }
                ]
            )
        )
        spec = DatasetSpec(
            dataset='market_bar_1m',
            domain='market',
            granularity='1m',
            source_name='akshare',
            adapter_name='market_minute_bar',
            request_spec={'period': '1', 'adjust': ''},
            schema_spec={},
            quality_spec={},
            storage_spec={},
        )
        request = IngestionRequest(
            dataset='market_bar_1m',
            start='2024-01-02',
            end='2024-01-02',
            extra={'symbols': ['000001.SZ']},
        )

        with mock.patch.dict(sys.modules, {'akshare': fake_ak}):
            df = AkshareMinuteBarAdapter().fetch(request, spec)

        self.assertEqual(df['instrument_id'].tolist(), ['000001'])
        self.assertIn('ts', df.columns)
        self.assertAlmostEqual(float(df.loc[0, 'close']), 10.1, places=6)

    def test_macro_adapter_uses_live_provider_for_supported_series(self):
        fake_ak = types.SimpleNamespace(
            macro_china_money_supply=lambda: pd.DataFrame(
                [{'月份': '2024-01', '货币和准货币(M2)-同比增长': 8.7}]
            ),
            macro_china_shibor_all=lambda: pd.DataFrame(
                [{'日期': '2024-01-02', '3M-定价': 2.4}]
            ),
            currency_boc_safe=lambda: pd.DataFrame(
                [{'日期': '2024-01-02', '美元': 720.0}]
            ),
            stock_index_pe_lg=lambda **kwargs: pd.DataFrame(
                [{'日期': '2024-01-02', '滚动市盈率': 12.3}]
            ),
        )
        spec = DatasetSpec(
            dataset='macro_series',
            domain='macro',
            granularity='mixed',
            source_name='akshare',
            adapter_name='macro_timeseries',
            request_spec={'series_catalog': ['m2_yoy', 'shibor_3m', 'usdcny', 'csi300_pe_ttm']},
            schema_spec={},
            quality_spec={},
            storage_spec={},
        )
        request = IngestionRequest(dataset='macro_series', start='2024-01-01', end='2024-01-31')

        with mock.patch.dict(sys.modules, {'akshare': fake_ak}):
            df = AkshareMacroAdapter().fetch(request, spec)

        self.assertTrue({'m2_yoy', 'shibor_3m', 'usdcny', 'csi300_pe_ttm'}.issubset(set(df['series_id'].tolist())))
        self.assertIn('available_time', df.columns)
        self.assertIn('release_time', df.columns)
        usdcny = df[df['series_id'] == 'usdcny'].iloc[0]
        self.assertAlmostEqual(float(usdcny['value']), 7.2, places=6)


if __name__ == '__main__':
    unittest.main()
