import io
import os
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ingestion.adapters.akshare_macro import AkshareMacroAdapter
from ingestion.adapters.akshare_minute import AkshareMinuteAdapter
from ingestion.adapters.baostock_daily import BaoStockDailyAdapter
from ingestion.adapters.baostock_daily import _BaoStockClient
from ingestion.models import DatasetSpec
from ingestion.models import IngestionRequest


def _make_spec(*, dataset: str, adapter_name: str, request_spec=None, schema_spec=None) -> DatasetSpec:
    return DatasetSpec(
        dataset=dataset,
        domain='market',
        granularity='1d',
        source_name='test',
        adapter_name=adapter_name,
        request_spec=request_spec or {},
        schema_spec=schema_spec or {},
        quality_spec={},
        storage_spec={},
    )


class _FakeBaoStockClient:
    def __init__(self):
        self.daily_calls = []

    def fetch_hs300_symbols(self, as_of_date: str):
        return ['sh.600000', 'sz.000001']

    def fetch_daily(self, symbol: str, *, start_date: str, end_date: str, frequency: str, adjustflag: str):
        self.daily_calls.append(
            {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'frequency': frequency,
                'adjustflag': adjustflag,
            }
        )
        return pd.DataFrame(
            [
                {
                    'date': '2024-01-02',
                    'code': symbol,
                    'open': '10.0',
                    'high': '11.0',
                    'low': '9.5',
                    'close': '10.5',
                    'preclose': '9.9',
                    'volume': '1000',
                    'amount': '10250',
                    'turn': '0.12',
                    'pctChg': '6.06',
                    'tradestatus': '1',
                }
            ]
        )


class _EmptyBaoStockClient(_FakeBaoStockClient):
    def fetch_daily(self, symbol: str, *, start_date: str, end_date: str, frequency: str, adjustflag: str):
        super().fetch_daily(
            symbol,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag=adjustflag,
        )
        return pd.DataFrame()


class _FakeMinuteClient:
    def fetch_minute(self, symbol: str, *, start_date: str, end_date: str, period: str, adjust: str):
        return pd.DataFrame(
            [
                {
                    '时间': '2024-01-02 09:31:00',
                    '开盘': '10.0',
                    '收盘': '10.2',
                    '最高': '10.3',
                    '最低': '9.9',
                    '成交量': '1200',
                    '成交额': '12240',
                },
                {
                    '时间': '2024-01-02 09:32:00',
                    '开盘': '10.2',
                    '收盘': '10.4',
                    '最高': '10.5',
                    '最低': '10.1',
                    '成交量': '800',
                    '成交额': '8320',
                },
            ]
        )


class _FakeMacroClient:
    def fetch_series(self, series_id: str, *, start: str, end: str, include_revisions: bool):
        if series_id != 'cpi_yoy':
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    '月份': '2024-01',
                    '发布时间': '2024-02-08 09:30:00',
                    '可用时间': '2024-02-08 09:31:00',
                    '全国-同比增长': '0.5',
                }
            ]
        )


class AdapterContractTests(unittest.TestCase):
    def test_default_baostock_client_suppresses_provider_stdout(self):
        frame = pd.DataFrame(
            [
                {
                    'date': '2024-01-02',
                    'code': 'sz.000001',
                    'open': '10.0',
                    'high': '10.5',
                    'low': '9.8',
                    'close': '10.2',
                    'preclose': '9.9',
                    'volume': '1000',
                    'amount': '10200',
                    'turn': '0.11',
                    'pctChg': '3.03',
                    'tradestatus': '1',
                }
            ]
        )
        fake_bs = types.SimpleNamespace(
            login=lambda: (print('login success!') or types.SimpleNamespace(error_code='0', error_msg='')),
            logout=lambda: print('logout success!'),
            query_history_k_data_plus=lambda *args, **kwargs: types.SimpleNamespace(
                error_code='0',
                error_msg='',
                get_data=lambda: frame,
            ),
            query_hs300_stocks=lambda date='': types.SimpleNamespace(
                error_code='0',
                error_msg='',
                get_data=lambda: pd.DataFrame([{'code': 'sz.000001'}]),
            ),
        )

        buffer = io.StringIO()
        with patch.dict(sys.modules, {'baostock': fake_bs}):
            with redirect_stdout(buffer):
                result = _BaoStockClient().fetch_daily(
                    'sz.000001',
                    start_date='2024-01-01',
                    end_date='2024-01-02',
                    frequency='d',
                    adjustflag='1',
                )

        self.assertEqual(buffer.getvalue(), '')
        self.assertEqual(list(result.columns), list(frame.columns))

    def test_baostock_daily_adapter_returns_canonical_rows(self):
        spec = _make_spec(
            dataset='market_bar_1d',
            adapter_name='baostock_daily',
            request_spec={'universe_ref': 'universe_membership.hs300'},
        )
        request = IngestionRequest(dataset='market_bar_1d', start='2024-01-01', end='2024-01-31')
        adapter = BaoStockDailyAdapter(client=_FakeBaoStockClient())

        df = adapter.fetch(request, spec)

        self.assertEqual(
            list(df.columns),
            [
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
            ],
        )
        self.assertEqual(df['instrument_id'].tolist(), ['000001', '600000'])
        self.assertEqual(df['trade_date'].tolist(), ['2024-01-02', '2024-01-02'])
        self.assertAlmostEqual(float(df.loc[0, 'close']), 10.5)

    def test_baostock_daily_adapter_returns_empty_canonical_frame(self):
        spec = _make_spec(
            dataset='market_bar_1d',
            adapter_name='baostock_daily',
            request_spec={'universe_ref': 'universe_membership.hs300'},
        )
        request = IngestionRequest(dataset='market_bar_1d', start='2024-01-01', end='2024-01-31')
        adapter = BaoStockDailyAdapter(client=_EmptyBaoStockClient())

        df = adapter.fetch(request, spec)

        self.assertEqual(
            list(df.columns),
            [
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
            ],
        )
        self.assertTrue(df.empty)

    def test_akshare_minute_adapter_returns_canonical_rows(self):
        spec = _make_spec(dataset='market_bar_1m', adapter_name='market_minute_bar', request_spec={'symbols': ['000001']})
        request = IngestionRequest(dataset='market_bar_1m', start='2024-01-02', end='2024-01-02')
        adapter = AkshareMinuteAdapter(client=_FakeMinuteClient())

        df = adapter.fetch(request, spec)

        self.assertEqual(
            list(df.columns),
            ['instrument_id', 'ts', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount'],
        )
        self.assertEqual(df['instrument_id'].tolist(), ['000001', '000001'])
        self.assertEqual(df['trade_date'].tolist(), ['2024-01-02', '2024-01-02'])
        self.assertEqual(df['ts'].tolist(), ['2024-01-02 09:31:00', '2024-01-02 09:32:00'])

    def test_akshare_macro_adapter_returns_canonical_rows(self):
        spec = _make_spec(
            dataset='macro_series',
            adapter_name='macro_timeseries',
            request_spec={'series_catalog': ['cpi_yoy'], 'include_revisions': True},
            schema_spec={'primary_key': ['series_id', 'observation_date', 'release_time']},
        )
        request = IngestionRequest(dataset='macro_series', start='2024-01-01', end='2024-03-31')
        adapter = AkshareMacroAdapter(client=_FakeMacroClient())

        df = adapter.fetch(request, spec)

        self.assertEqual(
            list(df.columns),
            ['series_id', 'observation_date', 'release_time', 'available_time', 'frequency', 'vintage', 'value'],
        )
        self.assertEqual(df.loc[0, 'series_id'], 'cpi_yoy')
        self.assertEqual(df.loc[0, 'observation_date'], '2024-01-01')
        self.assertEqual(df.loc[0, 'release_time'], '2024-02-08 09:30:00')
        self.assertEqual(df.loc[0, 'available_time'], '2024-02-08 09:31:00')
        self.assertEqual(df.loc[0, 'frequency'], 'monthly')
        self.assertEqual(df.loc[0, 'vintage'], 'latest')
        self.assertAlmostEqual(float(df.loc[0, 'value']), 0.5)


if __name__ == '__main__':
    unittest.main()
