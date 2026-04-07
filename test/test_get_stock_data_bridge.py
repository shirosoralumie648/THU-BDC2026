import os
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock
import json

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import get_stock_data
from ingestion.compat import run_stock_data_bridge


class _FakeService:
    def __init__(self, runtime_root: str):
        self.runtime_root = runtime_root

    def create_and_run(self, request):
        curated_path = Path(self.runtime_root) / 'curated' / 'job-bridge.csv'
        curated_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    'instrument_id': '000001',
                    'trade_date': '2024-01-02',
                    'open': 10.0,
                    'high': 10.8,
                    'low': 9.9,
                    'close': 10.5,
                    'preclose': 10.0,
                    'volume': 1000,
                    'amount': 10500,
                    'turnover': 1.2,
                    'pct_chg': 5.0,
                }
            ]
        ).to_csv(curated_path, index=False)

        manifest_path = Path(self.runtime_root) / 'manifests' / 'job-bridge.json'
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            '{"job_id":"job-bridge","curated_paths":["'
            + str(curated_path).replace('\\', '\\\\')
            + '"],"status":"succeeded"}',
            encoding='utf-8',
        )
        return types.SimpleNamespace(
            job_id='job-bridge',
            status='succeeded',
            manifest_path=str(manifest_path),
        )

    def load_manifest(self, job_or_id):
        target = job_or_id
        if not isinstance(target, str):
            target = getattr(target, 'manifest_path', '')
        import json

        with open(target, 'r', encoding='utf-8') as f:
            return json.load(f)


class StockDataBridgeTests(unittest.TestCase):
    def test_run_stock_data_bridge_exports_legacy_stock_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / 'stock_data.csv'
            runtime_root = Path(tmp) / 'runtime'
            args = Namespace(
                pipeline_config_dir='./config',
                dataset_name='market_bar_1d',
                start_date='2024-01-01',
                end_date='2024-01-31',
                index_date='2024-01-31',
                output_path=str(output_path),
                manifest_path='',
                adjustflag='1',
                legacy_direct_fetch=False,
            )

            result = run_stock_data_bridge(
                args,
                runtime_root=str(runtime_root),
                service_builder=lambda **_: _FakeService(str(runtime_root)),
            )

            self.assertEqual(result['job_id'], 'job-bridge')
            self.assertTrue(output_path.exists())
            df = pd.read_csv(output_path, dtype={'股票代码': str})
            self.assertEqual(
                df.columns.tolist(),
                ['股票代码', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅'],
            )
            self.assertEqual(df.loc[0, '股票代码'], '000001')
            self.assertEqual(df.loc[0, '日期'], '2024/1/2')
            self.assertAlmostEqual(float(df.loc[0, '涨跌额']), 0.5, places=6)
            self.assertGreater(float(df.loc[0, '振幅']), 0.0)
            self.assertTrue(os.path.exists(result['manifest_path']))

    def test_script_main_delegates_to_compat_bridge(self):
        args = Namespace(
            pipeline_config_dir='./config',
            dataset_name='market_bar_1d',
            start_date='2024-01-01',
            end_date='2024-01-31',
            index_date='2024-01-31',
            output_path='./temp/stock_data.csv',
            manifest_path='',
            adjustflag='1',
            legacy_direct_fetch=False,
            runtime_root='./temp/ingestion_runtime',
        )
        bridge_result = {
            'job_id': 'job-bridge',
            'status': 'succeeded',
            'output_path': '/tmp/stock_data.csv',
            'manifest_path': '/tmp/data_manifest_stock_fetch.json',
        }

        with mock.patch.object(get_stock_data, 'parse_args', return_value=args):
            with mock.patch.object(
                get_stock_data,
                'run_stock_data_bridge',
                create=True,
                return_value=bridge_result,
            ) as bridge_mock:
                with mock.patch('builtins.print') as print_mock:
                    get_stock_data.main()

        bridge_mock.assert_called_once()
        called_args, called_kwargs = bridge_mock.call_args
        self.assertIs(called_args[0], args)
        self.assertTrue(os.path.isabs(called_kwargs['runtime_root']))
        self.assertTrue(called_kwargs['runtime_root'].endswith(os.path.join('temp', 'ingestion_runtime')))
        self.assertEqual(json.loads(print_mock.call_args.args[0]), bridge_result)


if __name__ == '__main__':
    unittest.main()
