import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')


class IngestionCliTests(unittest.TestCase):
    def test_manage_data_ingest_datasets_lists_registry_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dir = Path(tmp)
            (cfg_dir / 'datasets.yaml').write_text(
                '\n'.join(
                    [
                        'version: 1',
                        'datasets:',
                        '  market_bar_1d:',
                        '    domain: market',
                        '    granularity: 1d',
                        '    source: {name: baostock, adapter: baostock_daily}',
                        '    request: {}',
                        '    schema: {primary_key: [instrument_id, trade_date], columns: {instrument_id: {source: code}}}',
                        '    quality: {required_columns: [instrument_id, trade_date, close]}',
                        '    storage: {raw_uri: data/raw/mock.csv, curated_uri: data/curated/mock.csv}',
                    ]
                ),
                encoding='utf-8',
            )
            (cfg_dir / 'factors.yaml').write_text('version: 1\nlayer_order: []\nfactor_nodes: []\n', encoding='utf-8')
            (cfg_dir / 'storage.yaml').write_text(
                'version: 1\nlayers: {raw: {}, curated: {}, feature_long: {}, feature_wide: {}, datasets: {}, manifests: {}}\n',
                encoding='utf-8',
            )

            result = subprocess.run(
                [
                    sys.executable,
                    os.path.join(SRC_ROOT, 'manage_data.py'),
                    'ingest',
                    'datasets',
                    '--pipeline-config-dir',
                    str(cfg_dir),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + '\n' + result.stderr)
            self.assertIn('market_bar_1d', result.stdout)

    def test_manage_data_ingest_factor_pipeline_runs_orchestration(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_path = tmp_path / 'stock_data.csv'
            minute_path = tmp_path / 'minute.csv'
            macro_path = tmp_path / 'macro.csv'
            out_dir = tmp_path / 'out'

            base_rows = []
            minute_rows = []
            dates = ['2024-01-02', '2024-01-03']
            for code, base_price in [('000001', 10.0), ('000002', 20.0)]:
                for d_idx, dt in enumerate(dates):
                    close_daily = base_price + d_idx * 0.2
                    base_rows.append(
                        {
                            '股票代码': code,
                            '日期': dt,
                            '开盘': close_daily - 0.1,
                            '收盘': close_daily,
                            '最高': close_daily + 0.2,
                            '最低': close_daily - 0.2,
                            '成交量': 1000,
                            '成交额': close_daily * 1000,
                            '换手率': 1.0,
                            '涨跌幅': 0.01,
                        }
                    )
                    for i in range(30):
                        minute_rows.append(
                            {
                                '股票代码': code,
                                'datetime': f'{dt} 14:{i:02d}:00',
                                'close': close_daily + i * 0.001,
                                'amount': (500 + i) * (close_daily + i * 0.001),
                            }
                        )
            import pandas as pd

            pd.DataFrame(base_rows).to_csv(base_path, index=False, encoding='utf-8')
            pd.DataFrame(minute_rows).to_csv(minute_path, index=False, encoding='utf-8')
            pd.DataFrame(
                [
                    {'series_id': 'm2_yoy', 'available_time': '2024-01-02 09:00:00', 'value': 8.0},
                    {'series_id': 'shibor_3m', 'available_time': '2024-01-02 09:00:00', 'value': 2.5},
                    {'series_id': 'usdcny', 'available_time': '2024-01-02 09:00:00', 'value': 7.1},
                    {'series_id': 'm2_yoy', 'available_time': '2024-01-03 09:00:00', 'value': 8.1},
                    {'series_id': 'shibor_3m', 'available_time': '2024-01-03 09:00:00', 'value': 2.6},
                    {'series_id': 'usdcny', 'available_time': '2024-01-03 09:00:00', 'value': 7.2},
                ]
            ).to_csv(macro_path, index=False, encoding='utf-8')

            result = subprocess.run(
                [
                    sys.executable,
                    os.path.join(SRC_ROOT, 'manage_data.py'),
                    'ingest',
                    'factor-pipeline',
                    '--pipeline-config-dir',
                    './config',
                    '--base-input',
                    str(base_path),
                    '--hf-minute-input',
                    str(minute_path),
                    '--macro-input',
                    str(macro_path),
                    '--output-dir',
                    str(out_dir),
                    '--feature-set-version',
                    'vcli',
                    '--run-id',
                    'cli-run',
                    '--strict',
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + '\n' + result.stderr)
            self.assertTrue((out_dir / 'factor_graph.csv').exists())


if __name__ == '__main__':
    unittest.main()
