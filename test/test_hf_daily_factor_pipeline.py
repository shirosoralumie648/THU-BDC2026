import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from data_manager import merge_hf_daily_factors
from data_manager import normalize_stock_code_series


class HFDailyFactorPipelineTests(unittest.TestCase):
    def test_normalize_stock_code_series_handles_common_formats(self):
        series = pd.Series(['000001.SZ', 'SZ000002', '000003', 'sh600519', None, ''])
        normalized = normalize_stock_code_series(series).tolist()
        self.assertEqual(normalized, ['000001', '000002', '000003', '600519', '', ''])

    def test_merge_hf_daily_factors_merges_by_normalized_stock_and_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            hf_path = os.path.join(tmp, 'hf_daily_factors.csv')
            pd.DataFrame(
                [
                    {'股票代码': '000001', '日期': '2024-01-02', 'hf_alpha': 1.1},
                    {'股票代码': '000002', '日期': '2024-01-02', 'hf_alpha': 2.2},
                ]
            ).to_csv(hf_path, index=False)

            base = pd.DataFrame(
                [
                    {'股票代码': '000001.SZ', '日期': '2024-01-02', 'label': 0.1},
                    {'股票代码': 'SZ000002', '日期': '2024-01-02', 'label': 0.2},
                ]
            )
            cfg = {
                'use_hf_daily_factor_merge': True,
                'hf_daily_factor_path': hf_path,
                'hf_factor_required': True,
                'hf_factor_prefix': '',
                'hf_factor_merge_how': 'left',
                'hf_factor_drop_duplicate_keep': 'last',
                'hf_factor_allow_overwrite_columns': False,
                'hf_factor_columns': [],
                'hf_factor_stock_col': '股票代码',
                'hf_factor_date_col': '日期',
            }

            merged, meta = merge_hf_daily_factors(base, cfg)

            self.assertTrue(meta.get('used', False))
            self.assertEqual(meta.get('factor_count'), 1)
            self.assertAlmostEqual(float(meta.get('coverage', 0.0)), 1.0, places=6)
            self.assertIn('hf_alpha', merged.columns)
            result = merged[['股票代码', 'hf_alpha']].sort_values('股票代码').reset_index(drop=True)
            self.assertEqual(result['股票代码'].tolist(), ['000001.SZ', 'SZ000002'])
            self.assertEqual(result['hf_alpha'].tolist(), [1.1, 2.2])

    def test_merge_hf_daily_factors_required_missing_file_raises(self):
        base = pd.DataFrame([{'股票代码': '000001', '日期': '2024-01-02'}])
        cfg = {
            'use_hf_daily_factor_merge': True,
            'hf_daily_factor_path': '/tmp/not_exists_hf_daily_factors.csv',
            'hf_factor_required': True,
            'data_path': '/tmp',
        }
        with self.assertRaises(FileNotFoundError):
            merge_hf_daily_factors(base, cfg)

    def test_build_hf_daily_factors_script_multi_source_resample(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in1 = tmp_path / 'hf_part1.csv'
            in2 = tmp_path / 'hf_part2.csv'
            out = tmp_path / 'hf_daily_factors.csv'
            manifest = tmp_path / 'manifest.json'

            rows_part1 = []
            for code, start in [('000001.SZ', 10.0), ('000002.SZ', 20.0)]:
                for i in range(15):
                    ts = pd.Timestamp('2024-01-02 09:30:00') + pd.Timedelta(minutes=i)
                    close = start + i * 0.1
                    vol = 1000 + i * 10
                    amt = close * vol
                    rows_part1.append(
                        {
                            '股票代码': code,
                            'datetime': ts.strftime('%Y-%m-%d %H:%M:%S'),
                            'close': close,
                            'volume': vol,
                            'amount': amt,
                        }
                    )
            pd.DataFrame(rows_part1).to_csv(in1, index=False)

            rows_part2 = []
            for i in range(15):
                ts = pd.Timestamp('2024-01-03 09:30:00') + pd.Timedelta(minutes=i)
                close = 11.0 + i * 0.05
                vol = 1200 + i * 5
                amt = close * vol
                rows_part2.append(
                    {
                        '股票代码': '000001.SZ',
                        'datetime': ts.strftime('%Y-%m-%d %H:%M:%S'),
                        'close': close,
                        'volume': vol,
                        'amount': amt,
                    }
                )
            pd.DataFrame(rows_part2).to_csv(in2, index=False)

            cmd = [
                sys.executable,
                os.path.join(SRC_ROOT, 'build_hf_daily_factors.py'),
                '--input',
                str(in1),
                '--input',
                str(in2),
                '--resample-minutes',
                '5',
                '--min-bars',
                '2',
                '--output',
                str(out),
                '--manifest-path',
                str(manifest),
            ]
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
            if result.returncode != 0:
                raise AssertionError(
                    f'build_hf_daily_factors failed: rc={result.returncode}\n'
                    f'stdout={result.stdout}\n'
                    f'stderr={result.stderr}'
                )

            self.assertTrue(out.exists())
            self.assertTrue(manifest.exists())

            df = pd.read_csv(out, dtype={'股票代码': str})
            stock_codes = set(df['股票代码'].astype(str).str.zfill(6).tolist())
            self.assertEqual(stock_codes, {'000001', '000002'})
            self.assertEqual(len(df), 3)
            self.assertIn('hf_bar_count_raw', df.columns)
            self.assertIn('hf_bar_count_m05', df.columns)

            with open(manifest, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            variants = payload.get('summary', {}).get('variants', [])
            self.assertEqual(variants, ['raw', 'm05'])


if __name__ == '__main__':
    unittest.main()
