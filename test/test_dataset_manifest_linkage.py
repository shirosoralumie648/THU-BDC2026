import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import manage_data as manage_data_module
from data_manager import load_train_dataset_from_build_manifest
from manage_data import _resolve_factor_fingerprint_from_feature_input


class DatasetManifestLinkageTests(unittest.TestCase):
    def test_build_dataset_manifest_infers_factor_fingerprint_from_adjacent_factor_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_dir = tmp_path / 'config'
            cfg_dir.mkdir(parents=True, exist_ok=True)
            output_dir = tmp_path / 'dataset_output'
            output_dir.mkdir(parents=True, exist_ok=True)
            features_dir = tmp_path / 'features'
            features_dir.mkdir(parents=True, exist_ok=True)

            (cfg_dir / 'datasets.yaml').write_text('version: 1\ndatasets: {}\n', encoding='utf-8')
            (cfg_dir / 'storage.yaml').write_text('version: 1\nlayers: {raw: {}, curated: {}, feature_long: {}, feature_wide: {}, datasets: {}, manifests: {}}\n', encoding='utf-8')
            (cfg_dir / 'factors.yaml').write_text(
                '\n'.join(
                    [
                        'version: 1',
                        'factor_views:',
                        '  - layout: wide',
                        '    export:',
                        '      csv_compat_uri: data/datasets/features/{feature_set_version}.csv',
                        'build_manifest:',
                        '  output_uri: data/manifests/factor_build/{feature_set_version}/{run_date}/{run_id}.json',
                    ]
                ),
                encoding='utf-8',
            )

            base_path = tmp_path / 'stock_data.csv'
            feature_path = features_dir / 'features.csv'
            factor_manifest_path = features_dir / 'factor_build_manifest.json'

            pd.DataFrame(
                [
                    {'股票代码': '000001.SZ', '日期': '2024-01-02', '收盘': 10.0},
                    {'股票代码': '000001.SZ', '日期': '2024-01-03', '收盘': 10.5},
                    {'股票代码': '000001.SZ', '日期': '2024-03-11', '收盘': 10.9},
                ]
            ).to_csv(base_path, index=False, encoding='utf-8')
            pd.DataFrame(
                [
                    {'股票代码': '000001.SZ', '日期': '2024-01-02', 'f_ret_1d': 0.01},
                    {'股票代码': '000001.SZ', '日期': '2024-01-03', 'f_ret_1d': 0.02},
                    {'股票代码': '000001.SZ', '日期': '2024-03-11', 'f_ret_1d': 0.03},
                ]
            ).to_csv(feature_path, index=False, encoding='utf-8')
            factor_manifest_path.write_text(
                json.dumps(
                    {
                        'action': 'build_factor_graph',
                        'feature_set_version': 'vtest',
                        'factor_fingerprint': 'fingerprint-from-factor-manifest',
                        'output_paths': {
                            'wide_csv': str(feature_path),
                            'wide_csv_snapshot': {'path': str(feature_path)},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding='utf-8',
            )

            cmd = [
                sys.executable,
                os.path.join(SRC_ROOT, 'manage_data.py'),
                'build-dataset',
                '--base-input',
                str(base_path),
                '--feature-input',
                str(feature_path),
                '--feature-set-version',
                'vtest',
                '--pipeline-config-dir',
                str(cfg_dir),
                '--output-dir',
                str(output_dir),
                '--train-start',
                '2024-01-01',
                '--train-end',
                '2024-01-31',
                '--test-start',
                '2024-03-01',
                '--test-end',
                '2024-03-31',
            ]
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
            if result.returncode != 0:
                raise AssertionError(
                    f'build-dataset failed: rc={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}'
                )

            manifest_path = output_dir / 'data_manifest_dataset_build.json'
            self.assertTrue(manifest_path.exists())
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            self.assertEqual(manifest.get('factor_fingerprint'), 'fingerprint-from-factor-manifest')
            self.assertEqual(manifest.get('params', {}).get('factor_fingerprint'), 'fingerprint-from-factor-manifest')

            config = {
                'output_dir': str(output_dir),
                'dataset_build_manifest_path': str(manifest_path),
                'use_dataset_build_manifest': True,
                'dataset_manifest_strict': False,
                'dataset_manifest_require_factor_fingerprint': True,
            }
            train_path, info = load_train_dataset_from_build_manifest(
                config,
                {'factor_fingerprint': 'fingerprint-from-factor-manifest'},
            )
            self.assertEqual(train_path, str((output_dir / 'train.csv').resolve()))
            self.assertEqual(info.get('factor_fingerprint'), 'fingerprint-from-factor-manifest')
            self.assertEqual(info.get('errors'), [])

            _, mismatch_info = load_train_dataset_from_build_manifest(
                config,
                {'factor_fingerprint': 'different-fingerprint'},
            )
            self.assertTrue(
                any('当前激活因子指纹不一致' in msg for msg in mismatch_info.get('errors', [])),
                msg=str(mismatch_info),
            )

    def test_load_train_dataset_from_build_manifest_reuses_cached_manifest_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / 'dataset_output'
            output_dir.mkdir(parents=True, exist_ok=True)
            train_csv = output_dir / 'train.csv'
            manifest_path = output_dir / 'data_manifest_dataset_build.json'

            pd.DataFrame(
                [
                    {'股票代码': '000001.SZ', '日期': '2024-01-02', '收盘': 10.0},
                    {'股票代码': '000001.SZ', '日期': '2024-01-03', '收盘': 10.5},
                ]
            ).to_csv(train_csv, index=False, encoding='utf-8')
            manifest_path.write_text(
                json.dumps(
                    {
                        'action': 'build_dataset',
                        'build_id': 'cached-build',
                        'feature_set_version': 'vcache',
                        'factor_fingerprint': 'cache-fingerprint',
                        'outputs': {
                            'train_csv': {
                                'path': str(train_csv),
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding='utf-8',
            )

            config = {
                'output_dir': str(output_dir),
                'dataset_build_manifest_path': str(manifest_path),
                'use_dataset_build_manifest': True,
                'dataset_manifest_strict': False,
                'dataset_manifest_require_factor_fingerprint': True,
            }

            with patch('data_manager.json.load', wraps=json.load) as mocked:
                first_path, first_info = load_train_dataset_from_build_manifest(
                    config,
                    {'factor_fingerprint': 'cache-fingerprint'},
                )
                second_path, second_info = load_train_dataset_from_build_manifest(
                    config,
                    {'factor_fingerprint': 'cache-fingerprint'},
                )

            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(first_path, second_path)
            self.assertEqual(first_info.get('factor_fingerprint'), 'cache-fingerprint')
            self.assertEqual(second_info.get('factor_fingerprint'), 'cache-fingerprint')

    def test_resolve_factor_fingerprint_reuses_cached_manifest_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            features_dir = tmp_path / 'features'
            features_dir.mkdir(parents=True, exist_ok=True)
            feature_path = features_dir / 'features.csv'
            manifest_path = features_dir / 'factor_build_manifest.json'

            pd.DataFrame(
                [
                    {'股票代码': '000001.SZ', '日期': '2024-01-02', 'f_ret_1d': 0.01},
                ]
            ).to_csv(feature_path, index=False, encoding='utf-8')
            manifest_path.write_text(
                json.dumps(
                    {
                        'action': 'build_factor_graph',
                        'feature_set_version': 'vcache',
                        'factor_fingerprint': 'fingerprint-cached',
                        'output_paths': {
                            'wide_csv': str(feature_path),
                            'wide_csv_snapshot': {'path': str(feature_path)},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding='utf-8',
            )

            with patch('manage_data.json.load', wraps=manage_data_module.json.load) as mocked:
                first = _resolve_factor_fingerprint_from_feature_input(
                    str(feature_path),
                    feature_set_version='vcache',
                    pipeline_configs={},
                )
                second = _resolve_factor_fingerprint_from_feature_input(
                    str(feature_path),
                    feature_set_version='vcache',
                    pipeline_configs={},
                )

            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(first, 'fingerprint-cached')
            self.assertEqual(second, 'fingerprint-cached')


if __name__ == '__main__':
    unittest.main()
