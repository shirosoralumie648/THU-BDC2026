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

from data_manager import build_file_snapshot

from ingestion.manifests import build_ingestion_manifest
from ingestion.models import IngestionJob
from ingestion.models import IngestionRequest
from ingestion.models import IngestionResult
from predict import build_prediction_input_manifest


def _write_minimal_pipeline_config(cfg_dir: Path):
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / 'datasets.yaml').write_text(
        '\n'.join(
            [
                'version: 1',
                'datasets: {}',
            ]
        ),
        encoding='utf-8',
    )
    (cfg_dir / 'factors.yaml').write_text(
        '\n'.join(
            [
                'version: 1',
                'layer_order: []',
                'factor_nodes: []',
            ]
        ),
        encoding='utf-8',
    )
    (cfg_dir / 'storage.yaml').write_text(
        '\n'.join(
            [
                'version: 1',
                'layers:',
                '  raw: {uri_template: data/raw/{dataset}/{run_id}.csv}',
                '  curated: {uri_template: data/curated/{dataset}/{run_id}.csv}',
                '  feature_long: {uri_template: data/feature_long/{dataset}.csv}',
                '  feature_wide: {uri_template: data/feature_wide/{dataset}.csv}',
                '  datasets: {uri_template: data/datasets/{dataset}.csv}',
                '  manifests: {uri_template: data/manifests/{dataset}.json}',
            ]
        ),
        encoding='utf-8',
    )


class ManifestContractTests(unittest.TestCase):
    def test_ingestion_manifest_exposes_required_runtime_contract(self):
        job = IngestionJob(
            job_id='job-001',
            request=IngestionRequest(dataset='market_bar_1d', start='2024-01-01', end='2024-01-31'),
            status='succeeded',
            attempt=2,
            created_at='2026-04-04T00:00:00Z',
            started_at='2026-04-04T00:00:01Z',
            finished_at='2026-04-04T00:00:02Z',
            parent_job_id='job-parent',
        )
        result = IngestionResult(
            row_count=10,
            schema_hash='schema-abc',
            data_hash='data-def',
            quality_summary={'row_count': 10, 'rules': []},
            raw_paths=['/tmp/raw.csv'],
            curated_paths=['/tmp/curated.csv'],
            warnings=['empty_result'],
            errors=[],
            code_version='ignored-by-builder',
        )

        manifest = build_ingestion_manifest(job, result, code_version='deadbeef')

        self.assertEqual(manifest['action'], 'ingestion_job')
        self.assertEqual(manifest['job_id'], 'job-001')
        self.assertEqual(manifest['status'], 'succeeded')
        self.assertEqual(manifest['attempt'], 2)
        self.assertEqual(manifest['parent_job_id'], 'job-parent')
        self.assertEqual(manifest['row_count'], 10)
        self.assertEqual(manifest['schema_hash'], 'schema-abc')
        self.assertEqual(manifest['data_hash'], 'data-def')
        self.assertEqual(manifest['output_paths']['raw'], ['/tmp/raw.csv'])
        self.assertEqual(manifest['output_paths']['curated'], ['/tmp/curated.csv'])
        self.assertEqual(manifest['warnings'], ['empty_result'])
        self.assertEqual(manifest['errors'], [])
        self.assertEqual(manifest['code_version'], 'deadbeef')

    def test_build_dataset_manifest_contains_required_contract_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_dir = tmp_path / 'config'
            base_path = tmp_path / 'stock_data.csv'
            output_dir = tmp_path / 'dataset_output'
            _write_minimal_pipeline_config(cfg_dir)

            pd.DataFrame(
                [
                    {'股票代码': '000001.SZ', '日期': '2024-01-02', '收盘': 10.0},
                    {'股票代码': '000001.SZ', '日期': '2024-01-03', '收盘': 10.5},
                    {'股票代码': '000001.SZ', '日期': '2024-03-11', '收盘': 10.9},
                ]
            ).to_csv(base_path, index=False, encoding='utf-8')

            cmd = [
                sys.executable,
                os.path.join(SRC_ROOT, 'manage_data.py'),
                'build-dataset',
                '--base-input',
                str(base_path),
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
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            self.assertEqual(manifest['action'], 'build_dataset')
            self.assertTrue(bool(manifest['build_id']))
            self.assertIn('params', manifest)
            self.assertIn('inputs', manifest)
            self.assertIn('outputs', manifest)
            self.assertIn('stats', manifest)
            self.assertIn('factor_merge', manifest)
            self.assertIn('pipeline_config_validation', manifest)
            self.assertEqual(manifest['inputs']['base_input']['exists'], True)
            self.assertEqual(manifest['outputs']['train_csv']['exists'], True)
            self.assertEqual(manifest['outputs']['test_csv']['exists'], True)
            self.assertEqual(manifest['stats']['train_rows'], 2)
            self.assertEqual(manifest['stats']['test_rows'], 1)
            self.assertFalse(manifest['factor_merge']['used'])

    def test_factor_graph_manifest_contains_required_contract_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_path = tmp_path / 'stock_data.csv'
            hf_path = tmp_path / 'hf_daily.csv'
            macro_path = tmp_path / 'macro_series.csv'
            output_path = tmp_path / 'features.csv'
            manifest_path = tmp_path / 'manifest.json'

            dates = pd.date_range('2024-01-02', periods=25, freq='D')
            base_rows = []
            hf_rows = []
            for code, base_price in [('000001.SZ', 10.0), ('000002.SZ', 20.0)]:
                for i, dt in enumerate(dates):
                    close = base_price + i * 0.2
                    base_rows.append(
                        {
                            '股票代码': code,
                            '日期': dt.strftime('%Y-%m-%d'),
                            '开盘': close - 0.1,
                            '收盘': close,
                            '最高': close + 0.2,
                            '最低': close - 0.2,
                            '成交量': 1000 + i * 5,
                            '成交额': (1000 + i * 5) * close,
                            '换手率': 0.8 + i * 0.01,
                            '涨跌幅': 0.01,
                        }
                    )
                    hf_rows.append(
                        {
                            '股票代码': code,
                            '日期': dt.strftime('%Y-%m-%d'),
                            'hf_realized_vol': 0.05 + i * 0.001,
                            'hf_last_tail_ret': 0.002 + i * 0.0001,
                            'hf_last_tail_amount_share': 0.20 + i * 0.001,
                        }
                    )
            pd.DataFrame(base_rows).to_csv(base_path, index=False, encoding='utf-8')
            pd.DataFrame(hf_rows).to_csv(hf_path, index=False, encoding='utf-8')

            macro_rows = []
            for series_id, base_val in [('m2_yoy', 8.0), ('shibor_3m', 2.5), ('usdcny', 7.1)]:
                for i, dt in enumerate(dates):
                    macro_rows.append(
                        {
                            'series_id': series_id,
                            'available_time': (dt + pd.Timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'),
                            'value': base_val + i * 0.01,
                        }
                    )
            pd.DataFrame(macro_rows).to_csv(macro_path, index=False, encoding='utf-8')

            cmd = [
                sys.executable,
                os.path.join(SRC_ROOT, 'build_factor_graph.py'),
                '--pipeline-config-dir',
                './config',
                '--feature-set-version',
                'vcontract',
                '--base-input',
                str(base_path),
                '--hf-daily-input',
                str(hf_path),
                '--macro-input',
                str(macro_path),
                '--output',
                str(output_path),
                '--manifest-path',
                str(manifest_path),
                '--strict',
                '--run-id',
                'contract-run',
            ]
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
            if result.returncode != 0:
                raise AssertionError(
                    f'build_factor_graph failed: rc={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}'
                )

            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            self.assertEqual(manifest['action'], 'build_factor_graph')
            self.assertEqual(manifest['run_id'], 'contract-run')
            self.assertEqual(manifest['feature_set_version'], 'vcontract')
            self.assertTrue(bool(manifest['factor_fingerprint']))
            self.assertIn('input_data_versions', manifest)
            self.assertIn('node_status', manifest)
            self.assertIn('quality_summary', manifest)
            self.assertIn('output_paths', manifest)
            self.assertIn('execution_plan', manifest)
            self.assertTrue(manifest['output_paths']['wide_csv_snapshot']['exists'])
            self.assertEqual(manifest['pipeline_config_validation']['valid'], True)
            self.assertGreater(manifest['execution_plan']['total_nodes'], 0)

    def test_prediction_input_manifest_contains_dependency_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = tmp_path / 'data'
            data_root.mkdir(parents=True, exist_ok=True)
            structured_root = data_root / 'datasets'
            (structured_root / 'splits').mkdir(parents=True, exist_ok=True)
            (structured_root / 'raw').mkdir(parents=True, exist_ok=True)

            pd.DataFrame([{'股票代码': '000001', '日期': '2024-01-02', '收盘': 10.0}]).to_csv(
                structured_root / 'splits' / 'train.csv',
                index=False,
                encoding='utf-8',
            )
            pd.DataFrame([{'股票代码': '000001', '日期': '2024-03-11', '收盘': 10.9}]).to_csv(
                structured_root / 'splits' / 'test.csv',
                index=False,
                encoding='utf-8',
            )
            pd.DataFrame([{'股票代码': '000001', '日期': '2024-01-02', '收盘': 10.0}]).to_csv(
                structured_root / 'raw' / 'stock_data.csv',
                index=False,
                encoding='utf-8',
            )

            runtime_config = {
                'data_path': str(data_root),
                'structured_data_root': 'datasets',
            }

            manifest = build_prediction_input_manifest(runtime_config)

            self.assertTrue(bool(manifest['generated_at_utc']))
            self.assertIn('dataset_candidates', manifest)
            self.assertIn('train_csv', manifest)
            self.assertIn('test_csv', manifest)
            self.assertIn('stock_data_csv', manifest)
            self.assertEqual(manifest['train_csv']['exists'], True)
            self.assertEqual(manifest['test_csv']['exists'], True)
            self.assertEqual(manifest['stock_data_csv']['exists'], True)

    def test_build_file_snapshot_records_structured_csv_parse_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_csv = Path(tmp) / 'bad.csv'
            bad_csv.write_text('股票代码,日期\n"unterminated,2024-01-02\n', encoding='utf-8')

            snapshot = build_file_snapshot(str(bad_csv), inspect_csv=True)

            self.assertEqual(snapshot['path'], str(bad_csv.resolve()))
            self.assertTrue(snapshot['exists'])
            self.assertIn('csv', snapshot)
            self.assertEqual(snapshot['csv']['status'], 'error')
            self.assertEqual(snapshot['csv']['error_code'], 'csv_parse_error')
            self.assertTrue(snapshot['csv']['message'])
            self.assertIn('errors', snapshot)
            self.assertEqual(snapshot['errors'][0]['code'], 'csv_parse_error')
            self.assertEqual(snapshot['errors'][0]['message'], snapshot['csv']['message'])

    def test_build_file_snapshot_marks_valid_csv_metadata_as_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / 'ok.csv'
            csv_path.write_text('股票代码,日期\n000001.SZ,2024-01-02\n', encoding='utf-8')

            snapshot = build_file_snapshot(str(csv_path), inspect_csv=True)

        self.assertTrue(snapshot['exists'])
        self.assertIn('csv', snapshot)
        self.assertEqual(snapshot['csv']['status'], 'ok')
        self.assertNotIn('error_code', snapshot['csv'])
        self.assertNotIn('message', snapshot['csv'])
        self.assertNotIn('errors', snapshot)

    def test_build_file_snapshot_records_non_parse_csv_read_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / 'read-error.csv'
            csv_path.write_text('股票代码,日期\n000001.SZ,2024-01-02\n', encoding='utf-8')

            with patch('data_manager.pd.read_csv', side_effect=PermissionError('read boom')):
                snapshot = build_file_snapshot(str(csv_path), inspect_csv=True)

        self.assertTrue(snapshot['exists'])
        self.assertIn('csv', snapshot)
        self.assertEqual(snapshot['csv']['status'], 'error')
        self.assertEqual(snapshot['csv']['error_code'], 'csv_read_error')
        self.assertIn('errors', snapshot)
        self.assertEqual(snapshot['errors'][0]['code'], 'csv_read_error')
        self.assertIn('read boom', snapshot['errors'][0]['message'])

    def test_build_file_snapshot_records_stat_failure_without_hiding_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / 'good.csv'
            csv_path.write_text('股票代码,日期\n000001.SZ,2024-01-02\n', encoding='utf-8')
            with patch('data_manager.os.path.getsize', side_effect=OSError('stat boom')):
                snapshot = build_file_snapshot(str(csv_path), inspect_csv=False)

        self.assertTrue(snapshot['exists'])
        self.assertNotIn('size_bytes', snapshot)
        self.assertIn('errors', snapshot)
        self.assertEqual(snapshot['errors'][0]['code'], 'stat_failed')
        self.assertIn('stat boom', snapshot['errors'][0]['message'])


if __name__ == '__main__':
    unittest.main()
