import os
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')


def _write_minimal_pipeline_config(cfg_dir: Path, *, manifest_uri: str = ''):
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / 'datasets.yaml').write_text('version: 1\ndatasets: {}\n', encoding='utf-8')
    factor_lines = ['version: 1', 'layer_order: []', 'factor_nodes: []']
    if manifest_uri:
        factor_lines.extend(['build_manifest:', f'  output_uri: "{manifest_uri}"'])
    (cfg_dir / 'factors.yaml').write_text('\n'.join(factor_lines) + '\n', encoding='utf-8')
    (cfg_dir / 'storage.yaml').write_text(
        '\n'.join(
            [
                'version: 1',
                'layers:',
                '  raw: {uri_template: "data/raw/{dataset}/{run_id}.csv"}',
                '  curated: {uri_template: "data/curated/{dataset}/{run_id}.csv"}',
                '  feature_long: {uri_template: "data/feature_long/{dataset}.csv"}',
                '  feature_wide: {uri_template: "data/feature_wide/{dataset}.csv"}',
                '  datasets: {uri_template: "data/datasets/{dataset}.csv"}',
                '  manifests: {uri_template: "data/manifests/{dataset}.json"}',
            ]
        ),
        encoding='utf-8',
    )


class CliErrorPathTests(unittest.TestCase):
    def test_build_dataset_reports_missing_base_input_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / 'out'
            cmd = [
                sys.executable,
                os.path.join(SRC_ROOT, 'manage_data.py'),
                'build-dataset',
                '--base-input',
                str(Path(tmp) / 'missing.csv'),
                '--output-dir',
                str(output_dir),
            ]
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

            self.assertEqual(result.returncode, 2)
            self.assertIn('未找到基础输入文件', result.stderr)
            self.assertNotIn('Traceback', result.stderr)

    def test_build_dataset_reports_missing_feature_input_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_dir = tmp_path / 'config'
            base_path = tmp_path / 'stock_data.csv'
            output_dir = tmp_path / 'out'
            _write_minimal_pipeline_config(cfg_dir)
            pd.DataFrame([{'股票代码': '000001.SZ', '日期': '2024-01-02', '收盘': 10.0}]).to_csv(
                base_path,
                index=False,
                encoding='utf-8',
            )

            cmd = [
                sys.executable,
                os.path.join(SRC_ROOT, 'manage_data.py'),
                'build-dataset',
                '--base-input',
                str(base_path),
                '--feature-input',
                str(tmp_path / 'missing_features.csv'),
                '--pipeline-config-dir',
                str(cfg_dir),
                '--output-dir',
                str(output_dir),
            ]
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

            self.assertEqual(result.returncode, 2)
            self.assertIn('未找到因子输入文件', result.stderr)
            self.assertNotIn('Traceback', result.stderr)

    def test_build_dataset_reports_corrupt_factor_manifest_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_dir = tmp_path / 'config'
            base_path = tmp_path / 'stock_data.csv'
            feature_path = tmp_path / 'features.csv'
            output_dir = tmp_path / 'out'
            manifest_dir = tmp_path / 'manifests'
            manifest_path = manifest_dir / 'factor_build.json'

            manifest_dir.mkdir(parents=True, exist_ok=True)
            _write_minimal_pipeline_config(cfg_dir, manifest_uri=str(manifest_path))
            manifest_path.write_text('{broken', encoding='utf-8')

            pd.DataFrame([{'股票代码': '000001.SZ', '日期': '2024-01-02', '收盘': 10.0}]).to_csv(
                base_path,
                index=False,
                encoding='utf-8',
            )
            pd.DataFrame([{'股票代码': '000001.SZ', '日期': '2024-01-02', 'alpha_001': 1.0}]).to_csv(
                feature_path,
                index=False,
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
                '--pipeline-config-dir',
                str(cfg_dir),
                '--output-dir',
                str(output_dir),
                '--train-start',
                '2024-01-01',
                '--train-end',
                '2024-01-31',
                '--test-start',
                '2024-02-01',
                '--test-end',
                '2024-02-29',
            ]
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

            self.assertEqual(result.returncode, 2)
            self.assertIn('解析 factor build manifest 失败', result.stderr)
            self.assertNotIn('Traceback', result.stderr)

    def test_build_dataset_ignores_unrelated_broken_json_when_manifest_uri_points_to_valid_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_dir = tmp_path / 'config'
            base_path = tmp_path / 'stock_data.csv'
            feature_path = tmp_path / 'features.csv'
            output_dir = tmp_path / 'out'
            manifest_dir = tmp_path / 'manifests'
            manifest_path = manifest_dir / 'factor_build.json'
            unrelated_broken_json = tmp_path / 'broken.json'

            manifest_dir.mkdir(parents=True, exist_ok=True)
            _write_minimal_pipeline_config(cfg_dir, manifest_uri=str(manifest_path))
            unrelated_broken_json.write_text('{broken', encoding='utf-8')

            pd.DataFrame([{'股票代码': '000001.SZ', '日期': '2024-01-02', '收盘': 10.0}]).to_csv(
                base_path,
                index=False,
                encoding='utf-8',
            )
            pd.DataFrame([{'股票代码': '000001.SZ', '日期': '2024-01-02', 'alpha_001': 1.0}]).to_csv(
                feature_path,
                index=False,
                encoding='utf-8',
            )

            manifest_path.write_text(
                json.dumps(
                    {
                        'action': 'build_factor_graph',
                        'feature_set_version': 'v1',
                        'factor_fingerprint': 'fp-valid',
                        'output_paths': {
                            'wide_csv': str(feature_path),
                        },
                    },
                    ensure_ascii=False,
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
                '--pipeline-config-dir',
                str(cfg_dir),
                '--output-dir',
                str(output_dir),
                '--train-start',
                '2024-01-01',
                '--train-end',
                '2024-01-31',
                '--test-start',
                '2024-02-01',
                '--test-end',
                '2024-02-29',
            ]
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

            self.assertEqual(result.returncode, 0)
            self.assertNotIn('解析 factor build manifest 失败', result.stderr)
            self.assertNotIn('Traceback', result.stderr)

    def test_build_dataset_release_profile_requires_factor_fingerprint_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_dir = tmp_path / 'config'
            base_path = tmp_path / 'stock_data.csv'
            feature_path = tmp_path / 'features.csv'
            output_dir = tmp_path / 'out'

            _write_minimal_pipeline_config(cfg_dir)
            pd.DataFrame([{'股票代码': '000001.SZ', '日期': '2024-01-02', '收盘': 10.0}]).to_csv(
                base_path,
                index=False,
                encoding='utf-8',
            )
            pd.DataFrame([{'股票代码': '000001.SZ', '日期': '2024-01-02', 'alpha_001': 1.0}]).to_csv(
                feature_path,
                index=False,
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
                '--pipeline-config-dir',
                str(cfg_dir),
                '--output-dir',
                str(output_dir),
                '--train-start',
                '2024-01-01',
                '--train-end',
                '2024-01-31',
                '--test-start',
                '2024-02-01',
                '--test-end',
                '2024-02-29',
                '--profile',
                'release',
            ]
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

            self.assertEqual(result.returncode, 2)
            self.assertIn('release profile preflight failed', result.stderr)
            self.assertIn('factor_fingerprint', result.stderr)
            self.assertNotIn('Traceback', result.stderr)

    def test_ingest_get_reports_missing_job_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / 'runtime'
            cmd = [
                sys.executable,
                os.path.join(SRC_ROOT, 'manage_data.py'),
                'ingest',
                'get',
                '--job-id',
                'job-missing',
                '--config-dir',
                './config',
                '--runtime-root',
                str(runtime_root),
            ]
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

            self.assertEqual(result.returncode, 2)
            self.assertIn('job not found: job-missing', result.stderr)
            self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
