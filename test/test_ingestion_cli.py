import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')


class IngestionCliTests(unittest.TestCase):
    def test_manage_data_ingest_create_and_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / 'runtime'
            create_cmd = [
                sys.executable,
                os.path.join(SRC_ROOT, 'manage_data.py'),
                'ingest',
                'create',
                '--dataset',
                'market_bar_1d',
                '--start',
                '2024-01-01',
                '--end',
                '2024-01-31',
                '--config-dir',
                './config',
                '--runtime-root',
                str(runtime_root),
            ]
            create_result = subprocess.run(create_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
            if create_result.returncode != 0:
                raise AssertionError(f'create failed\nstdout={create_result.stdout}\nstderr={create_result.stderr}')
            payload = json.loads(create_result.stdout)
            self.assertEqual(payload['dataset'], 'market_bar_1d')
            job_id = payload['job_id']

            get_cmd = [
                sys.executable,
                os.path.join(SRC_ROOT, 'manage_data.py'),
                'ingest',
                'get',
                '--job-id',
                job_id,
                '--config-dir',
                './config',
                '--runtime-root',
                str(runtime_root),
            ]
            get_result = subprocess.run(get_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
            if get_result.returncode != 0:
                raise AssertionError(f'get failed\nstdout={get_result.stdout}\nstderr={get_result.stderr}')
            loaded = json.loads(get_result.stdout)
            self.assertEqual(loaded['job_id'], job_id)
            self.assertEqual(loaded['status'], 'queued')


if __name__ == '__main__':
    unittest.main()
