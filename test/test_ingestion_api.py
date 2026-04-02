import os
import sys
import tempfile
import unittest

import pandas as pd
from fastapi.testclient import TestClient


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ingestion.api.app import create_app
from ingestion.models import DatasetSpec
from ingestion.service import IngestionService


class _FakeAdapter:
    adapter_name = 'fake_adapter'

    def fetch(self, request, spec):
        return pd.DataFrame(
            [
                {'instrument_id': '000001', 'trade_date': '2024-01-02', 'close': 10.5},
            ]
        )


class IngestionApiTests(unittest.TestCase):
    def test_http_endpoints_delegate_to_shared_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = DatasetSpec(
                dataset='market_bar_1d',
                domain='market',
                granularity='1d',
                source_name='fake',
                adapter_name='fake_adapter',
                request_spec={},
                schema_spec={'primary_key': ['instrument_id', 'trade_date'], 'columns': {'instrument_id': {'source': 'instrument_id'}}},
                quality_spec={'required_columns': ['instrument_id', 'trade_date', 'close']},
                storage_spec={'raw_uri': 'data/raw/{dataset}/{run_id}.csv', 'curated_uri': 'data/curated/{dataset}/{run_id}.csv'},
            )
            service = IngestionService.for_testing(
                specs={'market_bar_1d': spec},
                adapters={'fake_adapter': _FakeAdapter()},
                runtime_root=tmp,
            )
            client = TestClient(create_app(service=service, runtime_root=tmp))

            health = client.get('/health')
            self.assertEqual(health.status_code, 200)

            created = client.post('/ingestion/jobs', json={'dataset': 'market_bar_1d', 'start': '2024-01-01', 'end': '2024-01-31'})
            self.assertEqual(created.status_code, 200)
            payload = created.json()
            self.assertEqual(payload['status'], 'queued')

            job_id = payload['job_id']
            run_resp = client.post(f'/ingestion/jobs/{job_id}/run')
            self.assertEqual(run_resp.status_code, 200)
            self.assertEqual(run_resp.json()['status'], 'succeeded')

            replay_resp = client.post('/ingestion/replay', json={'job_id': job_id})
            self.assertEqual(replay_resp.status_code, 200)
            self.assertEqual(replay_resp.json()['status'], 'succeeded')


if __name__ == '__main__':
    unittest.main()
