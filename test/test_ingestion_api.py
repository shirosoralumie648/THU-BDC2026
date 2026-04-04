import asyncio
import os
import sys
import tempfile
import unittest

import httpx
import pandas as pd


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
        asyncio.run(self._exercise_http_endpoints())

    async def _exercise_http_endpoints(self):
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
            app = create_app(service=service, runtime_root=tmp)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:
                health = await client.get('/health')
                self.assertEqual(health.status_code, 200)

                created = await client.post(
                    '/ingestion/jobs',
                    json={'dataset': 'market_bar_1d', 'start': '2024-01-01', 'end': '2024-01-31'},
                )
                self.assertEqual(created.status_code, 200)
                payload = created.json()
                self.assertEqual(payload['status'], 'queued')

                job_id = payload['job_id']
                run_resp = await client.post(f'/ingestion/jobs/{job_id}/run')
                self.assertEqual(run_resp.status_code, 200)
                self.assertEqual(run_resp.json()['status'], 'succeeded')

                replay_resp = await client.post('/ingestion/replay', json={'job_id': job_id})
                self.assertEqual(replay_resp.status_code, 200)
                self.assertEqual(replay_resp.json()['status'], 'succeeded')


if __name__ == '__main__':
    unittest.main()
