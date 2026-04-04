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


class _FailingAdapter:
    adapter_name = 'failing_adapter'

    def fetch(self, request, spec):
        raise RuntimeError('provider unavailable')


class IngestionApiTests(unittest.TestCase):
    def _build_service(self, runtime_root: str, *, adapter_name: str = 'fake_adapter', adapter=None):
        spec = DatasetSpec(
            dataset='market_bar_1d',
            domain='market',
            granularity='1d',
            source_name='fake',
            adapter_name=adapter_name,
            request_spec={},
            schema_spec={'primary_key': ['instrument_id', 'trade_date'], 'columns': {'instrument_id': {'source': 'instrument_id'}}},
            quality_spec={'required_columns': ['instrument_id', 'trade_date', 'close']},
            storage_spec={'raw_uri': 'data/raw/{dataset}/{run_id}.csv', 'curated_uri': 'data/curated/{dataset}/{run_id}.csv'},
        )
        return IngestionService.for_testing(
            specs={'market_bar_1d': spec},
            adapters={adapter_name: adapter or _FakeAdapter()},
            runtime_root=runtime_root,
        )

    def test_http_endpoints_delegate_to_shared_service(self):
        asyncio.run(self._exercise_http_endpoints())

    async def _exercise_http_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._build_service(tmp)
            app = create_app(service=service, runtime_root=tmp)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:
                health = await client.get('/health')
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json(), {'status': 'ok'})

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
                replay_payload = replay_resp.json()
                self.assertEqual(replay_payload['status'], 'succeeded')
                self.assertEqual(replay_payload['result']['parent_job_id'], job_id)

    def test_http_errors_map_to_stable_contract(self):
        asyncio.run(self._exercise_http_errors())

    async def _exercise_http_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._build_service(tmp, adapter_name='failing_adapter', adapter=_FailingAdapter())
            app = create_app(service=service, runtime_root=tmp)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:
                create_missing = await client.post(
                    '/ingestion/jobs',
                    json={'dataset': 'missing', 'start': '2024-01-01', 'end': '2024-01-31'},
                )
                self.assertEqual(create_missing.status_code, 400)
                self.assertEqual(
                    create_missing.json()['detail'],
                    {'code': 'invalid_request', 'message': 'unknown dataset: missing'},
                )

                get_missing = await client.get('/ingestion/jobs/job-missing')
                self.assertEqual(get_missing.status_code, 404)
                self.assertEqual(
                    get_missing.json()['detail'],
                    {'code': 'job_not_found', 'message': 'job not found: job-missing'},
                )

                created = await client.post(
                    '/ingestion/jobs',
                    json={'dataset': 'market_bar_1d', 'start': '2024-01-01', 'end': '2024-01-31'},
                )
                self.assertEqual(created.status_code, 200)
                job_id = created.json()['job_id']

                run_failed = await client.post(f'/ingestion/jobs/{job_id}/run')
                self.assertEqual(run_failed.status_code, 400)
                self.assertEqual(
                    run_failed.json()['detail'],
                    {'code': 'job_run_failed', 'message': 'provider unavailable'},
                )

                replay_missing_id = await client.post('/ingestion/replay', json={})
                self.assertEqual(replay_missing_id.status_code, 400)
                self.assertEqual(
                    replay_missing_id.json()['detail'],
                    {'code': 'invalid_request', 'message': 'job_id is required'},
                )


if __name__ == '__main__':
    unittest.main()
