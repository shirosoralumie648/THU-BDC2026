import os
import sys
import tempfile
import unittest

import pandas as pd
from fastapi import HTTPException


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ingestion.api.app import create_app
from ingestion.models import DatasetSpec
from ingestion.service import IngestionService


def _route_handler(app, path: str, method: str):
    for route in app.routes:
        if getattr(route, 'path', None) == path and method in getattr(route, 'methods', set()):
            return route.endpoint
    raise AssertionError(f'route not found: {method} {path}')


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

    def test_route_handlers_delegate_to_shared_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._build_service(tmp)
            app = create_app(service=service, runtime_root=tmp)
            health = _route_handler(app, '/health', 'GET')
            create_job = _route_handler(app, '/ingestion/jobs', 'POST')
            run_job = _route_handler(app, '/ingestion/jobs/{job_id}/run', 'POST')
            replay_job = _route_handler(app, '/ingestion/replay', 'POST')

            self.assertEqual(health(), {'status': 'ok'})

            created = create_job({'dataset': 'market_bar_1d', 'start': '2024-01-01', 'end': '2024-01-31'})
            self.assertEqual(created['status'], 'queued')

            job_id = created['job_id']
            run_resp = run_job(job_id)
            self.assertEqual(run_resp['status'], 'succeeded')

            replay_resp = replay_job({'job_id': job_id})
            self.assertEqual(replay_resp['status'], 'succeeded')
            self.assertEqual(replay_resp['result']['parent_job_id'], job_id)

    def test_route_handlers_map_errors_to_stable_http_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._build_service(tmp, adapter_name='failing_adapter', adapter=_FailingAdapter())
            app = create_app(service=service, runtime_root=tmp)
            create_job = _route_handler(app, '/ingestion/jobs', 'POST')
            get_job = _route_handler(app, '/ingestion/jobs/{job_id}', 'GET')
            run_job = _route_handler(app, '/ingestion/jobs/{job_id}/run', 'POST')
            replay_job = _route_handler(app, '/ingestion/replay', 'POST')

            with self.assertRaises(HTTPException) as create_ctx:
                create_job({'dataset': 'missing', 'start': '2024-01-01', 'end': '2024-01-31'})
            self.assertEqual(create_ctx.exception.status_code, 400)
            self.assertEqual(
                create_ctx.exception.detail,
                {'code': 'invalid_request', 'message': 'unknown dataset: missing'},
            )

            with self.assertRaises(HTTPException) as get_ctx:
                get_job('job-missing')
            self.assertEqual(get_ctx.exception.status_code, 404)
            self.assertEqual(
                get_ctx.exception.detail,
                {'code': 'job_not_found', 'message': 'job not found: job-missing'},
            )

            created = create_job({'dataset': 'market_bar_1d', 'start': '2024-01-01', 'end': '2024-01-31'})
            with self.assertRaises(HTTPException) as run_ctx:
                run_job(created['job_id'])
            self.assertEqual(run_ctx.exception.status_code, 400)
            self.assertEqual(
                run_ctx.exception.detail,
                {'code': 'job_run_failed', 'message': 'provider unavailable'},
            )

            with self.assertRaises(HTTPException) as replay_ctx:
                replay_job({})
            self.assertEqual(replay_ctx.exception.status_code, 400)
            self.assertEqual(
                replay_ctx.exception.detail,
                {'code': 'invalid_request', 'message': 'job_id is required'},
            )


if __name__ == '__main__':
    unittest.main()
