import os
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ingestion.models import DatasetSpec
from ingestion.models import IngestionRequest
from ingestion.service import IngestionService


class _FakeAdapter:
    adapter_name = 'fake_adapter'

    def fetch(self, request, spec):
        return pd.DataFrame(
            [
                {'instrument_id': '000001', 'trade_date': '2024-01-02', 'open': 10.0, 'high': 11.0, 'low': 9.0, 'close': 10.5},
                {'instrument_id': '000002', 'trade_date': '2024-01-02', 'open': 20.0, 'high': 21.0, 'low': 19.0, 'close': 20.5},
            ]
        )


class IngestionServiceTests(unittest.TestCase):
    def test_run_sync_executes_adapter_quality_and_job_persistence(self):
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

            job = service.create_job(IngestionRequest(dataset='market_bar_1d', start='2024-01-01', end='2024-01-31'))
            finished = service.run_job(job.job_id)

            self.assertEqual(finished.status, 'succeeded')
            self.assertTrue(finished.manifest_path.endswith('.json'))
            self.assertTrue(os.path.exists(finished.manifest_path))
            self.assertTrue(os.path.exists(os.path.join(tmp, 'raw', f'{finished.job_id}.csv')))
            self.assertTrue(os.path.exists(os.path.join(tmp, 'curated', f'{finished.job_id}.csv')))

            replayed = service.replay_job(finished.job_id)
            self.assertEqual(replayed.parent_job_id, finished.job_id)
            self.assertEqual(replayed.status, 'queued')


if __name__ == '__main__':
    unittest.main()
