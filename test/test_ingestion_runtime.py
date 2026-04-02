import os
import sys
import tempfile
import unittest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ingestion.job_store import LocalJobStore
from ingestion.manifests import build_ingestion_manifest
from ingestion.models import IngestionJob
from ingestion.models import IngestionRequest
from ingestion.models import IngestionResult
from ingestion.storage import render_storage_target


class IngestionRuntimeTests(unittest.TestCase):
    def test_local_job_store_roundtrip_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalJobStore(tmp)
            req = IngestionRequest(dataset='market_bar_1d', start='2024-01-01', end='2024-01-31')
            job = IngestionJob(job_id='job-001', request=req, status='queued')
            store.save(job)

            loaded = store.get('job-001')
            self.assertEqual(loaded.job_id, 'job-001')
            self.assertEqual(loaded.status, 'queued')

            target = render_storage_target(
                'data/raw/source={source}/dataset={dataset}/ingest_date={ingest_date}/part-{run_id}.parquet',
                source='baostock',
                dataset='market_bar_1d',
                ingest_date='2026-03-28',
                run_id='job-001',
            )
            self.assertIn('market_bar_1d', target)

            result = IngestionResult(
                row_count=10,
                schema_hash='abc',
                data_hash='def',
                quality_summary={'row_count': 10},
                raw_paths=['/tmp/raw.parquet'],
                curated_paths=['/tmp/curated.parquet'],
            )
            manifest = build_ingestion_manifest(job, result, code_version='deadbeef')
            self.assertEqual(manifest['job_id'], 'job-001')
            self.assertEqual(manifest['row_count'], 10)
            self.assertEqual(manifest['code_version'], 'deadbeef')


if __name__ == '__main__':
    unittest.main()
