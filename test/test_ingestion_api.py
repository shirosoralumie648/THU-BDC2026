import os
import sys
import unittest

from fastapi.testclient import TestClient


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ingestion.api.app import create_app


class IngestionApiTests(unittest.TestCase):
    def test_health_and_dataset_endpoints(self):
        app = create_app(runtime_root='./temp/ingestion_api_test_runtime')
        client = TestClient(app)

        health = client.get('/healthz')
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()['status'], 'ok')

        datasets = client.get('/ingestion/datasets')
        self.assertEqual(datasets.status_code, 200)
        self.assertTrue(any(item['dataset'] == 'market_bar_1d' for item in datasets.json()))

        created = client.post(
            '/ingestion/jobs',
            json={'dataset': 'market_bar_1d', 'start': '2024-01-01', 'end': '2024-01-31'},
        )
        self.assertEqual(created.status_code, 200)
        job_id = created.json()['job_id']

        details = client.get(f'/ingestion/jobs/{job_id}')
        self.assertEqual(details.status_code, 200)
        self.assertEqual(details.json()['job_id'], job_id)

        replay = client.post('/ingestion/replay', json={'job_id': job_id})
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay.json()['source_job_id'], job_id)


if __name__ == '__main__':
    unittest.main()
