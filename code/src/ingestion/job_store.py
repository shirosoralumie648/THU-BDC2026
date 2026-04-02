from __future__ import annotations

import json
import os
from dataclasses import asdict

from ingestion.models import IngestionJob
from ingestion.models import IngestionRequest


class LocalJobStore:
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)

    def _path(self, job_id: str) -> str:
        return os.path.join(self.root_dir, f'{job_id}.json')

    def save(self, job: IngestionJob) -> str:
        path = self._path(job.job_id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(job), f, ensure_ascii=False, indent=2)
        return path

    def get(self, job_id: str) -> IngestionJob:
        with open(self._path(job_id), 'r', encoding='utf-8') as f:
            payload = json.load(f)
        payload['request'] = IngestionRequest(**payload['request'])
        return IngestionJob(**payload)

    def list_jobs(self):
        jobs = []
        for name in sorted(os.listdir(self.root_dir)):
            if not name.endswith('.json'):
                continue
            jobs.append(self.get(name[:-5]))
        return jobs
