import json
import os
from datetime import datetime
from datetime import timezone
from uuid import uuid4

from ingestion.job_store import LocalJobStore
from ingestion.models import IngestionJob
from ingestion.quality import QualityGate
from ingestion.runner import JobRunner


class IngestionService:
    def __init__(self, *, specs, adapters, runtime_root: str):
        self.specs = specs
        self.adapters = adapters
        self.runtime_root = runtime_root
        self.job_store = LocalJobStore(os.path.join(runtime_root, 'jobs'))
        self.quality_gate = QualityGate()
        self.runner = JobRunner(
            quality_gate=self.quality_gate,
            job_store=self.job_store,
            manifest_writer=self._write_manifest,
            runtime_root=runtime_root,
        )

    @classmethod
    def for_testing(cls, *, specs, adapters, runtime_root: str):
        return cls(specs=specs, adapters=adapters, runtime_root=runtime_root)

    def _write_manifest(self, job_id: str, manifest: dict) -> str:
        path = os.path.join(self.runtime_root, 'manifests', f'{job_id}.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return path

    def create_job(self, request):
        job = IngestionJob(
            job_id=f'job-{uuid4().hex[:12]}',
            request=request,
            status='queued',
            created_at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        )
        self.job_store.save(job)
        return job

    def get_job(self, job_id: str):
        return self.job_store.get(job_id)

    def create_and_run(self, request):
        job = self.create_job(request)
        return self.run_job(job.job_id)

    def run_job(self, job_id: str):
        job = self.get_job(job_id)
        spec = self.specs[job.request.dataset]
        adapter = self.adapters[spec.adapter_name]
        return self.runner.run(job, spec, adapter)

    def load_manifest(self, job_or_id):
        if isinstance(job_or_id, str):
            job = self.get_job(job_or_id)
            manifest_path = job.manifest_path
        else:
            manifest_path = getattr(job_or_id, 'manifest_path', '')
        if not manifest_path:
            raise ValueError('manifest path is empty')
        with open(manifest_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}

    def replay_job(self, job_id: str):
        prior = self.get_job(job_id)
        replayed = self.create_job(prior.request)
        replayed.parent_job_id = prior.job_id
        self.job_store.save(replayed)
        return replayed
