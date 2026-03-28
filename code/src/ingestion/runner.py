import hashlib
import os
from datetime import datetime
from datetime import timezone

from ingestion.manifests import build_ingestion_manifest
from ingestion.models import IngestionResult


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


class JobRunner:
    def __init__(self, *, quality_gate, job_store, manifest_writer, runtime_root: str):
        self.quality_gate = quality_gate
        self.job_store = job_store
        self.manifest_writer = manifest_writer
        self.runtime_root = runtime_root

    def run(self, job, spec, adapter):
        job.status = 'running'
        job.started_at = _utc_now()
        self.job_store.save(job)

        df = adapter.fetch(job.request, spec)
        self.quality_gate.validate(df, spec)

        raw_path = os.path.join(self.runtime_root, 'raw', f'{job.job_id}.csv')
        curated_path = os.path.join(self.runtime_root, 'curated', f'{job.job_id}.csv')
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        os.makedirs(os.path.dirname(curated_path), exist_ok=True)
        df.to_csv(raw_path, index=False)
        df.to_csv(curated_path, index=False)

        payload = df.to_csv(index=False).encode('utf-8')
        result = IngestionResult(
            row_count=len(df),
            schema_hash=hashlib.md5(','.join(df.columns).encode('utf-8')).hexdigest(),
            data_hash=hashlib.md5(payload).hexdigest(),
            quality_summary={'row_count': int(len(df))},
            raw_paths=[raw_path],
            curated_paths=[curated_path],
        )
        manifest = build_ingestion_manifest(job, result, code_version='')
        job.manifest_path = self.manifest_writer(job.job_id, manifest)
        job.status = 'succeeded'
        job.finished_at = _utc_now()
        self.job_store.save(job)
        return job
