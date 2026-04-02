from __future__ import annotations

import hashlib
import os
from datetime import datetime
from datetime import timezone

from ingestion.manifests import build_ingestion_manifest
from ingestion.models import IngestionResult


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _hash_text(payload: str) -> str:
    return hashlib.md5(payload.encode('utf-8')).hexdigest()


class JobRunner:
    def __init__(self, *, quality_gate, job_store, storage_layout, manifest_writer, runtime_root: str):
        self.quality_gate = quality_gate
        self.job_store = job_store
        self.storage_layout = storage_layout
        self.manifest_writer = manifest_writer
        self.runtime_root = os.path.abspath(runtime_root)

    def run(self, job, spec, adapter):
        job.status = 'running'
        job.started_at = _utc_now()
        job.attempt += 1
        self.job_store.save(job)

        try:
            df = adapter.fetch(job.request, spec)
            quality_summary = self.quality_gate.validate(df, spec)
            paths = self.storage_layout.render_dataset_paths(
                spec,
                run_id=job.job_id,
                ingest_date=job.request.end,
                trade_date=job.request.end,
            )
            raw_path = paths.get('raw', os.path.join(self.runtime_root, 'raw', f'{job.job_id}.csv'))
            curated_path = paths.get('curated', os.path.join(self.runtime_root, 'curated', f'{job.job_id}.csv'))
            os.makedirs(os.path.dirname(raw_path), exist_ok=True)
            os.makedirs(os.path.dirname(curated_path), exist_ok=True)
            if raw_path.endswith('.parquet'):
                df.to_parquet(raw_path, index=False)
            else:
                df.to_csv(raw_path, index=False)
            if curated_path.endswith('.parquet'):
                df.to_parquet(curated_path, index=False)
            else:
                df.to_csv(curated_path, index=False)

            schema_hash = _hash_text(','.join(df.columns.astype(str).tolist()))
            data_hash = _hash_text(df.to_csv(index=False))
            result = IngestionResult(
                row_count=int(len(df)),
                schema_hash=schema_hash,
                data_hash=data_hash,
                quality_summary=quality_summary,
                raw_paths=[raw_path],
                curated_paths=[curated_path],
            )
            job.status = 'succeeded'
            job.finished_at = _utc_now()
            job.warnings = list(result.warnings)
            manifest = build_ingestion_manifest(job, result, code_version='')
            manifest_path = self.manifest_writer(job.job_id, manifest, dataset=spec.dataset)
            job.result = manifest
            job.manifest_path = manifest_path
            self.job_store.save(job)
            return job
        except Exception as exc:
            job.status = 'failed'
            job.finished_at = _utc_now()
            job.errors = [str(exc)]
            self.job_store.save(job)
            raise
