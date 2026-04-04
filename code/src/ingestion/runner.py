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

    def _exception_message(self, exc: Exception) -> str:
        if exc.args:
            first = exc.args[0]
            if isinstance(first, str) and first:
                return first
        return str(exc)

    def _is_retryable_error(self, exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        message = self._exception_message(exc).lower()
        retryable_tokens = ('timeout', 'temporar', 'throttle', 'rate limit', 'connection reset', 'connection aborted', 'gateway')
        return any(token in message for token in retryable_tokens)

    def _persist_frame(self, df, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if path.endswith('.parquet'):
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)

    def _cleanup_outputs(self, *paths: str):
        for path in paths:
            if path and os.path.exists(path):
                os.remove(path)

    def _mark_failed(self, job, *, status: str, exc: Exception):
        job.status = status
        job.finished_at = _utc_now()
        job.manifest_path = ''
        job.warnings = []
        job.result = {}
        job.errors = [self._exception_message(exc)]
        self.job_store.save(job)

    def run(self, job, spec, adapter):
        job.status = 'running'
        job.started_at = _utc_now()
        job.attempt += 1
        self.job_store.save(job)

        raw_path = ''
        curated_path = ''
        try:
            df = adapter.fetch(job.request, spec)
        except Exception as exc:
            failed_status = 'retryable_failed' if self._is_retryable_error(exc) else 'fetch_failed'
            self._mark_failed(job, status=failed_status, exc=exc)
            raise

        try:
            quality_summary = self.quality_gate.validate(df, spec)
        except Exception as exc:
            self._mark_failed(job, status='quality_failed', exc=exc)
            raise

        warnings = []
        if df.empty:
            warnings.append('empty_result')

        try:
            paths = self.storage_layout.render_dataset_paths(
                spec,
                run_id=job.job_id,
                ingest_date=job.request.end,
                trade_date=job.request.end,
            )
            raw_path = paths.get('raw', os.path.join(self.runtime_root, 'raw', f'{job.job_id}.csv'))
            curated_path = paths.get('curated', os.path.join(self.runtime_root, 'curated', f'{job.job_id}.csv'))
            self._persist_frame(df, raw_path)
            self._persist_frame(df, curated_path)

            schema_hash = _hash_text(','.join(df.columns.astype(str).tolist()))
            data_hash = _hash_text(df.to_csv(index=False))
            result = IngestionResult(
                row_count=int(len(df)),
                schema_hash=schema_hash,
                data_hash=data_hash,
                quality_summary=quality_summary,
                raw_paths=[raw_path],
                curated_paths=[curated_path],
                warnings=warnings,
            )
            job.status = 'succeeded'
            job.finished_at = _utc_now()
            job.warnings = list(result.warnings)
            job.errors = list(result.errors)
            manifest = build_ingestion_manifest(job, result, code_version='')
            manifest_path = self.manifest_writer(job.job_id, manifest, dataset=spec.dataset)
            job.result = manifest
            job.manifest_path = manifest_path
            self.job_store.save(job)
            return job
        except Exception as exc:
            self._cleanup_outputs(raw_path, curated_path)
            self._mark_failed(job, status='write_failed', exc=exc)
            raise
