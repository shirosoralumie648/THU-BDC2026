from __future__ import annotations

import json
import os
from datetime import datetime
from datetime import timezone
from uuid import uuid4

from ingestion.adapters.akshare_macro import AkshareMacroAdapter
from ingestion.adapters.akshare_minute import AkshareMinuteAdapter
from ingestion.adapters.baostock_daily import BaoStockDailyAdapter
from ingestion.job_store import LocalJobStore
from ingestion.manifests import write_ingestion_manifest
from ingestion.models import IngestionJob
from ingestion.models import IngestionRequest
from ingestion.quality import QualityGate
from ingestion.registry import DatasetRegistry
from ingestion.runner import JobRunner
from ingestion.storage import StorageLayout
from pipeline_config import load_pipeline_configs


class IngestionService:
    def __init__(self, *, specs, adapters, runtime_root: str, project_root: str):
        self.specs = dict(specs)
        self.adapters = dict(adapters)
        self.runtime_root = os.path.abspath(runtime_root)
        self.project_root = os.path.abspath(project_root)
        self.job_store = LocalJobStore(os.path.join(self.runtime_root, 'jobs'))
        self.quality_gate = QualityGate()
        self.storage_layout = StorageLayout.from_config({}, project_root=self.project_root)
        self.runner = JobRunner(
            quality_gate=self.quality_gate,
            job_store=self.job_store,
            storage_layout=self.storage_layout,
            manifest_writer=self._write_manifest,
            runtime_root=self.runtime_root,
        )

    @classmethod
    def from_config_dir(cls, config_dir: str, *, runtime_root: str, project_root: str):
        registry = DatasetRegistry.from_config_dir(config_dir)
        configs, _ = load_pipeline_configs(config_dir=config_dir, strict=False)
        storage_config = configs.get('storage', {}) if isinstance(configs, dict) else {}
        adapters = {
            'baostock_daily': BaoStockDailyAdapter(),
            'market_minute_bar': AkshareMinuteAdapter(),
            'macro_timeseries': AkshareMacroAdapter(),
        }
        service = cls(specs=registry.to_dict(), adapters=adapters, runtime_root=runtime_root, project_root=project_root)
        service.storage_layout = StorageLayout.from_config(storage_config, project_root=project_root)
        service.runner = JobRunner(
            quality_gate=service.quality_gate,
            job_store=service.job_store,
            storage_layout=service.storage_layout,
            manifest_writer=service._write_manifest,
            runtime_root=service.runtime_root,
        )
        return service

    @classmethod
    def for_testing(cls, *, specs, adapters, runtime_root: str, project_root: str | None = None):
        return cls(
            specs=specs,
            adapters=adapters,
            runtime_root=runtime_root,
            project_root=project_root or runtime_root,
        )

    def _write_manifest(self, job_id: str, manifest, *, dataset: str) -> str:
        manifest_dir = os.path.join(self.runtime_root, 'manifests', dataset)
        return write_ingestion_manifest(manifest_dir, manifest, filename=f'{job_id}.json')

    def create_job(self, request: IngestionRequest) -> IngestionJob:
        if request.dataset not in self.specs:
            raise KeyError(f'unknown dataset: {request.dataset}')
        job = IngestionJob(
            job_id=f'job-{uuid4().hex[:12]}',
            request=request,
            status='queued',
            created_at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        )
        self.job_store.save(job)
        return job

    def get_job(self, job_id: str) -> IngestionJob:
        return self.job_store.get(job_id)

    def list_jobs(self):
        return self.job_store.list_jobs()

    def run_job(self, job_id: str) -> IngestionJob:
        job = self.get_job(job_id)
        spec = self.specs[job.request.dataset]
        adapter = self.adapters[spec.adapter_name]
        return self.runner.run(job, spec, adapter)

    def replay_job(self, job_id: str) -> IngestionJob:
        original = self.get_job(job_id)
        replay = self.create_job(original.request)
        replay.parent_job_id = original.job_id
        self.job_store.save(replay)
        return self.run_job(replay.job_id)

    def create_and_run(self, request: IngestionRequest) -> IngestionJob:
        job = self.create_job(request)
        return self.run_job(job.job_id)

    def job_to_payload(self, job: IngestionJob):
        return {
            'job_id': job.job_id,
            'status': job.status,
            'dataset': job.request.dataset,
            'request': {
                'dataset': job.request.dataset,
                'start': job.request.start,
                'end': job.request.end,
                'source': job.request.source,
                'mode': job.request.mode,
                'universe': job.request.universe,
                'adjustment': job.request.adjustment,
                'extra': job.request.extra,
            },
            'manifest_path': job.manifest_path,
            'warnings': list(job.warnings),
            'errors': list(job.errors),
            'result': dict(job.result),
        }
