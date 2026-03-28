from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

from ingestion.adapters import build_default_adapters
from ingestion.models import IngestionRequest
from ingestion.registry import DatasetRegistry
from ingestion.service import IngestionService


class CreateJobPayload(BaseModel):
    dataset: str
    start: str
    end: str
    source: str = ''
    mode: str = 'incremental'
    universe: str | None = None
    adjustment: str | None = None


class ReplayPayload(BaseModel):
    job_id: str


def create_app(config_dir: str = './config', runtime_root: str = './data/runtime/ingestion') -> FastAPI:
    app = FastAPI(title='THU-BDC2026 Ingestion API')
    registry = DatasetRegistry.from_config_dir(config_dir)
    specs = {spec.dataset: spec for spec in registry.list_datasets()}
    service = IngestionService(specs=specs, adapters=build_default_adapters(), runtime_root=runtime_root)

    @app.get('/healthz')
    def healthz():
        return {'status': 'ok'}

    @app.get('/ingestion/datasets')
    def list_datasets():
        return [
            {
                'dataset': spec.dataset,
                'source': spec.source_name,
                'adapter': spec.adapter_name,
                'granularity': spec.granularity,
            }
            for spec in registry.list_datasets()
        ]

    @app.post('/ingestion/jobs')
    def create_job(payload: CreateJobPayload):
        if payload.dataset not in specs:
            raise HTTPException(status_code=404, detail='dataset not found')
        job = service.create_job(
            IngestionRequest(
                dataset=payload.dataset,
                start=payload.start,
                end=payload.end,
                source=payload.source,
                mode=payload.mode,
                universe=payload.universe,
                adjustment=payload.adjustment,
            )
        )
        return {
            'job_id': job.job_id,
            'status': job.status,
            'dataset': job.request.dataset,
        }

    @app.get('/ingestion/jobs/{job_id}')
    def get_job(job_id: str):
        job = service.get_job(job_id)
        return {
            'job_id': job.job_id,
            'status': job.status,
            'dataset': job.request.dataset,
            'manifest_path': job.manifest_path,
            'warnings': job.warnings,
            'errors': job.errors,
        }

    @app.post('/ingestion/replay')
    def replay_job(payload: ReplayPayload):
        replayed = service.replay_job(payload.job_id)
        return {
            'job_id': replayed.job_id,
            'source_job_id': payload.job_id,
            'status': replayed.status,
        }

    return app
