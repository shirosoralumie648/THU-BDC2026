from __future__ import annotations

import os
from pathlib import Path

try:
    from fastapi import FastAPI
    from fastapi import HTTPException
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError('fastapi is required to use ingestion.api.app') from exc

from ingestion.models import IngestionRequest
from ingestion.service import IngestionService

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUNTIME_ROOT = str((PROJECT_ROOT / 'temp' / 'ingestion_runtime').resolve())
DEFAULT_CONFIG_DIR = str((PROJECT_ROOT / 'config').resolve())


def create_app(*, service: IngestionService | None = None, runtime_root: str | None = None, config_dir: str | None = None) -> FastAPI:
    app = FastAPI(title='THU BDC Ingestion API', version='0.1.0')
    resolved_runtime_root = runtime_root or os.environ.get('THU_BDC_INGESTION_RUNTIME_ROOT', DEFAULT_RUNTIME_ROOT)
    resolved_config_dir = config_dir or os.environ.get('THU_BDC_INGESTION_CONFIG_DIR', DEFAULT_CONFIG_DIR)
    ingestion_service = service or IngestionService.from_config_dir(
        resolved_config_dir,
        runtime_root=resolved_runtime_root,
        project_root=str(PROJECT_ROOT),
    )
    app.state.ingestion_service = ingestion_service

    @app.get('/health')
    def health():
        return {'status': 'ok'}

    @app.get('/ingestion/datasets')
    def list_datasets():
        return {'datasets': sorted(ingestion_service.specs.keys())}

    @app.post('/ingestion/jobs')
    def create_job(payload: dict):
        try:
            request = IngestionRequest(**payload)
            job = ingestion_service.create_job(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ingestion_service.job_to_payload(job)

    @app.get('/ingestion/jobs/{job_id}')
    def get_job(job_id: str):
        try:
            job = ingestion_service.get_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ingestion_service.job_to_payload(job)

    @app.post('/ingestion/jobs/{job_id}/run')
    def run_job(job_id: str):
        try:
            job = ingestion_service.run_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ingestion_service.job_to_payload(job)

    @app.post('/ingestion/replay')
    def replay_job(payload: dict):
        job_id = str(payload.get('job_id', '')).strip()
        if not job_id:
            raise HTTPException(status_code=400, detail='job_id is required')
        try:
            job = ingestion_service.replay_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ingestion_service.job_to_payload(job)

    return app


app = create_app()
