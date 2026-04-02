from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any
from typing import Dict


def build_ingestion_manifest(job, result, *, code_version: str) -> Dict[str, Any]:
    return {
        'action': 'ingestion_job',
        'job_id': job.job_id,
        'run_id': job.job_id,
        'dataset': job.request.dataset,
        'source': job.request.source,
        'start': job.request.start,
        'end': job.request.end,
        'request': asdict(job.request),
        'status': job.status,
        'attempt': int(getattr(job, 'attempt', 0)),
        'created_at': getattr(job, 'created_at', ''),
        'started_at': getattr(job, 'started_at', ''),
        'finished_at': getattr(job, 'finished_at', ''),
        'parent_job_id': getattr(job, 'parent_job_id', ''),
        'row_count': result.row_count,
        'schema_hash': result.schema_hash,
        'data_hash': result.data_hash,
        'quality_summary': dict(result.quality_summary),
        'output_paths': {
            'raw': list(result.raw_paths),
            'curated': list(result.curated_paths),
        },
        'raw_paths': list(result.raw_paths),
        'curated_paths': list(result.curated_paths),
        'warnings': list(result.warnings),
        'errors': list(result.errors),
        'code_version': code_version,
    }


def write_ingestion_manifest(output_dir: str, manifest: Dict[str, Any], *, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path
