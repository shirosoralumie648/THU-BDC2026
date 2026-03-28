from dataclasses import asdict


def build_ingestion_manifest(job, result, *, code_version: str):
    return {
        'action': 'ingestion_job',
        'job_id': job.job_id,
        'dataset': job.request.dataset,
        'request': asdict(job.request),
        'status': job.status,
        'row_count': result.row_count,
        'schema_hash': result.schema_hash,
        'data_hash': result.data_hash,
        'quality_summary': dict(result.quality_summary),
        'raw_paths': list(result.raw_paths),
        'curated_paths': list(result.curated_paths),
        'warnings': list(result.warnings),
        'errors': list(result.errors),
        'code_version': code_version,
    }
