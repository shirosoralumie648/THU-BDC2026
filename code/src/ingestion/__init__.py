__all__ = [
    'DatasetRegistry',
    'DatasetSpec',
    'IngestionJob',
    'IngestionRequest',
    'IngestionResult',
    'IngestionService',
]


def __getattr__(name):
    if name == 'DatasetRegistry':
        from ingestion.registry import DatasetRegistry
        return DatasetRegistry
    if name == 'DatasetSpec':
        from ingestion.models import DatasetSpec
        return DatasetSpec
    if name == 'IngestionJob':
        from ingestion.models import IngestionJob
        return IngestionJob
    if name == 'IngestionRequest':
        from ingestion.models import IngestionRequest
        return IngestionRequest
    if name == 'IngestionResult':
        from ingestion.models import IngestionResult
        return IngestionResult
    if name == 'IngestionService':
        from ingestion.service import IngestionService
        return IngestionService
    raise AttributeError(name)
