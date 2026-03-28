from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


@dataclass
class DatasetSpec:
    dataset: str
    domain: str
    granularity: str
    source_name: str
    adapter_name: str
    request_spec: Dict[str, Any]
    schema_spec: Dict[str, Any]
    quality_spec: Dict[str, Any]
    storage_spec: Dict[str, Any]

    @property
    def primary_key(self) -> List[str]:
        key = self.schema_spec.get('primary_key', [])
        return list(key) if isinstance(key, list) else []


@dataclass
class IngestionRequest:
    dataset: str
    start: str
    end: str
    source: str = ''
    mode: str = 'incremental'
    universe: Optional[str] = None
    adjustment: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionJob:
    job_id: str
    request: IngestionRequest
    status: str = 'queued'
    attempt: int = 0
    created_at: str = ''
    started_at: str = ''
    finished_at: str = ''
    parent_job_id: str = ''
    manifest_path: str = ''
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class IngestionResult:
    row_count: int
    schema_hash: str
    data_hash: str
    quality_summary: Dict[str, Any]
    raw_paths: List[str]
    curated_paths: List[str]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    code_version: str = ''
