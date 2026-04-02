from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Protocol

from ingestion.models import DatasetSpec
from ingestion.models import IngestionRequest


class BaseAdapter(Protocol):
    adapter_name: str

    def fetch(self, request: IngestionRequest, spec: DatasetSpec):
        ...


class DictBackedAdapter:
    adapter_name = 'dict_backed'

    def __init__(self, rows: Dict[str, Any]):
        self._rows = rows

    def fetch(self, request: IngestionRequest, spec: DatasetSpec):
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise RuntimeError('pandas is required for adapter fetch results') from exc
        dataset_rows = self._rows.get(request.dataset, [])
        return pd.DataFrame(dataset_rows)
