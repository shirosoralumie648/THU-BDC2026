from __future__ import annotations

from typing import Dict
from typing import Iterable

from ingestion.models import DatasetSpec
from pipeline_config import get_dataset_spec
from pipeline_config import load_pipeline_configs


class DatasetRegistry:
    def __init__(self, items: Dict[str, DatasetSpec]):
        self._items = dict(items)

    @classmethod
    def from_config_dir(cls, config_dir: str) -> 'DatasetRegistry':
        configs, _ = load_pipeline_configs(config_dir=config_dir, strict=False)
        datasets_cfg = configs.get('datasets', {}).get('datasets', {})
        items: Dict[str, DatasetSpec] = {}
        for dataset_name in datasets_cfg.keys():
            raw_spec = get_dataset_spec(configs, dataset_name)
            source = raw_spec.get('source', {}) if isinstance(raw_spec, dict) else {}
            items[dataset_name] = DatasetSpec(
                dataset=dataset_name,
                domain=str(raw_spec.get('domain', '')),
                granularity=str(raw_spec.get('granularity', '')),
                source_name=str(source.get('name', '')),
                adapter_name=str(source.get('adapter', '')),
                request_spec=raw_spec.get('request', {}) or {},
                schema_spec=raw_spec.get('schema', {}) or {},
                quality_spec=raw_spec.get('quality', {}) or {},
                storage_spec=raw_spec.get('storage', {}) or {},
                enabled=bool(raw_spec.get('enabled', True)),
            )
        return cls(items)

    def get(self, dataset: str) -> DatasetSpec:
        return self._items[dataset]

    def list_datasets(self) -> Iterable[DatasetSpec]:
        return list(self._items.values())

    def to_dict(self) -> Dict[str, DatasetSpec]:
        return dict(self._items)
