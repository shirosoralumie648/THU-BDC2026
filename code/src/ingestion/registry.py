from typing import Dict
from typing import List

from ingestion.models import DatasetSpec
from pipeline_config import get_dataset_spec
from pipeline_config import load_pipeline_configs


class DatasetRegistry:
    def __init__(self, items: Dict[str, DatasetSpec]):
        self._items = items

    @classmethod
    def from_config_dir(cls, config_dir: str) -> 'DatasetRegistry':
        configs, _ = load_pipeline_configs(config_dir=config_dir, strict=False)
        datasets_cfg = configs.get('datasets', {}).get('datasets', {})
        items: Dict[str, DatasetSpec] = {}
        for dataset_name in datasets_cfg.keys():
            spec = get_dataset_spec(configs, dataset_name)
            source = spec.get('source', {})
            items[dataset_name] = DatasetSpec(
                dataset=dataset_name,
                domain=str(spec.get('domain', '')),
                granularity=str(spec.get('granularity', '')),
                source_name=str(source.get('name', '')),
                adapter_name=str(source.get('adapter', '')),
                request_spec=spec.get('request', {}) or {},
                schema_spec=spec.get('schema', {}) or {},
                quality_spec=spec.get('quality', {}) or {},
                storage_spec=spec.get('storage', {}) or {},
            )
        return cls(items)

    def get(self, dataset: str) -> DatasetSpec:
        return self._items[dataset]

    def list_datasets(self) -> List[DatasetSpec]:
        return list(self._items.values())
