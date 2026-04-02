from __future__ import annotations

import os
from typing import Any
from typing import Dict


def render_storage_target(template: str, **kwargs: Any) -> str:
    return str(template).format(**kwargs)


class StorageLayout:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.config = config if isinstance(config, dict) else {}
        self.project_root = os.path.abspath(project_root)

    @classmethod
    def from_config(cls, config: Dict[str, Any], project_root: str) -> 'StorageLayout':
        return cls(config=config, project_root=project_root)

    def resolve(self, template: str, **kwargs: Any) -> str:
        rendered = render_storage_target(template, **kwargs)
        if os.path.isabs(rendered):
            return rendered
        return os.path.abspath(os.path.join(self.project_root, rendered))

    def _layer_template(self, layer_name: str) -> str:
        layers = self.config.get('layers', {}) if isinstance(self.config, dict) else {}
        spec = layers.get(layer_name, {}) if isinstance(layers, dict) else {}
        if isinstance(spec, dict):
            return str(spec.get('uri_template', '') or '')
        return ''

    def render_dataset_paths(self, spec, *, run_id: str, ingest_date: str, trade_date: str = '', observation_month: str = '', series_id: str = '') -> Dict[str, str]:
        raw_tpl = str(spec.storage_spec.get('raw_uri', '') or self._layer_template('raw'))
        curated_tpl = str(spec.storage_spec.get('curated_uri', '') or self._layer_template('curated'))
        values = {
            'source': spec.source_name,
            'dataset': spec.dataset,
            'table': spec.dataset,
            'ingest_date': ingest_date,
            'trade_date': trade_date or ingest_date,
            'run_id': run_id,
            'observation_month': observation_month or ingest_date[:7],
            'series_id': series_id or 'all',
        }
        out: Dict[str, str] = {}
        if raw_tpl:
            out['raw'] = self.resolve(raw_tpl, **values)
        if curated_tpl:
            out['curated'] = self.resolve(curated_tpl, **values)
        return out
