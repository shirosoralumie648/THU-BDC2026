# Unified Ingestion Platform Phase 1-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared ingestion core for `market_bar_1d`, `market_bar_1m`, and `macro_series`, expose it through both CLI and HTTP, and persist raw/curated outputs plus ingestion manifests through one contract.

**Architecture:** Add a new `code/src/ingestion/` package that owns dataset registry parsing, runtime job models, local job-state persistence, quality gates, storage path rendering, adapter execution, and manifest writing. Keep entry logic thin by wiring both `manage_data.py` and a new FastAPI app into the same `IngestionService`, while leaving backfill slicing, GUI integration, and observability for later plans.

**Tech Stack:** Python 3.10, pandas, PyYAML, FastAPI, uvicorn, unittest, tempfile, existing `data_manager.py` manifest helpers

---

## Scope Guard

This plan intentionally implements only spec phase 1 and phase 2:

- Included:
  - shared ingestion package
  - local file-backed job store
  - raw and curated writes
  - ingestion manifest generation
  - CLI integration
  - HTTP API skeleton
  - first dataset slice: `market_bar_1d`, `market_bar_1m`, `macro_series`
- Excluded:
  - sliced backfill planner
  - GUI data center migration
  - observability dashboards and metrics sinks
  - distributed job execution

Follow-up plans will be needed for spec phase 3 and phase 4.

## Target File Structure

### New Files

- `code/src/ingestion/__init__.py`
  Export package entry points.
- `code/src/ingestion/models.py`
  Dataclasses for dataset specs, ingestion requests, jobs, and results.
- `code/src/ingestion/registry.py`
  Load and resolve `DatasetSpec` objects from `datasets.yaml`.
- `code/src/ingestion/job_store.py`
  Local file-backed persistence for job state.
- `code/src/ingestion/storage.py`
  Render raw and curated storage targets from `storage.yaml`.
- `code/src/ingestion/manifests.py`
  Normalize ingestion result metadata into persisted manifests.
- `code/src/ingestion/quality.py`
  Execute pre-write and post-write dataset quality gates.
- `code/src/ingestion/runner.py`
  Execute a single job from adapter fetch through manifest write.
- `code/src/ingestion/service.py`
  Shared service layer for create/run/get/replay operations.
- `code/src/ingestion/adapters/__init__.py`
  Adapter registry and imports.
- `code/src/ingestion/adapters/base.py`
  Common adapter protocol and context types.
- `code/src/ingestion/adapters/baostock_daily.py`
  `market_bar_1d` source adapter.
- `code/src/ingestion/adapters/akshare_minute.py`
  `market_bar_1m` source adapter.
- `code/src/ingestion/adapters/akshare_macro.py`
  `macro_series` source adapter.
- `code/src/ingestion/api/__init__.py`
  API package export.
- `code/src/ingestion/api/app.py`
  FastAPI application exposing ingestion endpoints.
- `test/test_ingestion_imports.py`
  Smoke import test for new package and API app.
- `test/test_ingestion_registry.py`
  Dataset registry tests.
- `test/test_ingestion_runtime.py`
  Job-store, storage-path, and manifest tests.
- `test/test_ingestion_service.py`
  Service and runner tests with fake adapters.
- `test/test_ingestion_cli.py`
  CLI integration tests.
- `test/test_ingestion_api.py`
  HTTP API tests.

### Modified Files

- `pyproject.toml`
  Add API runtime dependencies.
- `requirements.txt`
  Keep direct install path in sync with `pyproject.toml`.
- `code/src/manage_data.py`
  Add `ingest` command family and wire to `IngestionService`.
- `get_stock_data.py`
  Convert into a compatibility wrapper over the new service for `market_bar_1d`.

## Task 1: Add Dependencies and Package Skeleton

**Files:**
- Create: `code/src/ingestion/__init__.py`
- Create: `code/src/ingestion/api/__init__.py`
- Test: `test/test_ingestion_imports.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

- [ ] **Step 1: Write the failing import smoke test**

```python
import os
import sys
import unittest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


class IngestionImportTests(unittest.TestCase):
    def test_ingestion_package_exports(self):
        import ingestion

        self.assertTrue(hasattr(ingestion, '__all__'))

    def test_api_package_exports(self):
        import ingestion.api

        self.assertTrue(hasattr(ingestion.api, '__all__'))


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python test/test_ingestion_imports.py`

Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion'`

- [ ] **Step 3: Add package initializers and dependencies**

```python
# code/src/ingestion/__init__.py
__all__ = []
```

```python
# code/src/ingestion/api/__init__.py
__all__ = []
```

```toml
# pyproject.toml
dependencies = [
    "akshare>=1.18.28",
    "baostock>=0.8.9",
    "docker>=7.1.0",
    "fastapi>=0.115.0",
    "httpx>=0.28.1",
    "joblib>=1.5.2",
    "pandas>=2.3.2",
    "plotly>=5.24.0",
    "pyyaml>=6.0.2",
    "scikit-learn>=1.7.2",
    "seaborn>=0.13.2",
    "streamlit>=1.38.0",
    "streamlit-autorefresh>=1.0.1",
    "ta-lib>=0.6.8",
    "tensorboard>=2.20.0",
    "tensorboardx>=2.6.4",
    "torch>=2.6.0",
    "tqdm>=4.67.1",
    "uvicorn>=0.32.0",
]
```

```text
# requirements.txt
fastapi>=0.115.0
httpx>=0.28.1
uvicorn>=0.32.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python test/test_ingestion_imports.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt code/src/ingestion/__init__.py code/src/ingestion/api/__init__.py test/test_ingestion_imports.py
git commit -m "feat: scaffold ingestion package and api runtime deps"
```

## Task 2: Implement Runtime Models and Dataset Registry

**Files:**
- Modify: `code/src/ingestion/__init__.py`
- Create: `code/src/ingestion/models.py`
- Create: `code/src/ingestion/registry.py`
- Test: `test/test_ingestion_registry.py`

- [ ] **Step 1: Write the failing registry test**

```python
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ingestion.registry import DatasetRegistry


class DatasetRegistryTests(unittest.TestCase):
    def test_loads_dataset_spec_from_config_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dir = Path(tmp)
            (cfg_dir / 'datasets.yaml').write_text(
                '\n'.join(
                    [
                        'version: 1',
                        'datasets:',
                        '  market_bar_1d:',
                        '    domain: market',
                        '    granularity: 1d',
                        '    source: {name: baostock, adapter: baostock_daily}',
                        '    request: {start: "${START_DATE}", end: "${END_DATE}"}',
                        '    schema: {primary_key: [instrument_id, trade_date], columns: {instrument_id: {source: code}}}',
                        '    quality: {required_columns: [instrument_id, trade_date]}',
                        '    storage: {raw_uri: data/raw/mock.parquet, curated_uri: data/curated/mock.parquet}',
                    ]
                ),
                encoding='utf-8',
            )
            (cfg_dir / 'factors.yaml').write_text('version: 1\nlayer_order: []\nfactor_nodes: []\n', encoding='utf-8')
            (cfg_dir / 'storage.yaml').write_text('version: 1\nlayers: {raw: {}, curated: {}, feature_long: {}, feature_wide: {}, datasets: {}, manifests: {}}\n', encoding='utf-8')

            registry = DatasetRegistry.from_config_dir(str(cfg_dir))
            spec = registry.get('market_bar_1d')

            self.assertEqual(spec.dataset, 'market_bar_1d')
            self.assertEqual(spec.source_name, 'baostock')
            self.assertEqual(spec.adapter_name, 'baostock_daily')
            self.assertEqual(spec.primary_key, ['instrument_id', 'trade_date'])


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python test/test_ingestion_registry.py`

Expected: FAIL with `ImportError` or `AttributeError` for missing registry/models

- [ ] **Step 3: Implement the models and registry**

```python
# code/src/ingestion/models.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
```

```python
# code/src/ingestion/registry.py
from typing import Dict

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

    def list_datasets(self):
        return list(self._items.values())
```

```python
# code/src/ingestion/__init__.py
from ingestion.models import DatasetSpec
from ingestion.models import IngestionJob
from ingestion.models import IngestionRequest
from ingestion.models import IngestionResult

__all__ = [
    'DatasetSpec',
    'IngestionJob',
    'IngestionRequest',
    'IngestionResult',
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python test/test_ingestion_registry.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add code/src/ingestion/models.py code/src/ingestion/registry.py test/test_ingestion_registry.py
git commit -m "feat: add ingestion dataset registry and runtime models"
```

## Task 3: Add Job Store, Storage Path Rendering, and Manifest Service

**Files:**
- Create: `code/src/ingestion/job_store.py`
- Create: `code/src/ingestion/storage.py`
- Create: `code/src/ingestion/manifests.py`
- Test: `test/test_ingestion_runtime.py`

- [ ] **Step 1: Write the failing runtime persistence test**

```python
import json
import os
import sys
import tempfile
import unittest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ingestion.job_store import LocalJobStore
from ingestion.manifests import build_ingestion_manifest
from ingestion.models import IngestionJob
from ingestion.models import IngestionRequest
from ingestion.models import IngestionResult
from ingestion.storage import render_storage_target


class IngestionRuntimeTests(unittest.TestCase):
    def test_local_job_store_roundtrip_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalJobStore(tmp)
            req = IngestionRequest(dataset='market_bar_1d', start='2024-01-01', end='2024-01-31')
            job = IngestionJob(job_id='job-001', request=req, status='queued')
            store.save(job)

            loaded = store.get('job-001')
            self.assertEqual(loaded.job_id, 'job-001')
            self.assertEqual(loaded.status, 'queued')

            target = render_storage_target(
                'data/raw/source={source}/dataset={dataset}/ingest_date={ingest_date}/part-{run_id}.parquet',
                source='baostock',
                dataset='market_bar_1d',
                ingest_date='2026-03-28',
                run_id='job-001',
            )
            self.assertIn('market_bar_1d', target)

            result = IngestionResult(
                row_count=10,
                schema_hash='abc',
                data_hash='def',
                quality_summary={'row_count': 10},
                raw_paths=['/tmp/raw.parquet'],
                curated_paths=['/tmp/curated.parquet'],
            )
            manifest = build_ingestion_manifest(job, result, code_version='deadbeef')
            self.assertEqual(manifest['job_id'], 'job-001')
            self.assertEqual(manifest['row_count'], 10)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python test/test_ingestion_runtime.py`

Expected: FAIL because runtime persistence helpers do not exist yet

- [ ] **Step 3: Implement local runtime persistence and manifest helpers**

```python
# code/src/ingestion/job_store.py
import json
import os
from dataclasses import asdict

from ingestion.models import IngestionJob
from ingestion.models import IngestionRequest


class LocalJobStore:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def _path(self, job_id: str) -> str:
        return os.path.join(self.root_dir, f'{job_id}.json')

    def save(self, job: IngestionJob) -> str:
        payload = asdict(job)
        with open(self._path(job.job_id), 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return self._path(job.job_id)

    def get(self, job_id: str) -> IngestionJob:
        with open(self._path(job_id), 'r', encoding='utf-8') as f:
            payload = json.load(f)
        payload['request'] = IngestionRequest(**payload['request'])
        return IngestionJob(**payload)
```

```python
# code/src/ingestion/storage.py
def render_storage_target(template: str, **kwargs) -> str:
    return str(template).format(**kwargs)
```

```python
# code/src/ingestion/manifests.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python test/test_ingestion_runtime.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add code/src/ingestion/job_store.py code/src/ingestion/storage.py code/src/ingestion/manifests.py test/test_ingestion_runtime.py
git commit -m "feat: add ingestion job store and manifest runtime helpers"
```

## Task 4: Implement Quality Gates, Runner, and Shared Service

**Files:**
- Create: `code/src/ingestion/adapters/base.py`
- Create: `code/src/ingestion/quality.py`
- Create: `code/src/ingestion/runner.py`
- Create: `code/src/ingestion/service.py`
- Test: `test/test_ingestion_service.py`

- [ ] **Step 1: Write the failing service-flow test**

```python
import os
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ingestion.models import DatasetSpec
from ingestion.models import IngestionRequest
from ingestion.service import IngestionService


class _FakeAdapter:
    adapter_name = 'fake_adapter'

    def fetch(self, request, spec):
        return pd.DataFrame(
            [
                {'instrument_id': '000001', 'trade_date': '2024-01-02', 'open': 10.0, 'high': 11.0, 'low': 9.0, 'close': 10.5},
                {'instrument_id': '000002', 'trade_date': '2024-01-02', 'open': 20.0, 'high': 21.0, 'low': 19.0, 'close': 20.5},
            ]
        )


class IngestionServiceTests(unittest.TestCase):
    def test_run_sync_executes_adapter_quality_and_job_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = DatasetSpec(
                dataset='market_bar_1d',
                domain='market',
                granularity='1d',
                source_name='fake',
                adapter_name='fake_adapter',
                request_spec={},
                schema_spec={'primary_key': ['instrument_id', 'trade_date'], 'columns': {'instrument_id': {'source': 'instrument_id'}}},
                quality_spec={'required_columns': ['instrument_id', 'trade_date', 'close']},
                storage_spec={'raw_uri': 'data/raw/{dataset}/{run_id}.csv', 'curated_uri': 'data/curated/{dataset}/{run_id}.csv'},
            )
            service = IngestionService.for_testing(
                specs={'market_bar_1d': spec},
                adapters={'fake_adapter': _FakeAdapter()},
                runtime_root=tmp,
            )

            job = service.create_job(IngestionRequest(dataset='market_bar_1d', start='2024-01-01', end='2024-01-31'))
            finished = service.run_job(job.job_id)

            self.assertEqual(finished.status, 'succeeded')
            self.assertTrue(finished.manifest_path.endswith('.json'))


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python test/test_ingestion_service.py`

Expected: FAIL because `IngestionService` and runner flow are not implemented

- [ ] **Step 3: Implement the shared execution flow**

```python
# code/src/ingestion/adapters/base.py
from typing import Protocol


class BaseAdapter(Protocol):
    adapter_name: str

    def fetch(self, request, spec):
        ...
```

```python
# code/src/ingestion/quality.py
class QualityGate:
    def validate(self, df, spec):
        required = spec.quality_spec.get('required_columns', [])
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f'missing required columns: {missing}')

        primary_key = spec.primary_key
        if primary_key and df.duplicated(subset=primary_key).any():
            raise ValueError(f'duplicate primary key rows for {primary_key}')
```

```python
# code/src/ingestion/runner.py
import hashlib
import json
import os
from datetime import datetime, timezone

from ingestion.manifests import build_ingestion_manifest
from ingestion.models import IngestionResult


def _utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


class JobRunner:
    def __init__(self, *, quality_gate, job_store, manifest_writer, runtime_root):
        self.quality_gate = quality_gate
        self.job_store = job_store
        self.manifest_writer = manifest_writer
        self.runtime_root = runtime_root

    def run(self, job, spec, adapter):
        job.status = 'running'
        job.started_at = _utc_now()
        self.job_store.save(job)

        df = adapter.fetch(job.request, spec)
        self.quality_gate.validate(df, spec)

        raw_path = os.path.join(self.runtime_root, 'raw', f'{job.job_id}.csv')
        curated_path = os.path.join(self.runtime_root, 'curated', f'{job.job_id}.csv')
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        os.makedirs(os.path.dirname(curated_path), exist_ok=True)
        df.to_csv(raw_path, index=False)
        df.to_csv(curated_path, index=False)

        payload = df.to_csv(index=False).encode('utf-8')
        result = IngestionResult(
            row_count=len(df),
            schema_hash=hashlib.md5(','.join(df.columns).encode('utf-8')).hexdigest(),
            data_hash=hashlib.md5(payload).hexdigest(),
            quality_summary={'row_count': int(len(df))},
            raw_paths=[raw_path],
            curated_paths=[curated_path],
        )

        manifest = build_ingestion_manifest(job, result, code_version='')
        manifest_path = self.manifest_writer(job.job_id, manifest)
        job.manifest_path = manifest_path
        job.status = 'succeeded'
        job.finished_at = _utc_now()
        self.job_store.save(job)
        return job
```

```python
# code/src/ingestion/service.py
import json
import os
from datetime import datetime, timezone
from uuid import uuid4

from ingestion.job_store import LocalJobStore
from ingestion.quality import QualityGate
from ingestion.runner import JobRunner


class IngestionService:
    def __init__(self, *, specs, adapters, runtime_root):
        self.specs = specs
        self.adapters = adapters
        self.runtime_root = runtime_root
        self.job_store = LocalJobStore(os.path.join(runtime_root, 'jobs'))
        self.quality_gate = QualityGate()
        self.runner = JobRunner(
            quality_gate=self.quality_gate,
            job_store=self.job_store,
            manifest_writer=self._write_manifest,
            runtime_root=runtime_root,
        )

    @classmethod
    def for_testing(cls, *, specs, adapters, runtime_root):
        return cls(specs=specs, adapters=adapters, runtime_root=runtime_root)

    def _write_manifest(self, job_id, manifest):
        path = os.path.join(self.runtime_root, 'manifests', f'{job_id}.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return path

    def create_job(self, request):
        from ingestion.models import IngestionJob

        job = IngestionJob(
            job_id=f'job-{uuid4().hex[:12]}',
            request=request,
            status='queued',
            created_at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        )
        self.job_store.save(job)
        return job

    def get_job(self, job_id):
        return self.job_store.get(job_id)

    def run_job(self, job_id):
        job = self.get_job(job_id)
        spec = self.specs[job.request.dataset]
        adapter = self.adapters[spec.adapter_name]
        return self.runner.run(job, spec, adapter)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python test/test_ingestion_service.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add code/src/ingestion/adapters/base.py code/src/ingestion/quality.py code/src/ingestion/runner.py code/src/ingestion/service.py test/test_ingestion_service.py
git commit -m "feat: add ingestion runner and shared service core"
```

## Task 5: Implement Concrete Adapters and CLI Integration

**Files:**
- Create: `code/src/ingestion/adapters/__init__.py`
- Create: `code/src/ingestion/adapters/baostock_daily.py`
- Create: `code/src/ingestion/adapters/akshare_minute.py`
- Create: `code/src/ingestion/adapters/akshare_macro.py`
- Modify: `code/src/manage_data.py`
- Modify: `get_stock_data.py`
- Test: `test/test_ingestion_cli.py`

- [ ] **Step 1: Write the failing CLI integration test**

```python
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')


class IngestionCliTests(unittest.TestCase):
    def test_manage_data_ingest_datasets_lists_registry_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dir = Path(tmp)
            (cfg_dir / 'datasets.yaml').write_text(
                '\n'.join(
                    [
                        'version: 1',
                        'datasets:',
                        '  market_bar_1d:',
                        '    domain: market',
                        '    granularity: 1d',
                        '    source: {name: baostock, adapter: baostock_daily}',
                        '    request: {}',
                        '    schema: {primary_key: [instrument_id, trade_date], columns: {instrument_id: {source: code}}}',
                        '    quality: {required_columns: [instrument_id, trade_date, close]}',
                        '    storage: {raw_uri: data/raw/mock.csv, curated_uri: data/curated/mock.csv}',
                    ]
                ),
                encoding='utf-8',
            )
            (cfg_dir / 'factors.yaml').write_text('version: 1\nlayer_order: []\nfactor_nodes: []\n', encoding='utf-8')
            (cfg_dir / 'storage.yaml').write_text('version: 1\nlayers: {raw: {}, curated: {}, feature_long: {}, feature_wide: {}, datasets: {}, manifests: {}}\n', encoding='utf-8')

            result = subprocess.run(
                [
                    sys.executable,
                    os.path.join(SRC_ROOT, 'manage_data.py'),
                    'ingest',
                    'datasets',
                    '--pipeline-config-dir',
                    str(cfg_dir),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + '\n' + result.stderr)
            self.assertIn('market_bar_1d', result.stdout)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python test/test_ingestion_cli.py`

Expected: FAIL because `manage_data.py ingest datasets` does not exist yet

- [ ] **Step 3: Implement adapter registry and CLI commands**

```python
# code/src/ingestion/adapters/__init__.py
from ingestion.adapters.akshare_macro import AkshareMacroAdapter
from ingestion.adapters.akshare_minute import AkshareMinuteBarAdapter
from ingestion.adapters.baostock_daily import BaostockDailyAdapter


def build_default_adapters():
    adapters = [
        BaostockDailyAdapter(),
        AkshareMinuteBarAdapter(),
        AkshareMacroAdapter(),
    ]
    return {adapter.adapter_name: adapter for adapter in adapters}
```

```python
# code/src/ingestion/adapters/baostock_daily.py
import baostock as bs
import pandas as pd


class BaostockDailyAdapter:
    adapter_name = 'baostock_daily'

    def fetch(self, request, spec):
        source_params = spec.request_spec or {}
        symbols = [str(value).strip() for value in source_params.get('symbols', []) if str(value).strip()]
        if request.extra.get('symbols'):
            symbols = [str(value).strip() for value in request.extra.get('symbols', []) if str(value).strip()]
        fields = 'date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,tradestatus'
        lg = bs.login()
        if lg.error_code != '0':
            raise RuntimeError(lg.error_msg)
        try:
            frames = []
            for symbol in symbols:
                rs = bs.query_history_k_data_plus(
                    symbol,
                    fields,
                    start_date=request.start,
                    end_date=request.end,
                    frequency='d',
                    adjustflag=request.adjustment or '1',
                )
                rows = []
                while rs.error_code == '0' and rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    continue
                part = pd.DataFrame(rows, columns=rs.fields).rename(columns={'code': 'instrument_id', 'date': 'trade_date'})
                frames.append(part)
            if not frames:
                return pd.DataFrame(columns=['instrument_id', 'trade_date', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg', 'tradestatus'])
            return pd.concat(frames, ignore_index=True)
        finally:
            bs.logout()
```

```python
# code/src/ingestion/adapters/akshare_minute.py
import os

import akshare as ak
import pandas as pd


class AkshareMinuteBarAdapter:
    adapter_name = 'market_minute_bar'

    def fetch(self, request, spec):
        source_params = spec.request_spec or {}
        input_path = str(source_params.get('input_path', '')).strip()
        if input_path and os.path.exists(input_path):
            df = pd.read_csv(input_path)
        else:
            symbols = [str(value).strip() for value in source_params.get('symbols', []) if str(value).strip()]
            frames = []
            for symbol in symbols:
                raw = ak.stock_zh_a_hist_min_em(
                    symbol=symbol,
                    period=str(source_params.get('period', '1')),
                    start_date=request.start.replace('-', '') + ' 09:30:00',
                    end_date=request.end.replace('-', '') + ' 15:00:00',
                    adjust=str(request.adjustment or source_params.get('adjust', '')),
                )
                if raw is None or raw.empty:
                    continue
                part = raw.rename(
                    columns={
                        '时间': 'ts',
                        '开盘': 'open',
                        '最高': 'high',
                        '最低': 'low',
                        '收盘': 'close',
                        '成交量': 'volume',
                        '成交额': 'amount',
                    }
                ).copy()
                part['instrument_id'] = symbol[-6:]
                frames.append(part[['instrument_id', 'ts', 'open', 'high', 'low', 'close', 'volume', 'amount']])
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
                columns=['instrument_id', 'ts', 'open', 'high', 'low', 'close', 'volume', 'amount']
            )

        if df.empty:
            return df

        out = df.copy()
        out['instrument_id'] = out['instrument_id'].astype(str).str.replace(r'[^0-9]', '', regex=True).str[-6:].str.zfill(6)
        out['ts'] = pd.to_datetime(out['ts'], errors='coerce')
        return out.dropna(subset=['instrument_id', 'ts']).reset_index(drop=True)
```

```python
# code/src/ingestion/adapters/akshare_macro.py
import os

import akshare as ak
import pandas as pd


class AkshareMacroAdapter:
    adapter_name = 'macro_timeseries'

    def fetch(self, request, spec):
        source_params = spec.request_spec or {}
        input_path = str(source_params.get('input_path', '')).strip()
        if input_path and os.path.exists(input_path):
            df = pd.read_csv(input_path)
        else:
            fetchers = {
                'm2_yoy': lambda: ak.macro_china_money_supply(),
                'shibor_3m': lambda: ak.rate_interbank(market='上海银行同业拆借市场', symbol='Shibor人民币', indicator='3月'),
                'usdcny': lambda: ak.currency_boc_safe(),
            }
            series_ids = [str(value).strip() for value in source_params.get('series_ids', ['m2_yoy', 'shibor_3m', 'usdcny'])]
            frames = []
            for series_id in series_ids:
                if series_id not in fetchers:
                    raise ValueError(f'unsupported macro series id: {series_id}')
                raw = fetchers[series_id]()
                if raw is None or raw.empty:
                    continue
                part = raw.copy()
                first_col = part.columns[0]
                value_col = part.columns[-1]
                part = part.rename(columns={first_col: 'observation_date', value_col: 'value'})
                part['series_id'] = series_id
                part['available_time'] = pd.to_datetime(part['observation_date'], errors='coerce')
                frames.append(part[['series_id', 'observation_date', 'available_time', 'value']])
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
                columns=['series_id', 'observation_date', 'available_time', 'value']
            )

        if df.empty:
            return df

        out = df.copy()
        out['series_id'] = out['series_id'].astype(str).str.strip()
        out['observation_date'] = pd.to_datetime(out['observation_date'], errors='coerce')
        out['available_time'] = pd.to_datetime(out['available_time'], errors='coerce')
        out['value'] = pd.to_numeric(out['value'], errors='coerce')
        return out.dropna(subset=['series_id', 'observation_date', 'available_time']).reset_index(drop=True)
```

```python
# code/src/manage_data.py
ingest_parser = subparsers.add_parser('ingest', help='统一抓取平台入口')
ingest_subparsers = ingest_parser.add_subparsers(dest='ingest_command', required=True)

ingest_run = ingest_subparsers.add_parser('run', help='运行一个 ingestion job')
ingest_run.add_argument('--dataset', required=True)
ingest_run.add_argument('--start', required=True)
ingest_run.add_argument('--end', required=True)
ingest_run.add_argument('--pipeline-config-dir', default='./config')

ingest_status = ingest_subparsers.add_parser('status', help='查看 ingestion job')
ingest_status.add_argument('--job-id', required=True)
ingest_status.add_argument('--runtime-root', default='./data/runtime/ingestion')

ingest_replay = ingest_subparsers.add_parser('replay', help='重放 ingestion job')
ingest_replay.add_argument('--job-id', required=True)
ingest_replay.add_argument('--runtime-root', default='./data/runtime/ingestion')

ingest_datasets = ingest_subparsers.add_parser('datasets', help='列出可抓取数据集')
ingest_datasets.add_argument('--pipeline-config-dir', default='./config')


def _build_ingestion_service(config_dir: str, runtime_root: str):
    from ingestion.adapters import build_default_adapters
    from ingestion.registry import DatasetRegistry
    from ingestion.service import IngestionService

    registry = DatasetRegistry.from_config_dir(config_dir)
    specs = {spec.dataset: spec for spec in registry.list_datasets()}
    return registry, IngestionService(specs=specs, adapters=build_default_adapters(), runtime_root=runtime_root)


def command_ingest_datasets(args):
    registry, _ = _build_ingestion_service(args.pipeline_config_dir, './data/runtime/ingestion')
    for spec in registry.list_datasets():
        print(f'{spec.dataset}\t{spec.adapter_name}\t{spec.granularity}')
    return 0


def command_ingest_run(args):
    _, service = _build_ingestion_service(args.pipeline_config_dir, args.runtime_root)
    from ingestion.models import IngestionRequest

    request = IngestionRequest(
        dataset=args.dataset,
        start=args.start,
        end=args.end,
        source=args.source,
        mode=args.mode,
        universe=args.universe,
        adjustment=args.adjustment,
    )
    job = service.create_job(request)
    finished = service.run_job(job.job_id)
    print(f'job_id={finished.job_id}')
    print(f'status={finished.status}')
    print(f'manifest_path={finished.manifest_path}')
    return 0


def command_ingest_status(args):
    _, service = _build_ingestion_service(args.pipeline_config_dir, args.runtime_root)
    job = service.get_job(args.job_id)
    print(f'job_id={job.job_id}')
    print(f'status={job.status}')
    print(f'manifest_path={job.manifest_path}')
    return 0


def command_ingest_replay(args):
    _, service = _build_ingestion_service(args.pipeline_config_dir, args.runtime_root)
    replayed = service.replay_job(args.job_id)
    print(f'job_id={replayed.job_id}')
    print(f'parent_job_id={replayed.parent_job_id}')
    print(f'status={replayed.status}')
    return 0
```

```python
# get_stock_data.py
from subprocess import run


def main():
    args = parse_args()
    cmd = [
        sys.executable,
        str(CODE_SRC_DIR / 'manage_data.py'),
        'ingest',
        'run',
        '--dataset',
        args.dataset_name,
        '--start',
        args.start_date,
        '--end',
        args.end_date,
        '--pipeline-config-dir',
        args.pipeline_config_dir,
    ]
    if args.adjustflag:
        cmd.extend(['--adjustment', str(args.adjustflag)])
    return run(cmd).returncode


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python test/test_ingestion_cli.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add code/src/ingestion/adapters/__init__.py code/src/ingestion/adapters/baostock_daily.py code/src/ingestion/adapters/akshare_minute.py code/src/ingestion/adapters/akshare_macro.py code/src/manage_data.py get_stock_data.py test/test_ingestion_cli.py
git commit -m "feat: wire ingestion adapters into cli entry points"
```

## Task 6: Add FastAPI HTTP API Skeleton on Top of the Shared Service

**Files:**
- Create: `code/src/ingestion/api/app.py`
- Test: `test/test_ingestion_api.py`
- Modify: `code/src/ingestion/service.py`

- [ ] **Step 1: Write the failing API test**

```python
import os
import sys
import unittest

from fastapi.testclient import TestClient


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ingestion.api.app import create_app


class IngestionApiTests(unittest.TestCase):
    def test_health_and_dataset_endpoints(self):
        app = create_app()
        client = TestClient(app)

        health = client.get('/healthz')
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python test/test_ingestion_api.py`

Expected: FAIL because `create_app()` does not exist yet

- [ ] **Step 3: Implement the HTTP API**

```python
# code/src/ingestion/api/app.py
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
        return {'job_id': job.job_id, 'status': job.status, 'dataset': job.request.dataset}

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
    def replay_job(payload: dict):
        source_job_id = payload['job_id']
        replayed = service.replay_job(source_job_id)
        return {'job_id': replayed.job_id, 'source_job_id': source_job_id, 'status': replayed.status}

    return app
```

```python
# code/src/ingestion/service.py
    def replay_job(self, job_id):
        prior = self.get_job(job_id)
        new_job = self.create_job(prior.request)
        new_job.parent_job_id = prior.job_id
        self.job_store.save(new_job)
        return new_job
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python test/test_ingestion_api.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add code/src/ingestion/api/app.py code/src/ingestion/service.py test/test_ingestion_api.py
git commit -m "feat: expose ingestion service through fastapi endpoints"
```

## Task 7: Verify the Whole Phase 1-2 Slice

**Files:**
- Modify: `test/test_ingestion_service.py`
- Modify: `test/test_ingestion_cli.py`
- Modify: `test/test_ingestion_api.py`

- [ ] **Step 1: Add verification coverage for replay and manifest persistence**

```python
# Extend existing tests with:
self.assertTrue(os.path.exists(finished.manifest_path))
self.assertTrue(os.path.exists(os.path.join(tmp, 'raw', f'{finished.job_id}.csv')))
self.assertTrue(os.path.exists(os.path.join(tmp, 'curated', f'{finished.job_id}.csv')))
```

```python
# Extend API test with:
created = client.post('/ingestion/jobs', json={'dataset': 'market_bar_1d', 'start': '2024-01-01', 'end': '2024-01-31'})
self.assertEqual(created.status_code, 200)
job_id = created.json()['job_id']
details = client.get(f'/ingestion/jobs/{job_id}')
self.assertEqual(details.status_code, 200)
```

- [ ] **Step 2: Run the full ingestion test suite before finalizing**

Run: `./.venv/bin/python test/test_ingestion_imports.py`

Expected: PASS

Run: `./.venv/bin/python test/test_ingestion_registry.py`

Expected: PASS

Run: `./.venv/bin/python test/test_ingestion_runtime.py`

Expected: PASS

Run: `./.venv/bin/python test/test_ingestion_service.py`

Expected: PASS

Run: `./.venv/bin/python test/test_ingestion_cli.py`

Expected: PASS

Run: `./.venv/bin/python test/test_ingestion_api.py`

Expected: PASS

- [ ] **Step 3: Run a manual CLI smoke for the registry path**

Run: `./.venv/bin/python code/src/manage_data.py ingest datasets --pipeline-config-dir ./config`

Expected: output includes `market_bar_1d`, `market_bar_1m`, and `macro_series`

- [ ] **Step 4: Run a manual API smoke**

Run: `./.venv/bin/python -c "import sys; sys.path.insert(0, 'code/src'); from ingestion.api.app import create_app; app = create_app(); print(app.title)"`

Expected: prints `THU-BDC2026 Ingestion API`

- [ ] **Step 5: Commit**

```bash
git add test/test_ingestion_imports.py test/test_ingestion_registry.py test/test_ingestion_runtime.py test/test_ingestion_service.py test/test_ingestion_cli.py test/test_ingestion_api.py
git commit -m "test: verify unified ingestion phase 1 and phase 2 slice"
```

## Self-Review Notes

### Spec Coverage

Covered in this plan:

- shared ingestion service core
- CLI and HTTP on the same contract
- raw and curated writes
- ingestion manifest generation
- initial dataset slice for `market_bar_1d`, `market_bar_1m`, `macro_series`

Deliberately deferred to later plans:

- sliced backfill planning
- GUI migration
- observability dashboards

This is consistent with the approved scope: service skeleton first.

### Placeholder Scan

No `TBD`, `TODO`, or unresolved placeholders should remain in implementation steps. Adapter implementations may return empty frames only when the upstream provider returns no data for the requested range, not as scaffolding stubs.

### Type Consistency

The plan consistently uses:

- `DatasetSpec`
- `IngestionRequest`
- `IngestionJob`
- `IngestionResult`
- `IngestionService`

Do not rename these while executing unless every task and test is updated together.
