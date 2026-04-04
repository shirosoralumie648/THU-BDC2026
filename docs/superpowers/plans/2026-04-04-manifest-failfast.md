# Manifest Fail-Fast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 dataset/factor manifest 与 CSV 快照链在关键损坏场景下 fail-fast，并把可降级错误变成结构化可观测信息。

**Architecture:** 先用测试锁住两类问题：`data_manager.py` 中的快照静默降级，以及 `manage_data.py` 中的 factor manifest 解析静默回退。然后分别做最小实现改动，把关键 manifest 错误升级为明确异常，把 CSV/stat 失败写入结构化 snapshot 错误字段，最后做 focused + combined regression。

**Tech Stack:** Python 3.10+, unittest, subprocess CLI tests, pandas, json, pathlib

---

## File Map

- Modify: `code/src/data_manager.py`
  - 为 `inspect_csv_metadata()` 和 `build_file_snapshot()` 增加结构化错误语义
  - 保持现有调用方兼容，不重写返回顶层形状
- Modify: `code/src/manage_data.py`
  - 为 factor manifest 解析引入 fail-fast 异常
  - 保持 CLI 顶层 `stderr + exit code 2` 协议不变
- Modify: `test/test_manifest_contracts.py`
  - 增加快照错误 contract 测试
- Modify: `test/test_cli_error_paths.py`
  - 增加损坏 factor manifest 的 CLI 错误路径测试

## Task 1: Lock Snapshot Error Semantics in Tests

**Files:**
- Modify: `test/test_manifest_contracts.py`

- [ ] **Step 1: Add failing test for CSV parse errors surfacing in file snapshots**

Add these imports near the top of `test/test_manifest_contracts.py`:

```python
from unittest.mock import patch

from data_manager import build_file_snapshot
```

Add this test method inside `ManifestContractTests`:

```python
def test_build_file_snapshot_records_structured_csv_parse_error(self):
    with tempfile.TemporaryDirectory() as tmp:
        bad_csv = Path(tmp) / 'broken.csv'
        bad_csv.write_text('股票代码,日期\n"unterminated,2024-01-02\n', encoding='utf-8')

        snapshot = build_file_snapshot(str(bad_csv), inspect_csv=True)

        self.assertEqual(snapshot['path'], str(bad_csv.resolve()))
        self.assertEqual(snapshot['exists'], True)
        self.assertIn('csv', snapshot)
        self.assertEqual(snapshot['csv']['status'], 'error')
        self.assertEqual(snapshot['csv']['error_code'], 'csv_parse_error')
        self.assertIn('errors', snapshot)
        self.assertEqual(snapshot['errors'][0]['code'], 'csv_parse_error')
```

- [ ] **Step 2: Add failing test for stat failures surfacing in file snapshots**

Add this test method immediately below the previous one:

```python
def test_build_file_snapshot_records_stat_failure_without_hiding_existing_file(self):
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / 'ok.csv'
        csv_path.write_text('股票代码,日期\n000001.SZ,2024-01-02\n', encoding='utf-8')

        with patch('data_manager.os.path.getsize', side_effect=OSError('stat boom')):
            snapshot = build_file_snapshot(str(csv_path), inspect_csv=False)

        self.assertEqual(snapshot['exists'], True)
        self.assertNotIn('size_bytes', snapshot)
        self.assertIn('errors', snapshot)
        self.assertEqual(snapshot['errors'][0]['code'], 'stat_failed')
        self.assertIn('stat boom', snapshot['errors'][0]['message'])
```

- [ ] **Step 3: Run the focused manifest contract test to verify it fails**

Run:

```bash
cd /home/shirosora/code_storage/THU-BDC2026/.worktrees/phase2-manifest-failfast
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python -m unittest test.test_manifest_contracts.ManifestContractTests -v
```

Expected:

- FAIL because `build_file_snapshot()` does not yet emit `csv.status` / `csv.error_code`
- FAIL because `build_file_snapshot()` currently swallows `os.path.getsize()` errors

- [ ] **Step 4: Implement the minimal snapshot error structure in `data_manager.py`**

Add a small helper near `inspect_csv_metadata()`:

```python
def _snapshot_error(code: str, message: str) -> Dict[str, str]:
    return {
        'code': str(code),
        'message': str(message),
    }
```

Update `inspect_csv_metadata()` to return structured status values:

```python
def inspect_csv_metadata(
    path: str,
    *,
    date_col_candidates: Optional[Iterable[str]] = None,
    stock_col_candidates: Optional[Iterable[str]] = None,
) -> Dict:
    if not path or not os.path.exists(path):
        return {}

    date_candidates = list(date_col_candidates or ['日期', 'date', 'datetime', 'trade_date'])
    stock_candidates = list(stock_col_candidates or ['股票代码', 'stock_id', 'code', 'ts_code'])
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return {
            'status': 'error',
            'error_code': 'csv_parse_error',
            'message': str(exc),
        }

    info = {
        'status': 'ok',
        'row_count': int(len(df)),
        'column_count': int(len(df.columns)),
    }

    date_col = infer_existing_column(df, date_candidates)
    if date_col is not None:
        date_values = pd.to_datetime(df[date_col], errors='coerce')
        date_values = date_values.dropna()
        if not date_values.empty:
            info['date_column'] = date_col
            info['date_min'] = str(date_values.min().date())
            info['date_max'] = str(date_values.max().date())

    stock_col = infer_existing_column(df, stock_candidates)
    if stock_col is not None:
        stock_norm = normalize_stock_code_series(df[stock_col]).replace('', np.nan).dropna()
        if not stock_norm.empty:
            info['stock_column'] = stock_col
            info['stock_count'] = int(stock_norm.nunique())

    return info
```

Update `build_file_snapshot()` so existing files keep working but errors become observable:

```python
def build_file_snapshot(
    path: str,
    *,
    inspect_csv: bool = False,
    date_col_candidates: Optional[Iterable[str]] = None,
    stock_col_candidates: Optional[Iterable[str]] = None,
) -> Dict:
    target_path = str(path or '').strip()
    if not target_path:
        return {'path': '', 'exists': False}

    abs_path = os.path.abspath(target_path)
    exists = os.path.exists(abs_path)
    snapshot = {'path': abs_path, 'exists': bool(exists)}
    if not exists:
        return snapshot

    errors = []
    try:
        snapshot['size_bytes'] = int(os.path.getsize(abs_path))
    except Exception as exc:
        errors.append(_snapshot_error('stat_failed', str(exc)))

    if inspect_csv and abs_path.lower().endswith('.csv'):
        csv_meta = inspect_csv_metadata(
            abs_path,
            date_col_candidates=date_col_candidates,
            stock_col_candidates=stock_col_candidates,
        )
        if csv_meta:
            snapshot['csv'] = csv_meta
            if csv_meta.get('status') == 'error':
                errors.append(
                    _snapshot_error(
                        str(csv_meta.get('error_code', 'csv_parse_error')),
                        str(csv_meta.get('message', 'unknown csv parse error')),
                    )
                )

    if errors:
        snapshot['errors'] = errors
    return snapshot
```

- [ ] **Step 5: Run the focused manifest contract test to verify it passes**

Run:

```bash
cd /home/shirosora/code_storage/THU-BDC2026/.worktrees/phase2-manifest-failfast
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python -m unittest test.test_manifest_contracts.ManifestContractTests -v
```

Expected:

- PASS
- Existing manifest contract tests remain green

- [ ] **Step 6: Commit the snapshot error handling slice**

```bash
cd /home/shirosora/code_storage/THU-BDC2026/.worktrees/phase2-manifest-failfast
git add test/test_manifest_contracts.py code/src/data_manager.py
git commit -m "fix(data): surface structured snapshot errors"
```

## Task 2: Make Corrupt Factor Manifests Fail Fast in CLI

**Files:**
- Modify: `test/test_cli_error_paths.py`
- Modify: `code/src/manage_data.py`

- [ ] **Step 1: Add a helper that can point `factors.yaml` at a manifest output location**

Change `_write_minimal_pipeline_config()` in `test/test_cli_error_paths.py` to accept `manifest_uri`:

```python
def _write_minimal_pipeline_config(cfg_dir: Path, *, manifest_uri: str = ''):
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / 'datasets.yaml').write_text('version: 1\ndatasets: {}\n', encoding='utf-8')

    factor_lines = ['version: 1', 'layer_order: []', 'factor_nodes: []']
    if manifest_uri:
        factor_lines.extend(['build_manifest:', f'  output_uri: "{manifest_uri}"'])
    (cfg_dir / 'factors.yaml').write_text('\n'.join(factor_lines) + '\n', encoding='utf-8')

    (cfg_dir / 'storage.yaml').write_text(
        '\n'.join(
            [
                'version: 1',
                'layers:',
                '  raw: {uri_template: data/raw/{dataset}/{run_id}.csv}',
                '  curated: {uri_template: data/curated/{dataset}/{run_id}.csv}',
                '  feature_long: {uri_template: data/feature_long/{dataset}.csv}',
                '  feature_wide: {uri_template: data/feature_wide/{dataset}.csv}',
                '  datasets: {uri_template: data/datasets/{dataset}.csv}',
                '  manifests: {uri_template: data/manifests/{dataset}.json}',
            ]
        ),
        encoding='utf-8',
    )
```

- [ ] **Step 2: Add a failing CLI test for a corrupt factor build manifest**

Add this test method in `CliErrorPathTests`:

```python
def test_build_dataset_reports_corrupt_factor_manifest_without_traceback(self):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg_dir = tmp_path / 'config'
        base_path = tmp_path / 'stock_data.csv'
        feature_path = tmp_path / 'features.csv'
        output_dir = tmp_path / 'out'
        manifest_dir = tmp_path / 'manifests'
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / 'factor_build.json'

        _write_minimal_pipeline_config(cfg_dir, manifest_uri=str(manifest_path))
        manifest_path.write_text('{broken', encoding='utf-8')

        pd.DataFrame([{'股票代码': '000001.SZ', '日期': '2024-01-02', '收盘': 10.0}]).to_csv(
            base_path, index=False, encoding='utf-8'
        )
        pd.DataFrame([{'股票代码': '000001.SZ', '日期': '2024-01-02', 'alpha_001': 1.0}]).to_csv(
            feature_path, index=False, encoding='utf-8'
        )

        cmd = [
            sys.executable,
            os.path.join(SRC_ROOT, 'manage_data.py'),
            'build-dataset',
            '--base-input', str(base_path),
            '--feature-input', str(feature_path),
            '--pipeline-config-dir', str(cfg_dir),
            '--output-dir', str(output_dir),
            '--train-start', '2024-01-01',
            '--train-end', '2024-01-31',
            '--test-start', '2024-02-01',
            '--test-end', '2024-02-29',
        ]
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

        self.assertEqual(result.returncode, 2)
        self.assertIn('解析 factor build manifest 失败', result.stderr)
        self.assertNotIn('Traceback', result.stderr)
```

- [ ] **Step 3: Run the focused CLI error test to verify it fails**

Run:

```bash
cd /home/shirosora/code_storage/THU-BDC2026/.worktrees/phase2-manifest-failfast
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python -m unittest test.test_cli_error_paths.CliErrorPathTests -v
```

Expected:

- FAIL because corrupt factor manifest still falls through to empty fingerprint warning
- `build-dataset` returns `0` instead of `2`

- [ ] **Step 4: Implement fail-fast factor manifest resolution in `manage_data.py`**

Add a dedicated exception near `_cli_error_message()`:

```python
class ManifestResolutionError(ValueError):
    """Critical factor/dataset manifest resolution failure."""
```

Update `_extract_factor_fingerprint_from_manifest()` so parse/type failures raise instead of returning `''`:

```python
def _extract_factor_fingerprint_from_manifest(
    manifest_path: str,
    *,
    feature_input: str,
    feature_set_version: str,
) -> str:
    try:
        payload = _load_json_payload(manifest_path)
    except Exception as exc:
        raise ManifestResolutionError(
            f'解析 factor build manifest 失败: {manifest_path} | {exc}'
        ) from exc

    if not isinstance(payload, dict):
        raise ManifestResolutionError(
            f'factor build manifest 顶层必须为对象: {manifest_path}'
        )

    if str(payload.get('action', '') or '').strip() not in {'', 'build_factor_graph'}:
        return ''

    manifest_feature_set_version = str(payload.get('feature_set_version', '') or '').strip()
    if feature_set_version and manifest_feature_set_version and manifest_feature_set_version != feature_set_version:
        return ''

    output_paths = payload.get('output_paths', {})
    if not isinstance(output_paths, dict):
        output_paths = {}
    snapshot = output_paths.get('wide_csv_snapshot', {})
    if not isinstance(snapshot, dict):
        snapshot = {}

    feature_candidates = {
        os.path.abspath(str(path))
        for path in [output_paths.get('wide_csv', ''), snapshot.get('path', '')]
        if str(path).strip()
    }
    if feature_candidates and os.path.abspath(feature_input) not in feature_candidates:
        return ''

    return str(payload.get('factor_fingerprint', '') or '').strip()
```

Keep `_resolve_factor_fingerprint_from_feature_input()` simple: let `ManifestResolutionError` bubble; only return `''` when all candidate manifests are structurally valid but none match.

Do **not** catch `ManifestResolutionError` inside `command_build_dataset()`; let `main()` translate it through the existing `except (FileNotFoundError, ValueError, PipelineConfigError, KeyError, RuntimeError)` branch into `stderr + exit 2`.

- [ ] **Step 5: Run the focused CLI error test to verify it passes**

Run:

```bash
cd /home/shirosora/code_storage/THU-BDC2026/.worktrees/phase2-manifest-failfast
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python -m unittest test.test_cli_error_paths.CliErrorPathTests -v
```

Expected:

- PASS
- The new corrupt factor manifest case exits with code `2`
- No traceback appears in `stderr`

- [ ] **Step 6: Commit the factor manifest fail-fast slice**

```bash
cd /home/shirosora/code_storage/THU-BDC2026/.worktrees/phase2-manifest-failfast
git add test/test_cli_error_paths.py code/src/manage_data.py
git commit -m "fix(cli): fail fast on corrupt factor manifests"
```

## Task 3: Run Combined Regression and Finalize

**Files:**
- Verify: `code/src/data_manager.py`
- Verify: `code/src/manage_data.py`
- Verify: `test/test_manifest_contracts.py`
- Verify: `test/test_cli_error_paths.py`

- [ ] **Step 1: Run the combined robustness regression suite**

Run:

```bash
cd /home/shirosora/code_storage/THU-BDC2026/.worktrees/phase2-manifest-failfast
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python -m unittest \
  test.test_pipeline_config_errors \
  test.test_cli_error_paths \
  test.test_manifest_contracts \
  test.test_ingestion_api -v
```

Expected:

- PASS
- No traceback leakage in CLI error-path tests

- [ ] **Step 2: Run the nearby data/ingestion regression suite**

Run:

```bash
cd /home/shirosora/code_storage/THU-BDC2026/.worktrees/phase2-manifest-failfast
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python -m unittest \
  test.test_ingestion_service \
  test.test_ingestion_runtime -v
```

Expected:

- PASS
- No regression in manifest consumers or ingestion runtime snapshots

- [ ] **Step 3: Verify the branch is clean after the two implementation commits**

```bash
cd /home/shirosora/code_storage/THU-BDC2026/.worktrees/phase2-manifest-failfast
git status --short
```

Expected:

- No unstaged code changes remain
- The last two commits are:
  - `fix(data): surface structured snapshot errors`
  - `fix(cli): fail fast on corrupt factor manifests`
