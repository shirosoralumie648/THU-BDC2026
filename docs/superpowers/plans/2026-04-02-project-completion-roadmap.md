# Project Completion Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the gap between the current contest-delivery baseline and the repository's documented target architecture, while preserving `train.sh`, `test.sh`, Docker packaging, and `output/result.csv` compatibility.

**Architecture:** Work in three tracks. First, stabilize the current repository so tests, manifests, and entrypoints are reliable. Second, complete the ingestion platform so the YAML-driven data layer is actually usable beyond tests. Third, finish the internal modularization of ranking, portfolio, and experiment logic without breaking the existing orchestration wrappers in `code/src/train.py` and `code/src/predict.py`.

**Tech Stack:** Python 3.12, pandas, PyTorch, FastAPI, unittest, argparse, existing YAML pipeline configs, Docker

---

## File Map

- [`code/src/manage_data.py`](/home/shirosora/THU-BDC2026/code/src/manage_data.py)
  Unified CLI for validation, factor graph build, dataset build, and ingestion job operations.
- [`code/src/build_factor_graph.py`](/home/shirosora/THU-BDC2026/code/src/build_factor_graph.py)
  YAML-driven factor DAG execution, intraday DSL evaluation, macro as-of join, factor build manifest output.
- [`code/src/data_manager.py`](/home/shirosora/THU-BDC2026/code/src/data_manager.py)
  Dataset resolution, manifest lookup, CSV stats, HF merge, industry mapping, compatibility path handling.
- [`code/src/train.py`](/home/shirosora/THU-BDC2026/code/src/train.py)
  Current training orchestrator and still the largest concentration of business logic.
- [`code/src/predict.py`](/home/shirosora/THU-BDC2026/code/src/predict.py)
  Current inference orchestrator and compatibility shell for contest output.
- [`code/src/model.py`](/home/shirosora/THU-BDC2026/code/src/model.py)
  Current monolithic ranking model implementation.
- [`code/src/features/feature_assembler.py`](/home/shirosora/THU-BDC2026/code/src/features/feature_assembler.py)
  Current thin wrapper over legacy feature builders; target area for extraction.
- [`code/src/models/rank_model.py`](/home/shirosora/THU-BDC2026/code/src/models/rank_model.py)
  Current alias export only; target area for real model-layer modularization.
- [`code/src/objectives/ranking_loss.py`](/home/shirosora/THU-BDC2026/code/src/objectives/ranking_loss.py)
  New objective module, but still duplicated with loss code in `train.py`.
- [`code/src/portfolio/policy.py`](/home/shirosora/THU-BDC2026/code/src/portfolio/policy.py)
  Final portfolio conversion, currently only `top_k + equal/softmax`.
- [`code/src/experiments/splits.py`](/home/shirosora/THU-BDC2026/code/src/experiments/splits.py)
  Rolling validation split helper.
- [`code/src/experiments/metrics.py`](/home/shirosora/THU-BDC2026/code/src/experiments/metrics.py)
  Strategy candidate generation and best-strategy selection.
- [`code/src/ingestion/service.py`](/home/shirosora/THU-BDC2026/code/src/ingestion/service.py)
  Shared ingestion execution core.
- [`code/src/ingestion/api/app.py`](/home/shirosora/THU-BDC2026/code/src/ingestion/api/app.py)
  FastAPI surface for ingestion jobs; currently has import-time initialization and path-resolution issues.
- [`code/src/ingestion/adapters/baostock_daily.py`](/home/shirosora/THU-BDC2026/code/src/ingestion/adapters/baostock_daily.py)
- [`code/src/ingestion/adapters/akshare_minute.py`](/home/shirosora/THU-BDC2026/code/src/ingestion/adapters/akshare_minute.py)
- [`code/src/ingestion/adapters/akshare_macro.py`](/home/shirosora/THU-BDC2026/code/src/ingestion/adapters/akshare_macro.py)
  Real source integrations still missing.
- [`test/test.py`](/home/shirosora/THU-BDC2026/test/test.py)
  Linux Docker validation script; must stay import-safe and executable.
- [`test/test_batch_validation_import.py`](/home/shirosora/THU-BDC2026/test/test_batch_validation_import.py)
  Regression test for import side effects.
- [`test/test_ingestion_api.py`](/home/shirosora/THU-BDC2026/test/test_ingestion_api.py)
  API contract test; currently failing because `ingestion.api.app` resolves the wrong project root at import time.
- [`config/datasets.yaml`](/home/shirosora/THU-BDC2026/config/datasets.yaml)
- [`config/factors.yaml`](/home/shirosora/THU-BDC2026/config/factors.yaml)
- [`config/storage.yaml`](/home/shirosora/THU-BDC2026/config/storage.yaml)
  Canonical pipeline design inputs.

## Completion Snapshot

### Already Complete Enough To Keep

- Contest delivery contract:
  - `train.sh`
  - `test.sh`
  - `output/result.csv`
  - `output/prediction_scores.csv`
- Ranking baseline:
  - mixed ranking loss
  - market gating
  - multi-scale temporal branches
  - cross-stock attention
  - prior graph support
  - industry virtual stock support
  - rolling validation and strategy reselection
- YAML-driven factor build:
  - config validation
  - factor DAG execution
  - intraday DSL evaluation
  - macro as-of join
  - factor build manifest generation
- Dataset build:
  - base table + feature table merge
  - CSV output
  - dataset manifest generation

### Implemented But Not Finished

- Ingestion platform:
  - CLI/service/job model exists
  - API surface exists
  - adapters are still empty
  - quality rules only cover the simplest checks
- Module split:
  - new package directories exist
  - most real logic still lives in `train.py`, `predict.py`, `model.py`, and `utils.py`
- New pipeline adoption:
  - capability exists
  - repository defaults and generated artifacts are still mixed between old and new flows

### Missing Relative To The Design Docs

- real external ingestion adapters
- full quality-rule execution from `datasets.yaml`
- real model-layer decomposition into encoder/router modules
- full portfolio constraint layer
- experiment runner, ensemble, diagnostics modules
- purged walk-forward validation with embargo
- one single source of truth for factor, dataset, model, and strategy lineage

## Delivery Strategy

Implement in this order:

1. Repository stabilization
2. Ingestion platform completion
3. Pipeline and manifest unification
4. Internal modularization without behavior change
5. Portfolio and experiment upgrades
6. Release hardening

Each phase should land in small reviewable commits and keep the current training/inference contract working.

## Task 1: Stabilize Tests, Paths, And Entrypoints

**Files:**
- Create: `test/__init__.py`
- Modify: `test/test.py`
- Modify: `test/test_batch_validation_import.py`
- Modify: `code/src/ingestion/api/app.py`
- Modify: `test/test_ingestion_api.py`

- [ ] **Step 1: Make the `test` directory importable**

Create:

```python
# test/__init__.py
"""Test package marker for import-based regression tests."""
```

Run:

```bash
./.venv/bin/python -c 'import importlib; importlib.import_module("test.test_batch_validation_import")'
```

Expected:
- exit code `0`

- [ ] **Step 2: Verify Linux Docker validation script stays import-safe**

Keep `test/test.py` structured around:

```python
def main() -> None:
    input_file = "./test/tar_files_list.txt"
    tar_files = read_tar_files(input_file)
    ...


if __name__ == "__main__":
    main()
```

Also change subprocess Python invocations inside the script from hardcoded `python` to `sys.executable` so the repo virtualenv is used consistently.

Run:

```bash
./.venv/bin/python -m unittest test.test_batch_validation_import -v
```

Expected:
- `PASS`
- no Docker batch execution noise

- [ ] **Step 3: Fix ingestion API project-root resolution**

Replace the current root logic in `code/src/ingestion/api/app.py`:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUNTIME_ROOT = str((PROJECT_ROOT / "temp" / "ingestion_runtime").resolve())
DEFAULT_CONFIG_DIR = str((PROJECT_ROOT / "config").resolve())
```

Keep `create_app(...)` as the constructor entrypoint. If module-level `app` is retained for uvicorn compatibility, make sure it resolves the correct repository root and does not depend on a nonexistent `code/config`.

Run:

```bash
./.venv/bin/python -m unittest test.test_ingestion_api -v
```

Expected:
- `PASS`

- [ ] **Step 4: Re-run the full targeted test suite**

Run:

```bash
./.venv/bin/python -m unittest discover -s test -p 'test_*.py' -v
```

Expected:
- all current tests green
- no import-time side-effect failures

- [ ] **Step 5: Commit**

```bash
git add test/__init__.py test/test.py test/test_batch_validation_import.py code/src/ingestion/api/app.py test/test_ingestion_api.py
git commit -m "test: stabilize imports and ingestion api bootstrap"
```

## Task 2: Complete Real Ingestion Adapters

**Files:**
- Modify: `code/src/ingestion/adapters/baostock_daily.py`
- Modify: `code/src/ingestion/adapters/akshare_minute.py`
- Modify: `code/src/ingestion/adapters/akshare_macro.py`
- Modify: `code/src/ingestion/service.py`
- Modify: `code/src/ingestion/runner.py`
- Modify: `code/src/ingestion/quality.py`
- Test: `test/test_ingestion_service.py`
- Create: `test/test_ingestion_adapters.py`

- [ ] **Step 1: Define adapter contract tests first**

Create tests that assert each adapter returns canonical columns expected by its dataset spec, for example:

```python
class AdapterContractTests(unittest.TestCase):
    def test_baostock_daily_adapter_returns_canonical_rows(self):
        adapter = BaoStockDailyAdapter(client=fake_client)
        df = adapter.fetch(request, spec)
        self.assertIn("instrument_id", df.columns)
        self.assertIn("trade_date", df.columns)
        self.assertIn("open", df.columns)
        self.assertIn("close", df.columns)
```

Run:

```bash
./.venv/bin/python -m unittest test.test_ingestion_adapters -v
```

Expected:
- `FAIL`

- [ ] **Step 2: Implement `BaoStockDailyAdapter` with explicit normalization**

Implementation shape:

```python
class BaoStockDailyAdapter:
    def __init__(self, client=None):
        self.client = client or _BaoStockClient()

    def fetch(self, request, spec):
        rows = self.client.fetch_daily(...)
        df = pd.DataFrame(rows)
        df["instrument_id"] = normalize_instrument_id(df["code"])
        df["trade_date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        ...
        return df[["instrument_id", "trade_date", "open", "high", "low", "close", "preclose", "volume", "amount", "turnover", "pct_chg", "trade_status"]]
```

Requirements:
- login/logout safety
- retry boundaries handled at service layer or client helper, not ad hoc in `fetch`
- empty provider response returns empty canonical frame, not arbitrary columns

- [ ] **Step 3: Implement `AkshareMinuteAdapter` and `AkshareMacroAdapter` with the same canonical contract**

Minute output target:

```python
["instrument_id", "ts", "trade_date", "open", "high", "low", "close", "volume", "amount"]
```

Macro output target:

```python
["series_id", "available_time", "value"]
```

Requirements:
- deterministic column naming
- explicit datetime normalization
- no provider-specific field names leaking into curated output

- [ ] **Step 4: Expand quality execution beyond required-columns + duplicate-PK**

Implement support for the currently configured rule families:
- `unique_key`
- `expression`
- `stock_coverage_vs_trade_days`
- `market_session_check`

Implementation shape:

```python
class QualityGate:
    def validate(self, df, spec):
        summary = {"row_count": int(len(df)), "rules": []}
        ...
        return summary
```

Requirements:
- hard-fail on required column absence
- hard-fail on duplicate PK
- per-rule summaries in manifest payload

- [ ] **Step 5: Add service-level smoke tests**

Run:

```bash
./.venv/bin/python -m unittest test.test_ingestion_service test.test_ingestion_api test.test_ingestion_adapters -v
```

Expected:
- all ingestion tests green

- [ ] **Step 6: Add opt-in live smoke commands**

Document but do not run by default in unit tests:

```bash
./.venv/bin/python code/src/manage_data.py ingest create --dataset market_bar_1d --start 2024-01-01 --end 2024-01-31
./.venv/bin/python code/src/manage_data.py ingest run --job-id <job_id>
```

Acceptance:
- `run` no longer fails with `missing required columns`

- [ ] **Step 7: Commit**

```bash
git add code/src/ingestion/adapters/baostock_daily.py code/src/ingestion/adapters/akshare_minute.py code/src/ingestion/adapters/akshare_macro.py code/src/ingestion/service.py code/src/ingestion/runner.py code/src/ingestion/quality.py test/test_ingestion_service.py test/test_ingestion_api.py test/test_ingestion_adapters.py
git commit -m "feat: complete ingestion adapters and quality rules"
```

## Task 3: Unify Dataset Build, Factor Build, And Manifest Lineage

**Files:**
- Modify: `code/src/manage_data.py`
- Modify: `code/src/build_factor_graph.py`
- Modify: `code/src/data_manager.py`
- Modify: `README.md`
- Modify: `GUIDE.md`
- Test: `test/test_factor_graph_pipeline.py`
- Create: `test/test_dataset_manifest_linkage.py`

- [ ] **Step 1: Write regression tests for manifest linkage**

Test cases:
- `build-dataset` should populate `factor_fingerprint`
- training should prefer dataset build manifest when available
- prediction should reject feature mismatch when manifest + effective feature set disagree

Skeleton:

```python
def test_build_dataset_manifest_contains_factor_fingerprint(self):
    ...
    self.assertTrue(payload["factor_fingerprint"])
```

- [ ] **Step 2: Auto-propagate factor fingerprint from factor-build outputs**

When `build-dataset` is called with `--feature-input` but no explicit `--factor-fingerprint`, resolve it in this order:

1. explicit CLI argument
2. factor-build manifest adjacent to rendered feature output
3. factor-build manifest in `data/manifests/factor_build/<feature_set_version>/...`

Implementation target inside `command_build_dataset(...)`:

```python
if not factor_fingerprint:
    factor_fingerprint = resolve_factor_fingerprint_from_feature_input(...)
```

- [ ] **Step 3: Make the YAML-driven path the default repository workflow**

Requirements:
- `README.md` and `GUIDE.md` already mention the new flow; make wording unambiguous
- clearly distinguish:
  - legacy raw split CSVs
  - feature-enriched build-dataset outputs
  - train-time manifest-resolved inputs

- [ ] **Step 4: Add one canonical dataset-build smoke command**

Command:

```bash
./.venv/bin/python code/src/manage_data.py build-factor-graph --pipeline-config-dir ./config --feature-set-version v1 --base-input ./data/stock_data.csv
./.venv/bin/python code/src/manage_data.py build-dataset --pipeline-config-dir ./config --feature-set-version v1 --base-input ./data/stock_data.csv --feature-input ./data/datasets/features/train_features_v1.csv --output-dir ./data
```

Acceptance:
- `data/data_manifest_dataset_build.json` exists
- `factor_fingerprint` is non-empty
- train/predict logs report manifest-based dataset usage

- [ ] **Step 5: Commit**

```bash
git add code/src/manage_data.py code/src/build_factor_graph.py code/src/data_manager.py README.md GUIDE.md test/test_dataset_manifest_linkage.py
git commit -m "feat: unify factor and dataset lineage via manifests"
```

## Task 4: Finish The Internal Module Split Without Breaking Behavior

**Files:**
- Create: `code/src/features/daily_features.py`
- Create: `code/src/features/relative_features.py`
- Create: `code/src/features/intraday_features.py`
- Create: `code/src/features/risk_features.py`
- Modify: `code/src/features/feature_assembler.py`
- Create: `code/src/models/daily_encoder.py`
- Create: `code/src/models/intraday_encoder.py`
- Create: `code/src/models/relation_encoder.py`
- Create: `code/src/models/regime_router.py`
- Modify: `code/src/models/rank_model.py`
- Create: `code/src/objectives/aux_losses.py`
- Create: `code/src/objectives/target_transforms.py`
- Modify: `code/src/objectives/ranking_loss.py`
- Modify: `code/src/train.py`
- Modify: `code/src/model.py`
- Test: `test/test_model_module_parity.py`

- [ ] **Step 1: Freeze current behavior with parity tests**

Create tests that compare:
- old loss vs new loss module on identical tensors
- old feature assembly vs new feature package entrypoints
- old model wrapper outputs vs refactored package wrapper outputs for fixed seeds and small synthetic input

- [ ] **Step 2: Extract feature builders by responsibility, not by novelty**

Target ownership:
- `daily_features.py`: base OHLCV-derived features
- `relative_features.py`: cross-sectional rank and industry-relative transforms
- `intraday_features.py`: HF-derived structured features
- `risk_features.py`: volatility and distributional risk features
- `feature_assembler.py`: dispatch + shared interface only

Keep public contract:

```python
def build_feature_table(df, feature_set: str):
    ...
```

- [ ] **Step 3: Extract model-layer building blocks while keeping `StockTransformer` as compatibility shell**

Target ownership:
- `daily_encoder.py`: temporal encoder and feature attention
- `intraday_encoder.py`: intraday branch logic if learned path is introduced
- `relation_encoder.py`: cross-stock attention and prior-graph interaction
- `regime_router.py`: market gating and macro context routing
- `rank_model.py`: assemble the blocks into the public ranking model

The external contract must stay:

```python
model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
```

- [ ] **Step 4: Remove duplicated objective logic**

Current duplication source:
- `code/src/train.py`
- `code/src/objectives/ranking_loss.py`

End state:
- `train.py` imports one canonical `PortfolioOptimizationLoss`
- auxiliary volatility loss and target transforms live in `objectives`

- [ ] **Step 5: Run parity and regression tests**

Run:

```bash
./.venv/bin/python -m unittest test.test_model_module_parity test.test_factor_store_engine test.test_factor_graph_pipeline -v
```

Expected:
- parity tests green
- no behavior regressions in existing data/factor tests

- [ ] **Step 6: Commit**

```bash
git add code/src/features code/src/models code/src/objectives code/src/train.py code/src/model.py test/test_model_module_parity.py
git commit -m "refactor: extract feature model and objective modules"
```

## Task 5: Implement Missing Portfolio And Experiment Layers

**Files:**
- Create: `code/src/portfolio/candidate_selector.py`
- Create: `code/src/portfolio/constraints.py`
- Create: `code/src/portfolio/weighting.py`
- Modify: `code/src/portfolio/policy.py`
- Create: `code/src/experiments/runner.py`
- Create: `code/src/experiments/ensemble.py`
- Create: `code/src/experiments/diagnostics.py`
- Modify: `code/src/experiments/splits.py`
- Modify: `code/src/experiments/metrics.py`
- Modify: `code/src/reselect_strategy.py`
- Test: `test/test_portfolio_policy.py`
- Test: `test/test_experiment_runner.py`

- [ ] **Step 1: Add portfolio unit tests before changing behavior**

Test matrix:
- top-k cap remains `<= 5`
- equal and softmax still supported
- optional industry concentration limit works
- optional turnover penalty works when previous holdings are provided

- [ ] **Step 2: Split current policy into selector, weighting, and constraints**

End-state shape:

```python
selected_ids, selected_scores = select_candidates(scores, stock_ids, strategy)
weights = compute_weights(selected_scores, strategy)
selected_ids, weights = apply_constraints(selected_ids, weights, metadata, strategy)
```

Compatibility rule:
- if no constraints configured, output must match current `scores_to_portfolio(...)`

- [ ] **Step 3: Upgrade experiment splits from simple rolling to purged rolling with embargo**

Requirements:
- keep current rolling behavior as fallback
- add explicit `label_horizon`-aware purge
- add embargo window for `T+5` label leakage protection

- [ ] **Step 4: Add experiment runner, ensemble, and diagnostics**

Runner responsibilities:
- multi-seed execution
- per-run metric collection
- best-run vs ensemble summary

Diagnostics responsibilities:
- export fold-level metrics
- export strategy comparison table
- export simple feature/regime summaries

- [ ] **Step 5: Rewire reselect strategy to the new experiment helpers**

Acceptance:
- `reselect_strategy.py` still writes `best_strategy_reselected.json`
- strategy selection logic remains backward compatible

- [ ] **Step 6: Run targeted tests**

```bash
./.venv/bin/python -m unittest test.test_portfolio_policy test.test_experiment_runner -v
```

Expected:
- all new tests green

- [ ] **Step 7: Commit**

```bash
git add code/src/portfolio code/src/experiments code/src/reselect_strategy.py test/test_portfolio_policy.py test/test_experiment_runner.py
git commit -m "feat: add portfolio constraints and experiment runner"
```

## Task 6: Release Hardening And Acceptance

**Files:**
- Modify: `README.md`
- Modify: `GUIDE.md`
- Modify: `Dockerfile` only if runtime dependencies changed
- Modify: `docker-compose.yml` only if validation flow changed

- [ ] **Step 1: Define acceptance commands**

Required green path:

```bash
./.venv/bin/python code/src/manage_data.py validate-pipeline-config --config-dir ./config
./.venv/bin/python -m unittest discover -s test -p 'test_*.py' -v
./.venv/bin/python code/src/manage_data.py build-factor-graph --pipeline-config-dir ./config --feature-set-version v1 --base-input ./data/stock_data.csv
./.venv/bin/python code/src/manage_data.py build-dataset --pipeline-config-dir ./config --feature-set-version v1 --base-input ./data/stock_data.csv --feature-input ./data/datasets/features/train_features_v1.csv --output-dir ./data
sh train.sh
sh test.sh
./.venv/bin/python test/score_self.py
docker compose up
```

- [ ] **Step 2: Define release checklist**

Release gate:
- all unit tests pass
- ingestion adapters return canonical columns
- build manifests contain `factor_fingerprint`
- `train.py` and `predict.py` use manifest-aware path resolution
- `output/result.csv` remains the final submission artifact

- [ ] **Step 3: Document known opt-in dependencies**

Explicitly document:
- network-required commands
- provider rate limits
- local Docker requirements
- Baostock/Akshare availability assumptions

- [ ] **Step 4: Commit**

```bash
git add README.md GUIDE.md Dockerfile docker-compose.yml
git commit -m "docs: finalize project acceptance and release checklist"
```

## Milestones

### Milestone A: Stable Baseline

Scope:
- Task 1

Success criteria:
- test suite green
- no import-time bootstrap failures

### Milestone B: Usable Ingestion Platform

Scope:
- Task 2

Success criteria:
- `manage_data.py ingest run` succeeds for at least one real dataset path
- manifests include quality summaries

### Milestone C: Single-Lineage Pipeline

Scope:
- Task 3

Success criteria:
- factor build and dataset build are linked by fingerprint
- training and prediction load manifest-aware inputs consistently

### Milestone D: Real Modular Architecture

Scope:
- Task 4
- Task 5

Success criteria:
- design-doc module groups exist as real modules, not aliases
- portfolio and experiment subsystems no longer live mostly in `train.py`

### Milestone E: Release-Ready Repository

Scope:
- Task 6

Success criteria:
- end-to-end commands pass
- docs reflect the actual default workflow

## Recommended Order And Effort

- Week 1:
  - Task 1
  - Task 3
- Week 2:
  - Task 2
- Week 3:
  - Task 4
- Week 4:
  - Task 5
  - Task 6

## Risks

- The ingestion adapter work has external dependency risk. Keep provider-specific client code isolated behind tiny wrappers so unit tests stay offline.
- The module split has regression risk because `train.py` currently owns too much logic. Add parity tests before moving code.
- The portfolio/experiment upgrade has score regression risk. Keep default strategy behavior identical unless a config flag enables the new logic.

## Self-Review

- Spec coverage:
  - baseline stabilization, ingestion completion, data/manifest unification, modularization, portfolio, and release hardening are all mapped to explicit tasks.
- Placeholder scan:
  - no `TODO` or `TBD` markers remain.
- Type consistency:
  - compatibility shell remains `StockTransformer` in `train.py` and `predict.py`.
  - compatibility output remains `output/result.csv`.

