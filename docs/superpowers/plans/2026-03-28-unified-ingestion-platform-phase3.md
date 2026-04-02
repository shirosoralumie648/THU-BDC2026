# Unified Ingestion Platform Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bridge legacy stock fetching into the ingestion platform, add live minute and macro adapters, and orchestrate HF daily factor plus factor-graph generation from the unified service layer.

**Architecture:** Keep source ingestion in `code/src/ingestion/`, add focused compatibility and orchestration modules there, and reuse `build_hf_daily_factors.py` plus `build_factor_graph.py` as computation backends invoked through shared service entry points. Preserve legacy CSV outputs while treating canonical curated tables and manifests as the internal contract.

**Tech Stack:** Python 3.10, pandas, unittest, tempfile, PyYAML, optional akshare/baostock providers, existing factor graph and manifest helpers

---

## Target File Structure

### New Files

- `code/src/ingestion/compat.py`
  Legacy export helpers for `stock_data.csv` and bridge manifests.
- `code/src/ingestion/pipeline_service.py`
  Orchestration for HF daily factor build and factor graph build.
- `test/test_ingestion_adapters_live.py`
  Fake-provider tests for minute and macro live adapter paths.
- `test/test_ingestion_pipeline_service.py`
  Orchestration tests for derived factor stages.
- `test/test_get_stock_data_bridge.py`
  Bridge logic tests for legacy stock export behavior.

### Modified Files

- `get_stock_data.py`
  Delegate to bridge path by default while preserving controlled legacy fallback.
- `code/src/ingestion/service.py`
  Add convenience execution helpers used by bridge and orchestration.
- `code/src/ingestion/adapters/akshare_minute.py`
  Replace file-only placeholder with live-provider-first implementation.
- `code/src/ingestion/adapters/akshare_macro.py`
  Replace file-only placeholder with live-provider-first implementation.
- `code/src/manage_data.py`
  Add CLI entry points for HF daily factor build and factor graph orchestration.
- `code/src/build_hf_daily_factors.py`
  Allow direct in-process invocation with argv for orchestration reuse.

## Task 1: Add Failing Tests for Bridge and Live Adapters

- [ ] Add a failing bridge test for legacy stock export against a fake ingestion service.
- [ ] Run the bridge test and verify it fails for missing compatibility module/behavior.
- [ ] Add failing minute adapter tests that inject a fake `akshare` provider and verify canonical minute output.
- [ ] Add failing macro adapter tests that inject a fake `akshare` provider and verify supported series normalization.
- [ ] Run the adapter tests and verify they fail for missing live-provider behavior.

## Task 2: Implement Bridge and Adapter Support

- [ ] Add `code/src/ingestion/compat.py` with legacy stock export helpers.
- [ ] Add convenience request execution helpers to `code/src/ingestion/service.py`.
- [ ] Update `get_stock_data.py` to prefer the bridge path and write legacy-compatible output plus manifest.
- [ ] Implement live minute fetching in `code/src/ingestion/adapters/akshare_minute.py` with file fallback.
- [ ] Implement live macro fetching in `code/src/ingestion/adapters/akshare_macro.py` with file fallback.
- [ ] Re-run the bridge and adapter tests until green.

## Task 3: Add Failing Tests for Derived Factor Orchestration

- [ ] Add a failing orchestration test that stages minute/macro inputs and expects HF daily factor plus factor graph outputs.
- [ ] Add a failing CLI test for the new orchestration command(s) in `manage_data.py`.
- [ ] Run the new tests and verify they fail for missing orchestration modules/commands.

## Task 4: Implement Pipeline Orchestration

- [ ] Add `code/src/ingestion/pipeline_service.py` to run HF daily factor generation and factor graph generation in-process.
- [ ] Update `code/src/build_hf_daily_factors.py` so orchestration can call it with explicit argv.
- [ ] Extend `code/src/manage_data.py` with orchestration entry points for HF daily factor build and factor graph build under the unified data management surface.
- [ ] Re-run orchestration and CLI tests until green.

## Task 5: Verify Regressions and Update Branch State

- [ ] Run the focused bridge, adapter, orchestration, ingestion, and factor regression tests.
- [ ] Inspect changed files and confirm no unrelated dirty synced files were modified.
- [ ] Commit the phase 3 changes onto `ingestion-platform-phase1-2`.
- [ ] Push the branch so PR `#3` picks up the new commits.
