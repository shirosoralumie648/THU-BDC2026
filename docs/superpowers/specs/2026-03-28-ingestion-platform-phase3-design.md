# Unified Ingestion Platform Phase 3 Design

**Date:** 2026-03-28  
**Status:** Approved in-terminal for direct implementation  
**Scope:** THU-BDC2026 ingestion bridge, live adapters, and factor orchestration follow-up

## Goal

Extend the phase 1-2 ingestion skeleton into a usable end-to-end market data pipeline by:

- turning `get_stock_data.py` into a compatibility bridge over the shared ingestion service
- replacing placeholder minute and macro adapters with live-provider-first implementations
- wiring high-frequency daily factor generation and factor DAG execution into a shared orchestration layer

This phase keeps the existing training and prediction interfaces stable while moving fetching and feature generation toward one service-oriented runtime.

## Non-Goals

- no scheduler, queue, or distributed worker system
- no training service refactor in this phase
- no forced migration of every legacy script into `ingestion/`
- no destructive cleanup of existing local data layout

## Chosen Direction

The selected direction is `Compatibility Bridge + Derived Pipeline Orchestration`.

Why this direction:

- the repository already contains working factor build scripts and training compatibility exports
- the fastest safe path is to reuse those scripts behind service-owned orchestration instead of rewriting factor computation from scratch
- legacy callers still need `stock_data.csv`, `train.csv`, and related outputs, so the new platform must export compatible artifacts while standardizing internals

## Incremental Architecture

### 1. Compatibility Layer

`get_stock_data.py` remains a supported entry point, but it should delegate to the ingestion platform by default.

Responsibilities:

- parse existing CLI arguments
- submit a `market_bar_1d` ingestion request
- export the canonical curated result back into the legacy `stock_data.csv` shape
- persist a compatibility manifest that records the source ingestion job

Fallback rule:

- if the bridge cannot be used, the script may keep a controlled legacy direct-fetch fallback instead of hard-failing

### 2. Live Provider Adapters

The adapters for `market_bar_1m` and `macro_series` become live-provider-first and file-fallback-second.

#### Minute adapter

- preferred provider: `akshare`
- preferred interface: A-share minute history endpoint
- symbol source: explicit `symbols` from request, otherwise HS300 universe resolution
- output schema: canonical `instrument_id`, `ts`, `open`, `high`, `low`, `close`, `volume`, `amount`

#### Macro adapter

- preferred provider: `akshare`
- supported series in this phase:
  - `cpi_yoy`
  - `ppi_yoy`
  - `m2_yoy`
  - `shibor_3m`
  - `usdcny`
  - `csi300_pe_ttm`
- output schema: canonical `series_id`, `observation_date`, `release_time`, `available_time`, `frequency`, `vintage`, `value`

Fallback rule:

- if `akshare` is unavailable or a requested series cannot be fetched live, adapter-level file input remains available for replay and local testing

### 3. Derived Pipeline Orchestration

Introduce a shared orchestration service that runs downstream derived artifacts from canonical inputs.

Primary stages:

1. `market_bar_1d` ingestion
2. `market_bar_1m` ingestion
3. `macro_series` ingestion
4. HF-to-daily aggregation via `build_hf_daily_factors.py`
5. factor DAG execution via `build_factor_graph.py`

The orchestration service owns:

- dataset selection
- input/output path handoff between stages
- manifest aggregation
- compatibility export paths for legacy scripts and downstream training

The factor scripts remain computation engines, but service entry points own when and how they run.

## Storage and Compatibility Rules

Internal rule:

- fetched tables are treated as canonical normalized inputs
- factor artifacts are treated as derived products

Compatibility rule:

- legacy CSV exports remain first-class outputs while the service matures

Expected artifact families:

- daily market compatibility export: `stock_data.csv`
- HF daily factor compatibility export: `hf_daily_factors.csv`
- factor wide compatibility export rendered from `factors.yaml`
- train/test compatibility exports rendered from `storage.yaml`

Every orchestration run must emit a manifest that records:

- source ingestion job IDs
- derived artifact paths
- selected feature set version
- factor fingerprint when available
- quality and lineage summary

## Testing Strategy

Phase 3 is still test-first.

Required coverage:

- bridge export logic for `get_stock_data.py`
- live-path minute adapter behavior with a fake `akshare` module
- live-path macro adapter behavior with a fake `akshare` module
- orchestration entry points that build HF daily factors and factor graph outputs from staged inputs
- regression coverage for existing factor scripts and CLI commands

## Delivery Boundary

This phase is complete when:

- `get_stock_data.py` can obtain `stock_data.csv` through the ingestion service path
- `manage_data.py` exposes orchestration commands for derived factor stages
- minute and macro adapters support real-provider fetch logic
- the existing PR can demonstrate an end-to-end data path from fetch to factor output without breaking current tests
