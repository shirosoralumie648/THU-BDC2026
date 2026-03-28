# Unified Ingestion Platform Design

**Date:** 2026-03-28  
**Status:** Draft approved in-terminal, pending final user review  
**Scope:** THU-BDC2026 unified ingestion platform, phase 1 architecture and delivery boundaries

## Goal

Build a unified ingestion platform that supports both local CLI execution and future HTTP service access, while keeping one shared execution core for dataset registry, adapter execution, quality checks, normalized storage, and ingestion manifests.

The first delivery target is a service-oriented skeleton, not a full workflow platform. The design must support later replay, backfill, scheduling, and observability without forcing a second architectural rewrite.

## Problem Statement

The current project has already improved the downstream pipeline:

- Pipeline configuration is now formalized in `datasets.yaml`, `factors.yaml`, and `storage.yaml`.
- Factor DAG construction is implemented.
- Dataset build manifests are integrated into training, prediction, and reselection.

The remaining weak point is the ingestion layer. Dataset contracts exist in configuration, but the runtime model for executing ingestion jobs, normalizing source payloads, persisting raw and curated data, and exposing reusable job semantics is not yet fully platformized.

Without a unified ingestion platform:

- CLI scripts and future API endpoints will drift into separate logic paths.
- Replay and backfill will be inconsistent.
- Source-specific field formats will leak into downstream logic.
- Manifest lineage and quality enforcement will remain partial.

## Chosen Direction

The selected direction is `Service Skeleton First`.

Alternatives that were considered:

- `CLI First`
  This would produce quick local execution but would turn the future service layer into a thin wrapper over script logic, making retries, replay, and lifecycle state unnatural.
- `Full Workflow Platform Immediately`
  This would include scheduler, monitoring, replay, backfill, and observability from day one, but it is too broad for the current repository stage and would slow delivery of the useful core.

The chosen direction keeps the first phase small enough to implement while still giving CLI and HTTP one shared ingestion core.

## Architecture

The ingestion platform is split into three layers.

### Entry Layer

This layer exposes ingestion operations to callers.

- CLI commands in `manage_data.py`
- Future HTTP endpoints such as `/ingestion/jobs`
- Existing or future GUI entry points

Responsibilities:

- Validate external input shape
- Translate input into service-layer request objects
- Format service-layer responses for the caller

Non-responsibilities:

- No source fetching logic
- No storage path decisions
- No retry or replay logic
- No manifest construction

### Application Layer

This is the shared execution core.

Primary components:

- `DatasetRegistry`
  Loads and resolves dataset definitions from `datasets.yaml`.
- `IngestionService`
  Main orchestration entry point for create-job, run-job, get-job, and replay-job flows.
- `JobPlanner`
  Turns one user request into executable ingestion jobs. In phase 1 this may resolve to a single direct job. In phase 3 it expands to sliced backfill planning.
- `JobRunner`
  Executes one concrete ingestion job.
- `QualityGate`
  Runs pre-write and post-write checks.
- `ManifestService`
  Produces ingestion manifests in a consistent contract.

Responsibilities:

- Resolve `DatasetSpec`
- Build job requests
- Transition job state
- Invoke source adapters
- Normalize schema into canonical form
- Run quality checks
- Write raw and curated outputs
- Write ingestion manifest

### Infrastructure Layer

This layer contains execution backends and source integrations.

Primary components:

- Source adapters such as `baostock_daily`, `market_minute_bar`, `macro_timeseries`
- Raw writer
- Curated writer
- Manifest persistence
- Local filesystem storage in phase 1

Responsibilities:

- Fetch source payloads
- Map source fields into intermediate raw records
- Persist files to configured storage targets

Non-responsibilities:

- No business decision about retries
- No dataset planning logic
- No caller-facing job API semantics

## Core Design Rules

- CLI and HTTP must call the same ingestion service core.
- Dataset adapters only fetch and map source payloads.
- All storage path and manifest decisions belong to the platform, not to adapters.
- Downstream factor and training logic must consume canonical curated data only.
- Replay must prefer historical manifest context over current mutable runtime defaults.

## Dataset Model

The platform uses three core runtime models.

### DatasetSpec

Derived from `datasets.yaml`.

Required semantics:

- `dataset`
- `domain`
- `granularity`
- `source.name`
- `source.adapter`
- `request` contract
- canonical `schema`
- `quality` rules
- `storage` targets

Purpose:

- This is the source of truth for what a dataset is and how it should be ingested.

### IngestionJob

Represents one executable ingestion unit.

Required fields:

- `job_id`
- `dataset`
- `source`
- `request`
- `status`
- `attempt`
- `created_at`
- `started_at`
- `finished_at`
- `parent_job_id` for replay or sliced jobs when applicable

Purpose:

- This is the unit of execution, retry, replay, and audit.

### IngestionResult

Represents the standardized output of one job execution.

Required fields:

- `row_count`
- `schema_hash`
- `data_hash`
- `quality_summary`
- `raw_paths`
- `curated_paths`
- `warnings`
- `errors`
- `code_version`

Purpose:

- This is the normalized result consumed by manifest generation and job reporting.

## API Contract

Phase 1 HTTP surface should be minimal.

### `POST /ingestion/jobs`

Purpose:

- Submit a dataset ingestion job.

Request shape:

- `dataset`
- `source`
- `start`
- `end`
- `mode`
- `universe`
- `adjustment`
- optional runtime overrides when needed

Response shape:

- `job_id`
- `status`
- `dataset`
- `submitted_at`

### `GET /ingestion/jobs/{job_id}`

Purpose:

- Query job lifecycle state and outputs.

Response shape:

- job metadata
- state
- request
- attempts
- output paths
- warnings
- errors
- manifest path

### `POST /ingestion/replay`

Purpose:

- Re-run a historical job using recorded context.

Request shape:

- `job_id` or manifest reference
- replay reason
- optional override mode if explicitly supported

Response shape:

- new `job_id`
- replay source reference
- status

### `GET /ingestion/datasets`

Purpose:

- List registered datasets and supported capabilities.

Response shape:

- dataset id
- source adapter
- granularity
- whether replay and backfill are supported

## CLI Contract

CLI remains first-class but no longer owns ingestion logic.

Proposed command family:

- `manage_data ingest run`
- `manage_data ingest status`
- `manage_data ingest replay`
- `manage_data ingest datasets`

Behavior:

- CLI commands map to the same service-layer methods that back the HTTP API.
- Local synchronous mode is allowed in phase 1 for developer usability.
- CLI output should still display job ids, manifest path, row count, and warnings.

## Job State Machine

The platform should not collapse all outcomes into simple pass or fail.

Recommended states:

- `queued`
- `running`
- `writing_raw`
- `writing_curated`
- `manifest_written`
- `succeeded`
- `fetch_failed`
- `quality_failed`
- `write_failed`
- `retryable_failed`

State semantics:

- `retryable_failed` is reserved for temporary faults such as network issues or vendor throttling.
- `fetch_failed` is for non-retryable source-side failures such as malformed payloads or unsupported response contracts.
- `quality_failed` and `write_failed` should not auto-retry by default because they indicate invariant or persistence problems.

## Dataset Scope

Phase 1 dataset scope should cover the current training and factor pipeline while staying bounded.

### Reference Datasets

- `instrument_dim`
- `trading_calendar`
- `universe_membership`
- `index_membership`

### Market Datasets

- `market_bar_1d`
- `market_bar_1m`
- `index_bar_1d`
- `adj_factor`
- `suspension_status`

### Macro / Fundamental / Event Datasets

- `macro_series`
- `financial_statement_pit`
- `corporate_action`

### Recommended First Implemented Slice

The smallest useful first implementation set is:

- `market_bar_1d`
- `market_bar_1m`
- `macro_series`

Reason:

- These three already support the existing daily training pipeline, high-frequency-to-daily factor derivation, and macro factor construction.

## Storage Design

The ingestion platform writes two storage layers in phase 1.

### Raw Layer

Purpose:

- Preserve vendor-origin payloads and original field naming.

Properties:

- Audit-friendly
- Replay-friendly
- Not directly consumed by factor or model pipelines

Path semantics:

- `source=...`
- `dataset=...`
- `ingest_date=...`
- `run_id=...`

### Curated Layer

Purpose:

- Provide canonical, normalized, downstream-safe tables.

Properties:

- Single canonical schema per dataset
- Stable primary keys
- Unified identifiers and time semantics
- The only ingestion layer consumed by factors and training

## Normalization Rules

Canonical normalization rules must be enforced during ingestion, not deferred downstream.

- All securities must use `instrument_id`.
- Daily data must use `trade_date`.
- Intraday data must use `ts`.
- Macro and financial datasets must retain `available_time`.
- Vendor-specific codes and field names must not leak into curated datasets.
- Numeric fields must be cast into the configured canonical types.
- Primary-key semantics must be enforced before curated write.

## Point-in-Time Discipline

Point-in-time correctness is mandatory for macro and financial datasets.

- Storing only observation date is insufficient.
- The curated schema must retain release or availability timestamps.
- Downstream factor joins must use `available_time` semantics, not observational time only.

This is required to prevent leakage in both factor generation and model training.

## Factor and Training Lineage

The ingestion platform sits upstream of the already-improved factor and dataset pipeline.

Required lineage chain:

- `raw`
- `curated`
- factor DAG build
- `feature_long`
- `feature_wide`
- dataset build
- train or predict

Rules:

- Factor computation may read curated market, macro, and event data.
- Factor computation must not read raw vendor payloads directly.
- High-frequency-derived daily factors should remain factor artifacts, not be written back as base market bars.
- Every transition point must keep a manifest reference wherever practical.

## Manifest Contract

Each ingestion job must write one ingestion manifest.

Required content:

- request parameters
- resolved dataset and adapter
- job state summary
- row count
- schema hash
- data hash
- quality summary
- raw output paths
- curated output paths
- code version
- warnings
- errors

Purpose:

- Replay
- Backfill audit
- Data lineage
- Failure diagnosis
- Downstream traceability

## Retry Policy

Retry behavior must be explicit.

Retryable cases:

- network interruption
- supplier temporary outage
- throttling
- transient authentication or gateway error if recoverable

Non-retryable cases:

- invalid schema mapping
- key uniqueness failure
- quality-gate hard failure
- malformed source payload incompatible with configured adapter contract

The retry policy should be configured centrally, not buried inside adapters.

## Replay Policy

Replay is not the same as retry.

Replay rules:

- A replay operation should use historical manifest or historical job context as its primary source.
- Replay should record a new job id and link to the prior job.
- Replay reason should be explicitly recorded.
- Replay should allow corrected configuration only when explicitly requested and recorded.

## Backfill Policy

Backfill should be planned, not improvised.

Rules:

- Large date ranges should be split into smaller job slices by `JobPlanner`.
- Slice strategy should depend on dataset granularity and source limits.
- Backfill execution should be resumable at the job-slice level.
- Failure handling should support hole-filling instead of forcing a full rerun of the entire range.

## Quality Gates

Quality enforcement should be split into pre-write and post-write stages.

### Pre-Write Gates

- required columns present
- canonical type validation
- primary-key uniqueness
- market-session validity for intraday data
- non-negative or valid price constraints where applicable

### Post-Write Gates

- row count
- partition write success
- time-range coverage
- stock or series coverage
- null-rate checks where configured
- persisted file snapshot capture

## Phased Delivery Plan

The delivery path should be incremental.

### Phase 1

- Implement `DatasetRegistry`
- Implement `IngestionService`
- Implement minimal direct-job planning inside `JobPlanner`
- Implement `JobRunner`
- Implement local job-state persistence
- Implement raw and curated write path handling
- Implement ingestion manifest writing
- Expose CLI entry points

### Phase 2

- Add HTTP API endpoints for jobs, replay, and dataset listing
- Keep service core unchanged
- Ensure CLI and HTTP both use the same service methods

### Phase 3

- Extend `JobPlanner` with slicing for backfill
- Add replay helpers
- Add GUI integration through the same service layer

### Phase 4

- Add observability
- Add job history views
- Add metrics and quality dashboards
- Add alerting or notification hooks

## Repository Integration

The existing repository already contains:

- `config/datasets.yaml`
- `config/storage.yaml`
- manifest helpers
- `manage_data.py`
- factor DAG and dataset build flows

The ingestion platform should extend these, not replace them.

Repository integration principles:

- Reuse `datasets.yaml` as registry input.
- Reuse storage-layer conventions already declared in `storage.yaml`.
- Add ingestion-oriented service modules under `code/src`.
- Keep current factor and dataset build logic downstream of curated outputs.
- Avoid writing provider-specific fetch logic directly inside `manage_data.py`.

## Testing Strategy

Testing must cover the platform at the contract level, not just utility functions.

### Unit Tests

- dataset registry parsing
- adapter contract validation
- job state transitions
- manifest generation
- storage path rendering

### Integration Tests

- `market_bar_1d` local ingest path
- `market_bar_1m` local ingest path
- `macro_series` local ingest path
- raw and curated file creation
- ingestion manifest generation

### Failure Tests

- retryable source failure
- non-retryable schema failure
- quality gate rejection
- replay from manifest

## Risks and Mitigations

### Risk: Adapter Logic Becomes Too Smart

Mitigation:

- Keep adapters limited to fetch and source-field mapping.

### Risk: CLI and HTTP Drift Again

Mitigation:

- Do not allow either entry layer to bypass `IngestionService`.

### Risk: Point-in-Time Leakage in Macro and Financial Data

Mitigation:

- Make `available_time` mandatory in canonical schema where applicable.

### Risk: Backfill Complexity Expands Too Early

Mitigation:

- Phase backfill after service core and basic API are working.

## Out of Scope for This Design

- Full production scheduler implementation
- Distributed queue execution
- External database-backed job store
- Full observability dashboard implementation
- Full feature-long governance implementation

These are future-compatible concerns but should not block phase 1.

## Acceptance Criteria

This design is successful when:

- One ingestion service core drives both CLI and HTTP entry points.
- At least `market_bar_1d`, `market_bar_1m`, and `macro_series` can be ingested through the unified flow.
- Raw and curated outputs are both written through platform-owned contracts.
- Each ingestion job produces a manifest with replay-safe metadata.
- Downstream factor generation can rely on curated canonical data only.
