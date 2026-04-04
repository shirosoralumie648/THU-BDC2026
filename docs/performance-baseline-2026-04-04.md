# Performance Baseline 2026-04-04

## Scope

本轮 Phase 3 当前已完成两类低风险、可量化工作：

- 建立训练、dataset build、prediction path、factor graph 四条轻量 benchmark
- 给因子 execution plan、factor manifest 解析、dataset build manifest 读取补缓存，先消除重复 JSON 解析与重复 DAG 编译

## Commands

```bash
/home/shirosora/code_storage/THU-BDC2026/.worktrees/phase1-core-algo/.venv/bin/python code/src/benchmarks/training_path.py
/home/shirosora/code_storage/THU-BDC2026/.worktrees/phase1-core-algo/.venv/bin/python code/src/benchmarks/factor_graph_path.py
/home/shirosora/code_storage/THU-BDC2026/.worktrees/phase1-core-algo/.venv/bin/python -c "import json, sys; sys.path.insert(0, 'code/src'); from benchmarks.training_path import run_dataset_build_benchmark; print(json.dumps(run_dataset_build_benchmark(), ensure_ascii=False, indent=2))"
/home/shirosora/code_storage/THU-BDC2026/.worktrees/phase1-core-algo/.venv/bin/python -c "import json, sys; sys.path.insert(0, 'code/src'); from benchmarks.training_path import run_prediction_path_benchmark; print(json.dumps(run_prediction_path_benchmark(), ensure_ascii=False, indent=2))"
```

## Baseline Metrics

| Benchmark | Input Shape | Wall Time | Throughput | Peak Memory |
| --- | --- | ---: | ---: | ---: |
| `training_path` | 1920 rows / 24 stocks / 12 features / `sequence_length=20` | `0.0562s` | `1086.27 samples/s` | `581.63 MB` |
| `dataset_build_path` | 72 base rows / 54 train rows / 18 test rows | `0.8715s` | `82.62 rows/s` | `574.04 MB` |
| `prediction_path` | 960 rows / 24 stocks / 12 features / `sequence_length=20` | `0.0376s` | `25529.49 rows/s` | `576.78 MB` |
| `factor_graph_path` | 240 rows / 8 stocks / 13 factor columns | `1.1531s` | `208.14 rows/s` | `107.54 MB` |

## Metrics Payloads

### `training_path`

```json
{
  "benchmark_name": "training_path",
  "rows": 1920,
  "stocks": 24,
  "features": 12,
  "sequence_length": 20,
  "samples": 61,
  "targets": 1464,
  "wall_time_seconds": 0.05615563498577103,
  "samples_per_second": 1086.2667658456085,
  "peak_memory_mb": 581.62890625
}
```

### `dataset_build_path`

```json
{
  "benchmark_name": "dataset_build_path",
  "base_rows": 72,
  "train_rows": 54,
  "test_rows": 18,
  "wall_time_seconds": 0.8714611970062833,
  "rows_per_second": 82.61985759932908,
  "peak_memory_mb": 574.03515625
}
```

### `prediction_path`

```json
{
  "benchmark_name": "prediction_path",
  "rows": 960,
  "stocks": 24,
  "features": 12,
  "sequence_length": 20,
  "sequence_count": 24,
  "wall_time_seconds": 0.03760356499697082,
  "rows_per_second": 25529.494346542237,
  "peak_memory_mb": 576.78125
}
```

### `factor_graph_path`

```json
{
  "benchmark_name": "factor_graph_path",
  "rows": 240,
  "stocks": 8,
  "factor_columns": 13,
  "wall_time_seconds": 1.1530674149980769,
  "samples_per_second": 208.14047546422105,
  "peak_memory_mb": 107.5390625,
  "factor_fingerprint": "d64f636142a41c76"
}
```

## First Optimization Batch

- `code/src/factor_store.py`
  - `build_factor_execution_plan(...)` 现在按因子规格缓存 DAG 编译结果，`resolve_factor_pipeline(...)` 与 `load_factor_snapshot(...)` 会复用同一 execution plan。
- `code/src/data_manager.py`
  - `load_train_dataset_from_build_manifest(...)` 现在按 manifest 文件签名缓存 JSON payload，避免训练与预测链路重复读取同一 manifest。
- `code/src/manage_data.py`
  - 解析 factor fingerprint 时现在按 manifest 文件签名缓存 JSON payload，避免 `build-dataset` 在同一进程内重复读取相同 factor build manifest。

## Regression Guards

- `test/test_performance_smoke.py`
  - 约束四条 benchmark 必须持续输出稳定字段
- `test/test_factor_store_engine.py`
  - 约束相同因子规格不会重复编译 execution plan
- `test/test_dataset_manifest_linkage.py`
  - 约束 dataset build manifest 与 factor build manifest 不重复 `json.load`

## Caveats

- 当前是 smoke baseline，不是严格微基准；主要用于发现数量级退化和验证缓存是否生效。
- `peak_memory_mb` 依赖进程级统计，适合回归对比，不适合当作绝对精确值。
- `prediction_path` 当前覆盖的是预测序列装配路径，还不是完整模型前向与 checkpoint 加载耗时。

## Second Optimization Batch 2026-04-05

- `code/src/factor_store.py`
  - 移除模块顶层对 `utils` 的静态导入，改为 `_get_feature_engineer(feature_set)` 按需加载 `engineer_features_39` / `engineer_features_158plus39`。
  - `resolve_factor_pipeline(...)`、`engineer_group_features(...)`、`load_factor_snapshot(...)` 统一复用同一懒加载入口。
- `test/test_factor_store_engine.py`
  - 新增回归测试，约束“仅导入 `factor_store`”时不会提前触发 `utils` 导入。

## Latest Observations 2026-04-05

### Verified Commands

```bash
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python -m unittest test.test_factor_store_engine test.test_factor_graph_pipeline test.test_manifest_contracts -v
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python code/src/benchmarks/factor_graph_path.py
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python -c "import json, sys; sys.path.insert(0, 'code/src'); from benchmarks.training_path import run_dataset_build_benchmark; print(json.dumps(run_dataset_build_benchmark(), ensure_ascii=False))"
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python -X importtime -c "import sys; sys.path.insert(0, 'code/src'); import build_factor_graph" 2>&1 | rg "utils|factor_store|build_factor_graph"
```

### Current Metrics

| Benchmark | Wall Time | Notes |
| --- | ---: | --- |
| `factor_graph_path` | `0.6392s` | 相比 2026-04-04 基线 `1.1531s` 继续下降 |
| `dataset_build_path` | `0.5287s` | 相比 2026-04-04 基线 `0.8715s` 继续下降 |

### Import-Time Snapshot

```text
import time:      3634 |       3634 |   factor_store
import time:      1089 |     436613 | build_factor_graph
```

- 本轮观测里 `utils` 已不再出现在 `build_factor_graph` 的导入链上，说明 `factor_store -> utils` 顶层依赖已被切断。
- 当前 `factor_graph_path` 的剩余主要成本来自 CLI 子进程与 `pandas` 冷启动，而不是 `factor_store` 自身导入。
