# Performance Baseline 2026-04-04

## Scope

本轮 Phase 3 先完成两类低风险、可量化工作：

- 建立训练路径与 factor graph 路径的轻量 benchmark 脚本
- 给因子 pipeline 解析和 dataset build manifest 读取补缓存，先消除重复 JSON 解析与重复 execution plan 构建

## Commands

```bash
/home/shirosora/code_storage/THU-BDC2026/.worktrees/phase1-core-algo/.venv/bin/python code/src/benchmarks/training_path.py
/home/shirosora/code_storage/THU-BDC2026/.worktrees/phase1-core-algo/.venv/bin/python code/src/benchmarks/factor_graph_path.py
```

## Baseline Metrics

| Benchmark | Input Shape | Wall Time | Throughput | Peak Memory |
| --- | --- | ---: | ---: | ---: |
| `training_path` | 1920 rows / 24 stocks / 12 features / `sequence_length=20` | `0.0384s` | `1590.04 samples/s` | `115.62 MB` |
| `factor_graph_path` | 240 rows / 8 stocks / 13 factor columns | `0.6536s` | `367.18 rows/s` | `106.98 MB` |

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
  "wall_time_seconds": 0.03836376700201072,
  "samples_per_second": 1590.041978849545,
  "peak_memory_mb": 115.6171875
}
```

### `factor_graph_path`

```json
{
  "benchmark_name": "factor_graph_path",
  "rows": 240,
  "stocks": 8,
  "factor_columns": 13,
  "wall_time_seconds": 0.6536242649890482,
  "samples_per_second": 367.1834306886102,
  "peak_memory_mb": 106.9765625,
  "factor_fingerprint": "d64f636142a41c76"
}
```

## First Optimization Batch

- `code/src/factor_store.py`
  - `resolve_factor_pipeline(...)` 现在按 `store_path` / `builtin_registry_path` 的文件签名缓存 execution plan，避免同一 run 内重复读取 JSON 并重复拓扑排序。
- `code/src/data_manager.py`
  - `load_train_dataset_from_build_manifest(...)` 现在按 manifest 文件签名缓存 JSON payload，避免训练与预测链路重复读取同一 manifest。

## Regression Guards

- `test/test_performance_smoke.py`
  - 约束 benchmark 脚本必须持续输出稳定字段
- `test/test_factor_store_engine.py`
  - 约束同一输入下不重复构建 factor execution plan
- `test/test_dataset_manifest_linkage.py`
  - 约束同一 manifest 不重复 `json.load`

## Caveats

- 当前是 smoke baseline，不是严格微基准；主要用于发现数量级退化和验证缓存是否生效。
- `peak_memory_mb` 依赖进程级统计，适合回归对比，不适合当作绝对精确值。
- dataset build 与 predict 的独立基线还没有拆成单独 benchmark 脚本，下一轮可继续补齐。
