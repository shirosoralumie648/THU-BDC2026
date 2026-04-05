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

## Third Optimization Batch 2026-04-05

- `code/src/data_manager.py`
  - 新增 `build_canonical_csv_metadata_from_dataframe(...)`，专门服务已经规范化的输出表，避免重复做日期推断和股票代码归一化。
- `code/src/build_factor_graph.py`
  - `output_df` 的 manifest CSV metadata 改走 canonical helper。
- `code/src/manage_data.py`
  - `train_df` / `test_df` 的 dataset manifest CSV metadata 改走 canonical helper。
- `test/test_manifest_contracts.py`
  - 新增 canonical metadata helper 合同测试，以及 `manage_data` 输出侧调用约束。
- `test/test_factor_graph_pipeline.py`
  - 新增 `build_factor_graph` 输出侧调用约束。

## Latest Observations After Third Batch 2026-04-05

### Verified Commands

```bash
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python -m unittest test.test_factor_store_engine test.test_factor_graph_pipeline test.test_manifest_contracts test.test_hf_daily_factor_pipeline test.test_cli_error_paths test.test_ingestion_runtime -v
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python code/src/benchmarks/factor_graph_path.py
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python -c "import json, sys; sys.path.insert(0, 'code/src'); from benchmarks.training_path import run_dataset_build_benchmark; print(json.dumps(run_dataset_build_benchmark(), ensure_ascii=False))"
```

### Current Metrics

| Benchmark | Wall Time | Previous Local Observation | Delta |
| --- | ---: | ---: | ---: |
| `factor_graph_path` | `0.5875s` | `0.7458s` | `-0.1583s` |
| `dataset_build_path` | `0.4887s` | `0.5457s` | `-0.0570s` |

### Process-Level Profile Snapshot

```text
build_factor_graph.main(...)                     0.220s
build_csv_metadata_from_dataframe(...) x3       0.016s
build_canonical_csv_metadata_from_dataframe(...) 0.003s
```

- 这一批优化没有改变 manifest 契约，只把“输出侧已经规范化的数据表”切换到更便宜的 metadata 路径。
- `factor_graph_path` 和 `dataset_build_path` 都出现了正向改善，说明这条快路径不只是进程内 profile 好看，也体现在端到端 benchmark 上。

## Fourth Optimization Batch 2026-04-05

- `code/src/data_manager.py`
  - 给 `build_csv_metadata_from_dataframe(...)` 增加 canonical fast-path：
    - 当日期列已经是 `YYYY-MM-DD` 文本时，直接走字典序 `min/max`
    - 当股票列已经是 6 位数字代码时，直接做 `nunique`
  - 仅在不满足上述条件时才回退到 `pd.to_datetime(...)` 和 `normalize_stock_code_series(...)`
- `test/test_manifest_contracts.py`
  - 新增回归测试，约束 canonical 列不会再触发 `pd.to_datetime` / `normalize_stock_code_series`

## Latest Observations After Fourth Batch 2026-04-05

### Verified Commands

```bash
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python -m unittest test.test_factor_store_engine test.test_factor_graph_pipeline test.test_manifest_contracts test.test_hf_daily_factor_pipeline test.test_cli_error_paths test.test_ingestion_runtime -v
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python code/src/benchmarks/factor_graph_path.py
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python -c "import json, sys; sys.path.insert(0, 'code/src'); from benchmarks.training_path import run_dataset_build_benchmark; print(json.dumps(run_dataset_build_benchmark(), ensure_ascii=False))"
```

### Current Metrics

| Benchmark | Wall Time | Notes |
| --- | ---: | --- |
| `factor_graph_path` | `0.6000s` | 与上一刀 `0.5875s` 接近，仍显著优于 `0.7458s` 以前的观测 |
| `dataset_build_path` | `0.3856s` | 单次观测明显下降，CLI 噪声仍存在 |

### Process-Level Profile Snapshot

```text
build_factor_graph.main(...)     0.199s
command_build_dataset(...)       0.045s
load_pipeline_configs(...)       0.014s
build_csv_metadata_from_dataframe(...) 0.007s
build_canonical_csv_metadata_from_dataframe(...) x2 0.004s
```

- 进程内 `command_build_dataset(...)` 已低于之前记录的大约 `0.066s`，说明这批优化确实压到了 dataset build 本体，而不仅仅是 benchmark 偶然波动。
- 当前端到端 benchmark 仍会受进程冷启动影响，所以更可靠的信号仍然是函数级 profile 与多次趋势对比。

## Fifth Optimization Batch 2026-04-05

- `code/src/build_factor_graph.py`
  - 给 `macro_asof_join` 引入 `_build_macro_cutoff_frame(...)`，把交易日 cutoff 时间框架从“每个宏观节点重复构造”改成“一次构造，多次复用”。
  - 在 `main(...)` 中预先按 `series_id` 构建 `macro_series_map`，节点循环里不再对整张 `macro_df` 反复做 `macro_df[macro_df['series_id'] == ...]` 过滤。
  - 继续复用 `macro_join_frame` 聚合所有宏观列，最后一次性回并到 `base_df`。
  - 修正 `_compute_macro_series_asof(...)` 的 staleness 边界：`fill_method=forward` 只允许填补有效窗口内的缺值，不能把超过 `max_staleness_days` 的旧值重新补活。
- `test/test_factor_graph_pipeline.py`
  - 新增源码级回归测试，约束 `macro_asof_join` 不再对完整 `macro_df` 做逐节点过滤。
  - 新增行为级回归测试，约束 `forward fill` 不会穿透 `max_staleness_days` 边界。

## Latest Observations After Fifth Batch 2026-04-05

### Verified Commands

```bash
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python -m unittest test.test_factor_store_engine test.test_factor_graph_pipeline test.test_manifest_contracts test.test_hf_daily_factor_pipeline test.test_cli_error_paths test.test_ingestion_runtime -v
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python code/src/benchmarks/factor_graph_path.py
/home/shirosora/code_storage/THU-BDC2026/.venv/bin/python - <<'PY'
import cProfile
import pstats
import sys
import tempfile
from pathlib import Path

ROOT = Path('/home/shirosora/code_storage/THU-BDC2026/.worktrees/phase3-performance-baseline')
SRC = ROOT / 'code' / 'src'
sys.path.insert(0, str(SRC))

from benchmarks.factor_graph_path import _write_benchmark_inputs
from build_factor_graph import main

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    base_path, hf_path, macro_path, output_path, manifest_path = _write_benchmark_inputs(tmp_path, num_stocks=8, num_days=30)
    argv = [
        '--pipeline-config-dir', './config',
        '--feature-set-version', 'vbench',
        '--base-input', str(base_path),
        '--hf-daily-input', str(hf_path),
        '--macro-input', str(macro_path),
        '--output', str(output_path),
        '--manifest-path', str(manifest_path),
        '--strict',
        '--run-id', 'profile-run',
    ]
    profiler = cProfile.Profile()
    profiler.enable()
    main(argv)
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime').print_stats(40)
PY
```

### Current Metrics

| Benchmark | Wall Time | Previous Local Observation | Delta |
| --- | ---: | ---: | ---: |
| `factor_graph_path` single run | `0.4902s` | `0.6000s` | `-0.1098s` |
| `factor_graph_path` 5-run mean | `0.4478s` | `0.6000s` | `-0.1522s` |
| `factor_graph_path` 5-run range | `0.4191s - 0.4792s` | - | - |

### Process-Level Profile Snapshot

```text
build_factor_graph.main(...)          0.197s
_compute_expression_factors(...)      0.057s
_compute_macro_series_asof(...) x3    0.017s
load_pipeline_configs(...)            0.025s
```

- 这批优化命中的是真实业务热点，而不是参数解析或测试夹具；在补上 staleness 语义修复后，`factor_graph_path` 5 次观测仍稳定落在 `0.42s - 0.48s` 区间，说明性能收益没有被吞回去。
- 当前 `macro_asof_join` 的主要浪费已经从“逐节点扫描整张 macro 表”收敛为“小规模 `merge_asof` 本身的必要成本”，后续更值得转向 `expression` 执行路径寻找下一刀。
