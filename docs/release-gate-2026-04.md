# Release Gate 2026-04

本文件是当前仓库唯一的 release validation bundle 说明。`README.md` 与 `GUIDE.md` 只保留摘要和入口，不再重复维护完整命令清单。

## Profile 约定

- `dev`：允许非关键路径回退，适合日常开发与局部排障。
- `release`：对 pipeline 配置 warning 和 dataset manifest 谱系约束执行 fail-fast。

当前 release profile 已落地到两个关键入口：

- `manage_data.py validate-pipeline-config --profile release`
  - 配置 warning 也会返回非零。
- `manage_data.py build-dataset --profile release`
  - pipeline 校验 warning/error 直接失败；
  - dataset manifest 必须能通过严格校验；
  - `feature_set_version` 与 `factor_fingerprint` 必须保持可追踪。

说明：

- `build-factor-graph` 暂未默认切到严格 profile。
- 原因是当前 baseline 允许缺少可选的 HF / macro 输入，并以 warning 记录；这部分会在后续更完整的数据流收敛后再收紧。
- 在 CPU-only 环境下，`train.sh` / `test.sh` 的 canonical release path 使用提交到仓库的 runtime override：
  - `./config/runtime/release_cpu_smoke.json`
  - 该 override 会收紧 epoch/batch 上限，并下调模型宽度，用于 release 验收烟测，不改变 `train.sh` / `test.sh` / `output/result.csv` 契约。
  - 该 override 使用独立模型目录 `./model/60_158+39_release_cpu_smoke`，避免和完整训练工件混用。
  - 若在 GPU 环境执行完整训练，可不设置该 override。

## Canonical Commands

按顺序执行：

```bash
./.venv/bin/python code/src/manage_data.py validate-pipeline-config --config-dir ./config --profile release
./.venv/bin/python -m unittest discover -s test -p 'test_*.py' -v
./.venv/bin/python code/src/manage_data.py build-factor-graph \
  --pipeline-config-dir ./config \
  --feature-set-version v1 \
  --base-input ./data/stock_data.csv
./.venv/bin/python code/src/manage_data.py build-dataset \
  --pipeline-config-dir ./config \
  --feature-set-version v1 \
  --base-input ./data/stock_data.csv \
  --feature-input ./data/datasets/features/train_features_v1.csv \
  --output-dir ./data \
  --profile release
./.venv/bin/python code/src/benchmarks/training_path.py
./.venv/bin/python code/src/benchmarks/factor_graph_path.py
./.venv/bin/python -c "import json, sys; sys.path.insert(0, 'code/src'); from benchmarks.training_path import run_dataset_build_benchmark; print(json.dumps(run_dataset_build_benchmark(), ensure_ascii=False, indent=2))"
./.venv/bin/python -c "import json, sys; sys.path.insert(0, 'code/src'); from benchmarks.training_path import run_prediction_path_benchmark; print(json.dumps(run_prediction_path_benchmark(), ensure_ascii=False, indent=2))"
rg -n "release-gate-2026-04.md|validate-pipeline-config|build-factor-graph|build-dataset|output/result.csv" README.md GUIDE.md
THU_BDC_CONFIG_OVERRIDE_PATH=./config/runtime/release_cpu_smoke.json sh train.sh
THU_BDC_CONFIG_OVERRIDE_PATH=./config/runtime/release_cpu_smoke.json sh test.sh
./.venv/bin/python test/score_self.py
```

## Pass Criteria

- `validate-pipeline-config --profile release` 返回 `0`
- 全量单测通过
- `build-factor-graph` 与 `build-dataset --profile release` 无 traceback
- `data/data_manifest_dataset_build.json` 包含 `feature_set_version` 与 `factor_fingerprint`
- benchmark smoke 命令均输出结构化指标
- 文档检索结果指向本文件作为 canonical release gate
- 最终预测产物仍为 `output/result.csv`

## Benchmark Policy

当前采用以下初始容忍带：

- `training_path` 不得比最新接受基线退化超过 `15%`
- `factor_graph_path` 不得退化超过 `15%`
- `dataset_build_path` 不得退化超过 `20%`
- `prediction_path` 不得退化超过 `15%`

基线参考：[docs/performance-baseline-2026-04-04.md](/home/shirosora/code_storage/THU-BDC2026/.worktrees/stage15-closeout/docs/performance-baseline-2026-04-04.md)
