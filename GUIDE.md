# 环境准备

推荐使用 `uv` 管理环境。

```bash
uv sync
source .venv/bin/activate
```

Windows:

```powershell
uv sync
.\.venv\Scripts\activate
```

# 当前默认流程

当前推荐的默认链路不是手工改源码，而是：

```text
validate-pipeline-config -> build-factor-graph -> build-dataset -> train -> predict
```

对应命令如下：

```bash
./.venv/bin/python code/src/manage_data.py validate-pipeline-config --config-dir ./config
./.venv/bin/python code/src/manage_data.py build-factor-graph \
  --pipeline-config-dir ./config \
  --feature-set-version v1 \
  --base-input ./data/stock_data.csv
./.venv/bin/python code/src/manage_data.py build-dataset \
  --pipeline-config-dir ./config \
  --feature-set-version v1 \
  --base-input ./data/stock_data.csv \
  --feature-input ./data/datasets/features/train_features_v1.csv \
  --output-dir ./data
sh train.sh
sh test.sh
```

Windows 下训练和预测可直接执行：

```powershell
python code/src/train.py
python code/src/predict.py
```

说明：

- `build-factor-graph` 默认会把宽表因子写到 `./data/datasets/features/train_features_v1.csv`。
- `build-dataset` 会生成 dataset build manifest；`train.py` 和 `predict.py` 会优先读取 manifest 指向的数据文件。
- 最终预测结果仍写到 `output/result.csv`。
- `train.sh` / `test.sh` 会优先使用仓库内 `.venv/bin/python`；如需覆盖解释器，可设置 `THU_BDC_PYTHON_BIN=/path/to/python`。

# 数据准备

仓库已经兼容 legacy 路径：

- `data/stock_data.csv`
- `data/train.csv`
- `data/test.csv`

如果这些文件已经准备好，可以直接走默认流程。

如果你需要重新抓取日线原始数据，使用当前的兼容入口：

```bash
python get_stock_data.py \
  --pipeline-config-dir ./config \
  --dataset-name market_bar_1d \
  --start-date 2015-01-01 \
  --end-date 2026-03-31 \
  --runtime-root ./temp/ingestion_runtime
```

说明：

- 这个命令当前通过 ingestion service 调 BaoStock provider。
- 该入口会优先走 compat bridge，并把 canonical 日线结果导出为 legacy `stock_data.csv`；如需显式指定输出位置，可传 `--output-path` 与 `--manifest-path`。
- `runtime-root` 用于保存底层 ingestion runtime、source manifest 与中间产物，不会直接覆写 `data/train.csv` / `data/test.csv`。
- 如果只是复现 baseline，优先使用仓库现有数据文件。

# 自评与验收

本地训练和预测完成后，执行：

```bash
./.venv/bin/python test/score_self.py
```

自评脚本会读取 `output/result.csv`，并把参考结果写到 `temp/tmp.csv`。

如果要执行当前 release gate，请以 [docs/release-gate-2026-04.md](/home/shirosora/code_storage/THU-BDC2026/.worktrees/stage15-closeout/docs/release-gate-2026-04.md) 为唯一 canonical 命令清单。

关键变化：

- `validate-pipeline-config` 使用 `--profile release`
- `build-dataset` 使用 `--profile release`
- benchmark smoke pack 已纳入 release gate
- CPU-only 环境执行 release gate 时，应按 canonical 命令显式设置 `THU_BDC_CONFIG_OVERRIDE_PATH=./config/runtime/release_cpu_smoke.json` 后再运行 `train.sh` / `test.sh`；该 override 会压缩 epoch/batch 和模型宽度
- GPU 环境如需完整训练，可不使用该 override

Release checklist：

- 全量单测通过。
- ingestion adapters 输出 canonical columns。
- build manifests 含 `factor_fingerprint`。
- `train.py` 与 `predict.py` 使用 manifest-aware 路径解析。
- `output/result.csv` 仍然是最终提交工件。

# Docker 验证与打包

构建镜像：

```bash
docker buildx build --platform linux/amd64 --load -t bdc2026:latest .
```

当前 `docker compose up` 的语义是“用镜像验证预测链路”：

- compose 使用本地镜像 `bdc2026:latest`
- 容器执行 `/app/data/run.sh`
- `run.sh` 会先调用 `init.sh`，再执行 `test.sh`
- 因此镜像里必须已经包含训练好的模型产物

导出镜像：

```bash
docker save -o 队伍名称.tar bdc2026:latest
```

如果要模拟赛事方批量评分：

```bash
python test/test.py
```

Windows:

```powershell
python test/test_windows.py
```

# 已知的可选依赖与外部前提

这些步骤依赖外部服务或额外本地环境：

- `uv sync` 在冷启动机器上需要联网下载依赖。
- `docker buildx build` 与首次 `docker compose up` 可能需要联网拉取镜像。
- `python get_stock_data.py ...` 和 `python code/src/manage_data.py ingest ...` 依赖 BaoStock / Akshare 在线可用。
- 当前仓库没有统一的 provider 自适应限流层；若 provider 限频、超时或空返回，应降低请求频率后重试。
- `docker-compose.yml` 当前按 GPU 验证路径配置，默认要求 Docker daemon 可用且本机安装 NVIDIA Container Toolkit。
