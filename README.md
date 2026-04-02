# THU-BigDataCompetition-2026-baseline

本项目是一个面向沪深300成分股的**排序学习选股**方案：
- 输入：每只股票过去一段时间（默认60个交易日）的量价与技术特征序列；
- 模型：`StockTransformer`，同时建模单股票时序模式与股票间交互；
- 输出：对同一天全部候选股票打分并排序，按 `best_strategy.json` 输出最多5只股票及权重（`equal` 或 `softmax`）。

---

## 1. 项目目标与整体流程

核心目标是学习“当天应优先持有哪些股票”的排序函数，而不是单只股票二分类。

训练与推理主流程如下：
1. 读取历史行情数据（`data/stock_data.csv`）；
2. 做特征工程（39特征或`158+39`特征）；
3. 构建标签：未来收益率（代码中为 `open_t1` 到 `open_t5` 的相对收益）；
4. 按“日期”组织排序样本：每个样本是一日内多只股票的序列与目标；
5. 训练排序模型，监控 `final_score` 并保存最优权重；
6. 使用训练好的 `best_model.pth` + `scaler.pkl` + `best_strategy.json` 在最新日期上生成持仓结果。

---

## 2. 代码结构说明

### [config.py](code/src/config.py)
统一管理训练与推理参数，包括：
- 序列长度 `sequence_length`（默认60）；
- 模型超参数（`d_model`、`nhead`、`num_layers` 等）；
- 训练超参数（`batch_size`、`num_epochs`、`learning_rate`）；
- 排序损失参数（`listnet_weight`、`pairwise_weight`、`lambda_ndcg_weight` 等）；
- 策略选择参数（`strategy_selection_mode`、`strategy_risk_lambda`、`prediction_top_k_candidates`）；
- 稳定性参数（特征/标签截面标准化、标签极值处理、RankIC早停）；
- 数据路径和输出路径（默认输出到 `output/`）。

### [model.py](code/src/model.py)
定义核心模型 `StockTransformer`，主要由以下模块组成：
- `PositionalEncoding`：时序位置编码；
- 时序编码器 `TransformerEncoder`：提取单股票历史序列表示；
- `FeatureAttention`：对时间维特征做注意力聚合；
- `CrossStockAttention`：在同一交易日内建模股票间关系；
- `ranking_layers` + `score_head`：输出每只股票的排序分数。

输入形状：`[batch, num_stocks, seq_len, feature_dim]`  
输出形状：`[batch, num_stocks]`。

### [data_manager.py](code/src/data_manager.py)
统一的数据来源管理模块，提供：
- 数据文件路径解析（`train.csv` / `test.csv` / `stock_data.csv`）；
- 行业映射解析与标准化（兼容列名自动推断）；
- 股票到行业索引映射构建（供行业虚拟股与先验图使用）；
- 数据源清单导出（`data_manifest*.json`，便于复现与排障）。

### [manage_data.py](code/src/manage_data.py)
统一数据管理 CLI：
- `manifest`：导出当前配置下的数据源清单；
- `validate`：按模式校验关键文件是否存在；
- `industry-index`：按输入股票池构建行业索引与词表。

### [gui_app.py](code/src/gui_app.py)
可视化控制台入口（Streamlit）：
- 数据中心（抓取/切分/预览）；
- 因子实验室（管理/编辑/分布分析）；
- 参数微调（配置覆盖文件）；
- 训练监控、回测预测、部署验证。

### [utils.py](code/src/utils.py)
包含特征工程与数据集构建逻辑：
- `engineer_features_39()`：39个技术指标特征；
- `engineer_features()`：158个Alpha类特征；
- `engineer_features_158plus39()`：合并 `158 + 39` 特征；
- `create_ranking_dataset_vectorized()`：向量化构建按日排序样本（训练核心加速点）。

说明：特征工程使用了 `TA-Lib`，若未正确安装会报错。

### [train.py](code/src/train.py)
训练主脚本，关键内容：
- 数据预处理：
	- `_preprocess_common()`：按股票分组并行特征工程、股票ID映射、标签构建；
	- `split_train_val_by_last_month()`：按最后阶段数据切分训练/验证集，并保留序列上下文。
- 数据集组织：
	- `RankingDataset` + `collate_fn`：处理每日股票数量不一致问题（padding + mask）。
- 损失函数：`PortfolioOptimizationLoss`
	- 组合 `ListNet + Pairwise RankNet + LambdaNDCG`；
	- 支持标签极值处理与标签截面标准化（仅用于训练 loss）。
- 评估指标：`calculate_ranking_metrics()`
	- 计算各策略 `mean/std/risk_adjusted return`；
	- 计算 `rank_ic_mean/rank_ic_ir`；
	- 训练过程中按配置选择最优策略并支持 RankIC 早停。

训练产物：
- `best_model.pth`：最佳模型参数；
- `scaler.pkl`：标准化器；
- `config.json`：训练时配置快照；
- `data_manifest.json`：训练使用的数据源清单；
- `active_factors.json`：本次训练实际启用的因子快照；
- `final_score.txt`：最佳分数记录；
- `log/`：TensorBoard日志。

### [predict.py](code/src/predict.py)
推理主脚本，流程：
1. 加载历史数据，取最新交易日；
2. 执行与训练一致的特征工程（含截面标准化）；
3. 加载 `scaler.pkl` 进行特征标准化；
4. 用 `best_model.pth` 对全部可预测股票打分；
5. 按 `best_strategy.json` 生成持仓并输出到 `output/result.csv`：
	 - `stock_id`
	 - `weight`

说明：
- 若模型目录下存在 `active_factors.json`，`predict.py` 会优先使用该快照中的因子配置；
- 这样即使之后你修改了 `config/factor_store.json`，旧模型推理仍会保持训练时的特征口径。
- 预测阶段会额外写出 `data_manifest_predict.json`，记录当前推理的数据来源。

### [get_stock_data.py](get_stock_data.py)
数据抓取脚本（Baostock）：
- 获取沪深300成分股；
- 抓取历史日线数据并保存为训练所需格式；
- 支持成分股快照日期、复权方式、重试/节流参数；
- 支持增量补齐与全量重建；
- 运行结束后自动输出 `data_manifest_stock_fetch.json`。

### [split_train_test.py](data/split_train_test.py)
数据切分脚本：
- 从 `stock_data.csv` 按日期区间切分 `train.csv` 和 `test.csv`；
- 运行结束后自动输出 `data_manifest_split.json`。

---

## 3. 数据与输入输出约定

默认训练数据文件：
- `data/train.csv`

结构化数据布局（可选，兼容旧路径）：
- `data/datasets/raw/stock_data.csv`
- `data/datasets/splits/train.csv`
- `data/datasets/splits/test.csv`

说明：
- 读取时会按候选路径自动回退（优先存在的文件）；
- 可通过 `config.py` 控制 `prefer_structured_data_layout`；
- 默认会镜像写入 legacy 与 structured 路径，避免旧脚本失效。

关键列：
- `股票代码`、`日期`、`开盘`、`收盘`、`最高`、`最低`、`成交量`、`成交额`、`换手率`、`涨跌幅` 等。

预测输出文件：
- output目录下 `result.csv`（由 `predict.py` 生成）。

---

## 4. 运行方法（推荐使用 uv）

按你要求的推荐方式如下：

1) 使用 `uv` 安装依赖

`uv sync`

2) 激活虚拟环境

`source .venv/bin/activate`

## 推荐执行流程（当前默认）

1) 校验 pipeline 配置

```bash
python code/src/manage_data.py validate-pipeline-config --config-dir ./config
```

2) 构建宽表因子

```bash
python code/src/manage_data.py build-factor-graph \
  --pipeline-config-dir ./config \
  --feature-set-version v1
```

3) 构建训练/测试集

```bash
python code/src/manage_data.py build-dataset \
  --pipeline-config-dir ./config \
  --feature-set-version v1
```

4) 训练模型

```bash
sh train.sh
```

5) 生成预测结果

```bash
sh test.sh
```

说明：legacy `data/train.csv` / `data/test.csv` 仍保持兼容；当前优先推荐路径是 `pipeline config -> factor graph -> dataset build -> train/predict`。

6) 打开 TensorBoard 查看训练过程与因子面板

```
sh tensorboard.sh
```

默认地址：

`http://127.0.0.1:6006`

如果想看其他模型目录：

```
sh tensorboard.sh ./model/60_158+39/log
```

6) 数据管理（可选）

生成数据源清单：

```
python code/src/manage_data.py manifest --include-csv-stats
```

校验训练数据是否齐备：

```
python code/src/manage_data.py validate --mode train
```

按训练集股票池构建行业索引：

```
python code/src/manage_data.py industry-index
```

抓取 HS300 行情（示例）：

```bash
python get_stock_data.py \
  --start-date 2015-01-01 \
  --end-date 2026-03-20 \
  --index-date 2026-03-20 \
  --adjustflag 1 \
  --max-retries 3
```

高频数据聚合为日因子（示例）：

```bash
python code/src/build_hf_daily_factors.py \
  --input ./data/hf_minute.csv \
  --output ./data/hf_daily_factors.csv \
  --tail-minutes 30 \
  --min-bars 10
```

多源输入（重复 `--input` 或一次逗号分隔）：

```bash
python code/src/build_hf_daily_factors.py \
  --input ./data/hf_part1.csv \
  --input ./data/hf_part2.csv \
  --output ./data/hf_daily_factors.csv
```

按 glob 批量读取：

```bash
python code/src/build_hf_daily_factors.py \
  --input-glob "./data/hf_*.csv" \
  --output ./data/hf_daily_factors.csv
```

生成多版本（日内原频 + 重采样）因子并自动加后缀：

```bash
python code/src/build_hf_daily_factors.py \
  --input-glob "./data/hf_*.csv" \
  --resample-minutes 5,15,30 \
  --min-bars 5 \
  --output ./data/hf_daily_factors.csv
```

仅保留重采样版本（不输出 raw）：

```bash
python code/src/build_hf_daily_factors.py \
  --input-glob "./data/hf_*.csv" \
  --resample-minutes 5,15 \
  --skip-raw \
  --min-bars 3 \
  --output ./data/hf_daily_factors.csv
```

说明：
- 脚本会自动推断 `股票代码/时间/价格/成交量/成交额` 列名，也可通过 `--stock-col` 等参数显式指定；
- 每次生成会写清单到 `<output_dir>/data_manifest_hf_daily_factors.json`；
- 使用重采样时，单日 bar 数会减少，通常需要同步调低 `--min-bars`（默认 10）。

启用高频日因子自动合并（通过 runtime override）：

```json
{
  "use_hf_daily_factor_merge": true,
  "hf_daily_factor_path": "./data/hf_daily_factors.csv",
  "hf_factor_prefix": "hf_",
  "hf_factor_required": true
}
```

保存为例如 `./temp/config_override_hf.json` 后运行：

```bash
THU_BDC_CONFIG_OVERRIDE_PATH=./temp/config_override_hf.json sh train.sh
THU_BDC_CONFIG_OVERRIDE_PATH=./temp/config_override_hf.json sh test.sh
```

说明：
- 高频日因子按 `股票代码 + 日期` 合并到 `train.csv` / `test.csv` 读取结果；
- 若 `hf_factor_required=true` 且文件缺失，训练/预测会直接报错；
- `manage_data validate --mode train` 会同时检查高频因子文件是否存在（当 required=true）。

7) 启动 GUI 控制台

```
streamlit run code/src/gui_app.py
```

8) 管理因子

列出当前因子：

```
sh factor.sh list
```

只看启用中的因子：

```
sh factor.sh list --enabled-only
```

关闭单个内置因子：

```
sh factor.sh disable RSQR60
```

重新启用：

```
sh factor.sh enable RSQR60
```

新建自定义因子：

```
sh factor.sh create gap_open_prev_close \
  --expression "(开盘 / (shift(收盘, 1) + 1e-12)) - 1" \
  --group custom
```

编辑自定义因子：

```
sh factor.sh update gap_open_prev_close \
  --expression "(开盘 / (shift(收盘, 1) + 1e-12)) - 1"
```

带输入映射（Aliasing）创建模板化因子：

```
sh factor.sh create my_custom_sma \
  --expression "sma(input_price, window)" \
  --inputs '{"input_price":"收盘","window":20}' \
  --author "alice"
```

说明：
- 因子表达式会自动做依赖解析和 DAG 拓扑排序，用户无需手动排列先后顺序；
- 支持截面函数 `cs_rank()` / `cs_zscore()`，会在全市场按日期统一计算；
- 表达式执行采用 AST 白名单校验（替代裸 `eval`）。

查看因子详情：

```
sh factor.sh show gap_open_prev_close
```

删除自定义因子：

```
sh factor.sh delete gap_open_prev_close
```

批量启用/关闭：

```
sh factor.sh enable-many RSQR60 ROC20 return_20
sh factor.sh disable-many VSUMD60 WVMA60
```

按分组启用/关闭：

```
sh factor.sh list-groups
sh factor.sh disable-group alpha_volume_behavior
sh factor.sh enable-group alpha_volume_behavior
```

仅保留指定因子（其他全部关闭）：

```
sh factor.sh activate-only --names "开盘,收盘,ROC20,RSQR20,return_20"
```

因子配置文件默认在：

`config/factor_store.json`

训练/预测的因子结果落盘目录：

- `model/<run>/factor_artifacts/train_factor_values.csv`
- `model/<run>/factor_artifacts/val_factor_values.csv`
- `model/<run>/factor_artifacts/predict_latest_factor_values.csv`
- 以及对应 `*_factor_stats.csv`、`*_factor_meta.json`

内置因子默认公式注册表在：

`config/builtin_factors.json`

恢复某个内置因子的默认公式：

```
sh factor.sh reset RSQR60
```

训练时 TensorBoard 会额外记录：
- 当前启用因子列表；
- 内置/自定义因子数量；
- 被 override 的内置因子数量；
- 因子分组统计；
- 前若干个因子的原始分布与标准化后分布直方图。
- 因子分组消融结果（去掉某一组因子后，验证收益变化多少）。

---

## 5. 常见问题

1) `TA-Lib` 安装失败  
本项目特征工程依赖 `TA-Lib`，需要先安装系统层面的 `ta-lib` 库，再安装Python包。
```
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
```

2) 多进程相关问题  
`train.py` 与 `predict.py` 均在入口使用了 `spawn` 模式，Linux/macOS下请保持通过脚本入口运行（不要在交互式环境里直接多进程调用主逻辑）。

3) GPU/CPU自动选择  
代码会按 `CUDA -> MPS -> CPU` 顺序自动选择设备；无GPU时可直接CPU运行。

---

## 6. 优化版（2026-03）新增能力

为了降低“验证集看起来好、测试集掉分”的问题，训练与推理新增了 6 个优化点：

1) **稳健策略选择（Risk-Adjusted）**  
- 对每个候选持仓策略同时统计 `mean/std`；
- 策略评分使用 `mean - λ * std`（`strategy_risk_lambda`），默认优先稳健组合。

2) **按日截面标准化（Features）**  
- 训练与推理都支持按日期对特征做截面标准化（`zscore` 或 `rank`）；
- 显著缓解跨时段分布漂移。

3) **标签极值处理 + 截面标签标准化（Loss Target）**  
- 训练损失前可做标签 `drop/clip`；
- 再做标签截面归一化（`zscore/rank`），提升训练稳定性与泛化。

4) **混合排序损失（ListNet + Pairwise + LambdaNDCG）**  
- 在原有 listwise + pairwise 基础上加入 LambdaNDCG；
- 更直接约束 Top-K 排序质量。

5) **市场状态引导门控（Market Gating）**  
- 在模型输入层增加“市场均值+波动”驱动的特征门控；
- 动态调整不同市场状态下的因子贡献。

6) **RankIC 早停**  
- 验证集新增 `rank_ic_mean/rank_ic_ir`；
- 默认使用 `rank_ic_mean` 做 early stopping，避免后期过拟合。

---

## 7. 核心配置项（优化版）

位置：`code/src/config.py`

- 策略选择
  - `strategy_selection_mode`: `risk_adjusted | return`
  - `strategy_risk_lambda`
  - `prediction_top_k_candidates`

- 特征截面标准化
  - `use_cross_sectional_feature_norm`
  - `feature_cs_norm_method`: `zscore | rank`
  - `feature_cs_clip_value`

- 标签处理（仅用于训练损失）
  - `label_extreme_mode`: `none | drop | clip | drop_clip`
  - `label_extreme_lower_quantile`
  - `label_extreme_upper_quantile`
  - `use_cross_sectional_label_norm`
  - `label_cs_norm_method`: `zscore | rank`
  - `label_cs_clip_value`

- 损失函数
  - `listnet_weight`
  - `pairwise_weight`
  - `lambda_ndcg_weight`
  - `lambda_ndcg_topk`
  - `pairwise_top_fraction`

- 模型门控
  - `use_market_gating`
  - `market_gate_hidden_dim`
  - `market_gate_residual`

- RankIC 早停
  - `early_stopping_enabled`
  - `early_stopping_patience`
  - `early_stopping_min_delta`
  - `early_stopping_monitor`
  - `early_stopping_mode`

---

## 8. 复现实验建议

1) 固定随机种子，先跑一版 baseline（旧参数）。
2) 只开启一个优化项做 ablation，对比 `final_score + rank_ic_mean`。
3) 最后开启全部优化项，观察：
   - 验证集：`rank_ic_mean` 和策略 `risk_adjusted return`；
   - 本地测试：`test/score_self.py` 得分是否回升且波动降低。
