# THU-BigDataCompetition-2026-baseline

本项目是一个面向沪深300成分股的**排序学习选股**方案：
- 输入：每只股票过去一段时间（默认60个交易日）的量价与技术特征序列；
- 模型：`StockTransformer`，同时建模单股票时序模式与股票间交互；
- 输出：对同一天全部候选股票打分并排序，最终输出前5只股票（等权重0.2）。

---

## 1. 项目目标与整体流程

核心目标是学习“当天应优先持有哪些股票”的排序函数，而不是单只股票二分类。

训练与推理主流程如下：
1. 读取历史行情数据（`data/hs300_history_2015_2026.csv`）；
2. 做特征工程（39特征或`158+39`特征）；
3. 构建标签：未来收益率（代码中为 `open_t1` 到 `close_t5` 的相对收益）；
4. 按“日期”组织排序样本：每个样本是一日内多只股票的序列与目标；
5. 训练排序模型，监控 `final_score` 并保存最优权重；
6. 使用训练好的 `best_model.pth` + `scaler.pkl` 在最新日期上生成Top5选股结果。

---

## 2. 代码结构说明

### [config.py](config.py)
统一管理训练与推理参数，包括：
- 序列长度 `sequence_length`（默认60）；
- 模型超参数（`d_model`、`nhead`、`num_layers` 等）；
- 训练超参数（`batch_size`、`num_epochs`、`learning_rate`）；
- 排序损失权重参数（`pairwise_weight`、`top5_weight`、`base_weight`）；
- 数据路径和输出路径（默认输出到 `output/60_158+39/`）。

### [model.py](model.py)
定义核心模型 `StockTransformer`，主要由以下模块组成：
- `PositionalEncoding`：时序位置编码；
- 时序编码器 `TransformerEncoder`：提取单股票历史序列表示；
- `FeatureAttention`：对时间维特征做注意力聚合；
- `CrossStockAttention`：在同一交易日内建模股票间关系；
- `ranking_layers` + `score_head`：输出每只股票的排序分数。

输入形状：`[batch, num_stocks, seq_len, feature_dim]`  
输出形状：`[batch, num_stocks]`。

### [utils.py](utils.py)
包含特征工程与数据集构建逻辑：
- `engineer_features_39()`：39个技术指标特征；
- `engineer_features()`：158个Alpha类特征；
- `engineer_features_158plus39()`：合并 `158 + 39` 特征；
- `create_ranking_dataset_vectorized()`：向量化构建按日排序样本（训练核心加速点）。

说明：特征工程使用了 `TA-Lib`，若未正确安装会报错。

### [train.py](train.py)
训练主脚本，关键内容：
- 数据预处理：
	- `_preprocess_common()`：按股票分组并行特征工程、股票ID映射、标签构建；
	- `split_train_val_by_last_month()`：按最后阶段数据切分训练/验证集，并保留序列上下文。
- 数据集组织：
	- `RankingDataset` + `collate_fn`：处理每日股票数量不一致问题（padding + mask）。
- 损失函数：`WeightedRankingLoss`
	- 组合了 `listwise_loss` 与 `pairwise_loss`；
	- 对真实Top-k样本施加更高权重。
- 评估指标：`calculate_ranking_metrics()`
	- 计算 `pred_return_sum`、`max_return_sum`、`ratio_pred`、`final_score` 等；
	- 训练过程中以验证集 `final_score` 选择最优模型。

训练产物：
- `best_model.pth`：最佳模型参数；
- `scaler.pkl`：标准化器；
- `config.json`：训练时配置快照；
- `final_score.txt`：最佳分数记录；
- `log/`：TensorBoard日志。

### [predict.py](predict.py)
推理主脚本，流程：
1. 加载历史数据，取最新交易日；
2. 执行与训练一致的特征工程；
3. 加载 `scaler.pkl` 进行特征标准化；
4. 用 `best_model.pth` 对全部可预测股票打分；
5. 按分数降序取前5只，输出到 `output.csv`：
	 - `stock_id`
	 - `weight`（固定 `0.2`）

### [get_stock_data.py](get_stock_data.py)
数据抓取脚本（Baostock）：
- 获取沪深300成分股；
- 抓取历史日线数据并保存为训练所需格式。

---

## 3. 数据与输入输出约定

默认训练数据文件：
- `data/hs300_history_2015_2026.csv`

关键列：
- `股票代码`、`日期`、`开盘`、`收盘`、`最高`、`最低`、`成交量`、`成交额`、`换手率`、`涨跌幅` 等。

预测输出文件：
- 根目录下 `output.csv`（由 `predict.py` 生成）。

---

## 4. 运行方法（推荐使用 uv）

按你要求的推荐方式如下：

1) 使用 `uv` 安装依赖

`uv sync`

2) 激活虚拟环境

`source .venv/bin/activate`

3) 训练模型

```
sh train.sh
```

4) 生成预测结果

```
sh test.sh
```

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
