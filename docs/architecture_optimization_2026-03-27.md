# THU-BDC2026 架构梳理与优化蓝图（2026-03-27）

## 1. 范围与目标

- 目标1：梳理现有项目端到端架构（数据 -> 因子 -> 训练 -> 推理 -> 打分）。
- 目标2：给出可落地的整体优化方案（质量、性能、可维护性、可观测性）。
- 目标3：结合相关论文给出“提升计算结果分数”的技术路线。

## 2. 现状架构（As-Is）

### 2.1 端到端流程

1. 数据接入：从 `data/train.csv`（及可选高频日聚合因子）加载。
2. 因子解析：从因子仓库解析 active factor pipeline 并快照。
3. 特征工程：按股票并行生成时序特征，按交易日生成截面特征。
4. 数据集构建：构建懒加载训练索引与滚动验证折。
5. 训练与选策：组合损失训练，按验证指标选择最优持仓策略并保存。
6. 推理：使用训练快照重放特征流程，输出全量分数与组合权重。
7. 评分：本地脚本按最后5个交易日开盘收益加权求分。

### 2.2 关键模块映射

- 数据治理与数据源清单：`code/src/data_manager.py`
  - `load_market_dataset`: 300
  - `merge_hf_daily_factors`: 162
  - `collect_data_sources`: 404
- 因子平台：`code/src/factor_store.py`
  - `resolve_factor_pipeline`: 249
  - `engineer_group_features`: 1321
  - `apply_factor_expressions`: 1267
  - `save/load_factor_snapshot`: 1664/1671
- 训练编排：`code/src/train.py`
  - `preprocess_data`: 575
  - `PortfolioOptimizationLoss`: 587
  - `build_strategy_candidates`: 817
  - `choose_best_strategy`: 928
  - `train_ranking_model`: 1348
  - `evaluate_ranking_folds`: 1591
  - `main`: 1941
- 模型：`code/src/model.py`
  - `StockTransformer`: 504
  - `forward`: 1002
- 推理：`code/src/predict.py`
  - `preprocess_predict_data`: 27
  - `load_prediction_strategy`: 105
  - `scores_to_portfolio`: 137
  - `main`: 237
- 增强特征与标准化：`code/src/utils.py`
  - `apply_cross_sectional_normalization`: 10
  - `_add_cross_sectional_rank_features`: 332
  - `augment_engineered_features`: 448

## 3. 差距分析（Gap Analysis）

### 3.1 已识别并已修复

1. 特征命名冲突风险（中文列清洗后同名覆盖）。
2. 梯度裁剪开关错误引用（实际可能未生效）。
3. 因子消融默认开启导致训练周期额外膨胀。

### 3.2 当前主要瓶颈

1. 训练吞吐偏低：单 epoch 训练耗时过长（与 batch 组装+模型复杂度耦合）。
2. 策略优化闭环较慢：需要完整训练后才能刷新 `best_strategy.json`。
3. 线上/离线一致性治理可进一步强化：特征版本、模型版本、策略版本应统一追踪。
4. 观测面不足：缺少“样本构建耗时、batch吞吐、GPU利用率、策略收益漂移”的统一看板。

## 4. 目标架构（To-Be）

### 4.1 逻辑分层

1. Data Layer：
- 数据源适配（行情、静态行业、市值、高频聚合因子）
- 数据质量校验（主键完整性、日期一致性、覆盖率）

2. Feature Layer：
- 因子注册中心 + 因子快照
- 统一特征命名与冲突规约
- 截面标准化与市场/行业中性处理

3. Model Layer：
- StockTransformer 主干
- 可插拔关系建模（行业先验图、相关性图、隐式概念图）
- 可插拔排序目标（Pairwise/Listwise/LambdaNDCG/IC）

4. Strategy Layer：
- 候选策略池（top_k × weighting × temp）
- 风险调整收益目标
- 策略版本化（`best_strategy.json`）

5. MLOps Layer：
- 数据清单、因子快照、模型、策略统一版本
- 训练/推理作业编排
- 可观测与告警

### 4.2 物理网络蓝图（推荐）

```
[Internet/行情源]
      |
      v
+--------------------+         +---------------------+
|  Ingest VPC (DMZ)  |  ---->  |  Object Storage     |
|  - Data Pull Jobs  |         |  raw/curated/factor |
+--------------------+         +---------------------+
                                      |
                                      v
                           +---------------------+
                           | Feature Compute VPC |
                           | - CPU Batch Cluster |
                           | - Factor Build Jobs |
                           +---------------------+
                                      |
                                      v
                           +---------------------+
                           | Training VPC (GPU)  |
                           | - Trainer           |
                           | - Experiment Track  |
                           +---------------------+
                                      |
                                      v
                           +---------------------+
                           | Inference VPC        |
                           | - Daily Batch Infer  |
                           | - Strategy Selector  |
                           +---------------------+
                                      |
                                      v
                           +---------------------+
                           | Result Store / API  |
                           +---------------------+
```

安全建议：VPC 间仅开放必要端口，模型与数据仓分离权限，产物仓采用只读推广流程。

## 5. 系统集成接口定义（Interface Contracts）

### 5.1 数据输入接口（批处理）

- 主键：`股票代码 + 日期`
- 必要字段：`开盘, 收盘, 最高, 最低, 成交量, 成交额, 换手率, 涨跌幅...`
- 校验：
  - `股票代码` 统一6位字符串
  - `日期` 可解析且交易日递增
  - 数值字段非字符串污染

### 5.2 因子快照接口

- 文件：`active_factors.json`
- 语义：训练/推理必须使用同一快照指纹（fingerprint）
- 失败策略：快照不存在时可回退当前因子配置，但需告警

### 5.3 模型与策略接口

- 模型文件：`best_model.pth`
- 策略文件：`best_strategy.json`
  - 字段：`name, top_k, weighting, temperature, validation_objective, rank_ic_mean...`

### 5.4 推理输出接口

- `output/prediction_scores.csv`: 全股票分数
- `output/result.csv`: 最终持仓组合（`stock_id, weight`）

## 6. 部署环境蓝图（Dev/Staging/Prod）

### 6.1 Dev

- 目标：快速迭代与局部回测
- 配置：小样本+短 epoch+高日志
- 产物：实验快照，不进入生产路径

### 6.2 Staging

- 目标：全量数据校验与策略回放
- 配置：全量特征、限制 epoch、固定随机种子
- 门禁：
  - 数据质量通过
  - 指标不退化（收益、RankIC、波动）

### 6.3 Prod

- 目标：稳定日批推理
- 推荐流程：
  1) T 日收盘后数据入湖
  2) 特征作业
  3) 推理作业
  4) 结果验收与归档
- 回滚：策略版本回滚优先，模型版本回滚次之

## 7. 分数优化路线图（与论文映射）

### P0（已执行）

1. 修复命名冲突与梯度裁剪开关。
2. 关闭默认因子消融，减少训练冗余。
3. 扩展策略候选到 `top2/top3/top5`。

### P1（1-2周）

1. 训练提速：
- 预构建/缓存 batch 索引
- AMP 混合精度
- 训练日志中加入 `samples/sec` 与 GPU 利用率

2. 策略快速重选：
- 将“策略重选”从完整训练流程中拆分为独立任务
- 固定模型仅重跑验证策略扫描

3. 关系建模增强：
- 结合行业先验图与收益相关图的 union/intersection 自适应选择

### P2（2-6周）

1. MASTER 思路：强化时间步级跨股交互与市场门控一致性。
2. HIST 思路：引入隐式概念图（动态聚类/图注意力）。
3. RSR 思路：引入显式关系编码（产业链/同题材）。
4. NeuralNDCG/ListNet/RankNet：构建可切换 listwise 目标并做稳定性对照。
5. DoubleAdapt：加入概念漂移自适应（滚动重权/元学习更新）。

## 8. 当前实验结论（2026-03-27）

1. 完整重训在当前实现下耗时较高，需先做训练吞吐优化再进行系统性超参搜索。
2. 在固定模型参数下，策略层快速扫描显示：`equal_top2` 优于当前 `equal_top3`。
3. 本地自评分示例：
- `equal_top3`: `-0.021844467653596496`
- `equal_top2`: `0.0023210178178508373`

> 注：这是策略层快速优化结论，正式上线应以“仅基于验证集”重选策略，避免未来信息泄漏。

## 9. 参考论文与文章

1. MASTER: Market-Guided Stock Transformer for Stock Price Forecasting and Ranking
   - https://arxiv.org/abs/2312.15235
2. RSR: Relational Stock Ranking
   - https://arxiv.org/abs/1809.09441
3. HIST: A Hidden Concept based Framework for Stock Trend Forecasting
   - https://arxiv.org/abs/2110.13716
4. ADB-TRM (IJCAI 2024)
   - https://www.ijcai.org/proceedings/2024/0221.pdf
5. DoubleAdapt
   - https://arxiv.org/abs/2306.09862
6. NeuralNDCG
   - https://arxiv.org/abs/2102.07831
7. TFT
   - https://arxiv.org/abs/1912.09363
8. ListNet
   - https://www.cs.nccu.edu.tw/~mftsai/papers/icml2007_tsai.pdf
9. RankNet
   - https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf
