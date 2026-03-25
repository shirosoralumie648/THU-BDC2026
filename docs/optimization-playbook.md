# 优化手册（6 项增强）

本文档对应 `code/src/config.py` 中的优化参数，目标是提升跨时间段稳定性，减少“验证好、测试差”的情况。

## 1. 稳健策略选择（Risk-Adjusted）

位置：`train.py -> calculate_ranking_metrics / choose_best_strategy`

- 每个候选策略都会输出：
  - `return_<strategy>`（均值）
  - `return_<strategy>_std`（波动）
  - `return_<strategy>_risk_adjusted = mean - λ * std`
- 选择逻辑由 `strategy_selection_mode` 控制：
  - `risk_adjusted`：默认，优先稳定性；
  - `return`：只看均值收益。

建议：
- 当出现 top1 高频翻车时，优先使用 `risk_adjusted`；
- `strategy_risk_lambda` 推荐从 `0.1 ~ 0.4` 网格搜索。

## 2. 按日截面标准化（Feature CS Norm）

位置：`utils.py -> apply_cross_sectional_normalization`

- 训练与推理都会执行，保证口径一致；
- 方法：
  - `zscore`: `(x - mean_t) / std_t`
  - `rank`: 将日内分位映射到 `[-1, 1]`

建议：
- 数据波动大时先用 `zscore`；
- 若异常值很多，可使用 `rank` 或设置 `feature_cs_clip_value`。

## 3. 标签极值处理 + 截面标签标准化

位置：`train.py -> transform_targets_for_loss`

- 仅用于损失训练目标，不改变评估时真实收益口径；
- 支持：
  - `label_extreme_mode`: `none/drop/clip/drop_clip`
  - `label_cs_norm_method`: `zscore/rank`

建议：
- 初始参数：`clip + zscore`；
- 如果仍不稳定，再尝试 `drop_clip`。

## 4. 混合排序损失

位置：`train.py -> PortfolioOptimizationLoss`

组合：
- ListNet：全局分布对齐；
- Pairwise RankNet：强调胜负 pair；
- LambdaNDCG：强调前排排序质量。

主要参数：
- `listnet_weight`
- `pairwise_weight`
- `lambda_ndcg_weight`
- `lambda_ndcg_topk`
- `pairwise_top_fraction`

建议：
- 先固定 `listnet_weight=1`；
- 再调 `lambda_ndcg_weight`（常见 `0.5~1.5`）。

## 5. 市场状态引导门控（Market Gating）

位置：`model.py -> StockTransformer.forward`

- 用日内“全市场均值 + 波动”得到 market state；
- 通过 gate 网络输出每个特征维的动态权重；
- 采用 residual 门控避免过抑制。

参数：
- `use_market_gating`
- `market_gate_hidden_dim`
- `market_gate_residual`

建议：
- `market_gate_residual` 不要太小（建议 `0.4~0.7`）。

## 6. RankIC 早停

位置：`train.py -> main`

- 新增指标：
  - `rank_ic_mean`
  - `rank_ic_std`
  - `rank_ic_ir`
- 默认以 `rank_ic_mean` 作为 early stopping monitor。

参数：
- `early_stopping_enabled`
- `early_stopping_patience`
- `early_stopping_min_delta`
- `early_stopping_monitor`
- `early_stopping_mode`

建议：
- 在滚动验证下，`patience=6~12` 较常用；
- 若曲线抖动大可适当调大 `min_delta`。

## 训练建议流程

1. 跑 baseline 保存结果；
2. 逐项打开优化做 ablation；
3. 选出稳定参数后再全量训练；
4. 用 `predict.py + test/score_self.py` 验证最终分数；
5. 再进行 Docker 打包验证。

## 参考方向（论文关键词）

- MASTER（市场引导 + 极值处理 + 截面标准化）
- LambdaLoss / LambdaRank（NDCG 驱动排序损失）
- RSR / RankIC 早停（排序任务与投资目标一致性）
- TFT（动态门控特征选择思想）
