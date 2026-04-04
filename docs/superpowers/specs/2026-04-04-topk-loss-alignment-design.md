# Top-K Loss Alignment Design

日期：2026-04-04

## 1. 背景

当前项目的真实目标不是通用排序精度，而是竞赛口径下的 `Final Score`：

- 每次提交最多选择 5 只股票。
- 需要输出组合权重。
- 本地分数按最后 5 个交易日开盘收益率加权求和。

现有训练目标位于 [train.py](/home/shirosora/code_storage/THU-BDC2026/code/src/train.py) 的 `PortfolioOptimizationLoss`，由 `ListNet + Pairwise + LambdaNDCG (+ optional IC)` 组成。该目标已经明显优于传统回归式训练，但仍然存在目标失配：

1. loss 仍对整段排序更敏感，而最终收益只强依赖极少数 `top-k` 样本。
2. 当前 `top_k` 约束主要在策略选择和推理阶段体现，训练阶段没有足够直接地感知“前排错误代价更高”。
3. 若直接重写成完全新的组合优化器，改动面太大，难以快速验证收益。

因此，本次设计选择最小侵入式改造：保留现有 loss 骨架，只增加一个可控的 `top-k` 导向分量。

## 2. 目标与非目标

### 2.1 目标

- 让训练 loss 对前排股票更敏感，优先提升 `top2/top3/top5` 排序质量。
- 保持与现有训练流程、策略选择逻辑和模型结构兼容。
- 通过配置开关控制新分量强度，支持直接做 ablation。
- 为后续 `LambdaLoss` 或 reranking 扩展保留接口空间。

### 2.2 非目标

- 本轮不改 [model.py](/home/shirosora/code_storage/THU-BDC2026/code/src/model.py) 主干结构。
- 本轮不改 [test/score_self.py](/home/shirosora/code_storage/THU-BDC2026/test/score_self.py) 的最终评分口径。
- 本轮不引入完整的 end-to-end portfolio optimization。
- 本轮不调整 `best_strategy.json` 的导出协议。

## 3. 方案选型

### 3.1 备选方案

方案 A：直接把现有 loss 替换为 LambdaLoss  
优点：理论上更贴近 metric-driven ranking。  
缺点：需要重做当前 loss 组合与数值稳定性验证，首轮改造风险高。

方案 B：保留现有 loss，增加 top-k focus 分量  
优点：最小改动、便于灰度验证、兼容当前配置和训练代码。  
缺点：仍属于启发式对齐，不是纯粹的 metric-driven 最优形式。

方案 C：直接引入可微 top-k / 组合层  
优点：目标对齐程度最高。  
缺点：工程复杂度高，改动范围会波及训练、推理和策略层。

### 3.2 选定方案

采用方案 B。

理由：

- 当前仓库已经有可用的 `ListNet + Pairwise + LambdaNDCG` 基础。
- 新分量可以直接建立在现有 mask、padding 和 batch 结构之上。
- 若效果不佳，可通过配置权重快速回退到当前行为。

## 4. 设计概述

## 4.1 新增能力

在 `PortfolioOptimizationLoss` 中增加一个 `top-k focus` 分量：

- 基于真实标签在每个横截面中识别“应位于前排”的股票。
- 对这些股票相关的排序误差施加更高权重。
- 该分量不改变输出接口，只作为总 loss 的附加项。

总目标形式保持为：

`total_loss = base_loss + topk_focus_weight * topk_focus_loss`

其中：

- `base_loss` 继续为现有 `ListNet + Pairwise + LambdaNDCG (+ IC)` 组合。
- `topk_focus_loss` 只在前排候选样本上放大错误。

## 4.2 top-k focus 机制

首版不做复杂可微排序，采用稳定、低风险的标签驱动加权：

1. 在每个样本内，根据有效标签和 mask 找出真实收益最高的前 `k` 个股票。
2. 生成 `topk_mask`。
3. 针对与 `topk_mask` 相关的项放大损失：
   - listwise 路线：对前排标签赋予更高 gain；
   - pairwise 路线：若 pair 中含 top-k 正样本，则提升 pair loss 权重；
   - 若实现复杂度过高，则首版只增强 pairwise 分量，保持 listwise 原样。

推荐实现顺序：

1. 先实现 `topk_pairwise_focus_loss`。
2. 若复杂度可控，再将 gain 融入现有 listwise 路径。

这样可以避免首轮改造过深。

## 4.3 配置项

在 [config.py](/home/shirosora/code_storage/THU-BDC2026/code/src/config.py) 中新增以下配置：

- `topk_focus_weight`
  - 类型：`float`
  - 含义：top-k focus 分量总权重
  - 建议默认：`0.0` 或较小值
- `topk_focus_k`
  - 类型：`int`
  - 含义：训练时重点关注的前排数量
  - 建议默认：`5`
- `topk_focus_gain_mode`
  - 类型：`str`
  - 可选：`binary | linear`
  - 含义：前排样本加权方式
- `topk_focus_normalize`
  - 类型：`bool`
  - 含义：是否对 focus 权重做归一化，避免 batch 间 scale 波动

兼容性要求：

- 当 `topk_focus_weight <= 0` 时，loss 行为必须与当前实现一致。
- 新配置缺失时必须走默认值，不影响旧 checkpoint 与旧训练脚本。

## 4.4 代码落点

主要修改文件：

- [train.py](/home/shirosora/code_storage/THU-BDC2026/code/src/train.py)
  - `PortfolioOptimizationLoss`
  - 如有必要，抽取内部 helper 以降低函数复杂度
- [config.py](/home/shirosora/code_storage/THU-BDC2026/code/src/config.py)
  - 新增 top-k focus 配置项
- `test/`
  - 新增或扩展针对 loss 的单元测试

不修改文件：

- [predict.py](/home/shirosora/code_storage/THU-BDC2026/code/src/predict.py)
- [code/src/experiments/metrics.py](/home/shirosora/code_storage/THU-BDC2026/code/src/experiments/metrics.py)
- [test/score_self.py](/home/shirosora/code_storage/THU-BDC2026/test/score_self.py)

## 5. 数据流与执行流程

训练时的执行流程保持不变：

1. 数据进入 batch。
2. 模型输出股票分数。
3. `PortfolioOptimizationLoss.forward()` 计算现有 base loss。
4. 额外计算 `topk_focus_loss`。
5. 合并为总 loss 并反向传播。

关键点：

- `topk_focus_loss` 必须复用现有 `mask` 逻辑。
- 对 padding 项不得产生梯度贡献。
- 对样本中股票数少于 `k` 的场景要自然退化为“对全部有效股票计算”。

## 6. 错误处理与鲁棒性要求

- 若某个 batch 中有效股票数量为 0，应安全返回 0 或跳过该样本，不得产生 NaN。
- 若有效股票数量小于 `topk_focus_k`，应自动取 `min(valid_count, k)`。
- 若标签存在全相等或无有效 pair 的情况，应返回稳定的零损失项，而不是 NaN/Inf。
- 新增权重归一化时，应保证极端小 batch 下不出现除零。

## 7. 测试策略

遵循最小可验证切片。

### 7.1 单元测试

至少覆盖：

1. `topk_focus_weight=0` 时，新旧 loss 一致。
2. `topk_focus_k=1/3/5` 时，前排样本被正确识别。
3. 含 padding/mask 时，padding 不参与 top-k 识别与 loss。
4. 有效股票数量小于 `k` 时，逻辑稳定。
5. 全相等标签或空 pair 场景不产生 NaN。

### 7.2 轻量集成验证

- 跑针对 loss 的测试文件。
- 如仓库现有训练测试可快速运行，可补一个 smoke check，验证训练入口仍能实例化 loss。

## 8. 验收标准

满足以下条件才可进入下一阶段：

1. 新配置默认不改变旧行为。
2. 单元测试覆盖 top-k focus 的核心边界。
3. 现有训练入口不会因新增参数报错。
4. 代码结构没有进一步恶化为更大的单体函数。

## 9. 风险与回滚

主要风险：

- 新分量可能过强，导致模型只关注极少数样本，损害整体稳定性。
- 若权重设计不当，可能放大标签噪声。
- 若直接在多个 loss 分量中同时加权，数值尺度可能失衡。

缓解策略：

- 首版默认 `topk_focus_weight=0.0`。
- 先只增强一个分量，再观察实验结果。
- 保持实现可开关，必要时单行配置回退。

## 10. 后续扩展

若该方案有效，下一步可顺延到：

1. 将启发式 top-k focus 升级为更标准的 LambdaLoss。
2. 在 `scores -> portfolio` 之间加入 reranking 层。
3. 再考虑可微 top-k 或直接组合优化。
