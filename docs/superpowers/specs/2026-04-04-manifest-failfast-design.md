# Manifest Fail-Fast Design

日期：2026-04-04

## 1. 背景

当前第二阶段的重点不是再扩展 ingestion/API 能力，而是压缩“结果能产出，但输入链或 manifest 链已经失真”的隐蔽失败面。

现状里，以下路径已经具备基本 contract 测试并且通过：

- `test.test_pipeline_config_errors`
- `test.test_cli_error_paths`
- `test.test_manifest_contracts`
- `test.test_ingestion_api`

但数据主链仍有若干静默降级点：

1. manifest 读取失败时，部分逻辑会回退为空字符串、空快照或普通 warning；
2. CSV 元数据探测失败时，调用侧很难区分“文件不存在”和“文件损坏”；
3. CLI 组装 manifest 时，会把“强约束失败”和“可选补充失败”混在一起，导致最终产物缺少可观测性。

这类问题不会立刻抛出 traceback，但会让训练/预测使用不完整的依赖快照，影响结果可信度。

## 2. 目标与非目标

### 2.1 目标

- 让 manifest / 数据快照链在关键失败场景下优先 fail-fast，而不是静默回退。
- 对允许降级的场景输出结构化 warning/error，保证调用侧可观测。
- 保持现有 CLI 协议、manifest 字段兼容，不破坏已有消费端。
- 用最小改动补齐边界测试，锁住新的错误语义。

### 2.2 非目标

- 本轮不重写 ingestion API contract。
- 本轮不调整 provider adapter 的抓取逻辑。
- 本轮不做性能优化或批量 I/O 重构。
- 本轮不引入新的 manifest 格式版本。

## 3. 方案选型

### 3.1 备选方案

方案 A：继续允许静默回退，只补日志  
优点：改动最小。  
缺点：失败仍然会进入结果产物，日志也不一定被消费。

方案 B：对关键 manifest/快照入口做 fail-fast，对可选探测保留结构化降级  
优点：风险和收益平衡，兼容当前 CLI 和测试体系。  
缺点：需要梳理每个入口的“关键”与“可选”边界。

方案 C：统一引入完整 Result/Error 对象并全链路改造  
优点：语义最清晰。  
缺点：改动范围过大，会波及训练、预测、CLI、manifest 生成多个模块。

### 3.2 选定方案

采用方案 B。

理由：

- 当前仓库已经有 manifest contract 和 CLI error path 测试，可以直接扩展；
- 关键入口 fail-fast 能立刻减少隐蔽坏结果；
- 不需要在第二阶段中途做大规模接口重写。

## 4. 设计概述

### 4.1 改造范围

主要文件：

- `code/src/data_manager.py`
- `code/src/manage_data.py`
- `test/test_manifest_contracts.py`
- `test/test_cli_error_paths.py`

必要时补充：

- `test/test_predict_pipeline.py`
- `test/test_train_pipeline.py`

### 4.2 关键原则

把错误分成两类：

1. **关键依赖错误**
   - manifest 文件存在但不可解析
   - manifest 顶层类型不合法
   - 关键字段存在但语义冲突
   - 应直接抛出明确异常或在 CLI 中转成退出码 2

2. **可选补充错误**
   - CSV 行列统计读取失败
   - 非关键 snapshot 无法补充 metadata
   - 允许保留主流程，但必须写入结构化 `warnings` 或 `errors`

### 4.3 具体落点

#### A. `data_manager.py`

重点收敛三个入口：

- dataset build manifest 读取
- CSV metadata 探测
- file snapshot 构建

要求：

- “manifest 不存在”和“manifest 损坏/不可解析”必须区分；
- `inspect_csv_metadata()` 不再只返回模糊字符串，应返回可断言的错误语义；
- `build_file_snapshot()` 对 `size_bytes`、`csv metadata` 的失败要保留结构化错误，而不是直接吞掉。

#### B. `manage_data.py`

重点收敛两个入口：

- factor manifest 指纹提取
- `build-dataset` 组装 pipeline validation / manifest 时的错误传播

要求：

- `_extract_factor_fingerprint_from_manifest()` 不能把“manifest 损坏”伪装成“没有 fingerprint”；
- `command_build_dataset()` 在关键 manifest 校验失败时应停止构建，而不是继续产出表面成功的 manifest；
- 对允许降级的场景，将错误写入 manifest 的 `warnings/errors`，保持可追踪。

## 5. 数据流与错误传播

新的传播规则：

1. CLI 接到路径参数；
2. 解析 manifest / snapshot；
3. 若失败属于关键依赖错误：
   - 业务函数抛出明确异常；
   - CLI 统一转为无 traceback 的用户错误；
4. 若失败属于可选补充错误：
   - 主流程继续；
   - 在 manifest snapshot 中记录 `error`/`warning` 字段。

这样可以保证“不能继续”的情况立即中止，“可以继续”的情况也不会静默丢失上下文。

## 6. 测试策略

### 6.1 扩展现有测试

- `test/test_manifest_contracts.py`
  - 覆盖 manifest 损坏时的 contract 输出
  - 覆盖 snapshot metadata 失败时的结构化错误字段

- `test/test_cli_error_paths.py`
  - 覆盖损坏 manifest 导致的 CLI 非 0 退出
  - 覆盖关键依赖缺失时不输出 traceback

### 6.2 回归测试

至少验证：

- `test.test_pipeline_config_errors`
- `test.test_cli_error_paths`
- `test.test_manifest_contracts`
- `test.test_ingestion_api`

若数据主链行为有外溢，再补跑：

- `test.test_ingestion_service`
- `test.test_ingestion_runtime`

## 7. 验收标准

满足以下条件才算完成：

1. manifest 关键错误不再静默降级；
2. 可选 metadata 错误可在产物中被观察到；
3. CLI 关键失败返回稳定错误消息且无 traceback；
4. 现有 manifest contract 测试和 CLI error path 测试全部通过；
5. 改动范围控制在 manifest / 数据链本身，不引入新的 API 语义漂移。

## 8. 风险与回滚

主要风险：

- fail-fast 边界划得过宽，导致原本可容忍的场景被错误中止；
- 某些调用方依赖旧的“返回空字符串”语义。

缓解方式：

- 首先用测试锁住关键/可选边界；
- 只对 manifest 和关键输入链做 fail-fast；
- 对非关键 metadata 继续兼容，但补充结构化错误信息。

若出现兼容性问题，可按函数级别回退到旧逻辑，而不影响第二阶段其余入口。
