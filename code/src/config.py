import json
import os

# 配置参数
sequence_length = 60
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 256,          # Transformer输入维度
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer层数
    'dim_feedforward': 512, # 前馈网络维度
    'batch_size': 4,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': 50,       # 排序任务可能需要更多epochs
    'learning_rate': 1e-5,  # 稍微降低学习率
    'weight_decay': 1e-5,
    'dropout': 0.1,
    'feature_num': feature_num,
    'feature_engineer_processes': 4,
    'enable_grad_clip': True,
    'max_grad_norm': 5.0,
    'use_amp': True,
    'use_amp_eval': True,
    'train_zero_grad_set_to_none': True,
    # 0 表示使用完整数据；>0 可用于快速实验
    'max_train_batches_per_epoch': 0,
    'max_eval_batches_per_fold': 0,

    'pairwise_weight': 1.0,  # 配对损失权重
    'listnet_weight': 1.0,   # ListNet 权重
    'lambda_ndcg_weight': 0.8,  # LambdaNDCG 权重
    'lambda_ndcg_topk': 60,  # LambdaNDCG 仅在真实头部样本上构造 pair，控制复杂度
    # IC 正则：增强预测分数与真实收益的全局相关性（默认 Pearson）
    'ic_weight': 0.2,
    'ic_mode': 'pearson',  # pearson | spearman
    'loss_temperature': 10.0,
    'pairwise_top_fraction': 0.1,
    'base_weight': 1.0,
    'top5_weight': 2.0,
    'pos_weight': 50.0,      # 正样本权重 (针对前2%妖股)
    'tail_multiplier': 10.0, 
    'tail_percentile': 0.95, 

    # 标签处理：市场中性化 + MAD去极值（按日） + 截面标准化（按日）
    'use_label_market_neutralization': True,
    # none | cross_sectional_mean | benchmark | industry | benchmark_then_industry | industry_then_benchmark
    'label_market_neutralization': 'cross_sectional_mean',
    # benchmark 中性化（可选）
    'label_benchmark_return_path': '',      # 例如: ./data/hs300_index.csv
    'label_benchmark_return_col': '',       # 留空自动识别: benchmark_return/return/收益率/涨跌幅/pct_chg
    # industry 中性化（可选）
    'label_industry_map_path': '',          # 例如: ./data/stock_industry_map.csv
    'label_industry_stock_col': '股票代码',
    'label_industry_col': '行业',
    'use_label_mad_clip': True,
    'label_mad_clip_n': 5.0,
    'label_mad_min_scale': 1e-6,
    'label_mad_min_group_size': 5,
    # 损失阶段的额外极值处理（默认关闭；保留兼容）
    'label_extreme_mode': 'none',   # none | drop | clip | drop_clip | mad_clip | mad_drop | mad_drop_clip
    'label_extreme_lower_quantile': 0.05,
    'label_extreme_upper_quantile': 0.95,
    'use_cross_sectional_label_norm': True,
    'label_cs_norm_method': 'zscore',  # zscore | rank
    'label_cs_clip_value': 5.0,

    # 特征稳定化：按日截面标准化（训练/推理同口径）
    'use_feature_enhancements': True,
    # 静态特征（行业/市值）来自外部映射文件，可选开启
    'use_static_stock_features': True,
    'stock_static_feature_path': '',  # 例如 ./data/stock_static_features.csv
    'stock_static_stock_col': '股票代码',
    'stock_static_industry_col': '行业',
    'stock_static_market_cap_col': '流通市值',
    'stock_static_industry_topk': 12,
    'stock_static_include_other_bucket': True,
    # 截面 rank 与行业内相对强弱
    'use_cross_sectional_rank_features': True,
    'cross_sectional_rank_source_features': [
        'rsi', 'return_1', 'return_5', 'volatility_20', 'atr_14', '换手率', '成交额', 'pv_intraday_range_pct'
    ],
    'use_industry_relative_z_features': True,
    'industry_relative_source_features': ['rsi', 'return_1', 'return_5', 'volatility_20', 'atr_14'],
    # 量价分布风险特征（基于日频 OHLCV）
    'use_price_volume_distribution_features': True,

    'use_cross_sectional_feature_norm': True,
    'feature_cs_norm_method': 'zscore',  # zscore | rank
    'feature_cs_clip_value': 5.0,
    # 股票间交互约束：默认启用“先验图 + 相似度”联合稀疏注意力
    'use_cross_stock_attention_mask': True,
    'cross_stock_mask_mode': 'prior_similarity',  # full | similarity | prior | prior_similarity
    'cross_stock_similarity_topk': 40,
    'prior_similarity_combine': 'intersection',  # intersection | union
    # 全局-局部交互：行业均值“虚拟股”参与注意力
    'use_industry_virtual_stock': True,
    'industry_virtual_connect_mode': 'same',  # same | all
    'industry_virtual_min_members': 1,
    'industry_virtual_on_temporal_cross_stock': False,
    # 先验图构建配置（行业关系 + 历史收益相关性）
    'prior_graph_use_industry': True,
    'prior_graph_use_correlation': True,
    'prior_graph_industry_map_path': '',  # 留空时回退到 label_industry_map_path/stock_static_feature_path
    'prior_graph_stock_col': '股票代码',
    'prior_graph_industry_col': '行业',
    'prior_graph_corr_source_col': 'label_raw',  # label_raw | label
    'prior_graph_corr_threshold': 0.2,
    'prior_graph_corr_topk': 20,
    'prior_graph_corr_min_periods': 20,

    # 多任务学习：收益率排序（主任务）+ 波动率预测（辅助任务）
    'use_multitask_volatility': True,
    'volatility_loss_weight': 0.2,
    'volatility_loss_type': 'huber',  # huber | mse | l1
    'volatility_horizon': 5,
    'use_volatility_label_log1p': True,
    'use_volatility_label_mad_clip': True,
    'volatility_label_mad_clip_n': 5.0,
    'volatility_label_mad_min_scale': 1e-6,
    'use_volatility_cs_norm': True,
    'volatility_cs_norm_method': 'zscore',  # zscore | rank
    'volatility_cs_clip_value': 5.0,

    # 市场状态引导门控（model input gating）
    'use_market_gating': True,
    'market_gate_hidden_dim': 128,
    'market_gate_residual': 0.5,
    'use_market_gating_macro_context': True,
    'market_gate_macro_hidden_dim': 64,
    'market_gate_macro_weight': 0.3,
    'market_gating_context_feature_names': [
        'market_median_return',
        'market_total_turnover_log',
        'market_limit_up_count_log',
        'market_limit_up_ratio',
    ],
    # Market Gating 增强：日级宏观情绪隐含特征
    'use_market_sentiment_features': True,
    'market_sentiment_return_col_candidates': ['涨跌幅', 'return_1'],
    'market_sentiment_turnover_col_candidates': ['成交额'],
    'market_sentiment_limit_up_threshold': 0.095,

    # 多尺度时序编码：超短(1-3日动量) + 短(5-10) + 长(20-60)并行分支
    'use_multi_scale_temporal': True,
    'use_ultra_short_branch': True,
    'multi_scale_ultra_short_windows': [1, 2, 3],
    'multi_scale_short_windows': [5, 10],
    'multi_scale_long_windows': [20, 40, 60],
    'multi_scale_window_reduce': 'mean',   # mean | last
    'multi_scale_fusion': 'gated',         # gated | weighted_sum
    # 时间步级跨股交互（MASTER风格）：先做每个时间切片内的股票关系，再做时序编码
    'use_temporal_cross_stock_attention': True,
    'temporal_cross_stock_nhead': 4,
    'use_temporal_cross_stock_attention_mask': True,
    'temporal_cross_stock_mask_mode': 'similarity',  # full | similarity | prior | prior_similarity
    'temporal_cross_stock_similarity_topk': 30,
    'temporal_prior_similarity_combine': 'intersection',  # intersection | union

    # 验证期策略选择：均值-波动惩罚，降低 top1 过拟合
    'validation_mode': 'rolling',
    'rolling_val_num_folds': 4,
    'rolling_val_window_size': 20,
    'rolling_val_step_size': 20,
    'strategy_selection_mode': 'risk_adjusted',  # risk_adjusted | return
    'strategy_risk_lambda': 0.2,
    'selection_metric': 'auto',
    'prediction_top_k_candidates': [2, 3, 5],
    'prediction_weighting_candidates': ['equal', 'softmax'],
    'softmax_temperature': 1.0,

    # RankIC 早停
    'early_stopping_enabled': True,
    'early_stopping_patience': 8,
    'early_stopping_min_delta': 1e-4,
    'early_stopping_monitor': 'rank_ic_mean',
    'early_stopping_mode': 'max',

    'builtin_factor_registry_path': './config/builtin_factors.json',
    'factor_store_path': './config/factor_store.json',
    'factor_histogram_max_features': 20,
    'factor_ablation_enabled': False,

    # 数据存储布局：保持兼容 legacy(data/*.csv)，可选 structured(data/datasets/*)
    'structured_data_root': 'datasets',
    'prefer_structured_data_layout': False,
    'mirror_legacy_and_structured_data': True,
    'dataset_paths': {},
    # 训练阶段可优先使用 build-dataset 生成的 manifest 定位 train.csv
    'use_dataset_build_manifest': True,
    # 留空时默认回退到 <train.csv 所在目录>/data_manifest_dataset_build.json
    'dataset_build_manifest_path': '',
    # 期望特征版本（例如 v1），留空表示不做版本匹配校验
    'expected_feature_set_version': '',
    # 期望因子指纹，留空表示不做显式指纹匹配
    'expected_factor_fingerprint': '',
    # 严格模式：manifest 不可用或校验失败直接报错，不回退
    'dataset_manifest_strict': False,
    # 要求 manifest 必须携带 factor_fingerprint
    'dataset_manifest_require_factor_fingerprint': False,

    # 因子结果落盘（训练/推理），便于排障与复盘
    'dump_factor_artifacts': True,
    'factor_artifact_max_rows': 100000,
    'factor_artifact_include_full_feature_stats': True,
    'save_predict_factor_snapshot': True,

    # 高频(日内)因子融合：将外部高频聚合后的日因子按 股票代码+日期 合并到训练/推理输入
    'use_hf_daily_factor_merge': False,
    'hf_daily_factor_path': '',  # 例如 ./data/hf_daily_factors.csv
    'hf_factor_stock_col': '股票代码',
    'hf_factor_date_col': '日期',
    'hf_factor_columns': [],  # 留空表示导入全部非主键列
    'hf_factor_prefix': '',   # 例如 'hf_'，可避免与现有列冲突
    'hf_factor_merge_how': 'left',  # left | inner
    'hf_factor_drop_duplicate_keep': 'last',  # first | last
    'hf_factor_allow_overwrite_columns': False,
    'hf_factor_required': False,

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
    'prediction_scores_path': './output/prediction_scores.csv',
}


def _apply_runtime_override(cfg):
    override_path = str(os.environ.get('THU_BDC_CONFIG_OVERRIDE_PATH', '')).strip()
    if not override_path:
        return cfg
    if not os.path.exists(override_path):
        print(f'[config] override 文件不存在，已忽略: {override_path}')
        return cfg

    try:
        with open(override_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError('override 内容必须是 JSON object')
    except Exception as exc:
        print(f'[config] 读取 override 失败，已忽略: {exc}')
        return cfg

    cfg.update(payload)
    if ('sequence_length' in payload or 'feature_num' in payload) and ('output_dir' not in payload):
        cfg['output_dir'] = f"./model/{cfg['sequence_length']}_{cfg['feature_num']}"

    print(f'[config] 已加载 runtime override: {override_path}')
    return cfg


config = _apply_runtime_override(config)
