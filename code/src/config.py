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
    'dropout': 0.1,
    'feature_num': feature_num,
    'feature_engineer_processes': 4,
    'max_grad_norm': 5.0,

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
    'prediction_top_k_candidates': [3, 5],
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
    'factor_ablation_enabled': True,

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
}
