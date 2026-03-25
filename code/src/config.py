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
    'loss_temperature': 10.0,
    'pairwise_top_fraction': 0.1,
    'base_weight': 1.0,
    'top5_weight': 2.0,
    'pos_weight': 50.0,      # 正样本权重 (针对前2%妖股)
    'tail_multiplier': 10.0, 
    'tail_percentile': 0.95, 

    # 训练目标稳健化：标签极值处理 + 截面标准化（按日）
    'label_extreme_mode': 'clip',   # none | drop | clip | drop_clip
    'label_extreme_lower_quantile': 0.05,
    'label_extreme_upper_quantile': 0.95,
    'use_cross_sectional_label_norm': True,
    'label_cs_norm_method': 'zscore',  # zscore | rank
    'label_cs_clip_value': 5.0,

    # 特征稳定化：按日截面标准化（训练/推理同口径）
    'use_cross_sectional_feature_norm': True,
    'feature_cs_norm_method': 'zscore',  # zscore | rank
    'feature_cs_clip_value': 5.0,

    # 市场状态引导门控（model input gating）
    'use_market_gating': True,
    'market_gate_hidden_dim': 128,
    'market_gate_residual': 0.5,

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
