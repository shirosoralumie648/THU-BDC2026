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

    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, 
    'top5_weight': 2.0,
    'pos_weight': 50.0,      # 正样本权重 (针对前2%妖股)
    'tail_multiplier': 10.0, 
    'tail_percentile': 0.95, 
    'validation_mode': 'rolling',
    'rolling_val_num_folds': 4,
    'rolling_val_window_size': 20,
    'rolling_val_step_size': 20,
    'selection_metric': 'auto',
    'prediction_top_k_candidates': [1, 2, 3, 4, 5],
    'prediction_weighting_candidates': ['equal', 'softmax'],
    'softmax_temperature': 1.0,
    'builtin_factor_registry_path': './config/builtin_factors.json',
    'factor_store_path': './config/factor_store.json',
    'factor_histogram_max_features': 20,
    'factor_ablation_enabled': True,

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
}
