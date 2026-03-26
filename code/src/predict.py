import os
import multiprocessing as mp
import json

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from factor_store import engineer_group_features
from factor_store import load_factor_snapshot
from factor_store import resolve_factor_pipeline
from model import StockTransformer
from data_manager import build_stock_industry_index as build_stock_industry_index_from_manager
from data_manager import collect_data_sources
from data_manager import load_market_dataset
from data_manager import load_stock_to_industry_map
from data_manager import save_data_manifest
from utils import apply_cross_sectional_normalization
from utils import augment_engineered_features
from utils import resolve_feature_indices


def preprocess_predict_data(df, stockid2idx, feature_pipeline):
	feature_columns = feature_pipeline['active_features']
	builtin_override_specs = feature_pipeline.get('builtin_override_specs', [])

	df = df.copy()
	df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
	groups = [group for _, group in df.groupby('股票代码', sort=False)]
	if len(groups) == 0:
		raise ValueError('输入数据为空，无法预测')

	num_processes = min(int(config.get('feature_engineer_processes', 4)), mp.cpu_count())
	with mp.Pool(processes=num_processes) as pool:
		tasks = [
			(group, feature_pipeline['feature_set'], builtin_override_specs, feature_pipeline['custom_specs'])
			for group in groups
		]
		processed_list = list(tqdm(pool.imap(engineer_group_features, tasks), total=len(groups), desc='预测集特征工程'))

	processed = pd.concat(processed_list).reset_index(drop=True)
	processed['instrument'] = processed['股票代码'].map(stockid2idx)
	processed = processed.dropna(subset=['instrument']).copy()
	processed['instrument'] = processed['instrument'].astype(np.int64)
	processed['日期'] = pd.to_datetime(processed['日期'])

	if bool(config.get('use_feature_enhancements', True)):
		processed, feature_columns = augment_engineered_features(
			processed,
			feature_columns,
			config=config,
			date_col='日期',
			stock_col='股票代码',
		)

	if config.get('use_cross_sectional_feature_norm', True):
		cs_exclude = [col for col in feature_columns if col.startswith('market_')]
		processed = apply_cross_sectional_normalization(
			processed,
			feature_columns,
			date_col='日期',
			method=config.get('feature_cs_norm_method', 'zscore'),
			clip_value=config.get('feature_cs_clip_value', None),
			exclude_columns=cs_exclude,
		)

	return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
	sequences, sequence_stock_ids = [], []
	for stock_id in stock_ids:
		stock_history = data[
			(data['股票代码'] == stock_id) &
			(data['日期'] <= latest_date)
		].sort_values('日期').tail(sequence_length)

		if len(stock_history) == sequence_length:
			sequences.append(stock_history[features].values.astype(np.float32))
			sequence_stock_ids.append(stock_id)

	if len(sequences) == 0:
		raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

	return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def load_prediction_strategy(output_dir):
	default_top_k = 5
	default_strategy = {
		'name': f'equal_top{default_top_k}',
		'top_k': default_top_k,
		'weighting': 'equal',
		'temperature': config.get('softmax_temperature', 1.0),
	}
	strategy_path = os.path.join(output_dir, 'best_strategy.json')

	if not os.path.exists(strategy_path):
		return default_strategy

	with open(strategy_path, 'r', encoding='utf-8') as f:
		strategy = json.load(f)

	top_k = int(strategy.get('top_k', default_top_k))
	if top_k < 1 or top_k > 5:
		raise ValueError(f'best_strategy.json 中的 top_k 非法: {top_k}')

	weighting = strategy.get('weighting', 'equal')
	if weighting not in {'equal', 'softmax'}:
		raise ValueError(f'best_strategy.json 中的 weighting 非法: {weighting}')

	return {
		'name': strategy.get('name', f'{weighting}_top{top_k}'),
		'top_k': top_k,
		'weighting': weighting,
		'temperature': float(strategy.get('temperature', config.get('softmax_temperature', 1.0))),
	}


def scores_to_portfolio(scores, stock_ids, strategy):
	top_k = min(int(strategy['top_k']), len(stock_ids), 5)
	if top_k <= 0:
		raise ValueError('持仓股票数量必须大于 0')

	order = np.argsort(scores)[::-1]
	top_indices = order[:top_k]
	selected_ids = [stock_ids[i] for i in top_indices]
	selected_scores = scores[top_indices]

	if strategy['weighting'] == 'equal' or top_k == 1:
		weights = np.full(top_k, 1.0 / top_k, dtype=np.float64)
	elif strategy['weighting'] == 'softmax':
		temperature = max(float(strategy.get('temperature', 1.0)), 1e-6)
		stable_scores = selected_scores - selected_scores.max()
		weights = np.exp(stable_scores / temperature)
		weights = weights / weights.sum()
	else:
		raise ValueError(f"不支持的权重方式: {strategy['weighting']}")

	return selected_ids, weights


def apply_optional_global_scaler(processed, features, scaler_path):
	"""
	兼容三种缩放状态：
	1) 新版默认：identity（仅截面标准化）；
	2) 旧版：sklearn StandardScaler；
	3) 缺失 scaler.pkl：回退 identity。
	"""
	if not os.path.exists(scaler_path):
		print(f'未找到 scaler 文件，回退为 identity: {scaler_path}')
		return processed

	scaler = joblib.load(scaler_path)
	if isinstance(scaler, dict):
		scaler_type = str(scaler.get('type', 'identity')).lower()
		if scaler_type == 'identity':
			print('检测到 identity scaler，跳过全局缩放（仅截面标准化）。')
			return processed
		raise ValueError(f'不支持的 scaler 元数据类型: {scaler_type}')

	if hasattr(scaler, 'transform'):
		processed[features] = scaler.transform(processed[features]).astype(np.float32)
		print('检测到旧版 StandardScaler，已应用全局缩放（兼容模式）。')
		return processed

	raise TypeError(f'无法识别的 scaler 类型: {type(scaler)}')


def main():
	model_path = os.path.join(config['output_dir'], 'best_model.pth')
	scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
	prior_graph_path = os.path.join(config['output_dir'], 'prior_graph_adj.npy')
	industry_index_path = os.path.join(config['output_dir'], 'stock_industry_idx.npy')
	output_path = os.path.join('./output/', 'result.csv')
	factor_snapshot_path = os.path.join(config['output_dir'], 'active_factors.json')
	effective_features_path = os.path.join(config['output_dir'], 'effective_features.json')

	if not os.path.exists(model_path):
		raise FileNotFoundError(f'未找到模型文件: {model_path}')

	data_manifest = collect_data_sources(config, include_csv_stats=True)
	manifest_path = save_data_manifest(config['output_dir'], data_manifest, filename='data_manifest_predict.json')
	print(f'已生成预测数据源清单: {manifest_path}')

	raw_df, data_file = load_market_dataset(config, 'train.csv', dtype={'股票代码': str})
	print(f'预测输入数据文件: {data_file}')
	raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
	raw_df['日期'] = pd.to_datetime(raw_df['日期'])
	latest_date = raw_df['日期'].max()

	stock_ids = sorted(raw_df['股票代码'].unique())
	stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
	stock_industry_idx = None
	if os.path.exists(industry_index_path):
		try:
			cached_index = np.load(industry_index_path)
			if cached_index.ndim == 1 and cached_index.shape[0] == len(stock_ids):
				stock_industry_idx = cached_index.astype(np.int64)
				print(f'加载行业索引映射: {industry_index_path}')
			else:
				print(
					f'行业索引映射形状不匹配，回退重建: '
					f'{cached_index.shape} vs ({len(stock_ids)},)'
				)
		except Exception as exc:
			print(f'读取行业索引映射失败，回退重建: {exc}')
	if stock_industry_idx is None:
		stock_to_industry = load_stock_to_industry_map(
			config,
			stock_col_key='prior_graph_stock_col',
			industry_col_key='prior_graph_industry_col',
		)
		stock_industry_idx, industry_vocab, matched = build_stock_industry_index_from_manager(
			stock_ids,
			stock_to_industry,
		)
		if matched > 0:
			coverage = matched / float(max(1, len(stock_ids)))
			print(
				f'重建行业索引映射: stocks={len(stock_ids)}, matched={matched}, '
				f'coverage={coverage:.2%}, industries={len(industry_vocab)}'
			)
		else:
			print('未构建到有效行业映射，行业虚拟股将回退为空映射。')
	if os.path.exists(factor_snapshot_path):
		feature_pipeline = load_factor_snapshot(factor_snapshot_path)
		print(f'加载训练因子快照: {factor_snapshot_path}')
	else:
		feature_pipeline = resolve_factor_pipeline(
			config['feature_num'],
			config['factor_store_path'],
			config['builtin_factor_registry_path'],
		)
		print(f'未找到训练因子快照，回退到当前因子配置: {config["factor_store_path"]}')

	processed, features = preprocess_predict_data(raw_df, stockid2idx, feature_pipeline)
	if os.path.exists(effective_features_path):
		with open(effective_features_path, 'r', encoding='utf-8') as f:
			saved_features = json.load(f)
		if not isinstance(saved_features, list) or not saved_features:
			raise ValueError(f'effective_features.json 格式非法: {effective_features_path}')
		missing_features = [name for name in saved_features if name not in processed.columns]
		if missing_features:
			raise ValueError(f'预测输入缺少训练特征: {missing_features[:10]}')
		features = saved_features
		print(f'加载训练特征清单: {effective_features_path} | 特征数: {len(features)}')
	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
	processed = apply_optional_global_scaler(processed, features, scaler_path)

	sequence_length = config['sequence_length']
	sequences_np, sequence_stock_ids = build_inference_sequences(
		processed,
		features,
		sequence_length,
		stock_ids,
		latest_date,
	)

	if torch.cuda.is_available():
		device = torch.device('cuda')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')

	model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
	market_context_feature_names = config.get(
		'market_gating_context_feature_names',
		[
			'market_median_return',
			'market_total_turnover_log',
			'market_limit_up_count_log',
			'market_limit_up_ratio',
		],
	)
	market_context_indices = resolve_feature_indices(features, market_context_feature_names)
	model.set_market_context_feature_indices(market_context_indices)
	model.set_stock_industry_index(torch.from_numpy(stock_industry_idx))
	mask_mode = str(config.get('cross_stock_mask_mode', 'similarity')).lower()
	if os.path.exists(prior_graph_path):
		prior_graph = np.load(prior_graph_path)
		if prior_graph.ndim == 2 and prior_graph.shape[0] == prior_graph.shape[1] == len(stock_ids):
			model.set_prior_graph(torch.from_numpy(prior_graph.astype(np.bool_)))
			print(f'加载先验图邻接矩阵: {prior_graph_path}')
		else:
			print(
				f'先验图形状与当前股票池不一致，忽略: '
				f'{prior_graph.shape} vs ({len(stock_ids)}, {len(stock_ids)})'
			)
	elif mask_mode in {'prior', 'prior_similarity'}:
		print('未找到先验图文件，预测阶段回退为 similarity 稀疏注意力。')
		model.cross_stock_attention.mask_mode = 'similarity'
	state_dict = torch.load(model_path, map_location=device)
	load_result = model.load_state_dict(state_dict, strict=False)
	multi_scale_prefixes = (
		'ultra_short_temporal_encoder.',
		'ultra_short_feature_attention.',
		'short_temporal_encoder.',
		'long_temporal_encoder.',
		'short_feature_attention.',
		'long_feature_attention.',
		'short_horizon_fusion_gate.',
		'short_horizon_norm.',
		'multi_scale_fusion_gate.',
		'multi_scale_branch_norm.',
	)
	temporal_cross_stock_prefixes = (
		'temporal_cross_stock_attention.',
	)
	compatible_prefixes = ('market_gate.', 'market_macro_proj.', 'volatility_head.') + multi_scale_prefixes
	compatible_prefixes = compatible_prefixes + temporal_cross_stock_prefixes
	compatible_exact_keys = {'multi_scale_branch_logits'}
	non_compatible_missing = [
		k for k in load_result.missing_keys
		if (k not in compatible_exact_keys) and (not k.startswith(compatible_prefixes))
	]
	if non_compatible_missing:
		raise RuntimeError(f'模型参数缺失且无法兼容: {non_compatible_missing[:10]}')
	if any(k.startswith('market_gate.') for k in load_result.missing_keys):
		print('检测到旧版checkpoint（缺少 market_gate 参数），已自动关闭 market gating 兼容推理')
		model.use_market_gating = False
	if any(k.startswith('market_macro_proj.') for k in load_result.missing_keys):
		print('检测到旧版checkpoint（缺少 market macro 参数），已自动关闭宏观情绪 gate 输入')
		model.use_market_gating_macro_context = False
	if any(k.startswith('volatility_head.') for k in load_result.missing_keys):
		print('检测到旧版checkpoint（缺少 volatility_head 参数），已自动关闭多任务辅助头')
		model.use_multitask_volatility = False
	if any((k == 'multi_scale_branch_logits') or k.startswith(multi_scale_prefixes) for k in load_result.missing_keys):
		print('检测到旧版checkpoint（缺少 multi-scale 参数），已自动关闭多尺度时序分支')
		model.use_multi_scale_temporal = False
	if any(k.startswith(temporal_cross_stock_prefixes) for k in load_result.missing_keys):
		print('检测到旧版checkpoint（缺少 temporal cross-stock 参数），已自动关闭时间步级跨股交互')
		model.use_temporal_cross_stock_attention = False
	if load_result.unexpected_keys:
		print(f'模型包含额外参数（将忽略）: {load_result.unexpected_keys[:10]}')
	model.to(device)
	model.eval()
	strategy = load_prediction_strategy(config['output_dir'])

	with torch.no_grad():
		x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)  # [1, N, L, F]
		stock_indices = torch.tensor(
			[[stockid2idx[sid] for sid in sequence_stock_ids]],
			dtype=torch.long,
			device=device,
		)
		stock_valid_mask = torch.ones_like(stock_indices, dtype=torch.bool, device=device)
		scores = model(
			x,
			stock_indices=stock_indices,
			stock_valid_mask=stock_valid_mask,
		).squeeze(0).detach().cpu().numpy()         # [N]

	selected_ids, weights = scores_to_portfolio(scores, sequence_stock_ids, strategy)

	output_df = pd.DataFrame({
		'stock_id': selected_ids,
		'weight': weights,
	})
	output_df.to_csv(output_path, index=False)

	print(f'预测日期: {latest_date.date()}')
	print(f'参与排序股票数: {len(sequence_stock_ids)}')
	print(f'使用持仓策略: {strategy["name"]}')
	print(f'结果已写入: {output_path}')


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()
