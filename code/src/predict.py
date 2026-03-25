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
from utils import apply_cross_sectional_normalization


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

	if config.get('use_cross_sectional_feature_norm', True):
		processed = apply_cross_sectional_normalization(
			processed,
			feature_columns,
			date_col='日期',
			method=config.get('feature_cs_norm_method', 'zscore'),
			clip_value=config.get('feature_cs_clip_value', None),
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


def main():
	data_file = os.path.join(config['data_path'], 'train.csv')
	model_path = os.path.join(config['output_dir'], 'best_model.pth')
	scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
	output_path = os.path.join('./output/', 'result.csv')
	factor_snapshot_path = os.path.join(config['output_dir'], 'active_factors.json')

	if not os.path.exists(model_path):
		raise FileNotFoundError(f'未找到模型文件: {model_path}')
	if not os.path.exists(scaler_path):
		raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

	raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
	raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
	raw_df['日期'] = pd.to_datetime(raw_df['日期'])
	latest_date = raw_df['日期'].max()

	stock_ids = sorted(raw_df['股票代码'].unique())
	stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
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
	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

	scaler = joblib.load(scaler_path)
	processed[features] = scaler.transform(processed[features])

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
	state_dict = torch.load(model_path, map_location=device)
	load_result = model.load_state_dict(state_dict, strict=False)
	non_gate_missing = [k for k in load_result.missing_keys if not k.startswith('market_gate.')]
	if non_gate_missing:
		raise RuntimeError(f'模型参数缺失且无法兼容: {non_gate_missing[:10]}')
	if load_result.missing_keys:
		print('检测到旧版checkpoint（缺少 market_gate 参数），已自动关闭 market gating 兼容推理')
		model.use_market_gating = False
	if load_result.unexpected_keys:
		print(f'模型包含额外参数（将忽略）: {load_result.unexpected_keys[:10]}')
	model.to(device)
	model.eval()
	strategy = load_prediction_strategy(config['output_dir'])

	with torch.no_grad():
		x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)  # [1, N, L, F]
		scores = model(x).squeeze(0).detach().cpu().numpy()         # [N]

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
