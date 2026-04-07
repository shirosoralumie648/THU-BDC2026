import os
import multiprocessing as mp

import numpy as np

from config import config
from features.feature_assembler import augment_feature_table
from features.feature_assembler import build_feature_table
from models.rank_model import StockTransformer
from training.inference import apply_optional_global_scaler
from training.inference import build_inference_sequences
from training.inference import build_prediction_input_manifest
from training.inference import dump_predict_factor_snapshot
from training.inference import load_prediction_inputs
from training.inference import load_prediction_model
from training.inference import load_prediction_runtime_config
from training.inference import resolve_effective_prediction_features
from training.inference import load_prediction_strategy
from training.inference import run_prediction_inference
from training.inference import write_prediction_outputs
from training.preprocessing import preprocess_predict_data
from training.runtime import resolve_prediction_stock_industry_index


def main():
	runtime_config = load_prediction_runtime_config(config['output_dir'], runtime_config=config)
	model_path = os.path.join(runtime_config['output_dir'], 'best_model.pth')
	scaler_path = os.path.join(runtime_config['output_dir'], 'scaler.pkl')
	prior_graph_path = os.path.join(runtime_config['output_dir'], 'prior_graph_adj.npy')
	industry_index_path = os.path.join(runtime_config['output_dir'], 'stock_industry_idx.npy')
	output_path = os.path.join('./output/', 'result.csv')
	scores_output_path = str(runtime_config.get('prediction_scores_path', os.path.join('./output/', 'prediction_scores.csv')))
	factor_snapshot_path = os.path.join(runtime_config['output_dir'], 'active_factors.json')
	effective_features_path = os.path.join(runtime_config['output_dir'], 'effective_features.json')

	if not os.path.exists(model_path):
		raise FileNotFoundError(f'未找到模型文件: {model_path}')

	feature_pipeline, raw_df, latest_date, stock_ids, stockid2idx = load_prediction_inputs(
		runtime_config,
		factor_snapshot_path,
	)
	stock_industry_idx = resolve_prediction_stock_industry_index(
		stock_ids,
		industry_index_path,
		runtime_config=runtime_config,
	)

	processed, features = preprocess_predict_data(raw_df, stockid2idx, feature_pipeline)
	features = resolve_effective_prediction_features(
		processed,
		features,
		effective_features_path,
	)
	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
	processed = apply_optional_global_scaler(processed, features, scaler_path)
	dump_predict_factor_snapshot(processed, features, runtime_config['output_dir'], latest_date, runtime_config=runtime_config)

	sequence_length = config['sequence_length']
	sequences_np, sequence_stock_ids = build_inference_sequences(
		processed,
		features,
		sequence_length,
		stock_ids,
		latest_date,
	)

	model, device = load_prediction_model(
		features=features,
		stock_ids=stock_ids,
		stock_industry_idx=stock_industry_idx,
		model_path=model_path,
		prior_graph_path=prior_graph_path,
		runtime_config=runtime_config,
	)
	strategy = load_prediction_strategy(runtime_config['output_dir'], runtime_config=runtime_config)
	scores = run_prediction_inference(
		model,
		sequences_np,
		sequence_stock_ids,
		stockid2idx,
		device,
	)
	write_prediction_outputs(
		scores,
		sequence_stock_ids,
		strategy,
		scores_output_path,
		output_path,
	)

	print(f'预测日期: {latest_date.date()}')
	print(f'参与排序股票数: {len(sequence_stock_ids)}')
	print(f'全量分数已写入: {scores_output_path}')
	print(f'使用持仓策略: {strategy["name"]}')
	print(f'结果已写入: {output_path}')


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()
