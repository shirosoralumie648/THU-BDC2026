import json
import os

import joblib
import numpy as np
import pandas as pd
import torch

from config import config
from data_manager import collect_data_sources
from data_manager import load_market_dataset
from data_manager import load_market_dataset_from_path
from data_manager import load_train_dataset_from_build_manifest
from data_manager import save_data_manifest
from factor_store import load_factor_snapshot
from factor_store import resolve_factor_pipeline
from models.rank_model import StockTransformer
from portfolio.policy import scores_to_portfolio
from utils import resolve_feature_indices


def predict_top_stocks(model, data, features, sequence_length, scaler, stockid2idx, device, top_k=5):
    """
    预测某一天涨幅前top_k的股票
    """
    model.eval()

    latest_date = data['日期'].max()

    day_sequences = []
    day_stock_codes = []
    day_stock_indices = []

    for stock_code in data['股票代码'].unique():
        stock_history = data[
            (data['股票代码'] == stock_code) &
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)

        if len(stock_history) == sequence_length:
            seq = stock_history[features].values
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)
            day_stock_indices.append(stockid2idx[stock_code])

    if len(day_sequences) == 0:
        return []

    sequences = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)
    stock_indices = torch.LongTensor(np.array(day_stock_indices, dtype=np.int64)).unsqueeze(0).to(device)
    stock_valid_mask = torch.ones_like(stock_indices, dtype=torch.bool, device=device)

    with torch.no_grad():
        outputs = model(
            sequences,
            stock_indices=stock_indices,
            stock_valid_mask=stock_valid_mask,
        )
        scores = outputs.squeeze().cpu().numpy()

        top_indices = np.argsort(scores)[::-1][:top_k]

        top_stocks = []
        for idx in top_indices:
            top_stocks.append({
                'stock_code': day_stock_codes[idx],
                'predicted_score': scores[idx],
                'rank': len(top_stocks) + 1,
            })

    return top_stocks


def save_predictions(top_stocks, output_path):
    """保存预测结果"""
    results = []
    for stock in top_stocks:
        results.append({
            '排名': stock['rank'],
            '股票代码': stock['stock_code'],
            '预测分数': stock['predicted_score'],
        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {output_path}")


def build_prediction_input_manifest(runtime_config):
    data_manifest = collect_data_sources(runtime_config, include_csv_stats=True)
    data_manifest['generated_at_utc'] = pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    return data_manifest


def load_prediction_runtime_config(output_dir, runtime_config=None):
    runtime_config = dict(runtime_config or config)
    config_path = os.path.join(output_dir, 'config.json')
    if not os.path.exists(config_path):
        return runtime_config

    with open(config_path, 'r', encoding='utf-8') as f:
        saved_config = json.load(f)
    if not isinstance(saved_config, dict):
        raise ValueError(f'训练配置快照格式非法: {config_path}')

    resolved_config = dict(saved_config)
    for key in ('output_dir', 'prediction_scores_path', 'data_path', 'dataset_build_manifest_path', 'dataset_manifest_strict'):
        if key in runtime_config:
            resolved_config[key] = runtime_config[key]
    return resolved_config


def load_prediction_inputs(runtime_config, factor_snapshot_path):
    output_dir = runtime_config['output_dir']
    data_manifest = build_prediction_input_manifest(runtime_config)
    manifest_path = save_data_manifest(output_dir, data_manifest, filename='data_manifest_predict.json')
    print(f'已生成预测数据源清单: {manifest_path}')

    if os.path.exists(factor_snapshot_path):
        feature_pipeline = load_factor_snapshot(factor_snapshot_path)
        print(
            f'加载训练因子快照: {factor_snapshot_path} | '
            f'fingerprint={feature_pipeline.get("factor_fingerprint", "")}'
        )
    else:
        feature_pipeline = resolve_factor_pipeline(
            runtime_config['feature_num'],
            runtime_config['factor_store_path'],
            runtime_config['builtin_factor_registry_path'],
        )
        print(f'未找到训练因子快照，回退到当前因子配置: {runtime_config["factor_store_path"]}')

    dataset_manifest_train_path, dataset_manifest_info = load_train_dataset_from_build_manifest(
        runtime_config,
        feature_pipeline,
    )
    if dataset_manifest_info.get('enabled', False):
        print(
            f"dataset build manifest: {dataset_manifest_info.get('manifest_path', '')} "
            f"(strict={dataset_manifest_info.get('strict', False)})"
        )
        for msg in dataset_manifest_info.get('warnings', []):
            print(f"[manifest-warning] {msg}")
        for msg in dataset_manifest_info.get('errors', []):
            print(f"[manifest-error] {msg}")
        if dataset_manifest_info.get('used', False):
            print(
                "manifest 元信息: "
                f"build_id={dataset_manifest_info.get('build_id', '')}, "
                f"feature_set_version={dataset_manifest_info.get('feature_set_version', '')}, "
                f"factor_fingerprint={dataset_manifest_info.get('factor_fingerprint', '')}"
            )

    if dataset_manifest_train_path:
        raw_df, data_file = load_market_dataset_from_path(
            runtime_config,
            dataset_manifest_train_path,
            dtype={'股票代码': str},
        )
        print(f'预测输入数据文件(manifest): {data_file}')
    else:
        raw_df, data_file = load_market_dataset(runtime_config, 'train.csv', dtype={'股票代码': str})
        print(f'预测输入数据文件: {data_file}')

    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    latest_date = raw_df['日期'].max()

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    return feature_pipeline, raw_df, latest_date, stock_ids, stockid2idx


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


def load_prediction_strategy(output_dir, runtime_config=None):
    runtime_config = runtime_config or config
    default_top_k = 5
    default_strategy = {
        'name': f'equal_top{default_top_k}',
        'top_k': default_top_k,
        'weighting': 'equal',
        'temperature': runtime_config.get('softmax_temperature', 1.0),
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
        'temperature': float(strategy.get('temperature', runtime_config.get('softmax_temperature', 1.0))),
    }


def resolve_effective_prediction_features(processed, features, effective_features_path):
    resolved_features = list(features)
    if not os.path.exists(effective_features_path):
        return resolved_features

    with open(effective_features_path, 'r', encoding='utf-8') as f:
        saved_features = json.load(f)
    if not isinstance(saved_features, list) or not saved_features:
        raise ValueError(f'effective_features.json 格式非法: {effective_features_path}')

    missing_features = [name for name in saved_features if name not in processed.columns]
    if missing_features:
        raise ValueError(f'预测输入缺少训练特征: {missing_features[:10]}')

    print(f'加载训练特征清单: {effective_features_path} | 特征数: {len(saved_features)}')
    return list(saved_features)


def _resolve_inference_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _apply_prediction_checkpoint_compatibility(model, load_result):
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


def load_prediction_model(features, stock_ids, stock_industry_idx, model_path, prior_graph_path, runtime_config=None):
    runtime_config = runtime_config or config
    device = _resolve_inference_device()

    model = StockTransformer(input_dim=len(features), config=runtime_config, num_stocks=len(stock_ids))
    market_context_feature_names = runtime_config.get(
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

    mask_mode = str(runtime_config.get('cross_stock_mask_mode', 'similarity')).lower()
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
    _apply_prediction_checkpoint_compatibility(model, load_result)
    model.to(device)
    model.eval()
    return model, device


def run_prediction_inference(model, sequences_np, sequence_stock_ids, stockid2idx, device):
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
        ).squeeze(0).detach().cpu().numpy()
    return scores


def write_prediction_outputs(scores, sequence_stock_ids, strategy, scores_output_path, output_path):
    score_df = pd.DataFrame({
        'stock_id': sequence_stock_ids,
        'score': scores,
    }).sort_values('score', ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(scores_output_path) or '.', exist_ok=True)
    score_df.to_csv(scores_output_path, index=False)

    selected_ids, weights = scores_to_portfolio(scores, sequence_stock_ids, strategy)
    output_df = pd.DataFrame({
        'stock_id': selected_ids,
        'weight': weights,
    })
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    output_df.to_csv(output_path, index=False)


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


def dump_predict_factor_snapshot(processed, features, output_dir, latest_date, runtime_config=None):
    runtime_config = runtime_config or config
    if not bool(runtime_config.get('save_predict_factor_snapshot', True)):
        return
    if processed is None or len(processed) == 0:
        return

    artifact_dir = os.path.join(output_dir, 'factor_artifacts')
    os.makedirs(artifact_dir, exist_ok=True)
    latest_ts = pd.to_datetime(latest_date, errors='coerce')
    if pd.isna(latest_ts):
        latest_df = processed.copy()
    else:
        latest_df = processed[pd.to_datetime(processed['日期'], errors='coerce') == latest_ts].copy()
        if latest_df.empty:
            latest_df = processed.copy()

    base_cols = [col for col in ['日期', '股票代码', 'instrument'] if col in latest_df.columns]
    feature_cols = [col for col in features if col in latest_df.columns]
    export_df = latest_df[base_cols + feature_cols].copy()

    values_path = os.path.join(artifact_dir, 'predict_latest_factor_values.csv')
    export_df.to_csv(values_path, index=False, encoding='utf-8')

    if feature_cols:
        stats_df = pd.DataFrame({
            'feature': feature_cols,
            'mean': export_df[feature_cols].mean(axis=0, skipna=True).values,
            'std': export_df[feature_cols].std(axis=0, skipna=True).values,
            'min': export_df[feature_cols].min(axis=0, skipna=True).values,
            'max': export_df[feature_cols].max(axis=0, skipna=True).values,
            'na_ratio': export_df[feature_cols].isna().mean(axis=0).values,
        })
    else:
        stats_df = pd.DataFrame(columns=['feature', 'mean', 'std', 'min', 'max', 'na_ratio'])
    stats_path = os.path.join(artifact_dir, 'predict_latest_factor_stats.csv')
    stats_df.to_csv(stats_path, index=False, encoding='utf-8')

    meta = {
        'latest_date': str(pd.to_datetime(latest_df['日期'], errors='coerce').max().date()) if '日期' in latest_df.columns else '',
        'rows_exported': int(len(export_df)),
        'feature_count': int(len(feature_cols)),
        'values_path': values_path,
        'stats_path': stats_path,
    }
    meta_path = os.path.join(artifact_dir, 'predict_latest_factor_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'已导出预测因子快照: {values_path} (rows={meta["rows_exported"]}, features={meta["feature_count"]})')
