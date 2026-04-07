import json
import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter

from config import config
from data_manager import build_stock_industry_index as build_stock_industry_index_from_manager
from data_manager import collect_data_sources
from data_manager import load_market_dataset
from data_manager import load_market_dataset_from_path
from data_manager import load_train_dataset_from_build_manifest
from data_manager import load_stock_to_industry_map
from data_manager import save_data_manifest
from factor_store import resolve_factor_pipeline
from factor_store import save_factor_snapshot
from graph.graph_builder import build_prior_graph_adjacency as build_prior_graph_adjacency_shared
from graph.industry_graph import build_stock_industry_index as build_stock_industry_index_shared
from models.rank_model import StockTransformer
from training.artifacts import format_factor_summary
from training.artifacts import print_active_factors
from utils import resolve_feature_indices


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def _resolve_torch_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _format_training_device_message(device):
    if device.type == 'cuda':
        return f"当前训练设备: cuda ({torch.cuda.get_device_name(device)})"
    if device.type == 'mps':
        return "当前训练设备: mps (Apple Silicon)"
    return "当前训练设备: cpu"


def initialize_training_runtime(output_dir, runtime_config=None, is_train=True):
    runtime_config = runtime_config or config
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(runtime_config, f, indent=4, ensure_ascii=False)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log')) if is_train else None
    device = _resolve_torch_device()
    print(_format_training_device_message(device))
    return writer, device


def build_rank_model(features, num_stocks, stock_industry_idx, runtime_config=None, prior_graph_adj=None):
    runtime_config = runtime_config or config
    model = StockTransformer(input_dim=len(features), config=runtime_config, num_stocks=num_stocks)
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
    if prior_graph_adj is not None:
        model.set_prior_graph(torch.as_tensor(prior_graph_adj))
    model.set_stock_industry_index(torch.as_tensor(stock_industry_idx, dtype=torch.long))
    return model


def build_optimizer_scheduler_scaler(model, device, runtime_config=None):
    runtime_config = runtime_config or config
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=runtime_config['learning_rate'],
        weight_decay=float(runtime_config.get('weight_decay', 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.2,
        total_iters=runtime_config['num_epochs'],
    )
    use_amp = bool(runtime_config.get('use_amp', True)) and (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"AMP 混合精度训练: {'开启' if use_amp else '关闭'}")
    return optimizer, scheduler, use_amp, scaler


def build_early_stopping_state(runtime_config=None):
    runtime_config = runtime_config or config
    early_stop_mode = str(runtime_config.get('early_stopping_mode', 'max')).lower()
    if early_stop_mode not in {'max', 'min'}:
        raise ValueError(f'early_stopping_mode 非法: {early_stop_mode}')
    return {
        'best_score': -float('inf'),
        'best_epoch': -1,
        'enabled': bool(runtime_config.get('early_stopping_enabled', True)),
        'patience': int(runtime_config.get('early_stopping_patience', 8)),
        'min_delta': float(runtime_config.get('early_stopping_min_delta', 1e-4)),
        'monitor': str(runtime_config.get('early_stopping_monitor', 'rank_ic_mean')),
        'mode': early_stop_mode,
        'best_monitor': -float('inf') if early_stop_mode == 'max' else float('inf'),
        'bad_epochs': 0,
    }


def update_early_stopping_state(eval_metrics, early_stop_state, epoch=None, writer=None):
    monitor = early_stop_state['monitor']
    monitor_value = eval_metrics.get(monitor, None)
    if monitor_value is None:
        next_state = dict(early_stop_state)
        next_state['monitor_value'] = None
        next_state['missing_monitor'] = True
        next_state['should_stop'] = False
        return next_state

    if early_stop_state['mode'] == 'max':
        improved = monitor_value > (early_stop_state['best_monitor'] + early_stop_state['min_delta'])
    else:
        improved = monitor_value < (early_stop_state['best_monitor'] - early_stop_state['min_delta'])

    next_state = dict(early_stop_state)
    next_state['monitor_value'] = monitor_value
    next_state['missing_monitor'] = False
    next_state['best_monitor'] = monitor_value if improved else early_stop_state['best_monitor']
    next_state['bad_epochs'] = 0 if improved else (early_stop_state['bad_epochs'] + 1)
    next_state['should_stop'] = next_state['enabled'] and next_state['bad_epochs'] >= next_state['patience']

    if writer:
        writer.add_scalar(f'early_stop/{monitor}', monitor_value, global_step=epoch)
        writer.add_scalar('early_stop/bad_epochs', next_state['bad_epochs'], global_step=epoch)

    return next_state


def print_early_stopping_summary(early_stop_state):
    if early_stop_state['missing_monitor']:
        print(f"早停监控指标缺失，跳过本轮监控: {early_stop_state['monitor']}")
        return

    monitor_value = early_stop_state['monitor_value']
    best_monitor = early_stop_state['best_monitor']
    bad_epochs = early_stop_state['bad_epochs']
    print(
        f"早停状态: monitor={early_stop_state['monitor']}, value={monitor_value:.6f}, "
        f"best={best_monitor:.6f}, bad_epochs={bad_epochs}/{early_stop_state['patience']}"
    )

    if early_stop_state['should_stop']:
        print(
            f"触发早停: monitor={early_stop_state['monitor']}, mode={early_stop_state['mode']}, "
            f"patience={early_stop_state['patience']}, best={best_monitor:.6f}"
        )


def build_prior_graph_adjacency(train_data, stockid2idx, runtime_config=None):
    runtime_config = runtime_config or config
    return build_prior_graph_adjacency_shared(train_data, stockid2idx, runtime_config)


def build_stock_industry_index(stockid2idx, runtime_config=None):
    runtime_config = runtime_config or config
    return build_stock_industry_index_shared(stockid2idx, runtime_config)


def load_training_inputs(output_dir, runtime_config=None):
    runtime_config = runtime_config or config
    data_manifest = collect_data_sources(runtime_config, include_csv_stats=True)
    manifest_path = save_data_manifest(output_dir, data_manifest)
    print(f"已生成数据源清单: {manifest_path}")

    factor_pipeline = resolve_factor_pipeline(
        runtime_config['feature_num'],
        runtime_config['factor_store_path'],
        runtime_config['builtin_factor_registry_path'],
    )
    dataset_manifest_train_path, dataset_manifest_info = load_train_dataset_from_build_manifest(
        runtime_config,
        factor_pipeline,
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
        full_df, data_file = load_market_dataset_from_path(runtime_config, dataset_manifest_train_path)
        print(f"训练数据文件(manifest): {data_file}")
    else:
        full_df, data_file = load_market_dataset(runtime_config, 'train.csv')
        print(f"训练数据文件: {data_file}")

    snapshot_path = os.path.join(output_dir, 'active_factors.runtime.json')
    snapshot_info = save_factor_snapshot(factor_pipeline, snapshot_path)
    factor_pipeline['snapshot_meta'] = snapshot_info.get('snapshot', {})
    factor_pipeline['factor_fingerprint'] = snapshot_info.get(
        'factor_fingerprint',
        factor_pipeline.get('factor_fingerprint', ''),
    )
    print(
        f"已保存因子快照: {snapshot_path} | "
        f"fingerprint={factor_pipeline.get('factor_fingerprint', '')}"
    )
    print("当前因子配置:", format_factor_summary(factor_pipeline))
    print_active_factors(factor_pipeline)
    return full_df, factor_pipeline


def resolve_prediction_stock_industry_index(stock_ids, industry_index_path, runtime_config=None):
    runtime_config = runtime_config or config
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

    if stock_industry_idx is not None:
        return stock_industry_idx

    stock_to_industry = load_stock_to_industry_map(
        runtime_config,
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
    return stock_industry_idx.astype(np.int64)
