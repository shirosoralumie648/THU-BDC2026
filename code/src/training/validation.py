import pandas as pd
from torch.utils.data import DataLoader

from config import config
from experiments.splits import build_rolling_validation_folds as build_rolling_validation_folds_shared
from training.datasets import RankingDataset
from training.datasets import collate_fn
from utils import create_ranking_dataset_vectorized


def build_factor_group_indices(feature_pipeline):
    group_indices = {}
    for feature_idx, spec in enumerate(feature_pipeline['active_specs']):
        group = spec.get('group', 'unknown')
        group_indices.setdefault(group, []).append(feature_idx)
    return group_indices


def evaluate_factor_group_ablation(
    model,
    fold_loaders,
    criterion,
    device,
    epoch,
    strategy_candidates,
    feature_pipeline,
    baseline_candidate,
    evaluate_ranking_folds_fn,
    runtime_config=None,
):
    runtime_config = runtime_config or config
    group_indices = build_factor_group_indices(feature_pipeline)
    ablation_results = []
    selection_mode = str(runtime_config.get('strategy_selection_mode', 'risk_adjusted')).lower()
    if selection_mode == 'risk_adjusted':
        baseline_metric_name = f'return_{baseline_candidate["name"]}_risk_adjusted'
    else:
        baseline_metric_name = f'return_{baseline_candidate["name"]}'

    for group_name, feature_indices in sorted(group_indices.items()):
        ablation_loss, ablation_metrics, _ = evaluate_ranking_folds_fn(
            model,
            fold_loaders,
            criterion,
            device,
            writer=None,
            epoch=epoch,
            strategy_candidates=strategy_candidates,
            ablation_feature_indices=feature_indices,
        )

        ablated_return = ablation_metrics.get(baseline_metric_name, 0.0)
        ablation_results.append({
            'group': group_name,
            'num_features': len(feature_indices),
            'loss': ablation_loss,
            'return': ablated_return,
            'metrics': ablation_metrics,
        })

    return ablation_results


def log_factor_ablation(writer, epoch, baseline_return, ablation_results):
    if writer is None:
        return

    writer.add_scalar('factors/ablation/baseline_return', baseline_return, global_step=epoch)
    for result in ablation_results:
        group = result['group']
        delta = result['return'] - baseline_return
        writer.add_scalar(f'factors/ablation/{group}/return', result['return'], global_step=epoch)
        writer.add_scalar(f'factors/ablation/{group}/delta', delta, global_step=epoch)
        writer.add_scalar(f'factors/ablation/{group}/num_features', result['num_features'], global_step=epoch)


def print_factor_ablation_summary(baseline_return, ablation_results):
    print("因子分组消融:")
    for result in ablation_results:
        delta = result['return'] - baseline_return
        print(
            f"  - {result['group']}: "
            f"features={result['num_features']}, "
            f"return={result['return']:.4f}, "
            f"delta={delta:.4f}"
        )


def resolve_training_validation_split(full_df, sequence_length, runtime_config=None):
    runtime_config = runtime_config or config
    validation_mode = runtime_config.get('validation_mode', 'rolling')
    if validation_mode == 'rolling':
        return build_rolling_validation_folds_shared(full_df, sequence_length, runtime_config)

    train_df, val_df, val_start = split_train_val_by_last_month(full_df, sequence_length)
    val_folds = [{
        'name': 'holdout',
        'start_date': pd.Timestamp(val_start).normalize(),
        'end_date': pd.to_datetime(val_df['日期']).max().normalize(),
    }]
    return train_df, val_df, val_folds


def split_train_val_by_last_month(df, sequence_length):
    """按最后一个月做验证集划分，并为验证集补充序列上下文。"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=2)).normalize()

    # 验证集需要保留前 sequence_length-1 个交易日作为序列上下文，
    # 这样第一个验证样本的窗口结束日就可以落在 val_start。
    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围(最后一个月): {val_start.date()} 到 {last_date.date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    # 恢复为字符串，保持与原流程一致
    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start


def build_validation_fold_loaders(
    val_data,
    features,
    val_folds,
    runtime_config=None,
    create_ranking_dataset_fn=create_ranking_dataset_vectorized,
    dataset_cls=RankingDataset,
    collate_fn_impl=collate_fn,
    data_loader_cls=DataLoader,
):
    """为每个滚动验证折构建独立的数据集与 DataLoader。"""
    runtime_config = runtime_config or config
    fold_loaders = []
    total_samples = 0

    for fold in val_folds:
        sequences, targets, relevance, stock_indices, vol_targets = create_ranking_dataset_fn(
            val_data,
            features,
            runtime_config['sequence_length'],
            ranking_data_path=None,
            min_window_end_date=fold['start_date'].strftime('%Y-%m-%d'),
            max_window_end_date=fold['end_date'].strftime('%Y-%m-%d'),
        )

        if len(sequences) == 0:
            raise ValueError(
                f"{fold['name']} ({fold['start_date'].date()} ~ {fold['end_date'].date()}) 未生成任何验证样本"
            )

        dataset = dataset_cls(
            sequences,
            targets,
            relevance,
            stock_indices,
            vol_targets=vol_targets,
        )
        loader = data_loader_cls(
            dataset,
            batch_size=runtime_config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn_impl,
            num_workers=0,
            pin_memory=False,
        )

        fold_loaders.append({
            'name': fold['name'],
            'start_date': fold['start_date'],
            'end_date': fold['end_date'],
            'num_samples': len(sequences),
            'loader': loader,
        })
        total_samples += len(sequences)

        print(
            f"验证折 {fold['name']} 样本数: {len(sequences)} "
            f"({fold['start_date'].date()} ~ {fold['end_date'].date()})"
        )

    print(f"滚动验证总样本数: {total_samples}")
    return fold_loaders
