import json
import os

import joblib
import numpy as np
import pandas as pd
import torch

from config import config
from experiments.runner import build_strategy_export_payload
from factor_store import save_factor_snapshot


def format_factor_summary(feature_pipeline):
    summary = feature_pipeline['summary']
    group_parts = [
        f'{group}={count}'
        for group, count in sorted(summary['group_counts'].items())
    ]
    return (
        f"feature_set={feature_pipeline['feature_set']}, "
        f"active={summary['active_total']}, "
        f"cross_sectional={summary.get('cross_sectional_total', 0)}, "
        f"builtin={summary['builtin_enabled']}/{summary['builtin_total']}, "
        f"builtin_overridden={summary.get('builtin_overridden', 0)}, "
        f"custom={summary['custom_enabled']}/{summary['custom_total']}, "
        f"groups=({', '.join(group_parts)})"
    )


def print_active_factors(feature_pipeline):
    grouped_specs = {}
    for spec in feature_pipeline['active_specs']:
        group = spec.get('group', 'unknown')
        label = spec['name']
        if spec.get('source') == 'custom':
            label = f'{label} [custom]'
        elif spec.get('overridden'):
            label = f'{label} [override]'
        grouped_specs.setdefault(group, []).append(label)

    print("当前启用因子明细:")
    for group, factor_names in sorted(grouped_specs.items()):
        print(f"  - {group} ({len(factor_names)}):")
        print("    " + ", ".join(factor_names))


def _build_factor_markdown(feature_pipeline):
    summary = feature_pipeline['summary']
    lines = [
        f"- feature_set: `{feature_pipeline['feature_set']}`",
        f"- factor_store: `{feature_pipeline['store_path']}`",
        f"- builtin_registry: `{feature_pipeline.get('builtin_registry_path', '')}`",
        f"- factor_fingerprint: `{feature_pipeline.get('factor_fingerprint', '')}`",
        f"- snapshot_created_at: `{feature_pipeline.get('snapshot_meta', {}).get('created_at', '')}`",
        f"- active_total: `{summary['active_total']}`",
        f"- builtin_enabled: `{summary['builtin_enabled']}/{summary['builtin_total']}`",
        f"- builtin_overridden: `{summary.get('builtin_overridden', 0)}`",
        f"- custom_enabled: `{summary['custom_enabled']}/{summary['custom_total']}`",
        f"- cross_sectional_total: `{summary.get('cross_sectional_total', 0)}`",
        f"- groups: `{json.dumps(summary['group_counts'], ensure_ascii=False)}`",
        "",
        "Active factors:",
        ", ".join(feature_pipeline['active_features']),
    ]
    if feature_pipeline['custom_specs']:
        lines.extend([
            "",
            "Custom factors:",
            json.dumps(feature_pipeline['custom_specs'], ensure_ascii=False, indent=2),
        ])
    return "\n".join(lines)


def log_factor_dashboard(writer, feature_pipeline, raw_hist_frame, scaled_hist_frame):
    if writer is None:
        return

    summary = feature_pipeline['summary']
    writer.add_text('factors/overview', _build_factor_markdown(feature_pipeline), global_step=0)
    writer.add_scalar('factors/active_total', summary['active_total'], global_step=0)
    writer.add_scalar('factors/builtin_enabled', summary['builtin_enabled'], global_step=0)
    writer.add_scalar('factors/builtin_overridden', summary.get('builtin_overridden', 0), global_step=0)
    writer.add_scalar('factors/custom_enabled', summary['custom_enabled'], global_step=0)

    for group, count in sorted(summary['group_counts'].items()):
        writer.add_scalar(f'factors/group_count/{group}', count, global_step=0)

    max_histograms = max(0, int(config.get('factor_histogram_max_features', 0)))
    if raw_hist_frame is None or scaled_hist_frame is None:
        return

    for feature_name in raw_hist_frame.columns[:max_histograms]:
        raw_values = raw_hist_frame[feature_name].to_numpy(dtype=np.float32, copy=True)
        scaled_values = scaled_hist_frame[feature_name].to_numpy(dtype=np.float32, copy=True)
        writer.add_histogram(f'factors/raw/{feature_name}', raw_values, global_step=0)
        writer.add_histogram(f'factors/scaled/{feature_name}', scaled_values, global_step=0)


def print_validation_run_summary(eval_loss, eval_metrics, run_summary):
    print(f"Eval Loss: {eval_loss:.4f}")
    for key, value in eval_metrics.items():
        print(f"Eval {key}: {value:.4f}")

    print("Eval 策略收益汇总: " + run_summary['strategy_summary'])
    print("Eval 策略对比:")
    for strategy_row in run_summary['strategy_comparison']:
        print(
            f"  - {strategy_row['name']}: "
            f"mean={strategy_row['mean_return']:.4f}, "
            f"std={strategy_row['return_std']:.4f}, "
            f"ra={strategy_row['risk_adjusted_return']:.4f}"
        )

    for fold_result in run_summary['fold_diagnostics']:
        fold_strategy_summary = ', '.join(
            (
                f"{row['name']}=mean:{row['mean_return']:.4f}"
                f"|std:{row['return_std']:.4f}"
                f"|ra:{row['risk_adjusted_return']:.4f}"
            )
            for row in fold_result['strategy_comparison']
        )
        print(
            f"Eval {fold_result['name']} "
            f"({fold_result['start_date'].date()} ~ {fold_result['end_date'].date()}) "
            f"样本数: {fold_result['num_samples']} | "
            f"Loss: {fold_result['loss']:.4f} | "
            f"best={fold_result['best_candidate']['name']}:{fold_result['best_score']:.4f}"
        )
        print("  策略收益: " + fold_strategy_summary)

    regime_summary = run_summary['regime_summary']
    print(
        "Eval Regime 摘要: "
        f"dominant={regime_summary['dominant_strategy']}, "
        f"positive={regime_summary['positive_return_fold_count']}, "
        f"negative={regime_summary['negative_return_fold_count']}, "
        f"flat={regime_summary['flat_return_fold_count']}"
    )


def print_best_strategy_summary(run_summary, eval_metrics):
    best_candidate = run_summary['best_candidate']
    current_final_score = run_summary['best_score']
    best_candidate_return = run_summary['best_return']
    print(
        f"当前最优持仓策略: {best_candidate['name']} | "
        f"验证目标值: {current_final_score:.4f} | "
        f"策略收益均值: {best_candidate_return:.4f} | "
        f"RankIC: {eval_metrics.get('rank_ic_mean', 0.0):.4f}"
    )


def print_train_epoch_summary(train_loss, train_metrics):
    print(f"Train Loss: {train_loss:.4f}")
    for key, value in train_metrics.items():
        print(f"Train {key}: {value:.4f}")


def print_best_checkpoint_summary(best_score):
    print(f"保存最佳模型 - objective: {best_score:.4f}")


def print_training_completion_summary(best_epoch, best_score):
    print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 objective: {best_score:.4f}")


def print_training_exit_banner(best_score):
    print(f"\n########## 训练完成！最佳 objective: {best_score:.4f} ##########")


def save_effective_feature_list(output_dir, features):
    feature_path = os.path.join(output_dir, 'effective_features.json')
    with open(feature_path, 'w', encoding='utf-8') as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    print(f'已保存训练特征清单: {feature_path} | 特征数: {len(features)}')
    return feature_path


def save_identity_scaler_artifact(output_dir):
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump({'type': 'identity', 'name': 'cross_sectional_only'}, scaler_path)
    print('已固定为截面标准化，禁用全局 StandardScaler。')
    return scaler_path


def save_prior_graph_artifact(output_dir, prior_graph_adj):
    prior_graph_path = os.path.join(output_dir, 'prior_graph_adj.npy')
    np.save(prior_graph_path, prior_graph_adj.astype(np.uint8))
    print(f"已保存先验图邻接矩阵: {prior_graph_path}")
    return prior_graph_path


def save_industry_index_artifacts(output_dir, stock_industry_idx, industry_vocab):
    industry_index_path = os.path.join(output_dir, 'stock_industry_idx.npy')
    industry_vocab_path = os.path.join(output_dir, 'industry_vocab.json')
    np.save(industry_index_path, stock_industry_idx.astype(np.int64))
    with open(industry_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(industry_vocab, f, ensure_ascii=False, indent=2)
    print(f"已保存行业索引映射: {industry_index_path} (行业数={len(industry_vocab)})")
    return industry_index_path, industry_vocab_path


def prepare_training_artifact_frames(
    train_data,
    val_data,
    features,
    output_dir,
    runtime_config=None,
):
    runtime_config = runtime_config or config
    train_data = train_data.copy()
    val_data = val_data.copy()

    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)

    dump_factor_artifacts('train', train_data, features, output_dir)
    dump_factor_artifacts('val', val_data, features, output_dir)

    histogram_features = features[:max(0, int(runtime_config.get('factor_histogram_max_features', 0)))]
    raw_train_hist_frame = train_data[histogram_features].copy() if histogram_features else None

    train_data[features] = train_data[features].astype(np.float32)
    val_data[features] = val_data[features].astype(np.float32)
    scaled_train_hist_frame = train_data[histogram_features] if histogram_features else None
    return train_data, val_data, raw_train_hist_frame, scaled_train_hist_frame


def _feature_stats_frame(df, feature_columns):
    if not feature_columns:
        return pd.DataFrame(columns=['feature', 'mean', 'std', 'min', 'max', 'na_ratio'])

    feature_df = df[feature_columns]
    stats = pd.DataFrame({
        'feature': feature_columns,
        'mean': feature_df.mean(axis=0, skipna=True).values,
        'std': feature_df.std(axis=0, skipna=True).values,
        'min': feature_df.min(axis=0, skipna=True).values,
        'max': feature_df.max(axis=0, skipna=True).values,
        'na_ratio': feature_df.isna().mean(axis=0).values,
    })
    return stats


def dump_factor_artifacts(split_name, df, feature_columns, output_dir):
    if not bool(config.get('dump_factor_artifacts', True)):
        return
    if df is None or len(df) == 0:
        return

    artifact_dir = os.path.join(output_dir, 'factor_artifacts')
    os.makedirs(artifact_dir, exist_ok=True)
    max_rows = int(config.get('factor_artifact_max_rows', 100000))
    max_rows = max(0, max_rows)

    base_cols = [
        col for col in ['日期', '股票代码', 'instrument', 'label', 'label_raw', 'vol_label', 'vol_label_raw']
        if col in df.columns
    ]
    export_cols = base_cols + [col for col in feature_columns if col in df.columns]

    export_df = df[export_cols].copy()
    if max_rows > 0 and len(export_df) > max_rows:
        export_df = (
            export_df.sample(n=max_rows, random_state=42)
            .sort_values([col for col in ['日期', '股票代码'] if col in export_df.columns])
            .reset_index(drop=True)
        )

    values_path = os.path.join(artifact_dir, f'{split_name}_factor_values.csv')
    export_df.to_csv(values_path, index=False, encoding='utf-8')

    stats_path = os.path.join(artifact_dir, f'{split_name}_factor_stats.csv')
    if bool(config.get('factor_artifact_include_full_feature_stats', True)):
        stats_df = _feature_stats_frame(df, [col for col in feature_columns if col in df.columns])
        stats_df.to_csv(stats_path, index=False, encoding='utf-8')

    meta = {
        'split': split_name,
        'rows_total': int(len(df)),
        'rows_exported': int(len(export_df)),
        'feature_count': int(len(feature_columns)),
        'feature_count_present': int(sum(1 for col in feature_columns if col in df.columns)),
        'values_path': values_path,
        'stats_path': stats_path if bool(config.get('factor_artifact_include_full_feature_stats', True)) else '',
    }
    meta_path = os.path.join(artifact_dir, f'{split_name}_factor_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(
        f'已导出 {split_name} 因子结果: values={values_path}, '
        f'rows={meta["rows_exported"]}/{meta["rows_total"]}, features={meta["feature_count_present"]}'
    )


def save_best_training_artifacts(
    model,
    output_dir,
    run_summary,
    validation_folds,
    runtime_config,
    best_epoch,
    features=None,
    factor_pipeline=None,
):
    model_path = os.path.join(output_dir, 'best_model.pth')
    strategy_path = os.path.join(output_dir, 'best_strategy.json')

    torch.save(model.state_dict(), model_path)
    with open(strategy_path, 'w', encoding='utf-8') as f:
        json.dump(
            build_strategy_export_payload(
                run_summary=run_summary,
                validation_folds=validation_folds,
                runtime_config=runtime_config,
                source='training_validation',
                exported_at_field='saved_at',
                best_epoch=best_epoch,
            ),
            f,
            indent=4,
            ensure_ascii=False,
        )
    if features is not None:
        save_effective_feature_list(output_dir, features)
    if factor_pipeline is not None:
        save_factor_snapshot(factor_pipeline, os.path.join(output_dir, 'active_factors.json'))
    return model_path, strategy_path


def write_final_training_score(output_dir, best_epoch, best_score):
    final_score_path = os.path.join(output_dir, 'final_score.txt')
    with open(final_score_path, 'w', encoding='utf-8') as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best objective: {best_score:.6f}\n")
    return final_score_path
