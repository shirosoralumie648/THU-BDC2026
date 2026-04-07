import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import config
from experiments.metrics import build_portfolio_weights
from experiments.metrics import build_strategy_candidates
from experiments.metrics import calculate_ranking_metrics
from experiments.metrics import choose_best_strategy
from experiments.metrics import format_strategy_metric_summary
from experiments.runner import summarize_experiment_run
from experiments.splits import build_rolling_validation_folds as build_rolling_validation_folds_shared
from features.feature_assembler import augment_feature_table
from features.feature_assembler import build_feature_table
from graph.graph_builder import build_prior_graph_adjacency as build_prior_graph_adjacency_shared
from graph.industry_graph import build_stock_industry_index as build_stock_industry_index_shared
from objectives.ranking_loss import build_portfolio_optimization_loss
from objectives.ranking_loss import PortfolioOptimizationLoss
from objectives.target_transforms import transform_targets_for_loss
from training.artifacts import dump_factor_artifacts
from training.artifacts import format_factor_summary
from training.artifacts import log_factor_dashboard
from training.artifacts import print_active_factors
from training.artifacts import print_best_checkpoint_summary
from training.artifacts import print_best_strategy_summary
from training.artifacts import print_train_epoch_summary
from training.artifacts import print_training_completion_summary
from training.artifacts import print_training_exit_banner
from training.artifacts import print_validation_run_summary
from training.artifacts import prepare_training_artifact_frames
from training.artifacts import save_effective_feature_list
from training.artifacts import save_identity_scaler_artifact
from training.artifacts import save_industry_index_artifacts
from training.artifacts import save_prior_graph_artifact
from training.artifacts import save_best_training_artifacts
from training.artifacts import write_final_training_score
from training.datasets import RankingDataset
from training.datasets import LazyRankingDataset
from training.datasets import build_lazy_ranking_index
from training.datasets import collate_fn
from training.inference import predict_top_stocks
from training.inference import save_predictions
from training.loops import evaluate_ranking_folds
from training.loops import evaluate_ranking_model
from training.loops import train_ranking_model
from training.preprocessing import preprocess_data
from training.preprocessing import preprocess_val_data
from training.runtime import build_early_stopping_state
from training.runtime import build_optimizer_scheduler_scaler
from training.runtime import build_prior_graph_adjacency
from training.runtime import build_rank_model
from training.runtime import build_stock_industry_index
from training.runtime import initialize_training_runtime
from training.runtime import load_training_inputs
from training.runtime import print_early_stopping_summary
from training.runtime import set_seed
from training.runtime import update_early_stopping_state
from training.validation import build_factor_group_indices
from training.validation import build_validation_fold_loaders
from training.validation import evaluate_factor_group_ablation
from training.validation import log_factor_ablation
from training.validation import print_factor_ablation_summary
from training.validation import resolve_training_validation_split
from training.validation import split_train_val_by_last_month
import os
import json
import multiprocessing as mp


def build_rolling_validation_folds(df, sequence_length):
    return build_rolling_validation_folds_shared(df, sequence_length, config)

# 主程序
def main():
    set_seed(config.get('seed', 42))
    output_dir = config['output_dir']
    is_train = True
    writer, device = initialize_training_runtime(output_dir, config, is_train=is_train)
    
    full_df, factor_pipeline = load_training_inputs(output_dir, config)
    train_df, val_df, val_folds = resolve_training_validation_split(
        full_df,
        config['sequence_length'],
        runtime_config=config,
    )
    
    # 获取所有股票ID，建立映射
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)
    
    # 2. 特征工程与预处理
    train_data, features = preprocess_data(train_df, factor_pipeline, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, factor_pipeline, stockid2idx=stockid2idx)
    
    # 3. 特征缩放（默认仅保留截面标准化，不做全局 StandardScaler）
    train_data, val_data, raw_train_hist_frame, scaled_train_hist_frame = prepare_training_artifact_frames(
        train_data,
        val_data,
        features,
        output_dir,
        runtime_config=config,
    )

    # 关键修正：仅按日截面标准化（已在 preprocess_* 中完成），
    # 这里明确禁用全局拟合缩放，避免时序泄露并保留日内相对强弱。
    save_identity_scaler_artifact(output_dir)
    log_factor_dashboard(writer, factor_pipeline, raw_train_hist_frame, scaled_train_hist_frame)

    
    # 4. 创建排序数据集
    train_stock_cache, train_day_entries = build_lazy_ranking_index(
        train_data,
        features,
        config['sequence_length'],
    )
    print(f"训练集样本数: {len(train_day_entries)}")
    val_fold_loaders = build_validation_fold_loaders(
        val_data,
        features,
        val_folds,
        runtime_config=config,
    )

    mask_mode = str(config.get('cross_stock_mask_mode', 'similarity')).lower()
    need_prior_graph = bool(config.get('use_cross_stock_attention_mask', True)) and mask_mode in {
        'prior',
        'prior_similarity',
    }
    prior_graph_adj = None
    if need_prior_graph:
        prior_graph_adj = build_prior_graph_adjacency(train_data, stockid2idx)
        save_prior_graph_artifact(output_dir, prior_graph_adj)

    use_industry_virtual = bool(config.get('use_industry_virtual_stock', False))
    use_industry_virtual_temporal = bool(config.get('industry_virtual_on_temporal_cross_stock', False))
    stock_industry_idx = np.full(num_stocks, -1, dtype=np.int64)
    if use_industry_virtual or use_industry_virtual_temporal:
        stock_industry_idx, industry_vocab = build_stock_industry_index(stockid2idx)
        save_industry_index_artifacts(output_dir, stock_industry_idx, industry_vocab)
    
    # 5. 创建排序数据集和数据加载器
    train_dataset = LazyRankingDataset(train_stock_cache, train_day_entries, config['sequence_length'])
    del train_data
    del val_data
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # 减少worker数量避免内存问题
        pin_memory=False
    )
    
    # 6. 模型初始化
    model = build_rank_model(
        features,
        num_stocks,
        stock_industry_idx,
        config,
        prior_graph_adj=prior_graph_adj,
    )
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    strategy_candidates = build_strategy_candidates()
    print("候选持仓策略:", ", ".join(candidate['name'] for candidate in strategy_candidates))
    
    # 7. 损失函数和优化器
    criterion = build_portfolio_optimization_loss(config)
    optimizer, scheduler, use_amp, scaler = build_optimizer_scheduler_scaler(model, device, config)
    
    # 8. 排序模型训练
    if is_train:
        early_stop_state = build_early_stopping_state(config)
        best_score = early_stop_state['best_score']
        best_epoch = early_stop_state['best_epoch']
        
        for epoch in range(config['num_epochs']):
            print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")
            
            # 训练
            train_loss, train_metrics = train_ranking_model(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                writer,
                strategy_candidates,
                use_amp=use_amp,
                scaler=scaler,
            )
            print_train_epoch_summary(train_loss, train_metrics)
            
            # 验证
            eval_loss, eval_metrics, fold_results = evaluate_ranking_folds(
                model, val_fold_loaders, criterion, device, writer, epoch, strategy_candidates
            )
            run_summary = summarize_experiment_run(
                eval_loss=eval_loss,
                eval_metrics=eval_metrics,
                fold_results=fold_results,
                strategy_candidates=strategy_candidates,
                runtime_config=config,
            )
            print_validation_run_summary(eval_loss, eval_metrics, run_summary)
            
            # 学习率调度
            scheduler.step()
            if writer:
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step=epoch)
            

            best_candidate = run_summary['best_candidate']
            current_final_score = run_summary['best_score']
            print_best_strategy_summary(run_summary, eval_metrics)

            if config.get('factor_ablation_enabled', True):
                ablation_results = evaluate_factor_group_ablation(
                    model,
                    val_fold_loaders,
                    criterion,
                    device,
                    epoch,
                    strategy_candidates,
                    factor_pipeline,
                    best_candidate,
                    evaluate_ranking_folds_fn=evaluate_ranking_folds,
                    runtime_config=config,
                )
                log_factor_ablation(writer, epoch, current_final_score, ablation_results)
                print_factor_ablation_summary(current_final_score, ablation_results)

            if current_final_score > best_score:
                best_score = current_final_score
                best_epoch = epoch + 1
                save_best_training_artifacts(
                    model,
                    output_dir,
                    run_summary,
                    val_folds,
                    config,
                    best_epoch,
                    features=features,
                    factor_pipeline=factor_pipeline,
                )
                print_best_checkpoint_summary(best_score)

            early_stop_state = update_early_stopping_state(
                eval_metrics,
                early_stop_state,
                epoch=epoch,
                writer=writer,
            )
            print_early_stopping_summary(early_stop_state)
            if early_stop_state['missing_monitor']:
                continue

            if early_stop_state['should_stop']:
                break
        print_training_completion_summary(best_epoch, best_score)
        write_final_training_score(output_dir, best_epoch, best_score)

        if writer:
            writer.close()

        return best_score

if __name__ == "__main__":
    # 多进程保护
    mp.set_start_method('spawn', force=True)
    best_score = main()
    print_training_exit_banner(best_score)
