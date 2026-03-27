import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from config import config
from data_manager import load_market_dataset
from factor_store import load_factor_snapshot
from factor_store import resolve_factor_pipeline
from model import StockTransformer
from train import PortfolioOptimizationLoss
from train import build_rolling_validation_folds
from train import build_strategy_candidates
from train import build_validation_fold_loaders
from train import choose_best_strategy
from train import evaluate_ranking_folds
from train import format_strategy_metric_summary
from train import preprocess_val_data
from train import set_seed
from train import split_train_val_by_last_month
from utils import resolve_feature_indices


def _resolve_device(device_name: str) -> torch.device:
    device_name = str(device_name or "auto").lower()
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def _load_feature_pipeline(output_dir: str):
    snapshot_path = os.path.join(output_dir, "active_factors.json")
    if os.path.exists(snapshot_path):
        pipeline = load_factor_snapshot(snapshot_path)
        print(
            f"加载训练因子快照: {snapshot_path} | "
            f"fingerprint={pipeline.get('factor_fingerprint', '')}"
        )
        return pipeline
    pipeline = resolve_factor_pipeline(
        config["feature_num"],
        config["factor_store_path"],
        config["builtin_factor_registry_path"],
    )
    print(f"未找到训练因子快照，回退当前配置: {config['factor_store_path']}")
    return pipeline


def _load_effective_features(output_dir: str):
    path = os.path.join(output_dir, "effective_features.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"effective_features.json 格式非法: {path}")
    print(f"加载训练特征清单: {path} | 特征数: {len(payload)}")
    return payload


def _load_stock_industry_idx(output_dir: str, num_stocks: int):
    path = os.path.join(output_dir, "stock_industry_idx.npy")
    if not os.path.exists(path):
        return np.full(num_stocks, -1, dtype=np.int64)
    try:
        idx = np.load(path)
        if idx.ndim == 1 and idx.shape[0] == num_stocks:
            print(f"加载行业索引映射: {path}")
            return idx.astype(np.int64)
        print(f"行业索引形状不匹配，回退空映射: {idx.shape} vs ({num_stocks},)")
    except Exception as exc:
        print(f"读取行业索引映射失败，回退空映射: {exc}")
    return np.full(num_stocks, -1, dtype=np.int64)


def _attach_prior_graph_if_available(model, output_dir: str, num_stocks: int):
    prior_graph_path = os.path.join(output_dir, "prior_graph_adj.npy")
    mask_mode = str(config.get("cross_stock_mask_mode", "similarity")).lower()
    if os.path.exists(prior_graph_path):
        try:
            prior_graph = np.load(prior_graph_path)
            if prior_graph.ndim == 2 and prior_graph.shape == (num_stocks, num_stocks):
                model.set_prior_graph(torch.from_numpy(prior_graph.astype(np.bool_)))
                print(f"加载先验图邻接矩阵: {prior_graph_path}")
                return
            print(
                "先验图形状不匹配，忽略: "
                f"{prior_graph.shape} vs ({num_stocks}, {num_stocks})"
            )
        except Exception as exc:
            print(f"加载先验图失败，忽略: {exc}")
    if mask_mode in {"prior", "prior_similarity"}:
        print("未找到可用先验图，回退为 similarity 稀疏注意力。")
        model.cross_stock_attention.mask_mode = "similarity"


def _load_checkpoint_with_compat(model, model_path: str, device: torch.device):
    state_dict = torch.load(model_path, map_location=device)
    load_result = model.load_state_dict(state_dict, strict=False)

    multi_scale_prefixes = (
        "ultra_short_temporal_encoder.",
        "ultra_short_feature_attention.",
        "short_temporal_encoder.",
        "long_temporal_encoder.",
        "short_feature_attention.",
        "long_feature_attention.",
        "short_horizon_fusion_gate.",
        "short_horizon_norm.",
        "multi_scale_fusion_gate.",
        "multi_scale_branch_norm.",
    )
    temporal_cross_stock_prefixes = ("temporal_cross_stock_attention.",)
    compatible_prefixes = (
        "market_gate.",
        "market_macro_proj.",
        "volatility_head.",
    ) + multi_scale_prefixes + temporal_cross_stock_prefixes
    compatible_exact_keys = {"multi_scale_branch_logits"}

    non_compatible_missing = [
        k
        for k in load_result.missing_keys
        if (k not in compatible_exact_keys) and (not k.startswith(compatible_prefixes))
    ]
    if non_compatible_missing:
        raise RuntimeError(f"模型参数缺失且无法兼容: {non_compatible_missing[:10]}")

    if any(k.startswith("market_gate.") for k in load_result.missing_keys):
        print("检测到旧版 checkpoint（缺少 market_gate 参数），关闭 market gating")
        model.use_market_gating = False
    if any(k.startswith("market_macro_proj.") for k in load_result.missing_keys):
        print("检测到旧版 checkpoint（缺少 market macro 参数），关闭宏观 gate 输入")
        model.use_market_gating_macro_context = False
    if any(k.startswith("volatility_head.") for k in load_result.missing_keys):
        print("检测到旧版 checkpoint（缺少 volatility_head 参数），关闭多任务辅助头")
        model.use_multitask_volatility = False
        config["use_multitask_volatility"] = False
    if any(
        (k == "multi_scale_branch_logits") or k.startswith(multi_scale_prefixes)
        for k in load_result.missing_keys
    ):
        print("检测到旧版 checkpoint（缺少 multi-scale 参数），关闭多尺度分支")
        model.use_multi_scale_temporal = False
    if any(k.startswith(temporal_cross_stock_prefixes) for k in load_result.missing_keys):
        print("检测到旧版 checkpoint（缺少 temporal cross-stock 参数），关闭时间步级跨股交互")
        model.use_temporal_cross_stock_attention = False
    if load_result.unexpected_keys:
        print(f"模型包含额外参数（将忽略）: {load_result.unexpected_keys[:10]}")


def _build_criterion():
    return PortfolioOptimizationLoss(
        temperature=float(config.get("loss_temperature", 10.0)),
        listnet_weight=float(config.get("listnet_weight", 1.0)),
        pairwise_weight=float(config.get("pairwise_weight", 1.0)),
        lambda_ndcg_weight=float(config.get("lambda_ndcg_weight", 1.0)),
        lambda_ndcg_topk=int(config.get("lambda_ndcg_topk", 50)),
        ic_weight=float(config.get("ic_weight", 0.0)),
        ic_mode=str(config.get("ic_mode", "pearson")),
    )


def _dump_strategy(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)
    print(f"已写入策略文件: {path}")


def main():
    parser = argparse.ArgumentParser(description="基于验证集重选 best strategy（不重训模型）")
    parser.add_argument("--output-dir", default=config["output_dir"], help="模型产物目录")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/mps")
    parser.add_argument("--apply", action="store_true", help="将重选结果覆盖写入 best_strategy.json")
    args = parser.parse_args()

    set_seed(config.get("seed", 42))
    output_dir = os.path.abspath(args.output_dir)
    model_path = os.path.join(output_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    full_df, data_file = load_market_dataset(config, "train.csv")
    print(f"重选策略数据文件: {data_file}")

    validation_mode = str(config.get("validation_mode", "rolling")).lower()
    if validation_mode == "rolling":
        _, val_df, val_folds = build_rolling_validation_folds(full_df, config["sequence_length"])
    else:
        _, val_df, val_start = split_train_val_by_last_month(full_df, config["sequence_length"])
        val_end = pd.to_datetime(val_df["日期"]).max().normalize()
        val_folds = [
            {
                "name": "holdout",
                "start_date": pd.Timestamp(val_start).normalize(),
                "end_date": pd.Timestamp(val_end).normalize(),
            }
        ]
    stock_ids = sorted(full_df["股票代码"].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    feature_pipeline = _load_feature_pipeline(output_dir)
    val_data, features = preprocess_val_data(val_df, feature_pipeline, stockid2idx=stockid2idx)

    saved_features = _load_effective_features(output_dir)
    if saved_features is not None:
        missing = [name for name in saved_features if name not in val_data.columns]
        if missing:
            raise ValueError(f"验证输入缺少训练特征: {missing[:10]}")
        features = saved_features

    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    val_data = val_data.dropna(subset=features).copy()
    val_data[features] = val_data[features].astype(np.float32)

    fold_loaders = build_validation_fold_loaders(val_data, features, val_folds)

    device = _resolve_device(args.device)
    if device.type == "cuda":
        print(f"当前设备: cuda ({torch.cuda.get_device_name(device)})")
    else:
        print(f"当前设备: {device}")

    model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stockid2idx))
    market_context_feature_names = config.get(
        "market_gating_context_feature_names",
        [
            "market_median_return",
            "market_total_turnover_log",
            "market_limit_up_count_log",
            "market_limit_up_ratio",
        ],
    )
    market_context_indices = resolve_feature_indices(features, market_context_feature_names)
    model.set_market_context_feature_indices(market_context_indices)
    stock_industry_idx = _load_stock_industry_idx(output_dir, len(stockid2idx))
    model.set_stock_industry_index(torch.from_numpy(stock_industry_idx))
    _attach_prior_graph_if_available(model, output_dir, len(stockid2idx))
    _load_checkpoint_with_compat(model, model_path, device)
    model.to(device)
    model.eval()

    criterion = _build_criterion()
    strategy_candidates = build_strategy_candidates()
    print("候选持仓策略:", ", ".join(candidate["name"] for candidate in strategy_candidates))

    eval_loss, eval_metrics, fold_results = evaluate_ranking_folds(
        model,
        fold_loaders,
        criterion,
        device,
        writer=None,
        epoch=0,
        strategy_candidates=strategy_candidates,
    )

    print(f"重选评估 Loss: {eval_loss:.6f}")
    print("策略收益汇总:", format_strategy_metric_summary(eval_metrics, strategy_candidates))

    best_candidate, best_score = choose_best_strategy(eval_metrics, strategy_candidates)
    best_candidate_return = eval_metrics.get(
        f"return_{best_candidate['name']}",
        best_score,
    )
    print(
        f"验证集最优策略: {best_candidate['name']} | "
        f"objective={best_score:.6f} | mean_return={best_candidate_return:.6f} | "
        f"rank_ic_mean={eval_metrics.get('rank_ic_mean', 0.0):.6f}"
    )

    for fold_result in fold_results:
        fold_best_candidate, fold_best_score = choose_best_strategy(
            fold_result["metrics"],
            strategy_candidates,
        )
        print(
            f"  - {fold_result['name']} "
            f"({str(fold_result['start_date'])[:10]} ~ {str(fold_result['end_date'])[:10]}): "
            f"best={fold_best_candidate['name']}:{fold_best_score:.6f}"
        )

    strategy_payload = {
        "name": best_candidate["name"],
        "top_k": int(best_candidate["top_k"]),
        "weighting": best_candidate["weighting"],
        "temperature": float(config.get("softmax_temperature", 1.0)),
        "validation_objective": float(best_score),
        "validation_return": float(best_candidate_return),
        "validation_mode": validation_mode,
        "strategy_selection_mode": config.get("strategy_selection_mode", "risk_adjusted"),
        "strategy_risk_lambda": float(config.get("strategy_risk_lambda", 0.2)),
        "rank_ic_mean": float(eval_metrics.get("rank_ic_mean", 0.0)),
        "rank_ic_ir": float(eval_metrics.get("rank_ic_ir", 0.0)),
        "validation_folds": [
            {
                "name": fold.get("name", ""),
                "start_date": str(fold.get("start_date", ""))[:10],
                "end_date": str(fold.get("end_date", ""))[:10],
            }
            for fold in val_folds
        ],
        "reselected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "validation_reselect",
    }

    reselect_path = os.path.join(output_dir, "best_strategy_reselected.json")
    _dump_strategy(reselect_path, strategy_payload)
    if args.apply:
        best_path = os.path.join(output_dir, "best_strategy.json")
        _dump_strategy(best_path, strategy_payload)


if __name__ == "__main__":
    main()
