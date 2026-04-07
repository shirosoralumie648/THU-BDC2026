import json
import os
import sys
import unittest
import tempfile

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import train
from objectives.ranking_loss import build_portfolio_optimization_loss as build_portfolio_optimization_loss_shared
from objectives.ranking_loss import PortfolioOptimizationLoss as SharedPortfolioOptimizationLoss

try:
    from objectives.target_transforms import mad_clip_bounds
    from objectives.target_transforms import rank_normalize_tensor
    from objectives.target_transforms import transform_targets_for_loss
except ModuleNotFoundError as exc:
    TARGET_TRANSFORM_IMPORT_ERROR = exc
    mad_clip_bounds = None
    rank_normalize_tensor = None
    transform_targets_for_loss = None
else:
    TARGET_TRANSFORM_IMPORT_ERROR = None

try:
    from features.feature_assembler import augment_feature_table
    from features.feature_assembler import build_feature_table
except ImportError as exc:
    FEATURE_ASSEMBLER_IMPORT_ERROR = exc
    augment_feature_table = None
    build_feature_table = None
else:
    FEATURE_ASSEMBLER_IMPORT_ERROR = None

try:
    from training.artifacts import dump_factor_artifacts as dump_factor_artifacts_shared
    from training.artifacts import format_factor_summary as format_factor_summary_shared
    from training.artifacts import log_factor_dashboard as log_factor_dashboard_shared
    from training.artifacts import print_best_checkpoint_summary as print_best_checkpoint_summary_shared
    from training.artifacts import print_best_strategy_summary as print_best_strategy_summary_shared
    from training.artifacts import print_train_epoch_summary as print_train_epoch_summary_shared
    from training.artifacts import print_training_completion_summary as print_training_completion_summary_shared
    from training.artifacts import print_training_exit_banner as print_training_exit_banner_shared
    from training.artifacts import print_validation_run_summary as print_validation_run_summary_shared
    from training.artifacts import prepare_training_artifact_frames as prepare_training_artifact_frames_shared
    from training.artifacts import save_effective_feature_list as save_effective_feature_list_shared
    from training.artifacts import save_identity_scaler_artifact as save_identity_scaler_artifact_shared
    from training.artifacts import save_industry_index_artifacts as save_industry_index_artifacts_shared
    from training.artifacts import save_prior_graph_artifact as save_prior_graph_artifact_shared
    from training.artifacts import save_best_training_artifacts as save_best_training_artifacts_shared
    from training.artifacts import write_final_training_score as write_final_training_score_shared
except ImportError as exc:
    TRAINING_ARTIFACT_IMPORT_ERROR = exc
    dump_factor_artifacts_shared = None
    format_factor_summary_shared = None
    log_factor_dashboard_shared = None
    print_best_checkpoint_summary_shared = None
    print_best_strategy_summary_shared = None
    print_train_epoch_summary_shared = None
    print_training_completion_summary_shared = None
    print_training_exit_banner_shared = None
    print_validation_run_summary_shared = None
    prepare_training_artifact_frames_shared = None
    save_effective_feature_list_shared = None
    save_identity_scaler_artifact_shared = None
    save_industry_index_artifacts_shared = None
    save_prior_graph_artifact_shared = None
    save_best_training_artifacts_shared = None
    write_final_training_score_shared = None
else:
    TRAINING_ARTIFACT_IMPORT_ERROR = None

try:
    from training.datasets import RankingDataset as RankingDatasetShared
    from training.datasets import LazyRankingDataset as LazyRankingDatasetShared
    from training.datasets import build_lazy_ranking_index as build_lazy_ranking_index_shared
    from training.datasets import collate_fn as collate_fn_shared
except ImportError as exc:
    TRAINING_DATASET_IMPORT_ERROR = exc
    RankingDatasetShared = None
    LazyRankingDatasetShared = None
    build_lazy_ranking_index_shared = None
    collate_fn_shared = None
else:
    TRAINING_DATASET_IMPORT_ERROR = None

try:
    from training.validation import build_factor_group_indices as build_factor_group_indices_shared
    from training.validation import build_validation_fold_loaders as build_validation_fold_loaders_shared
    from training.validation import evaluate_factor_group_ablation as evaluate_factor_group_ablation_shared
    from training.validation import log_factor_ablation as log_factor_ablation_shared
    from training.validation import print_factor_ablation_summary as print_factor_ablation_summary_shared
    from training.validation import resolve_training_validation_split as resolve_training_validation_split_shared
    from training.validation import split_train_val_by_last_month as split_train_val_by_last_month_shared
except ImportError as exc:
    TRAINING_VALIDATION_IMPORT_ERROR = exc
    build_factor_group_indices_shared = None
    build_validation_fold_loaders_shared = None
    evaluate_factor_group_ablation_shared = None
    log_factor_ablation_shared = None
    print_factor_ablation_summary_shared = None
    resolve_training_validation_split_shared = None
    split_train_val_by_last_month_shared = None
else:
    TRAINING_VALIDATION_IMPORT_ERROR = None

try:
    from experiments.metrics import build_portfolio_weights as build_portfolio_weights_shared
    from experiments.metrics import build_strategy_candidates as build_strategy_candidates_shared
    from experiments.metrics import calculate_ranking_metrics as calculate_ranking_metrics_shared
    from experiments.metrics import choose_best_strategy as choose_best_strategy_shared
    from experiments.metrics import format_strategy_metric_summary as format_strategy_metric_summary_shared
except ImportError as exc:
    EXPERIMENT_METRICS_IMPORT_ERROR = exc
    build_portfolio_weights_shared = None
    build_strategy_candidates_shared = None
    calculate_ranking_metrics_shared = None
    choose_best_strategy_shared = None
    format_strategy_metric_summary_shared = None
else:
    EXPERIMENT_METRICS_IMPORT_ERROR = None

try:
    from training.preprocessing import preprocess_data as preprocess_data_shared
    from training.preprocessing import preprocess_val_data as preprocess_val_data_shared
except ImportError as exc:
    TRAINING_PREPROCESS_IMPORT_ERROR = exc
    preprocess_data_shared = None
    preprocess_val_data_shared = None
else:
    TRAINING_PREPROCESS_IMPORT_ERROR = None

try:
    from training.loops import evaluate_ranking_folds as evaluate_ranking_folds_shared
    from training.loops import evaluate_ranking_model as evaluate_ranking_model_shared
    from training.loops import train_ranking_model as train_ranking_model_shared
except ImportError as exc:
    TRAINING_LOOPS_IMPORT_ERROR = exc
    evaluate_ranking_folds_shared = None
    evaluate_ranking_model_shared = None
    train_ranking_model_shared = None
else:
    TRAINING_LOOPS_IMPORT_ERROR = None

try:
    from training.inference import predict_top_stocks as predict_top_stocks_shared
    from training.inference import save_predictions as save_predictions_shared
except ImportError as exc:
    TRAINING_INFERENCE_IMPORT_ERROR = exc
    predict_top_stocks_shared = None
    save_predictions_shared = None
else:
    TRAINING_INFERENCE_IMPORT_ERROR = None

try:
    from training.runtime import build_optimizer_scheduler_scaler as build_optimizer_scheduler_scaler_shared
    from training.runtime import build_early_stopping_state as build_early_stopping_state_shared
    from training.runtime import update_early_stopping_state as update_early_stopping_state_shared
    from training.runtime import build_rank_model as build_rank_model_shared
    from training.runtime import build_prior_graph_adjacency as build_prior_graph_adjacency_shared
    from training.runtime import build_stock_industry_index as build_stock_industry_index_shared
    from training.runtime import initialize_training_runtime as initialize_training_runtime_shared
    from training.runtime import load_training_inputs as load_training_inputs_shared
    from training.runtime import print_early_stopping_summary as print_early_stopping_summary_shared
    from training.runtime import set_seed as set_seed_shared
except ImportError as exc:
    TRAINING_RUNTIME_IMPORT_ERROR = exc
    build_optimizer_scheduler_scaler_shared = None
    build_early_stopping_state_shared = None
    update_early_stopping_state_shared = None
    build_rank_model_shared = None
    build_prior_graph_adjacency_shared = None
    build_stock_industry_index_shared = None
    initialize_training_runtime_shared = None
    load_training_inputs_shared = None
    print_early_stopping_summary_shared = None
    set_seed_shared = None
else:
    TRAINING_RUNTIME_IMPORT_ERROR = None


class TrainPipelineModularizationTests(unittest.TestCase):
    def test_train_uses_shared_portfolio_optimization_loss(self):
        self.assertIs(train.PortfolioOptimizationLoss, SharedPortfolioOptimizationLoss)
        self.assertIs(train.build_portfolio_optimization_loss, build_portfolio_optimization_loss_shared)
        self.assertEqual(train.PortfolioOptimizationLoss.__module__, 'objectives.ranking_loss')
        self.assertEqual(train.build_portfolio_optimization_loss.__module__, 'objectives.ranking_loss')

    def test_train_uses_shared_target_transform_module(self):
        self.assertIsNone(
            TARGET_TRANSFORM_IMPORT_ERROR,
            f'objectives.target_transforms should exist: {TARGET_TRANSFORM_IMPORT_ERROR}',
        )
        self.assertIs(train.transform_targets_for_loss, transform_targets_for_loss)
        self.assertEqual(train.transform_targets_for_loss.__module__, 'objectives.target_transforms')
        self.assertTrue(callable(rank_normalize_tensor))
        self.assertTrue(callable(mad_clip_bounds))

    def test_train_uses_feature_assembler_entrypoints(self):
        self.assertIsNone(
            FEATURE_ASSEMBLER_IMPORT_ERROR,
            f'features.feature_assembler should expose entrypoints: {FEATURE_ASSEMBLER_IMPORT_ERROR}',
        )
        self.assertIs(train.build_feature_table, build_feature_table)
        self.assertIs(train.augment_feature_table, augment_feature_table)

    def test_feature_assembler_augment_entrypoint_supports_noop_contract(self):
        self.assertIsNone(
            FEATURE_ASSEMBLER_IMPORT_ERROR,
            f'features.feature_assembler should expose entrypoints: {FEATURE_ASSEMBLER_IMPORT_ERROR}',
        )
        df = pd.DataFrame({'日期': ['2024-01-02'], '股票代码': ['000001']})
        out_df, feature_columns = augment_feature_table(df, [], runtime_config={})
        self.assertEqual(feature_columns, [])
        self.assertListEqual(list(out_df.columns), list(df.columns))

    def test_train_reuses_training_artifact_helpers(self):
        self.assertIsNone(
            TRAINING_ARTIFACT_IMPORT_ERROR,
            f'training.artifacts should expose artifact helpers: {TRAINING_ARTIFACT_IMPORT_ERROR}',
        )
        self.assertIs(train.format_factor_summary, format_factor_summary_shared)
        self.assertIs(train.log_factor_dashboard, log_factor_dashboard_shared)
        self.assertIs(train.dump_factor_artifacts, dump_factor_artifacts_shared)
        self.assertIs(train.print_best_checkpoint_summary, print_best_checkpoint_summary_shared)
        self.assertIs(train.print_best_strategy_summary, print_best_strategy_summary_shared)
        self.assertIs(train.print_train_epoch_summary, print_train_epoch_summary_shared)
        self.assertIs(train.print_training_completion_summary, print_training_completion_summary_shared)
        self.assertIs(train.print_training_exit_banner, print_training_exit_banner_shared)
        self.assertIs(train.print_validation_run_summary, print_validation_run_summary_shared)
        self.assertIs(train.prepare_training_artifact_frames, prepare_training_artifact_frames_shared)
        self.assertIs(train.save_effective_feature_list, save_effective_feature_list_shared)
        self.assertIs(train.save_identity_scaler_artifact, save_identity_scaler_artifact_shared)
        self.assertIs(train.save_industry_index_artifacts, save_industry_index_artifacts_shared)
        self.assertIs(train.save_prior_graph_artifact, save_prior_graph_artifact_shared)
        self.assertIs(train.save_best_training_artifacts, save_best_training_artifacts_shared)
        self.assertIs(train.write_final_training_score, write_final_training_score_shared)
        self.assertEqual(train.format_factor_summary.__module__, 'training.artifacts')
        self.assertEqual(train.print_best_checkpoint_summary.__module__, 'training.artifacts')
        self.assertEqual(train.print_best_strategy_summary.__module__, 'training.artifacts')
        self.assertEqual(train.print_train_epoch_summary.__module__, 'training.artifacts')
        self.assertEqual(train.print_training_completion_summary.__module__, 'training.artifacts')
        self.assertEqual(train.print_training_exit_banner.__module__, 'training.artifacts')
        self.assertEqual(train.print_validation_run_summary.__module__, 'training.artifacts')
        self.assertEqual(train.prepare_training_artifact_frames.__module__, 'training.artifacts')
        self.assertEqual(train.save_effective_feature_list.__module__, 'training.artifacts')
        self.assertEqual(train.save_identity_scaler_artifact.__module__, 'training.artifacts')
        self.assertEqual(train.save_industry_index_artifacts.__module__, 'training.artifacts')
        self.assertEqual(train.save_prior_graph_artifact.__module__, 'training.artifacts')
        self.assertEqual(train.save_best_training_artifacts.__module__, 'training.artifacts')
        self.assertEqual(train.write_final_training_score.__module__, 'training.artifacts')

    def test_train_reuses_training_dataset_helpers(self):
        self.assertIsNone(
            TRAINING_DATASET_IMPORT_ERROR,
            f'training.datasets should expose dataset helpers: {TRAINING_DATASET_IMPORT_ERROR}',
        )
        self.assertIs(train.RankingDataset, RankingDatasetShared)
        self.assertIs(train.LazyRankingDataset, LazyRankingDatasetShared)
        self.assertIs(train.collate_fn, collate_fn_shared)
        self.assertIs(train.build_lazy_ranking_index, build_lazy_ranking_index_shared)
        self.assertEqual(train.RankingDataset.__module__, 'training.datasets')

    def test_train_reuses_training_validation_helpers(self):
        self.assertIsNone(
            TRAINING_VALIDATION_IMPORT_ERROR,
            f'training.validation should expose validation helpers: {TRAINING_VALIDATION_IMPORT_ERROR}',
        )
        self.assertIs(train.split_train_val_by_last_month, split_train_val_by_last_month_shared)
        self.assertIs(train.build_validation_fold_loaders, build_validation_fold_loaders_shared)
        self.assertIs(train.build_factor_group_indices, build_factor_group_indices_shared)
        self.assertIs(train.evaluate_factor_group_ablation, evaluate_factor_group_ablation_shared)
        self.assertIs(train.log_factor_ablation, log_factor_ablation_shared)
        self.assertIs(train.print_factor_ablation_summary, print_factor_ablation_summary_shared)
        self.assertIs(train.resolve_training_validation_split, resolve_training_validation_split_shared)
        self.assertEqual(train.split_train_val_by_last_month.__module__, 'training.validation')
        self.assertEqual(train.print_factor_ablation_summary.__module__, 'training.validation')
        self.assertEqual(train.resolve_training_validation_split.__module__, 'training.validation')

    def test_train_reuses_experiment_metric_helpers(self):
        self.assertIsNone(
            EXPERIMENT_METRICS_IMPORT_ERROR,
            f'experiments.metrics should expose strategy helpers: {EXPERIMENT_METRICS_IMPORT_ERROR}',
        )
        self.assertIs(train.build_strategy_candidates, build_strategy_candidates_shared)
        self.assertIs(train.build_portfolio_weights, build_portfolio_weights_shared)
        self.assertIs(train.calculate_ranking_metrics, calculate_ranking_metrics_shared)
        self.assertIs(train.choose_best_strategy, choose_best_strategy_shared)
        self.assertIs(train.format_strategy_metric_summary, format_strategy_metric_summary_shared)
        self.assertEqual(train.calculate_ranking_metrics.__module__, 'experiments.metrics')

    def test_train_reuses_training_preprocessing_helpers(self):
        self.assertIsNone(
            TRAINING_PREPROCESS_IMPORT_ERROR,
            f'training.preprocessing should expose preprocess helpers: {TRAINING_PREPROCESS_IMPORT_ERROR}',
        )
        self.assertIs(train.preprocess_data, preprocess_data_shared)
        self.assertIs(train.preprocess_val_data, preprocess_val_data_shared)
        self.assertEqual(train.preprocess_data.__module__, 'training.preprocessing')

    def test_train_reuses_training_loop_helpers(self):
        self.assertIsNone(
            TRAINING_LOOPS_IMPORT_ERROR,
            f'training.loops should expose train/eval helpers: {TRAINING_LOOPS_IMPORT_ERROR}',
        )
        self.assertIs(train.train_ranking_model, train_ranking_model_shared)
        self.assertIs(train.evaluate_ranking_model, evaluate_ranking_model_shared)
        self.assertIs(train.evaluate_ranking_folds, evaluate_ranking_folds_shared)
        self.assertEqual(train.train_ranking_model.__module__, 'training.loops')

    def test_train_reuses_training_inference_helpers(self):
        self.assertIsNone(
            TRAINING_INFERENCE_IMPORT_ERROR,
            f'training.inference should expose inference helpers: {TRAINING_INFERENCE_IMPORT_ERROR}',
        )
        self.assertIs(train.predict_top_stocks, predict_top_stocks_shared)
        self.assertIs(train.save_predictions, save_predictions_shared)
        self.assertEqual(train.predict_top_stocks.__module__, 'training.inference')

    def test_train_reuses_training_runtime_helpers(self):
        self.assertIsNone(
            TRAINING_RUNTIME_IMPORT_ERROR,
            f'training.runtime should expose runtime helpers: {TRAINING_RUNTIME_IMPORT_ERROR}',
        )
        self.assertIs(train.set_seed, set_seed_shared)
        self.assertIs(train.build_optimizer_scheduler_scaler, build_optimizer_scheduler_scaler_shared)
        self.assertIs(train.build_early_stopping_state, build_early_stopping_state_shared)
        self.assertIs(train.update_early_stopping_state, update_early_stopping_state_shared)
        self.assertIs(train.build_rank_model, build_rank_model_shared)
        self.assertIs(train.initialize_training_runtime, initialize_training_runtime_shared)
        self.assertIs(train.load_training_inputs, load_training_inputs_shared)
        self.assertIs(train.build_prior_graph_adjacency, build_prior_graph_adjacency_shared)
        self.assertIs(train.build_stock_industry_index, build_stock_industry_index_shared)
        self.assertIs(train.print_early_stopping_summary, print_early_stopping_summary_shared)
        self.assertEqual(train.set_seed.__module__, 'training.runtime')
        self.assertEqual(train.build_optimizer_scheduler_scaler.__module__, 'training.runtime')
        self.assertEqual(train.build_early_stopping_state.__module__, 'training.runtime')
        self.assertEqual(train.update_early_stopping_state.__module__, 'training.runtime')
        self.assertEqual(train.build_rank_model.__module__, 'training.runtime')
        self.assertEqual(train.initialize_training_runtime.__module__, 'training.runtime')
        self.assertEqual(train.load_training_inputs.__module__, 'training.runtime')
        self.assertEqual(train.print_early_stopping_summary.__module__, 'training.runtime')

    def test_save_best_training_artifacts_publishes_model_compatible_metadata(self):
        self.assertIsNone(
            TRAINING_ARTIFACT_IMPORT_ERROR,
            f'training.artifacts should expose artifact helpers: {TRAINING_ARTIFACT_IMPORT_ERROR}',
        )
        model = train.torch.nn.Linear(2, 1)
        run_summary = {
            'best_candidate': {'name': 'equal_top3', 'top_k': 3, 'weighting': 'equal', 'temperature': 1.0},
            'best_score': 0.1234,
            'best_return': 0.2345,
            'strategy_summary': 'equal_top3=0.2345',
            'strategy_comparison': [],
            'fold_diagnostics': [],
            'regime_summary': {
                'dominant_strategy': 'equal_top3',
                'positive_return_fold_count': 1,
                'negative_return_fold_count': 0,
                'flat_return_fold_count': 0,
            },
        }
        validation_folds = [{'name': 'fold_1', 'start_date': '2025-01-01', 'end_date': '2025-01-31'}]
        runtime_config = {
            'softmax_temperature': 1.0,
            'validation_mode': 'rolling',
            'strategy_selection_mode': 'risk_adjusted',
            'strategy_risk_lambda': 0.2,
        }
        factor_pipeline = {
            'feature_set': 'v1',
            'store_path': './config/factor_store.json',
            'builtin_registry_path': './config/builtin_factors.json',
            'summary': {'active_total': 2, 'builtin_enabled': 2, 'builtin_total': 2, 'custom_enabled': 0, 'custom_total': 0, 'group_counts': {'base': 2}},
            'active_features': ['alpha_1', 'alpha_2'],
            'builtin_specs': [{'name': 'alpha_1', 'enabled': True}, {'name': 'alpha_2', 'enabled': True}],
            'custom_specs': [],
            'dependency_graph': {},
            'factor_fingerprint': 'fingerprint-123',
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_best_training_artifacts_shared(
                model,
                tmpdir,
                run_summary,
                validation_folds,
                runtime_config,
                best_epoch=3,
                features=['alpha_1', 'alpha_2'],
                factor_pipeline=factor_pipeline,
            )

            with open(os.path.join(tmpdir, 'effective_features.json'), 'r', encoding='utf-8') as f:
                saved_features = json.load(f)
            with open(os.path.join(tmpdir, 'active_factors.json'), 'r', encoding='utf-8') as f:
                saved_snapshot = json.load(f)

        self.assertEqual(saved_features, ['alpha_1', 'alpha_2'])
        self.assertEqual(saved_snapshot['factor_fingerprint'], 'fingerprint-123')
        self.assertEqual(saved_snapshot['active_features'], ['alpha_1', 'alpha_2'])


if __name__ == '__main__':
    unittest.main()
