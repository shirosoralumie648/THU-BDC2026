import os
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import predict
from models.rank_model import StockTransformer as ModularStockTransformer

try:
    from features.feature_assembler import build_feature_table as FeatureAssemblerBuildFeatureTable
    from features.feature_assembler import augment_feature_table as FeatureAssemblerAugmentFeatureTable
except ImportError as exc:
    FEATURE_ASSEMBLER_IMPORT_ERROR = exc
    FeatureAssemblerBuildFeatureTable = None
    FeatureAssemblerAugmentFeatureTable = None
else:
    FEATURE_ASSEMBLER_IMPORT_ERROR = None

try:
    from training.inference import apply_optional_global_scaler as apply_optional_global_scaler_shared
    from training.inference import build_inference_sequences as build_inference_sequences_shared
    from training.inference import build_prediction_input_manifest as build_prediction_input_manifest_shared
    from training.inference import dump_predict_factor_snapshot as dump_predict_factor_snapshot_shared
    from training.inference import load_prediction_inputs as load_prediction_inputs_shared
    from training.inference import load_prediction_model as load_prediction_model_shared
    from training.inference import load_prediction_runtime_config as load_prediction_runtime_config_shared
    from training.inference import resolve_effective_prediction_features as resolve_effective_prediction_features_shared
    from training.inference import run_prediction_inference as run_prediction_inference_shared
    from training.inference import write_prediction_outputs as write_prediction_outputs_shared
    from training.inference import load_prediction_strategy as load_prediction_strategy_shared
except ImportError as exc:
    TRAINING_INFERENCE_IMPORT_ERROR = exc
    apply_optional_global_scaler_shared = None
    build_inference_sequences_shared = None
    build_prediction_input_manifest_shared = None
    dump_predict_factor_snapshot_shared = None
    load_prediction_inputs_shared = None
    load_prediction_model_shared = None
    load_prediction_runtime_config_shared = None
    resolve_effective_prediction_features_shared = None
    run_prediction_inference_shared = None
    write_prediction_outputs_shared = None
    load_prediction_strategy_shared = None
else:
    TRAINING_INFERENCE_IMPORT_ERROR = None

try:
    from training.preprocessing import preprocess_predict_data as preprocess_predict_data_shared
except ImportError as exc:
    TRAINING_PREPROCESS_IMPORT_ERROR = exc
    preprocess_predict_data_shared = None
else:
    TRAINING_PREPROCESS_IMPORT_ERROR = None

try:
    from training.runtime import resolve_prediction_stock_industry_index as resolve_prediction_stock_industry_index_shared
except ImportError as exc:
    TRAINING_RUNTIME_IMPORT_ERROR = exc
    resolve_prediction_stock_industry_index_shared = None
else:
    TRAINING_RUNTIME_IMPORT_ERROR = None


class PredictPipelineModularizationTests(unittest.TestCase):
    def test_predict_uses_modular_rank_model_import(self):
        self.assertIs(predict.StockTransformer, ModularStockTransformer)
        self.assertEqual(predict.StockTransformer.__module__, 'models.rank_model')

    def test_predict_uses_feature_assembler_entrypoints(self):
        self.assertIsNone(
            FEATURE_ASSEMBLER_IMPORT_ERROR,
            f'features.feature_assembler should expose entrypoints: {FEATURE_ASSEMBLER_IMPORT_ERROR}',
        )
        self.assertIs(predict.build_feature_table, FeatureAssemblerBuildFeatureTable)
        self.assertIs(predict.augment_feature_table, FeatureAssemblerAugmentFeatureTable)

    def test_predict_reuses_training_inference_helpers(self):
        self.assertIsNone(
            TRAINING_INFERENCE_IMPORT_ERROR,
            f'training.inference should expose predict helpers: {TRAINING_INFERENCE_IMPORT_ERROR}',
        )
        self.assertIs(predict.build_prediction_input_manifest, build_prediction_input_manifest_shared)
        self.assertIs(predict.load_prediction_runtime_config, load_prediction_runtime_config_shared)
        self.assertIs(predict.load_prediction_inputs, load_prediction_inputs_shared)
        self.assertIs(predict.build_inference_sequences, build_inference_sequences_shared)
        self.assertIs(predict.load_prediction_model, load_prediction_model_shared)
        self.assertIs(predict.resolve_effective_prediction_features, resolve_effective_prediction_features_shared)
        self.assertIs(predict.run_prediction_inference, run_prediction_inference_shared)
        self.assertIs(predict.write_prediction_outputs, write_prediction_outputs_shared)
        self.assertIs(predict.load_prediction_strategy, load_prediction_strategy_shared)
        self.assertIs(predict.apply_optional_global_scaler, apply_optional_global_scaler_shared)
        self.assertIs(predict.dump_predict_factor_snapshot, dump_predict_factor_snapshot_shared)
        self.assertEqual(predict.load_prediction_inputs.__module__, 'training.inference')
        self.assertEqual(predict.load_prediction_runtime_config.__module__, 'training.inference')
        self.assertEqual(predict.load_prediction_model.__module__, 'training.inference')
        self.assertEqual(predict.resolve_effective_prediction_features.__module__, 'training.inference')
        self.assertEqual(predict.run_prediction_inference.__module__, 'training.inference')
        self.assertEqual(predict.write_prediction_outputs.__module__, 'training.inference')
        self.assertEqual(predict.load_prediction_strategy.__module__, 'training.inference')

    def test_predict_reuses_training_preprocessing_helper(self):
        self.assertIsNone(
            TRAINING_PREPROCESS_IMPORT_ERROR,
            f'training.preprocessing should expose predict preprocess helper: {TRAINING_PREPROCESS_IMPORT_ERROR}',
        )
        self.assertIs(predict.preprocess_predict_data, preprocess_predict_data_shared)
        self.assertEqual(predict.preprocess_predict_data.__module__, 'training.preprocessing')

    def test_predict_reuses_training_runtime_helper_for_industry_index(self):
        self.assertIsNone(
            TRAINING_RUNTIME_IMPORT_ERROR,
            f'training.runtime should expose industry index helper: {TRAINING_RUNTIME_IMPORT_ERROR}',
        )
        self.assertIs(predict.resolve_prediction_stock_industry_index, resolve_prediction_stock_industry_index_shared)
        self.assertEqual(predict.resolve_prediction_stock_industry_index.__module__, 'training.runtime')

    def test_load_prediction_strategy_keeps_legacy_default_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy = predict.load_prediction_strategy(tmpdir)

        self.assertEqual(strategy['top_k'], 5)
        self.assertEqual(strategy['weighting'], 'equal')
        self.assertIn('temperature', strategy)

    def test_resolve_effective_prediction_features_prefers_saved_feature_list(self):
        self.assertIsNone(
            TRAINING_INFERENCE_IMPORT_ERROR,
            f'training.inference should expose predict helpers: {TRAINING_INFERENCE_IMPORT_ERROR}',
        )
        self.assertIsNotNone(
            resolve_effective_prediction_features_shared,
            'training.inference should expose resolve_effective_prediction_features',
        )
        if resolve_effective_prediction_features_shared is None:
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            feature_path = os.path.join(tmpdir, 'effective_features.json')
            with open(feature_path, 'w', encoding='utf-8') as f:
                f.write('["alpha_2", "alpha_1"]')

            processed = pd.DataFrame({'alpha_1': [1.0], 'alpha_2': [2.0], '日期': ['2024-01-02']})
            features = predict.resolve_effective_prediction_features(
                processed,
                ['alpha_1'],
                feature_path,
            )

        self.assertEqual(features, ['alpha_2', 'alpha_1'])

    def test_load_prediction_runtime_config_prefers_saved_training_snapshot(self):
        self.assertIsNone(
            TRAINING_INFERENCE_IMPORT_ERROR,
            f'training.inference should expose predict helpers: {TRAINING_INFERENCE_IMPORT_ERROR}',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write('{"d_model": 64, "nhead": 2, "num_layers": 1, "output_dir": "ignored"}')

            resolved = predict.load_prediction_runtime_config(
                tmpdir,
                runtime_config={'d_model': 256, 'nhead': 4, 'output_dir': tmpdir, 'prediction_scores_path': './output/custom.csv'},
            )

        self.assertEqual(resolved['d_model'], 64)
        self.assertEqual(resolved['nhead'], 2)
        self.assertEqual(resolved['num_layers'], 1)
        self.assertEqual(resolved['output_dir'], tmpdir)
        self.assertEqual(resolved['prediction_scores_path'], './output/custom.csv')


if __name__ == '__main__':
    unittest.main()
