import os
import sys
import tempfile
import unittest


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

    def test_load_prediction_strategy_keeps_legacy_default_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy = predict.load_prediction_strategy(tmpdir)

        self.assertEqual(strategy['top_k'], 5)
        self.assertEqual(strategy['weighting'], 'equal')
        self.assertIn('temperature', strategy)


if __name__ == '__main__':
    unittest.main()
