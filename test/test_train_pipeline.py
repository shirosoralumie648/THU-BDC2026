import os
import sys
import unittest

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import train
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


class TrainPipelineModularizationTests(unittest.TestCase):
    def test_train_uses_shared_portfolio_optimization_loss(self):
        self.assertIs(train.PortfolioOptimizationLoss, SharedPortfolioOptimizationLoss)
        self.assertEqual(train.PortfolioOptimizationLoss.__module__, 'objectives.ranking_loss')

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


if __name__ == '__main__':
    unittest.main()
