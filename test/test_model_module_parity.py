import os
import sys
import unittest

import pandas as pd
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from config import config
from features.daily_features import build_daily_feature_table
from features.feature_assembler import augment_feature_table
from features.feature_assembler import build_feature_table
from features.intraday_features import add_intraday_feature_blocks
from features.relative_features import add_relative_feature_blocks
from features.risk_features import add_risk_feature_blocks
from features.risk_features import normalize_feature_table
from model import StockTransformer as LegacyStockTransformer
from models.rank_model import StockTransformer as ModularStockTransformer
from objectives.aux_losses import compute_volatility_huber_loss
from objectives.ranking_loss import PortfolioOptimizationLoss
from objectives.ranking_loss import build_portfolio_optimization_loss


class ModuleParityTests(unittest.TestCase):
    def _model_runtime_config(self):
        runtime_config = dict(config)
        runtime_config.update({
            'sequence_length': 4,
            'd_model': 16,
            'nhead': 4,
            'num_layers': 1,
            'dim_feedforward': 32,
            'dropout': 0.0,
            'use_market_gating': False,
            'use_market_gating_macro_context': False,
            'use_multi_scale_temporal': False,
            'use_ultra_short_branch': False,
            'use_temporal_cross_stock_attention': False,
            'use_multitask_volatility': False,
            'use_cross_stock_attention_mask': False,
            'cross_stock_mask_mode': 'full',
        })
        return runtime_config

    def _feature_runtime_config(self):
        runtime_config = dict(config)
        runtime_config.update({
            'use_feature_enhancements': True,
            'use_market_sentiment_features': True,
            'use_cross_sectional_rank_features': True,
            'use_industry_relative_z_features': False,
            'use_price_volume_distribution_features': True,
            'use_cross_sectional_feature_norm': True,
            'feature_cs_norm_method': 'zscore',
            'feature_cs_clip_value': None,
            'use_static_stock_features': False,
        })
        return runtime_config

    def _sample_feature_frame(self):
        return pd.DataFrame({
            '日期': ['2024-01-02', '2024-01-02', '2024-01-03', '2024-01-03'],
            '股票代码': ['000001', '000002', '000001', '000002'],
            '开盘': [10.0, 20.0, 10.5, 19.5],
            '收盘': [10.2, 19.8, 10.7, 19.9],
            '最高': [10.4, 20.2, 10.9, 20.1],
            '最低': [9.9, 19.6, 10.3, 19.2],
            '成交量': [1000.0, 1500.0, 1200.0, 1700.0],
            '成交额': [10200.0, 29700.0, 12840.0, 33830.0],
            '换手率': [0.03, 0.05, 0.04, 0.06],
            'rsi': [45.0, 60.0, 50.0, 62.0],
            'return_1': [0.01, -0.02, 0.015, 0.01],
            'return_5': [0.03, -0.01, 0.04, 0.02],
            'volatility_20': [0.2, 0.35, 0.25, 0.3],
            'atr_14': [0.4, 0.5, 0.45, 0.55],
        })

    def _sample_daily_feature_frame(self):
        rows = 80
        dates = pd.date_range('2024-01-02', periods=rows, freq='B')
        base = pd.Series(range(rows), dtype='float64')
        close = 10.0 + base * 0.1
        open_ = close - 0.05
        high = close + 0.2
        low = close - 0.2
        volume = 1000.0 + base * 10.0
        amount = close * volume
        return pd.DataFrame({
            '日期': dates.strftime('%Y-%m-%d'),
            '股票代码': ['000001'] * rows,
            '开盘': open_,
            '收盘': close,
            '最高': high,
            '最低': low,
            '成交量': volume,
            '成交额': amount,
        })

    def test_loss_builder_matches_direct_loss_on_identical_tensors(self):
        runtime_config = dict(config)
        runtime_config.update({
            'loss_temperature': 7.0,
            'listnet_weight': 0.7,
            'pairwise_weight': 1.2,
            'lambda_ndcg_weight': 0.9,
            'lambda_ndcg_topk': 3,
            'ic_weight': 0.2,
            'ic_mode': 'spearman',
            'topk_focus_weight': 0.4,
            'topk_focus_k': 2,
            'topk_focus_gain_mode': 'linear',
            'topk_focus_normalize': True,
        })
        predictions = torch.tensor([0.2, -0.1, 0.7, 0.3], dtype=torch.float32)
        returns = torch.tensor([0.01, -0.02, 0.05, 0.03], dtype=torch.float32)

        direct_loss = PortfolioOptimizationLoss(
            temperature=runtime_config['loss_temperature'],
            listnet_weight=runtime_config['listnet_weight'],
            pairwise_weight=runtime_config['pairwise_weight'],
            lambda_ndcg_weight=runtime_config['lambda_ndcg_weight'],
            lambda_ndcg_topk=runtime_config['lambda_ndcg_topk'],
            ic_weight=runtime_config['ic_weight'],
            ic_mode=runtime_config['ic_mode'],
            topk_focus_weight=runtime_config['topk_focus_weight'],
            topk_focus_k=runtime_config['topk_focus_k'],
            topk_focus_gain_mode=runtime_config['topk_focus_gain_mode'],
            topk_focus_normalize=runtime_config['topk_focus_normalize'],
            runtime_config=runtime_config,
        )
        built_loss = build_portfolio_optimization_loss(runtime_config)

        self.assertAlmostEqual(
            float(direct_loss(predictions, returns).item()),
            float(built_loss(predictions, returns).item()),
            places=6,
        )

    def test_loss_volatility_term_matches_aux_loss_helper(self):
        predictions = torch.tensor([0.2, -0.1, 0.7, 0.3], dtype=torch.float32)
        returns = torch.tensor([0.01, -0.02, 0.05, 0.03], dtype=torch.float32)
        vol_targets = torch.tensor([0.3, float('nan'), 0.2, 0.25], dtype=torch.float32)
        vol_pred = torch.tensor([0.28, 0.31, 0.22, 0.21], dtype=torch.float32)

        loss = PortfolioOptimizationLoss()
        base_total = loss(predictions, returns)
        total_with_vol = loss(
            predictions,
            returns,
            volatility_targets=vol_targets,
            volatility_pred=vol_pred,
        )
        expected_delta = compute_volatility_huber_loss(vol_pred, vol_targets)

        self.assertAlmostEqual(
            float((total_with_vol - base_total).item()),
            float(expected_delta.item()),
            places=6,
        )

    def test_build_feature_table_matches_daily_feature_builder(self):
        try:
            import talib  # noqa: F401
        except ImportError:
            self.skipTest('TA-Lib unavailable')

        df = self._sample_daily_feature_frame()
        actual = build_feature_table(df, feature_set='39')
        expected = build_daily_feature_table(df, feature_set='39')
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    def test_augment_feature_table_matches_composed_feature_modules(self):
        runtime_config = self._feature_runtime_config()
        df = self._sample_feature_frame()
        feature_columns = ['rsi', 'return_1', 'return_5', 'volatility_20', 'atr_14']

        actual_df, actual_features = augment_feature_table(
            df,
            feature_columns,
            runtime_config=runtime_config,
            feature_pipeline=None,
            date_col='日期',
            stock_col='股票代码',
            apply_factor_pipeline=False,
            apply_feature_enhancements=True,
            apply_cross_sectional_norm=True,
        )

        expected_df, expected_features = add_intraday_feature_blocks(
            df,
            feature_columns,
            runtime_config,
            date_col='日期',
            stock_col='股票代码',
        )
        expected_df, expected_features = add_relative_feature_blocks(
            expected_df,
            expected_features,
            runtime_config,
            date_col='日期',
            stock_col='股票代码',
        )
        expected_df, expected_features = add_risk_feature_blocks(
            expected_df,
            expected_features,
            runtime_config,
            date_col='日期',
            stock_col='股票代码',
        )
        expected_df = normalize_feature_table(
            expected_df,
            expected_features,
            runtime_config,
            date_col='日期',
        )

        self.assertListEqual(actual_features, expected_features)
        pd.testing.assert_frame_equal(
            actual_df.sort_index(axis=1),
            expected_df.sort_index(axis=1),
            check_dtype=False,
        )

    def test_legacy_and_modular_stock_transformer_outputs_match_for_fixed_seed(self):
        runtime_config = self._model_runtime_config()
        input_dim = 6
        batch_size = 2
        num_stocks = 3
        src = torch.randn(batch_size, num_stocks, runtime_config['sequence_length'], input_dim)
        valid_mask = torch.ones(batch_size, num_stocks, dtype=torch.bool)

        torch.manual_seed(1234)
        modular_model = ModularStockTransformer(
            input_dim=input_dim,
            config=runtime_config,
            num_stocks=num_stocks,
        )
        torch.manual_seed(1234)
        legacy_model = LegacyStockTransformer(
            input_dim=input_dim,
            config=runtime_config,
            num_stocks=num_stocks,
        )
        modular_model.eval()
        legacy_model.eval()

        with torch.no_grad():
            modular_output = modular_model(src, stock_valid_mask=valid_mask)
            legacy_output = legacy_model(src, stock_valid_mask=valid_mask)

        self.assertTrue(torch.allclose(modular_output, legacy_output, atol=1e-6, rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
