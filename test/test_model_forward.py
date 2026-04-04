import os
import sys
import unittest
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from config import config
import models.rank_model as rank_model_module
from model import StockTransformer as LegacyStockTransformer
from models.rank_model import StockTransformer as ModularStockTransformer

try:
    from models.daily_encoder import FeatureAttention as DailyFeatureAttention
    from models.daily_encoder import PositionalEncoding as DailyPositionalEncoding
    from models.daily_encoder import _normalize_scale_windows as daily_normalize_scale_windows
    from models.intraday_encoder import build_multi_scale_fusion_gate
    from models.intraday_encoder import build_short_horizon_fusion_gate
    from models.regime_router import build_market_gate
    from models.regime_router import build_market_macro_proj
    from models.relation_encoder import CrossStockAttention as RelationCrossStockAttention
except ImportError as exc:
    MODEL_SUBMODULE_IMPORT_ERROR = exc
    DailyFeatureAttention = None
    DailyPositionalEncoding = None
    daily_normalize_scale_windows = None
    build_multi_scale_fusion_gate = None
    build_short_horizon_fusion_gate = None
    build_market_gate = None
    build_market_macro_proj = None
    RelationCrossStockAttention = None
else:
    MODEL_SUBMODULE_IMPORT_ERROR = None


class RankModelCompatibilityTests(unittest.TestCase):
    def _runtime_config(self):
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

    def test_models_rank_model_is_primary_stock_transformer_definition(self):
        self.assertEqual(ModularStockTransformer.__module__, 'models.rank_model')
        self.assertEqual(LegacyStockTransformer.__module__, 'models.rank_model')

    def test_rank_model_file_contains_primary_stock_transformer_definition(self):
        rank_model_path = os.path.join(SRC_ROOT, 'models', 'rank_model.py')
        with open(rank_model_path, 'r', encoding='utf-8') as f:
            source = f.read()

        self.assertIn('class StockTransformer', source)
        self.assertNotIn('from model import StockTransformer', source)

    def test_rank_model_reuses_extracted_model_submodules(self):
        self.assertIsNone(
            MODEL_SUBMODULE_IMPORT_ERROR,
            f'model submodules should be importable: {MODEL_SUBMODULE_IMPORT_ERROR}',
        )
        self.assertIs(rank_model_module.PositionalEncoding, DailyPositionalEncoding)
        self.assertIs(rank_model_module.FeatureAttention, DailyFeatureAttention)
        self.assertIs(rank_model_module.CrossStockAttention, RelationCrossStockAttention)
        self.assertIs(rank_model_module._normalize_scale_windows, daily_normalize_scale_windows)
        self.assertIs(rank_model_module.build_market_gate, build_market_gate)
        self.assertIs(rank_model_module.build_market_macro_proj, build_market_macro_proj)
        self.assertIs(rank_model_module.build_short_horizon_fusion_gate, build_short_horizon_fusion_gate)
        self.assertIs(rank_model_module.build_multi_scale_fusion_gate, build_multi_scale_fusion_gate)

    def test_legacy_and_modular_import_paths_keep_forward_shape_compatible(self):
        runtime_config = self._runtime_config()
        input_dim = 6
        batch_size = 2
        num_stocks = 3
        sequence_length = runtime_config['sequence_length']
        src = torch.randn(batch_size, num_stocks, sequence_length, input_dim)
        valid_mask = torch.ones(batch_size, num_stocks, dtype=torch.bool)

        modular_model = ModularStockTransformer(
            input_dim=input_dim,
            config=runtime_config,
            num_stocks=num_stocks,
        )
        legacy_model = LegacyStockTransformer(
            input_dim=input_dim,
            config=runtime_config,
            num_stocks=num_stocks,
        )

        modular_output = modular_model(src, stock_valid_mask=valid_mask)
        legacy_output = legacy_model(src, stock_valid_mask=valid_mask)

        self.assertEqual(tuple(modular_output.shape), (batch_size, num_stocks))
        self.assertEqual(tuple(legacy_output.shape), (batch_size, num_stocks))


if __name__ == '__main__':
    unittest.main()
