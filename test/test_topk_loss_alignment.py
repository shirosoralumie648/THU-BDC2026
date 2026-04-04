import math
import os
import sys
import unittest

import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from train import PortfolioOptimizationLoss


class TopKLossAlignmentTests(unittest.TestCase):
    def _batch(self, predictions, targets):
        y_pred = torch.tensor([predictions], dtype=torch.float32)
        y_true = torch.tensor([targets], dtype=torch.float32)
        return y_pred, y_true

    def test_zero_topk_focus_weight_matches_legacy_loss(self):
        y_pred, y_true = self._batch(
            [0.15, 0.60, 0.20, -0.05],
            [0.80, 0.70, 0.10, -0.20],
        )
        legacy = PortfolioOptimizationLoss(
            temperature=10.0,
            listnet_weight=1.0,
            pairwise_weight=1.0,
            lambda_ndcg_weight=0.8,
            lambda_ndcg_topk=3,
            ic_weight=0.0,
        )(y_pred, y_true)
        aligned = PortfolioOptimizationLoss(
            temperature=10.0,
            listnet_weight=1.0,
            pairwise_weight=1.0,
            lambda_ndcg_weight=0.8,
            lambda_ndcg_topk=3,
            ic_weight=0.0,
            topk_focus_weight=0.0,
            topk_focus_k=2,
            topk_focus_gain_mode='binary',
            topk_focus_normalize=True,
        )(y_pred, y_true)
        self.assertAlmostEqual(float(aligned.item()), float(legacy.item()), places=7)

    def test_topk_focus_penalizes_misordered_head_more_than_plain_loss(self):
        y_pred, y_true = self._batch(
            [0.05, 0.10, 0.90, 0.85],
            [0.95, 0.90, 0.20, 0.10],
        )
        plain = PortfolioOptimizationLoss(
            temperature=10.0,
            listnet_weight=0.0,
            pairwise_weight=1.0,
            lambda_ndcg_weight=0.0,
            ic_weight=0.0,
        )(y_pred, y_true)
        focused = PortfolioOptimizationLoss(
            temperature=10.0,
            listnet_weight=0.0,
            pairwise_weight=1.0,
            lambda_ndcg_weight=0.0,
            ic_weight=0.0,
            topk_focus_weight=1.0,
            topk_focus_k=2,
            topk_focus_gain_mode='binary',
            topk_focus_normalize=True,
        )(y_pred, y_true)
        self.assertGreater(float(focused.item()) - float(plain.item()), 1e-6)

    def test_topk_focus_is_finite_when_focus_k_exceeds_item_count(self):
        y_pred, y_true = self._batch(
            [0.40, 0.30, 0.20],
            [0.50, 0.10, -0.10],
        )
        loss = PortfolioOptimizationLoss(
            temperature=10.0,
            listnet_weight=0.0,
            pairwise_weight=0.0,
            lambda_ndcg_weight=0.0,
            ic_weight=0.0,
            topk_focus_weight=1.0,
            topk_focus_k=5,
            topk_focus_gain_mode='linear',
            topk_focus_normalize=True,
        )(y_pred, y_true)
        self.assertTrue(math.isfinite(float(loss.item())))

    def test_topk_focus_handles_equal_labels_without_nan(self):
        y_pred, y_true = self._batch(
            [0.10, 0.20, 0.30, 0.40],
            [0.25, 0.25, 0.25, 0.25],
        )
        loss = PortfolioOptimizationLoss(
            temperature=10.0,
            listnet_weight=0.0,
            pairwise_weight=0.0,
            lambda_ndcg_weight=0.0,
            ic_weight=0.0,
            topk_focus_weight=1.0,
            topk_focus_k=3,
            topk_focus_gain_mode='binary',
            topk_focus_normalize=True,
        )(y_pred, y_true)
        self.assertTrue(math.isfinite(float(loss.item())))
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_negative_topk_focus_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            PortfolioOptimizationLoss(topk_focus_weight=-0.1)

    def test_non_positive_focus_k_is_rejected_when_focus_enabled(self):
        with self.assertRaises(ValueError):
            PortfolioOptimizationLoss(topk_focus_weight=1.0, topk_focus_k=0)

    def test_invalid_topk_focus_gain_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            PortfolioOptimizationLoss(topk_focus_weight=1.0, topk_focus_gain_mode='unexpected')

    def test_parse_bool_recognizes_extended_string_values(self):
        self.assertFalse(PortfolioOptimizationLoss._parse_bool('off'))
        self.assertFalse(PortfolioOptimizationLoss._parse_bool('0.0'))
        self.assertTrue(PortfolioOptimizationLoss._parse_bool('on'))
        self.assertTrue(PortfolioOptimizationLoss._parse_bool('1.0'))


if __name__ == '__main__':
    unittest.main()
