import os
import sys
import unittest

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from portfolio.policy import scores_to_portfolio


class PortfolioPolicyTests(unittest.TestCase):
    def test_equal_weighting_keeps_top_k_compatibility(self):
        scores = np.array([0.1, 0.4, 0.2], dtype=np.float64)
        stock_ids = ['AAA', 'BBB', 'CCC']

        selected_ids, weights = scores_to_portfolio(
            scores,
            stock_ids,
            {'top_k': 2, 'weighting': 'equal'},
        )

        self.assertEqual(selected_ids, ['BBB', 'CCC'])
        np.testing.assert_allclose(weights, np.array([0.5, 0.5], dtype=np.float64))

    def test_softmax_weighting_keeps_descending_preference(self):
        scores = np.array([1.2, 0.5, 0.1], dtype=np.float64)
        stock_ids = ['AAA', 'BBB', 'CCC']

        selected_ids, weights = scores_to_portfolio(
            scores,
            stock_ids,
            {'top_k': 2, 'weighting': 'softmax', 'temperature': 1.0},
        )

        self.assertEqual(selected_ids, ['AAA', 'BBB'])
        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=8)

    def test_industry_limit_skips_second_name_in_same_industry(self):
        scores = np.array([0.95, 0.90, 0.80, 0.70], dtype=np.float64)
        stock_ids = ['AAA', 'BBB', 'CCC', 'DDD']
        strategy = {
            'top_k': 3,
            'weighting': 'equal',
            'max_per_industry': 1,
            'metadata': {
                'stock_to_industry': {
                    'AAA': 'tech',
                    'BBB': 'tech',
                    'CCC': 'finance',
                    'DDD': 'energy',
                },
            },
        }

        selected_ids, weights = scores_to_portfolio(scores, stock_ids, strategy)

        self.assertEqual(selected_ids, ['AAA', 'CCC', 'DDD'])
        np.testing.assert_allclose(weights, np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64))

    def test_turnover_penalty_prefers_existing_holdings_when_scores_are_close(self):
        scores = np.array([0.92, 0.91, 0.90], dtype=np.float64)
        stock_ids = ['AAA', 'BBB', 'CCC']
        strategy = {
            'top_k': 2,
            'weighting': 'equal',
            'turnover_penalty': 0.05,
            'metadata': {
                'previous_holdings': ['CCC'],
            },
        }

        selected_ids, weights = scores_to_portfolio(scores, stock_ids, strategy)

        self.assertEqual(selected_ids, ['CCC', 'AAA'])
        np.testing.assert_allclose(weights, np.array([0.5, 0.5], dtype=np.float64))


if __name__ == '__main__':
    unittest.main()
