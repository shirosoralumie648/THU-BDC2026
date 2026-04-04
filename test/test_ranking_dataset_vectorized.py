import os
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from utils import _aggregate_ranking_windows_by_date
from utils import create_ranking_dataset_vectorized


class RankingDatasetVectorizedTests(unittest.TestCase):
    def test_aggregate_ranking_windows_by_date_filters_dates_and_preserves_stock_order(self):
        date_a = pd.Timestamp('2024-01-03')
        date_b = pd.Timestamp('2024-01-04')
        window_buckets = {
            date_a: {
                'seqs': [np.full((2, 2), float(idx), dtype=np.float32) for idx in range(10)],
                'targets': [float(idx) for idx in range(10)],
                'vol_targets': [float(idx) / 10.0 for idx in range(10)],
                'stock_codes': [f'stock_{idx:02d}' for idx in range(10)],
            },
            date_b: {
                'seqs': [np.full((2, 2), float(idx + 10), dtype=np.float32) for idx in range(10)],
                'targets': [float(idx + 10) for idx in range(10)],
                'vol_targets': [float(idx + 10) / 10.0 for idx in range(10)],
                'stock_codes': [f'stock_{idx:02d}' for idx in range(10)],
            },
        }

        sequences, targets, relevance_scores, stock_indices, vol_targets = _aggregate_ranking_windows_by_date(
            window_buckets,
            min_window_end_date=date_a,
            max_window_end_date=date_a,
        )

        self.assertEqual(len(sequences), 1)
        self.assertEqual(len(targets), 1)
        self.assertEqual(len(relevance_scores), 1)
        self.assertEqual(len(stock_indices), 1)
        self.assertEqual(len(vol_targets), 1)
        self.assertEqual(sequences[0].shape, (10, 2, 2))
        self.assertEqual(stock_indices[0], [f'stock_{idx:02d}' for idx in range(10)])
        self.assertEqual(int(np.sum(relevance_scores[0])), 1)
        self.assertEqual(float(targets[0][-1]), 9.0)
        self.assertAlmostEqual(float(vol_targets[0][-1]), 0.9, places=6)

    def test_create_ranking_dataset_vectorized_returns_daily_samples_without_dataframe_groupby_contract_change(self):
        dates = pd.date_range('2024-01-02', periods=4, freq='B')
        rows = []
        for stock_idx in range(10):
            instrument = f'{stock_idx:06d}'
            for day_idx, trade_date in enumerate(dates):
                rows.append(
                    {
                        'instrument': instrument,
                        '日期': trade_date.strftime('%Y-%m-%d'),
                        'label': float(stock_idx),
                        'vol_label': float(stock_idx) / 10.0,
                        'feat_a': float(day_idx + stock_idx / 100.0),
                        'feat_b': float((day_idx + 1) * 2 + stock_idx / 100.0),
                    }
                )
        df = pd.DataFrame(rows)

        sequences, targets, relevance_scores, stock_indices, vol_targets = create_ranking_dataset_vectorized(
            df,
            ['feat_a', 'feat_b'],
            sequence_length=2,
            min_window_end_date=dates[1],
            max_window_end_date=dates[1],
        )

        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0].shape, (10, 2, 2))
        self.assertEqual(len(targets[0]), 10)
        self.assertEqual(len(vol_targets[0]), 10)
        self.assertEqual(len(stock_indices[0]), 10)
        self.assertEqual(int(np.sum(relevance_scores[0])), 1)


if __name__ == '__main__':
    unittest.main()
