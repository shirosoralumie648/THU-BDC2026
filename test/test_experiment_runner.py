import os
import sys
import unittest

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from experiments.ensemble import summarize_multi_seed_runs
from experiments.runner import summarize_experiment_run
from experiments.splits import build_rolling_validation_folds


class RollingValidationSplitTests(unittest.TestCase):
    def _build_frame(self, num_dates=40, stock_ids=('AAA', 'BBB')):
        rows = []
        for offset in range(num_dates):
            trade_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=offset)
            for stock_id in stock_ids:
                rows.append({'日期': trade_date.strftime('%Y-%m-%d'), '股票代码': stock_id, 'label': float(offset)})
        return pd.DataFrame(rows)

    def test_build_rolling_validation_folds_supports_purge_and_embargo(self):
        df = self._build_frame()
        runtime_config = {
            'rolling_val_num_folds': 2,
            'rolling_val_window_size': 4,
            'rolling_val_step_size': 4,
            'rolling_val_purge_days': 3,
            'rolling_val_embargo_days': 2,
            'label_horizon': 3,
        }

        train_df, val_df, folds = build_rolling_validation_folds(df, sequence_length=3, runtime_config=runtime_config)

        unique_dates = [pd.Timestamp(value).normalize() for value in sorted(pd.to_datetime(df['日期']).unique())]
        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}

        self.assertEqual(len(folds), 2)
        gap_days = date_to_idx[folds[1]['start_date']] - date_to_idx[folds[0]['end_date']] - 1
        self.assertEqual(gap_days, 2)

        earliest_start_idx = date_to_idx[folds[0]['start_date']]
        train_max_idx = date_to_idx[pd.to_datetime(train_df['日期']).max().normalize()]
        val_min_idx = date_to_idx[pd.to_datetime(val_df['日期']).min().normalize()]

        self.assertEqual(train_max_idx, earliest_start_idx - runtime_config['rolling_val_purge_days'] - 1)
        self.assertEqual(val_min_idx, earliest_start_idx - 2)
        self.assertEqual(folds[0]['purge_days'], 3)
        self.assertEqual(folds[0]['embargo_days'], 2)


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.strategy_candidates = [
            {'name': 'equal_top2', 'top_k': 2, 'weighting': 'equal'},
            {'name': 'softmax_top2', 'top_k': 2, 'weighting': 'softmax'},
        ]
        self.runtime_config = {'strategy_selection_mode': 'risk_adjusted', 'selection_metric': 'auto'}

    def test_summarize_experiment_run_reports_best_strategy_and_fold_diagnostics(self):
        eval_metrics = {
            'return_equal_top2': 0.08,
            'return_equal_top2_std': 0.02,
            'return_equal_top2_risk_adjusted': 0.06,
            'return_softmax_top2': 0.09,
            'return_softmax_top2_std': 0.05,
            'return_softmax_top2_risk_adjusted': 0.04,
        }
        fold_results = [
            {
                'name': 'fold_1',
                'start_date': pd.Timestamp('2024-02-01'),
                'end_date': pd.Timestamp('2024-02-05'),
                'num_samples': 12,
                'loss': 0.31,
                'metrics': dict(eval_metrics),
            },
            {
                'name': 'fold_2',
                'start_date': pd.Timestamp('2024-02-08'),
                'end_date': pd.Timestamp('2024-02-12'),
                'num_samples': 10,
                'loss': 0.29,
                'metrics': {
                    'return_equal_top2': 0.03,
                    'return_equal_top2_std': 0.01,
                    'return_equal_top2_risk_adjusted': 0.02,
                    'return_softmax_top2': 0.11,
                    'return_softmax_top2_std': 0.03,
                    'return_softmax_top2_risk_adjusted': 0.08,
                },
            },
        ]

        summary = summarize_experiment_run(
            eval_loss=0.30,
            eval_metrics=eval_metrics,
            fold_results=fold_results,
            strategy_candidates=self.strategy_candidates,
            runtime_config=self.runtime_config,
        )

        self.assertEqual(summary['best_candidate']['name'], 'equal_top2')
        self.assertAlmostEqual(summary['best_score'], 0.06, places=8)
        self.assertEqual(len(summary['fold_diagnostics']), 2)
        self.assertEqual(summary['fold_diagnostics'][1]['best_candidate']['name'], 'softmax_top2')
        self.assertIn('equal_top2=mean:0.0800', summary['strategy_summary'])

    def test_summarize_multi_seed_runs_aggregates_metrics_and_selects_best_run(self):
        run_summaries = [
            {
                'run_name': 'seed_1',
                'loss': 0.30,
                'metrics': {
                    'return_equal_top2': 0.08,
                    'return_equal_top2_std': 0.02,
                    'return_equal_top2_risk_adjusted': 0.06,
                    'return_softmax_top2': 0.09,
                    'return_softmax_top2_std': 0.05,
                    'return_softmax_top2_risk_adjusted': 0.04,
                },
            },
            {
                'run_name': 'seed_2',
                'loss': 0.28,
                'metrics': {
                    'return_equal_top2': 0.07,
                    'return_equal_top2_std': 0.01,
                    'return_equal_top2_risk_adjusted': 0.06,
                    'return_softmax_top2': 0.12,
                    'return_softmax_top2_std': 0.02,
                    'return_softmax_top2_risk_adjusted': 0.10,
                },
            },
        ]

        summary = summarize_multi_seed_runs(
            run_summaries=run_summaries,
            strategy_candidates=self.strategy_candidates,
            runtime_config=self.runtime_config,
        )

        self.assertEqual(summary['num_runs'], 2)
        self.assertAlmostEqual(summary['aggregate_metrics']['return_softmax_top2'], 0.105, places=8)
        self.assertEqual(summary['best_candidate']['name'], 'softmax_top2')
        self.assertEqual(summary['best_run']['run_name'], 'seed_2')


if __name__ == '__main__':
    unittest.main()
