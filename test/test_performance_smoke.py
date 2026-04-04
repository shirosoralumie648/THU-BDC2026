import os
import sys
import unittest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from benchmarks.factor_graph_path import run_factor_graph_benchmark
from benchmarks.training_path import run_dataset_build_benchmark
from benchmarks.training_path import run_prediction_path_benchmark
from benchmarks.training_path import run_training_path_benchmark


class PerformanceSmokeTests(unittest.TestCase):
    def test_training_benchmark_emits_required_metrics(self):
        metrics = run_training_path_benchmark()

        self.assertEqual(metrics['benchmark_name'], 'training_path')
        self.assertGreater(metrics['wall_time_seconds'], 0.0)
        self.assertGreater(metrics['rows'], 0)
        self.assertGreater(metrics['stocks'], 0)
        self.assertGreater(metrics['features'], 0)
        self.assertGreater(metrics['samples'], 0)
        self.assertGreater(metrics['samples_per_second'], 0.0)
        self.assertIn('peak_memory_mb', metrics)

    def test_factor_graph_benchmark_emits_required_metrics(self):
        metrics = run_factor_graph_benchmark()

        self.assertEqual(metrics['benchmark_name'], 'factor_graph_path')
        self.assertGreater(metrics['wall_time_seconds'], 0.0)
        self.assertGreater(metrics['rows'], 0)
        self.assertGreater(metrics['stocks'], 0)
        self.assertGreater(metrics['factor_columns'], 0)
        self.assertGreater(metrics['samples_per_second'], 0.0)
        self.assertIn('peak_memory_mb', metrics)

    def test_dataset_build_benchmark_emits_required_metrics(self):
        metrics = run_dataset_build_benchmark()

        self.assertEqual(metrics['benchmark_name'], 'dataset_build_path')
        self.assertGreater(metrics['wall_time_seconds'], 0.0)
        self.assertGreater(metrics['base_rows'], 0)
        self.assertGreater(metrics['train_rows'], 0)
        self.assertGreater(metrics['test_rows'], 0)
        self.assertGreater(metrics['rows_per_second'], 0.0)
        self.assertIn('peak_memory_mb', metrics)

    def test_prediction_benchmark_emits_required_metrics(self):
        metrics = run_prediction_path_benchmark()

        self.assertEqual(metrics['benchmark_name'], 'prediction_path')
        self.assertGreater(metrics['wall_time_seconds'], 0.0)
        self.assertGreater(metrics['rows'], 0)
        self.assertGreater(metrics['stocks'], 0)
        self.assertGreater(metrics['features'], 0)
        self.assertGreater(metrics['sequence_count'], 0)
        self.assertGreater(metrics['rows_per_second'], 0.0)
        self.assertIn('peak_memory_mb', metrics)


if __name__ == '__main__':
    unittest.main()
