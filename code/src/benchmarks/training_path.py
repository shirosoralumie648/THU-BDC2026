import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import create_ranking_dataset_vectorized


def _peak_memory_mb():
    try:
        import resource

        usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return float(usage_kb) / (1024.0 * 1024.0)
        return float(usage_kb) / 1024.0
    except Exception:
        return None


def _build_training_frame(num_stocks: int, num_days: int, num_features: int) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(20260404)
    dates = pd.date_range('2024-01-02', periods=num_days, freq='B')
    feature_names = [f'feat_{idx:02d}' for idx in range(num_features)]
    rows = []

    for stock_idx in range(num_stocks):
        base_price = 10.0 + stock_idx * 0.7
        stock_bias = stock_idx * 0.001
        for day_idx, trade_date in enumerate(dates):
            row = {
                'instrument': f'{stock_idx:06d}',
                '日期': trade_date.strftime('%Y-%m-%d'),
                'label': float(np.sin(day_idx / 7.0) * 0.02 + stock_bias),
                'vol_label': float(0.01 + abs(np.cos(day_idx / 9.0)) * 0.015 + stock_bias),
            }
            for feature_idx, feature_name in enumerate(feature_names):
                seasonal = np.sin((day_idx + feature_idx) / 5.0)
                noise = rng.normal(0.0, 0.01)
                row[feature_name] = float(base_price * 0.01 + seasonal + noise)
            rows.append(row)

    return pd.DataFrame(rows), feature_names


def run_training_path_benchmark(
    *,
    num_stocks: int = 24,
    num_days: int = 80,
    num_features: int = 12,
    sequence_length: int = 20,
):
    df, feature_names = _build_training_frame(num_stocks, num_days, num_features)

    started_at = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sequences, targets, relevance, stock_indices, vol_targets = create_ranking_dataset_vectorized(
            df,
            feature_names,
            sequence_length,
        )
    wall_time_seconds = time.perf_counter() - started_at
    samples = len(sequences)

    return {
        'benchmark_name': 'training_path',
        'rows': int(len(df)),
        'stocks': int(num_stocks),
        'features': int(len(feature_names)),
        'sequence_length': int(sequence_length),
        'samples': int(samples),
        'targets': int(sum(len(day_targets) for day_targets in targets)),
        'wall_time_seconds': float(wall_time_seconds),
        'samples_per_second': float(samples / max(wall_time_seconds, 1e-9)),
        'peak_memory_mb': _peak_memory_mb(),
    }


def main():
    print(json.dumps(run_training_path_benchmark(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
