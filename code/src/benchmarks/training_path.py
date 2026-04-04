import contextlib
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from predict import build_inference_sequences
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


def _write_minimal_pipeline_config(cfg_dir: Path):
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / 'datasets.yaml').write_text('version: 1\ndatasets: {}\n', encoding='utf-8')
    (cfg_dir / 'factors.yaml').write_text('version: 1\nlayer_order: []\nfactor_nodes: []\n', encoding='utf-8')
    (cfg_dir / 'storage.yaml').write_text(
        '\n'.join(
            [
                'version: 1',
                'layers:',
                '  raw: {uri_template: data/raw/{dataset}/{run_id}.csv}',
                '  curated: {uri_template: data/curated/{dataset}/{run_id}.csv}',
                '  feature_long: {uri_template: data/feature_long/{dataset}.csv}',
                '  feature_wide: {uri_template: data/feature_wide/{dataset}.csv}',
                '  datasets: {uri_template: data/datasets/{dataset}.csv}',
                '  manifests: {uri_template: data/manifests/{dataset}.json}',
            ]
        ),
        encoding='utf-8',
    )


def _build_dataset_frame() -> pd.DataFrame:
    rows = []
    for stock_idx, code in enumerate(['000001.SZ', '000002.SZ', '000003.SZ']):
        for trade_date in pd.date_range('2024-01-02', periods=18, freq='B'):
            close = 10.0 + stock_idx + trade_date.day * 0.01
            rows.append({'股票代码': code, '日期': trade_date.strftime('%Y-%m-%d'), '收盘': close})
        for trade_date in pd.date_range('2024-03-04', periods=6, freq='B'):
            close = 11.0 + stock_idx + trade_date.day * 0.01
            rows.append({'股票代码': code, '日期': trade_date.strftime('%Y-%m-%d'), '收盘': close})
    return pd.DataFrame(rows)


def _build_prediction_frame(num_stocks: int, num_days: int, num_features: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    rng = np.random.default_rng(20260405)
    dates = pd.date_range('2024-01-02', periods=num_days, freq='B')
    feature_names = [f'pred_feat_{idx:02d}' for idx in range(num_features)]
    rows = []
    stock_ids = []

    for stock_idx in range(num_stocks):
        stock_id = f'{stock_idx + 1:06d}'
        stock_ids.append(stock_id)
        base_price = 8.0 + stock_idx * 0.5
        for day_idx, trade_date in enumerate(dates):
            row = {
                '股票代码': stock_id,
                '日期': trade_date,
            }
            for feature_idx, feature_name in enumerate(feature_names):
                seasonal = np.cos((day_idx + feature_idx) / 6.0)
                noise = rng.normal(0.0, 0.01)
                row[feature_name] = float(base_price * 0.02 + seasonal + noise)
            rows.append(row)

    return pd.DataFrame(rows), feature_names, stock_ids


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


def run_dataset_build_benchmark():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg_dir = tmp_path / 'config'
        output_dir = tmp_path / 'dataset_output'
        base_path = tmp_path / 'stock_data.csv'
        _write_minimal_pipeline_config(cfg_dir)

        base_df = _build_dataset_frame()
        base_df.to_csv(base_path, index=False, encoding='utf-8')

        cmd = [
            sys.executable,
            str(SRC_ROOT / 'manage_data.py'),
            'build-dataset',
            '--base-input',
            str(base_path),
            '--pipeline-config-dir',
            str(cfg_dir),
            '--output-dir',
            str(output_dir),
            '--train-start',
            '2024-01-01',
            '--train-end',
            '2024-01-31',
            '--test-start',
            '2024-03-01',
            '--test-end',
            '2024-03-31',
        ]

        started_at = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        wall_time_seconds = time.perf_counter() - started_at
        if result.returncode != 0:
            raise RuntimeError(
                'dataset build benchmark failed:\n'
                f'stdout={result.stdout}\n'
                f'stderr={result.stderr}'
            )

        train_df = pd.read_csv(output_dir / 'train.csv', dtype={'股票代码': str})
        test_df = pd.read_csv(output_dir / 'test.csv', dtype={'股票代码': str})
        base_rows = len(base_df)

        return {
            'benchmark_name': 'dataset_build_path',
            'base_rows': int(base_rows),
            'train_rows': int(len(train_df)),
            'test_rows': int(len(test_df)),
            'wall_time_seconds': float(wall_time_seconds),
            'rows_per_second': float(base_rows / max(wall_time_seconds, 1e-9)),
            'peak_memory_mb': _peak_memory_mb(),
        }


def run_prediction_path_benchmark(
    *,
    num_stocks: int = 24,
    num_days: int = 40,
    num_features: int = 12,
    sequence_length: int = 20,
):
    df, feature_names, stock_ids = _build_prediction_frame(num_stocks, num_days, num_features)
    latest_date = pd.to_datetime(df['日期']).max()

    started_at = time.perf_counter()
    sequences, sequence_stock_ids = build_inference_sequences(
        df,
        feature_names,
        sequence_length,
        stock_ids,
        latest_date,
    )
    wall_time_seconds = time.perf_counter() - started_at

    return {
        'benchmark_name': 'prediction_path',
        'rows': int(len(df)),
        'stocks': int(len(stock_ids)),
        'features': int(len(feature_names)),
        'sequence_length': int(sequence_length),
        'sequence_count': int(len(sequence_stock_ids)),
        'wall_time_seconds': float(wall_time_seconds),
        'rows_per_second': float(len(df) / max(wall_time_seconds, 1e-9)),
        'peak_memory_mb': _peak_memory_mb(),
    }


def main():
    print(json.dumps(run_training_path_benchmark(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
