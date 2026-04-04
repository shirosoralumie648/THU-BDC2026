import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _peak_memory_mb():
    try:
        import resource

        usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return float(usage_kb) / (1024.0 * 1024.0)
        return float(usage_kb) / 1024.0
    except Exception:
        return None


def _write_benchmark_inputs(tmp_path: Path, *, num_stocks: int, num_days: int):
    base_path = tmp_path / 'stock_data.csv'
    hf_path = tmp_path / 'hf_daily.csv'
    macro_path = tmp_path / 'macro_series.csv'
    output_path = tmp_path / 'features.csv'
    manifest_path = tmp_path / 'manifest.json'

    dates = pd.date_range('2024-01-02', periods=num_days, freq='B')
    base_rows = []
    hf_rows = []
    for stock_idx in range(num_stocks):
        code = f'{stock_idx + 1:06d}.SZ'
        base_price = 10.0 + stock_idx
        for day_idx, trade_date in enumerate(dates):
            close = base_price + day_idx * 0.15
            base_rows.append(
                {
                    '股票代码': code,
                    '日期': trade_date.strftime('%Y-%m-%d'),
                    '开盘': close - 0.1,
                    '收盘': close,
                    '最高': close + 0.2,
                    '最低': close - 0.2,
                    '成交量': 1000 + day_idx * 5,
                    '成交额': (1000 + day_idx * 5) * close,
                    '换手率': 0.8 + day_idx * 0.01,
                    '涨跌幅': 0.01,
                }
            )
            hf_rows.append(
                {
                    '股票代码': code,
                    '日期': trade_date.strftime('%Y-%m-%d'),
                    'hf_realized_vol': 0.05 + day_idx * 0.001,
                    'hf_last_tail_ret': 0.002 + day_idx * 0.0001,
                    'hf_last_tail_amount_share': 0.20 + day_idx * 0.001,
                }
            )

    macro_rows = []
    for series_id, base_val in [('m2_yoy', 8.0), ('shibor_3m', 2.5), ('usdcny', 7.1)]:
        for day_idx, trade_date in enumerate(dates):
            macro_rows.append(
                {
                    'series_id': series_id,
                    'available_time': (trade_date + pd.Timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'),
                    'value': base_val + day_idx * 0.01,
                }
            )

    pd.DataFrame(base_rows).to_csv(base_path, index=False, encoding='utf-8')
    pd.DataFrame(hf_rows).to_csv(hf_path, index=False, encoding='utf-8')
    pd.DataFrame(macro_rows).to_csv(macro_path, index=False, encoding='utf-8')
    return base_path, hf_path, macro_path, output_path, manifest_path


def run_factor_graph_benchmark(*, num_stocks: int = 8, num_days: int = 30):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        base_path, hf_path, macro_path, output_path, manifest_path = _write_benchmark_inputs(
            tmp_path,
            num_stocks=num_stocks,
            num_days=num_days,
        )

        cmd = [
            sys.executable,
            str(SRC_ROOT / 'build_factor_graph.py'),
            '--pipeline-config-dir',
            './config',
            '--feature-set-version',
            'vbench',
            '--base-input',
            str(base_path),
            '--hf-daily-input',
            str(hf_path),
            '--macro-input',
            str(macro_path),
            '--output',
            str(output_path),
            '--manifest-path',
            str(manifest_path),
            '--strict',
            '--run-id',
            'benchmark-run',
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
                'factor graph benchmark failed:\n'
                f'stdout={result.stdout}\n'
                f'stderr={result.stderr}'
            )

        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        out_df = pd.read_csv(output_path, dtype={'股票代码': str})
        factor_columns = [
            col
            for col in out_df.columns
            if col not in {'股票代码', '日期', 'instrument_id', 'trade_date'}
        ]

        return {
            'benchmark_name': 'factor_graph_path',
            'rows': int(len(out_df)),
            'stocks': int(num_stocks),
            'factor_columns': int(len(factor_columns)),
            'wall_time_seconds': float(wall_time_seconds),
            'samples_per_second': float(len(out_df) / max(wall_time_seconds, 1e-9)),
            'peak_memory_mb': _peak_memory_mb(),
            'factor_fingerprint': manifest.get('factor_fingerprint', ''),
        }


def main():
    print(json.dumps(run_factor_graph_benchmark(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
