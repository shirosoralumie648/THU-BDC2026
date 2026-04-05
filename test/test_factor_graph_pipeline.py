import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from build_factor_graph import _build_macro_cutoff_frame
from build_factor_graph import _compute_macro_series_asof


class FactorGraphPipelineTests(unittest.TestCase):
    def test_compute_macro_series_asof_forward_fill_does_not_revive_stale_values(self):
        macro_cutoff_frame = _build_macro_cutoff_frame(
            pd.Series(pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']))
        )
        macro_series_df = pd.DataFrame(
            {
                'available_time': pd.to_datetime(['2024-01-01 15:00:00']),
                'value': [2.0],
            }
        )

        out = _compute_macro_series_asof(
            macro_cutoff_frame,
            macro_series_df,
            max_staleness_days=2,
            fill_method='forward',
        )

        self.assertEqual(out['value'].iloc[0], 2.0)
        self.assertEqual(out['value'].iloc[1], 2.0)
        self.assertEqual(out['value'].iloc[2], 2.0)
        self.assertTrue(pd.isna(out['value'].iloc[3]))

    def test_intraday_aggregation_source_no_longer_uses_groupby_apply(self):
        source_path = os.path.join(SRC_ROOT, 'build_factor_graph.py')
        with open(source_path, 'r', encoding='utf-8') as f:
            source = f.read()

        self.assertNotIn('groupby(keys, sort=False).apply', source)

    def test_macro_asof_join_source_no_longer_merges_base_frame_per_node(self):
        source_path = os.path.join(SRC_ROOT, 'build_factor_graph.py')
        with open(source_path, 'r', encoding='utf-8') as f:
            source = f.read()

        self.assertNotIn(
            "base_df = base_df.merge(series_asof.rename(columns={'value': output_col}), on='trade_date', how='left')",
            source,
        )

    def test_macro_asof_join_source_no_longer_filters_full_macro_frame_per_series(self):
        source_path = os.path.join(SRC_ROOT, 'build_factor_graph.py')
        with open(source_path, 'r', encoding='utf-8') as f:
            source = f.read()

        self.assertNotIn(
            "right = macro_df[macro_df['series_id'] == str(series_id)].copy()",
            source,
        )

    def test_factor_graph_source_uses_canonical_metadata_helper_for_output(self):
        source_path = os.path.join(SRC_ROOT, 'build_factor_graph.py')
        with open(source_path, 'r', encoding='utf-8') as f:
            source = f.read()

        self.assertIn(
            'output_csv_meta = build_canonical_csv_metadata_from_dataframe(',
            source,
        )

    def test_build_factor_graph_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_path = tmp_path / 'stock_data.csv'
            hf_path = tmp_path / 'hf_daily.csv'
            macro_path = tmp_path / 'macro_series.csv'
            output_path = tmp_path / 'features.csv'
            manifest_path = tmp_path / 'manifest.json'

            dates = pd.date_range('2024-01-02', periods=25, freq='D')
            base_rows = []
            hf_rows = []
            for code, base_price in [('000001.SZ', 10.0), ('000002.SZ', 20.0)]:
                for i, dt in enumerate(dates):
                    close = base_price + i * 0.2
                    base_rows.append(
                        {
                            '股票代码': code,
                            '日期': dt.strftime('%Y-%m-%d'),
                            '开盘': close - 0.1,
                            '收盘': close,
                            '最高': close + 0.2,
                            '最低': close - 0.2,
                            '成交量': 1000 + i * 5,
                            '成交额': (1000 + i * 5) * close,
                            '换手率': 0.8 + i * 0.01,
                            '涨跌幅': 0.01,
                        }
                    )
                    hf_rows.append(
                        {
                            '股票代码': code,
                            '日期': dt.strftime('%Y-%m-%d'),
                            'hf_realized_vol': 0.05 + i * 0.001,
                            'hf_last_tail_ret': 0.002 + i * 0.0001,
                            'hf_last_tail_amount_share': 0.20 + i * 0.001,
                        }
                    )
            pd.DataFrame(base_rows).to_csv(base_path, index=False, encoding='utf-8')
            pd.DataFrame(hf_rows).to_csv(hf_path, index=False, encoding='utf-8')

            macro_rows = []
            for series_id, base_val in [('m2_yoy', 8.0), ('shibor_3m', 2.5), ('usdcny', 7.1)]:
                for i, dt in enumerate(dates):
                    macro_rows.append(
                        {
                            'series_id': series_id,
                            'available_time': (dt + pd.Timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'),
                            'value': base_val + i * 0.01,
                        }
                    )
            pd.DataFrame(macro_rows).to_csv(macro_path, index=False, encoding='utf-8')

            cmd = [
                sys.executable,
                os.path.join(SRC_ROOT, 'build_factor_graph.py'),
                '--pipeline-config-dir',
                './config',
                '--feature-set-version',
                'vtest',
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
                'unit-test-run',
            ]
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
            if result.returncode != 0:
                raise AssertionError(
                    f'build_factor_graph failed: rc={result.returncode}\n'
                    f'stdout={result.stdout}\n'
                    f'stderr={result.stderr}'
                )

            self.assertTrue(output_path.exists())
            self.assertTrue(manifest_path.exists())

            out_df = pd.read_csv(output_path, dtype={'股票代码': str})
            self.assertEqual(len(out_df), len(base_rows))
            self.assertIn('f_ret_1d', out_df.columns)
            self.assertIn('f_hf_realized_vol_1d', out_df.columns)
            self.assertIn('f_macro_m2_yoy', out_df.columns)
            self.assertIn('f_macro_shibor_3m', out_df.columns)
            self.assertIn('f_macro_usdcny', out_df.columns)
            self.assertIn('f_momentum_quality_score', out_df.columns)
            self.assertIn('f_regime_risk_on', out_df.columns)

            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            self.assertEqual(manifest.get('action'), 'build_factor_graph')
            self.assertEqual(manifest.get('run_id'), 'unit-test-run')
            self.assertTrue(bool(manifest.get('factor_fingerprint')))
            self.assertEqual(manifest.get('row_count'), len(base_rows))

            statuses = manifest.get('node_status', [])
            self.assertTrue(any(item.get('id') == 'ret_1d' and item.get('status') == 'ok' for item in statuses))
            self.assertTrue(any(item.get('id') == 'macro_m2_yoy_asof' and item.get('status') == 'ok' for item in statuses))
            self.assertTrue(any(item.get('id') == 'macro_shibor_3m_asof' and item.get('status') == 'ok' for item in statuses))
            self.assertTrue(any(item.get('id') == 'macro_usdcny_asof' and item.get('status') == 'ok' for item in statuses))

    def test_build_factor_graph_with_hf_minute_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_path = tmp_path / 'stock_data.csv'
            hf_minute_path = tmp_path / 'hf_minute.csv'
            macro_path = tmp_path / 'macro_series.csv'
            output_path = tmp_path / 'features_minute.csv'
            manifest_path = tmp_path / 'manifest_minute.json'

            dates = pd.date_range('2024-02-01', periods=8, freq='D')
            base_rows = []
            minute_rows = []
            for code, base_price in [('000001.SZ', 11.0), ('000002.SZ', 22.0)]:
                for d_idx, dt in enumerate(dates):
                    close_daily = base_price + d_idx * 0.15
                    base_rows.append(
                        {
                            '股票代码': code,
                            '日期': dt.strftime('%Y-%m-%d'),
                            '开盘': close_daily - 0.1,
                            '收盘': close_daily,
                            '最高': close_daily + 0.2,
                            '最低': close_daily - 0.2,
                            '成交量': 1000 + d_idx * 3,
                            '成交额': (1000 + d_idx * 3) * close_daily,
                            '换手率': 0.7 + d_idx * 0.01,
                            '涨跌幅': 0.008,
                        }
                    )
                    for i in range(60):
                        ts = dt + pd.Timedelta(hours=14, minutes=i)
                        close_1m = close_daily + i * 0.002
                        minute_rows.append(
                            {
                                '股票代码': code,
                                'datetime': ts.strftime('%Y-%m-%d %H:%M:%S'),
                                'close': close_1m,
                                'amount': (500 + i * 2) * close_1m,
                            }
                        )
            pd.DataFrame(base_rows).to_csv(base_path, index=False, encoding='utf-8')
            pd.DataFrame(minute_rows).to_csv(hf_minute_path, index=False, encoding='utf-8')

            macro_rows = []
            for series_id, base_val in [('m2_yoy', 7.8), ('shibor_3m', 2.2), ('usdcny', 7.0)]:
                for i, dt in enumerate(dates):
                    macro_rows.append(
                        {
                            'series_id': series_id,
                            'available_time': (dt + pd.Timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'),
                            'value': base_val + i * 0.02,
                        }
                    )
            pd.DataFrame(macro_rows).to_csv(macro_path, index=False, encoding='utf-8')

            cmd = [
                sys.executable,
                os.path.join(SRC_ROOT, 'build_factor_graph.py'),
                '--pipeline-config-dir',
                './config',
                '--feature-set-version',
                'vminute',
                '--base-input',
                str(base_path),
                '--hf-minute-input',
                str(hf_minute_path),
                '--macro-input',
                str(macro_path),
                '--output',
                str(output_path),
                '--manifest-path',
                str(manifest_path),
                '--strict',
                '--run-id',
                'unit-test-minute',
            ]
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
            if result.returncode != 0:
                raise AssertionError(
                    f'build_factor_graph failed: rc={result.returncode}\n'
                    f'stdout={result.stdout}\n'
                    f'stderr={result.stderr}'
                )

            out_df = pd.read_csv(output_path, dtype={'股票代码': str})
            self.assertIn('f_hf_realized_vol_1d', out_df.columns)
            self.assertIn('f_hf_tail_ret_30m', out_df.columns)
            self.assertIn('f_hf_tail_amount_share_30m', out_df.columns)
            self.assertTrue(out_df['f_hf_realized_vol_1d'].notna().any())
            self.assertTrue(out_df['f_hf_tail_ret_30m'].notna().any())
            self.assertTrue(out_df['f_hf_tail_amount_share_30m'].notna().any())

            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            self.assertEqual(manifest.get('run_id'), 'unit-test-minute')
            src_map = manifest.get('intraday_source_map', {})
            self.assertTrue(str(src_map.get('f_hf_realized_vol_1d', '')).startswith('hf_minute_input'))

            statuses = manifest.get('node_status', [])
            self.assertTrue(
                any(
                    item.get('id') == 'hf_realized_vol_1d'
                    and item.get('status') == 'ok'
                    and 'hf_minute_input' in item.get('message', '')
                    for item in statuses
                )
            )

    def test_custom_intraday_expression_from_temp_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_dir = tmp_path / 'config'
            cfg_dir.mkdir(parents=True, exist_ok=True)
            base_path = tmp_path / 'stock_data.csv'
            hf_minute_path = tmp_path / 'hf_minute.csv'
            output_path = tmp_path / 'features_custom.csv'
            manifest_path = tmp_path / 'manifest_custom.json'

            (cfg_dir / 'datasets.yaml').write_text(
                '\n'.join(
                    [
                        'version: 1',
                        'datasets:',
                        '  market_bar_1d:',
                        '    source: {name: mock}',
                        '    schema:',
                        '      columns: {instrument_id: {source: code}, trade_date: {source: date}}',
                        '    storage: {curated_uri: data/curated/mock.parquet}',
                    ]
                ),
                encoding='utf-8',
            )
            (cfg_dir / 'storage.yaml').write_text(
                '\n'.join(
                    [
                        'version: 1',
                        'layers:',
                        '  raw: {uri_template: data/raw/mock.parquet}',
                        '  curated: {uri_template: data/curated/mock.parquet}',
                        '  feature_long: {uri_template: data/feature/long/mock.parquet}',
                        '  feature_wide: {uri_template: data/feature/wide/mock.parquet}',
                        '  datasets: {uri_template: data/datasets/mock.parquet}',
                        '  manifests: {uri_template: data/manifests/mock.json}',
                    ]
                ),
                encoding='utf-8',
            )
            (cfg_dir / 'factors.yaml').write_text(
                '\n'.join(
                    [
                        'version: 1',
                        'layer_order: [L1_base_hf2d]',
                        'factor_nodes:',
                        '  - id: close_tail_share_15m',
                        '    layer: L1_base_hf2d',
                        '    source: market_bar_1m',
                        '    dependencies: []',
                        '    compute:',
                        '      engine: intraday_aggregate',
                        '      expression: "sum(close, last=15m) / sum(close, full_day)"',
                        '      min_bars: 5',
                        '    output:',
                        '      column: f_hf_close_tail_share_15m',
                        '      dtype: float32',
                        'factor_views:',
                        '  - id: factor_wide_v1',
                        '    layout: wide',
                        '    key: [trade_date, instrument_id]',
                        '    include_factor_columns: [f_hf_close_tail_share_15m]',
                        '    export:',
                        '      csv_compat_uri: "data/datasets/features/train_features_{feature_set_version}.csv"',
                    ]
                ),
                encoding='utf-8',
            )

            dates = pd.date_range('2024-03-01', periods=4, freq='D')
            base_rows = []
            minute_rows = []
            for code, base_price in [('000001.SZ', 10.0), ('000002.SZ', 20.0)]:
                for d_idx, dt in enumerate(dates):
                    close_daily = base_price + d_idx * 0.2
                    base_rows.append(
                        {
                            '股票代码': code,
                            '日期': dt.strftime('%Y-%m-%d'),
                            '开盘': close_daily - 0.1,
                            '收盘': close_daily,
                            '最高': close_daily + 0.2,
                            '最低': close_daily - 0.2,
                            '成交量': 1000 + d_idx * 4,
                            '成交额': (1000 + d_idx * 4) * close_daily,
                            '换手率': 0.6 + d_idx * 0.01,
                            '涨跌幅': 0.007,
                        }
                    )
                    for i in range(40):
                        ts = dt + pd.Timedelta(hours=14, minutes=i)
                        close_1m = close_daily + i * 0.003
                        minute_rows.append(
                            {
                                '股票代码': code,
                                'datetime': ts.strftime('%Y-%m-%d %H:%M:%S'),
                                'close': close_1m,
                                'amount': (300 + i) * close_1m,
                            }
                        )
            pd.DataFrame(base_rows).to_csv(base_path, index=False, encoding='utf-8')
            pd.DataFrame(minute_rows).to_csv(hf_minute_path, index=False, encoding='utf-8')

            cmd = [
                sys.executable,
                os.path.join(SRC_ROOT, 'build_factor_graph.py'),
                '--pipeline-config-dir',
                str(cfg_dir),
                '--feature-set-version',
                'vcustom',
                '--base-input',
                str(base_path),
                '--hf-minute-input',
                str(hf_minute_path),
                '--output',
                str(output_path),
                '--manifest-path',
                str(manifest_path),
                '--strict',
                '--run-id',
                'unit-test-custom-intraday',
            ]
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
            if result.returncode != 0:
                raise AssertionError(
                    f'build_factor_graph failed: rc={result.returncode}\n'
                    f'stdout={result.stdout}\n'
                    f'stderr={result.stderr}'
                )

            out_df = pd.read_csv(output_path, dtype={'股票代码': str})
            self.assertIn('f_hf_close_tail_share_15m', out_df.columns)
            self.assertTrue(out_df['f_hf_close_tail_share_15m'].notna().any())

            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            src_map = manifest.get('intraday_source_map', {})
            expr = str(src_map.get('f_hf_close_tail_share_15m', ''))
            self.assertIn('sum(close, last=', expr)

    def test_invalid_intraday_expression_strict_and_non_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_dir = tmp_path / 'config'
            cfg_dir.mkdir(parents=True, exist_ok=True)
            base_path = tmp_path / 'stock_data.csv'
            hf_minute_path = tmp_path / 'hf_minute.csv'
            output_non_strict = tmp_path / 'features_non_strict.csv'
            manifest_non_strict = tmp_path / 'manifest_non_strict.json'
            output_strict = tmp_path / 'features_strict.csv'
            manifest_strict = tmp_path / 'manifest_strict.json'

            (cfg_dir / 'datasets.yaml').write_text(
                '\n'.join(
                    [
                        'version: 1',
                        'datasets:',
                        '  market_bar_1d:',
                        '    source: {name: mock}',
                        '    schema:',
                        '      columns: {instrument_id: {source: code}, trade_date: {source: date}}',
                        '    storage: {curated_uri: data/curated/mock.parquet}',
                    ]
                ),
                encoding='utf-8',
            )
            (cfg_dir / 'storage.yaml').write_text(
                '\n'.join(
                    [
                        'version: 1',
                        'layers:',
                        '  raw: {uri_template: data/raw/mock.parquet}',
                        '  curated: {uri_template: data/curated/mock.parquet}',
                        '  feature_long: {uri_template: data/feature/long/mock.parquet}',
                        '  feature_wide: {uri_template: data/feature/wide/mock.parquet}',
                        '  datasets: {uri_template: data/datasets/mock.parquet}',
                        '  manifests: {uri_template: data/manifests/mock.json}',
                    ]
                ),
                encoding='utf-8',
            )
            (cfg_dir / 'factors.yaml').write_text(
                '\n'.join(
                    [
                        'version: 1',
                        'layer_order: [L1_base_hf2d]',
                        'factor_nodes:',
                        '  - id: invalid_intraday_expr',
                        '    layer: L1_base_hf2d',
                        '    source: market_bar_1m',
                        '    dependencies: []',
                        '    compute:',
                        '      engine: intraday_aggregate',
                        '      expression: "bad_func(close)"',
                        '      min_bars: 5',
                        '    output:',
                        '      column: f_hf_invalid_expr',
                        '      dtype: float32',
                        'factor_views:',
                        '  - id: factor_wide_v1',
                        '    layout: wide',
                        '    key: [trade_date, instrument_id]',
                        '    include_factor_columns: [f_hf_invalid_expr]',
                        '    export:',
                        '      csv_compat_uri: "data/datasets/features/train_features_{feature_set_version}.csv"',
                    ]
                ),
                encoding='utf-8',
            )

            dates = pd.date_range('2024-04-01', periods=3, freq='D')
            base_rows = []
            minute_rows = []
            for code, base_price in [('000001.SZ', 10.0), ('000002.SZ', 20.0)]:
                for d_idx, dt in enumerate(dates):
                    close_daily = base_price + d_idx * 0.1
                    base_rows.append(
                        {
                            '股票代码': code,
                            '日期': dt.strftime('%Y-%m-%d'),
                            '开盘': close_daily - 0.1,
                            '收盘': close_daily,
                            '最高': close_daily + 0.2,
                            '最低': close_daily - 0.2,
                            '成交量': 1000 + d_idx * 2,
                            '成交额': (1000 + d_idx * 2) * close_daily,
                            '换手率': 0.5 + d_idx * 0.01,
                            '涨跌幅': 0.005,
                        }
                    )
                    for i in range(20):
                        ts = dt + pd.Timedelta(hours=14, minutes=i)
                        close_1m = close_daily + i * 0.001
                        minute_rows.append(
                            {
                                '股票代码': code,
                                'datetime': ts.strftime('%Y-%m-%d %H:%M:%S'),
                                'close': close_1m,
                                'amount': (200 + i) * close_1m,
                            }
                        )

            pd.DataFrame(base_rows).to_csv(base_path, index=False, encoding='utf-8')
            pd.DataFrame(minute_rows).to_csv(hf_minute_path, index=False, encoding='utf-8')

            non_strict_cmd = [
                sys.executable,
                os.path.join(SRC_ROOT, 'build_factor_graph.py'),
                '--pipeline-config-dir',
                str(cfg_dir),
                '--feature-set-version',
                'vbadexpr',
                '--base-input',
                str(base_path),
                '--hf-minute-input',
                str(hf_minute_path),
                '--output',
                str(output_non_strict),
                '--manifest-path',
                str(manifest_non_strict),
                '--run-id',
                'unit-test-badexpr-nonstrict',
            ]
            result_non_strict = subprocess.run(non_strict_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
            if result_non_strict.returncode != 0:
                raise AssertionError(
                    f'build_factor_graph(non-strict) failed: rc={result_non_strict.returncode}\n'
                    f'stdout={result_non_strict.stdout}\n'
                    f'stderr={result_non_strict.stderr}'
                )

            out_df = pd.read_csv(output_non_strict, dtype={'股票代码': str})
            self.assertIn('f_hf_invalid_expr', out_df.columns)

            with open(manifest_non_strict, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            statuses = manifest.get('node_status', [])
            self.assertTrue(
                any(
                    item.get('id') == 'invalid_intraday_expr'
                    and item.get('status') == 'skipped'
                    and 'expression 非法' in item.get('message', '')
                    for item in statuses
                )
            )
            warnings = manifest.get('warnings', [])
            self.assertTrue(any('expression 非法' in str(msg) for msg in warnings))

            strict_cmd = [
                sys.executable,
                os.path.join(SRC_ROOT, 'build_factor_graph.py'),
                '--pipeline-config-dir',
                str(cfg_dir),
                '--feature-set-version',
                'vbadexpr',
                '--base-input',
                str(base_path),
                '--hf-minute-input',
                str(hf_minute_path),
                '--output',
                str(output_strict),
                '--manifest-path',
                str(manifest_strict),
                '--strict',
                '--run-id',
                'unit-test-badexpr-strict',
            ]
            result_strict = subprocess.run(strict_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
            self.assertNotEqual(result_strict.returncode, 0)
            combined_output = (result_strict.stdout or '') + '\n' + (result_strict.stderr or '')
            self.assertIn('intraday_aggregate expression 非法', combined_output)
            self.assertIn('节点 invalid_intraday_expr', combined_output)
            self.assertIn('engine=intraday_aggregate', combined_output)
            self.assertIn('strict=True', combined_output)


if __name__ == '__main__':
    unittest.main()
