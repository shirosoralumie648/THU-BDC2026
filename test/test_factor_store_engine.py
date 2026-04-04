import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import factor_store as factor_store_module
from factor_store import apply_factor_expressions
from factor_store import build_factor_execution_plan
from factor_store import build_factor_snapshot
from factor_store import resolve_factor_pipeline


class FactorStoreEngineTests(unittest.TestCase):
    def _base_df(self):
        return pd.DataFrame(
            {
                '日期': pd.to_datetime(
                    ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02']
                ),
                '股票代码': ['000001', '000002', '000001', '000002'],
                '收盘': [10.0, 20.0, 11.0, 18.0],
                '开盘': [9.5, 19.5, 10.5, 18.5],
                '成交量': [100.0, 200.0, 130.0, 160.0],
            }
        )

    def test_dag_topological_order_auto_resolves_dependency(self):
        df = self._base_df()
        specs = [
            {'name': 'b_factor', 'expression': 'a_factor * 2'},
            {'name': 'a_factor', 'expression': '收盘 + 1'},
        ]
        out = apply_factor_expressions(df, specs, error_prefix='测试因子')

        np.testing.assert_allclose(out['a_factor'].to_numpy(), (df['收盘'] + 1).to_numpy())
        np.testing.assert_allclose(out['b_factor'].to_numpy(), ((df['收盘'] + 1) * 2).to_numpy())

    def test_cycle_dependency_raises(self):
        specs = [
            {'name': 'a_factor', 'expression': 'b_factor + 1'},
            {'name': 'b_factor', 'expression': 'a_factor + 1'},
        ]
        with self.assertRaisesRegex(ValueError, '循环依赖'):
            build_factor_execution_plan(specs, error_prefix='测试因子')

    def test_inputs_aliasing_works(self):
        df = self._base_df()
        specs = [
            {
                'name': 'my_custom_sma',
                'expression': 'sma(input_price, window)',
                'inputs': {
                    'input_price': '收盘',
                    'window': 2,
                },
            }
        ]
        out = apply_factor_expressions(df, specs, error_prefix='测试因子')
        expected = df['收盘'].rolling(2).mean().fillna(0.0).to_numpy(dtype=np.float32)
        np.testing.assert_allclose(out['my_custom_sma'].to_numpy(), expected)

    def test_build_factor_execution_plan_reuses_cached_compile_for_identical_specs(self):
        specs = [
            {'name': 'ret_1', 'expression': 'pct_change(收盘, 1)'},
            {'name': 'ret_1_rank', 'expression': 'cs_rank(ret_1)'},
        ]

        with patch(
            'factor_store._compile_expression',
            wraps=factor_store_module._compile_expression,
        ) as mocked:
            first = build_factor_execution_plan(specs, error_prefix='测试因子')
            second = build_factor_execution_plan(specs, error_prefix='测试因子')

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(
            [spec['name'] for spec in first['ordered_specs']],
            [spec['name'] for spec in second['ordered_specs']],
        )

    def test_cross_sectional_factor_plan_and_compute(self):
        df = self._base_df().sort_values(['股票代码', '日期']).reset_index(drop=True)
        specs = [
            {'name': 'ret_1', 'expression': 'pct_change(收盘, 1)'},
            {'name': 'ret_1_rank', 'expression': 'cs_rank(ret_1)'},
        ]
        plan = build_factor_execution_plan(specs, error_prefix='测试因子')
        self.assertEqual([s['name'] for s in plan['time_series_specs']], ['ret_1'])
        self.assertEqual([s['name'] for s in plan['cross_sectional_specs']], ['ret_1_rank'])

        grouped = []
        for _, g in df.groupby('股票代码', sort=False):
            grouped.append(apply_factor_expressions(g.copy(), plan['time_series_specs']))
        ts_df = pd.concat(grouped, ignore_index=True).sort_values(['日期', '股票代码']).reset_index(drop=True)
        out = apply_factor_expressions(ts_df, plan['cross_sectional_specs'], date_col='日期')

        day1 = out[out['日期'] == pd.Timestamp('2024-01-02')]['ret_1_rank'].to_numpy()
        np.testing.assert_allclose(day1, np.array([1.0, 0.5], dtype=np.float32))

    def test_cross_sectional_stage_rejects_stateful_helper(self):
        specs = [
            {'name': 'bad_cs', 'expression': 'cs_rank(sma(收盘, 5))'},
        ]
        with self.assertRaisesRegex(ValueError, '截面阶段调用了时序函数'):
            build_factor_execution_plan(specs, error_prefix='测试因子')

    def test_expression_security_blocks_unsafe_call(self):
        df = self._base_df()
        specs = [
            {'name': 'hack', 'expression': '__import__("os").system("echo hacked")'},
        ]
        with self.assertRaisesRegex(ValueError, '未授权函数|白名单'):
            apply_factor_expressions(df, specs, error_prefix='测试因子')

    def test_snapshot_contains_traceability_metadata(self):
        pipeline = {
            'feature_set': '39',
            'store_path': './config/factor_store.json',
            'builtin_registry_path': './config/builtin_factors.json',
            'summary': {
                'builtin_total': 0,
                'builtin_enabled': 0,
                'builtin_overridden': 0,
                'custom_total': 1,
                'custom_enabled': 1,
                'active_total': 1,
                'cross_sectional_total': 0,
                'group_counts': {'custom': 1},
            },
            'active_features': ['my_factor'],
            'builtin_specs': [],
            'custom_specs': [{'name': 'my_factor', 'expression': '收盘', 'inputs': {}, 'enabled': True}],
            'ordered_specs': [{'name': 'my_factor', 'expression': '收盘', 'inputs': {}, 'enabled': True}],
            'active_specs': [{'name': 'my_factor', 'expression': '收盘', 'inputs': {}, 'enabled': True}],
        }
        snapshot = build_factor_snapshot(pipeline)
        self.assertIn('factor_fingerprint', snapshot)
        self.assertIn('snapshot', snapshot)
        self.assertIn('created_at', snapshot['snapshot'])
        self.assertEqual(snapshot['snapshot']['factor_fingerprint'], snapshot['factor_fingerprint'])

    def test_resolve_factor_pipeline_reuses_cached_execution_plan_for_same_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry_path = tmp_path / 'builtin_factors.json'
            store_path = tmp_path / 'factor_store.json'

            registry_path.write_text(
                json.dumps(
                    {
                        'feature_sets': {
                            '39': [
                                {
                                    'name': 'f_close_copy',
                                    'group': 'price',
                                    'expression': '收盘',
                                    'inputs': {},
                                }
                            ],
                            '158+39': [],
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding='utf-8',
            )
            store_path.write_text(
                json.dumps(
                    {
                        'version': 1,
                        'feature_sets': {
                            '39': {
                                'disabled_builtin_factors': [],
                                'builtin_overrides': [],
                                'custom_factors': [],
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding='utf-8',
            )

            with patch('factor_store.build_factor_execution_plan', wraps=build_factor_execution_plan) as mocked:
                first = resolve_factor_pipeline('39', str(store_path), str(registry_path))
                second = resolve_factor_pipeline('39', str(store_path), str(registry_path))

            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(first['factor_fingerprint'], second['factor_fingerprint'])
            self.assertEqual(first['active_features'], second['active_features'])


if __name__ == '__main__':
    unittest.main()
