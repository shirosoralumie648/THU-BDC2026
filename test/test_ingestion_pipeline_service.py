import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ingestion.pipeline_service import FactorPipelineService


class FactorPipelineServiceTests(unittest.TestCase):
    def test_run_factor_pipeline_builds_hf_daily_and_factor_graph_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_dir = tmp_path / 'out'
            base_path = tmp_path / 'stock_data.csv'
            minute_path = tmp_path / 'minute.csv'
            macro_path = tmp_path / 'macro.csv'

            base_rows = []
            minute_rows = []
            dates = pd.date_range('2024-01-02', periods=25, freq='D')
            for code, base_price in [('000001', 10.0), ('000002', 20.0)]:
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
                            '成交量': 1000 + d_idx * 5,
                            '成交额': (1000 + d_idx * 5) * close_daily,
                            '换手率': 1.0,
                            '涨跌幅': 0.01,
                        }
                    )
                    for i in range(35):
                        ts = dt + pd.Timedelta(hours=14, minutes=i)
                        close_1m = close_daily + i * 0.001
                        minute_rows.append(
                            {
                                '股票代码': code,
                                'datetime': ts.strftime('%Y-%m-%d %H:%M:%S'),
                                'close': close_1m,
                                'amount': close_1m * (500 + i),
                            }
                        )
            pd.DataFrame(base_rows).to_csv(base_path, index=False, encoding='utf-8')
            pd.DataFrame(minute_rows).to_csv(minute_path, index=False, encoding='utf-8')

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

            service = FactorPipelineService(config_dir='./config', runtime_root=str(tmp_path / 'runtime'))
            result = service.run_factor_pipeline(
                base_input=str(base_path),
                hf_minute_input=str(minute_path),
                macro_input=str(macro_path),
                output_dir=str(out_dir),
                feature_set_version='vsvc',
                run_id='svc-run',
                strict=True,
            )

            self.assertTrue(os.path.exists(result['hf_daily_output']))
            self.assertTrue(os.path.exists(result['factor_graph_output']))
            self.assertTrue(os.path.exists(result['pipeline_manifest']))

            factor_df = pd.read_csv(result['factor_graph_output'], dtype={'股票代码': str})
            self.assertIn('f_hf_realized_vol_1d', factor_df.columns)
            self.assertIn('f_macro_m2_yoy', factor_df.columns)


if __name__ == '__main__':
    unittest.main()
