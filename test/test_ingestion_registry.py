import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ingestion.registry import DatasetRegistry


class DatasetRegistryTests(unittest.TestCase):
    def test_loads_dataset_spec_from_config_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dir = Path(tmp)
            (cfg_dir / 'datasets.yaml').write_text(
                '\n'.join(
                    [
                        'version: 1',
                        'datasets:',
                        '  market_bar_1d:',
                        '    domain: market',
                        '    granularity: 1d',
                        '    source: {name: baostock, adapter: baostock_daily}',
                        '    request: {start: "${START_DATE}", end: "${END_DATE}"}',
                        '    schema: {primary_key: [instrument_id, trade_date], columns: {instrument_id: {source: code}}}',
                        '    quality: {required_columns: [instrument_id, trade_date]}',
                        '    storage: {raw_uri: data/raw/mock.parquet, curated_uri: data/curated/mock.parquet}',
                    ]
                ),
                encoding='utf-8',
            )
            (cfg_dir / 'factors.yaml').write_text('version: 1\nlayer_order: []\nfactor_nodes: []\n', encoding='utf-8')
            (cfg_dir / 'storage.yaml').write_text(
                'version: 1\nlayers: {raw: {}, curated: {}, feature_long: {}, feature_wide: {}, datasets: {}, manifests: {}}\n',
                encoding='utf-8',
            )

            registry = DatasetRegistry.from_config_dir(str(cfg_dir))
            spec = registry.get('market_bar_1d')

            self.assertEqual(spec.dataset, 'market_bar_1d')
            self.assertEqual(spec.source_name, 'baostock')
            self.assertEqual(spec.adapter_name, 'baostock_daily')
            self.assertEqual(spec.primary_key, ['instrument_id', 'trade_date'])


if __name__ == '__main__':
    unittest.main()
