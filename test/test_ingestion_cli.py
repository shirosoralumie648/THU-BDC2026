import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')


class IngestionCliTests(unittest.TestCase):
    def test_manage_data_ingest_datasets_lists_registry_entries(self):
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
                        '    request: {}',
                        '    schema: {primary_key: [instrument_id, trade_date], columns: {instrument_id: {source: code}}}',
                        '    quality: {required_columns: [instrument_id, trade_date, close]}',
                        '    storage: {raw_uri: data/raw/mock.csv, curated_uri: data/curated/mock.csv}',
                    ]
                ),
                encoding='utf-8',
            )
            (cfg_dir / 'factors.yaml').write_text('version: 1\nlayer_order: []\nfactor_nodes: []\n', encoding='utf-8')
            (cfg_dir / 'storage.yaml').write_text(
                'version: 1\nlayers: {raw: {}, curated: {}, feature_long: {}, feature_wide: {}, datasets: {}, manifests: {}}\n',
                encoding='utf-8',
            )

            result = subprocess.run(
                [
                    sys.executable,
                    os.path.join(SRC_ROOT, 'manage_data.py'),
                    'ingest',
                    'datasets',
                    '--pipeline-config-dir',
                    str(cfg_dir),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + '\n' + result.stderr)
            self.assertIn('market_bar_1d', result.stdout)


if __name__ == '__main__':
    unittest.main()
