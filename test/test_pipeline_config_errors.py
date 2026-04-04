import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import yaml

from pipeline_config import PipelineConfigError
from pipeline_config import load_yaml_file
from pipeline_config import load_pipeline_configs
from pipeline_config import validate_datasets_config
from pipeline_config import validate_factors_config


class PipelineConfigErrorTests(unittest.TestCase):
    def test_validate_factors_config_rejects_unsupported_engine_and_missing_output_column(self):
        payload = {
            'version': 1,
            'layer_order': ['alpha'],
            'factor_nodes': [
                {
                    'id': 'bad_node',
                    'layer': 'alpha',
                    'compute': {
                        'engine': 'sql',
                    },
                    'output': {},
                }
            ],
        }

        report = validate_factors_config(payload)

        self.assertFalse(report.valid)
        self.assertIn('factors.yaml 节点 `bad_node` 使用了不支持的 compute.engine: sql', report.errors)
        self.assertIn('factors.yaml 节点 `bad_node` 缺少 output.column', report.errors)

    def test_validate_datasets_config_rejects_invalid_schema_contract(self):
        payload = {
            'version': 1,
            'datasets': {
                'market_bar_1d': {
                    'source': {'name': 'fake', 'adapter': 'fake_adapter'},
                    'schema': {
                        'primary_key': 'instrument_id',
                        'columns': [],
                    },
                    'storage': {'raw_uri': 'data/raw/mock.csv'},
                }
            },
        }

        report = validate_datasets_config(payload)

        self.assertFalse(report.valid)
        self.assertIn('datasets.yaml dataset `market_bar_1d` 的 schema.primary_key 必须为数组', report.errors)
        self.assertIn('datasets.yaml dataset `market_bar_1d` 的 schema.columns 必须为对象', report.errors)

    def test_load_pipeline_configs_strict_raises_combined_validation_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dir = Path(tmp)
            (cfg_dir / 'datasets.yaml').write_text(
                '\n'.join(
                    [
                        'version: 1',
                        'datasets:',
                        '  market_bar_1d:',
                        '    source: {name: fake, adapter: fake_adapter}',
                        '    schema:',
                        '      primary_key: instrument_id',
                        '      columns: []',
                        '    storage: {raw_uri: data/raw/mock.csv}',
                    ]
                ),
                encoding='utf-8',
            )
            (cfg_dir / 'factors.yaml').write_text(
                '\n'.join(
                    [
                        'version: 1',
                        'layer_order: [alpha]',
                        'factor_nodes:',
                        '  - id: bad_node',
                        '    layer: alpha',
                        '    compute: {engine: sql}',
                        '    output: {}',
                    ]
                ),
                encoding='utf-8',
            )
            (cfg_dir / 'storage.yaml').write_text(
                'version: 1\nlayers: {raw: {}, curated: {}, feature_long: {}, feature_wide: {}, datasets: {}, manifests: {}}\n',
                encoding='utf-8',
            )

            with self.assertRaises(PipelineConfigError) as ctx:
                load_pipeline_configs(config_dir=str(cfg_dir), strict=True)

            message = str(ctx.exception)
            self.assertIn('不支持的 compute.engine: sql', message)
            self.assertIn('缺少 output.column', message)
            self.assertIn('schema.primary_key 必须为数组', message)

    def test_load_yaml_file_prefers_c_safe_loader_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / 'config.yaml'
            cfg_path.write_text('version: 1\nitems: []\n', encoding='utf-8')

            with mock.patch.object(yaml, 'load', wraps=yaml.load) as load_mock:
                payload = load_yaml_file(str(cfg_path))

        self.assertEqual(payload['version'], 1)
        self.assertIs(load_mock.call_args.kwargs['Loader'], yaml.CSafeLoader)

    def test_load_yaml_file_falls_back_to_safe_loader_when_c_loader_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / 'config.yaml'
            cfg_path.write_text('version: 1\nitems: []\n', encoding='utf-8')

            with mock.patch.object(yaml, 'load', wraps=yaml.load) as load_mock:
                with mock.patch.object(yaml, 'CSafeLoader', None):
                    payload = load_yaml_file(str(cfg_path))

        self.assertEqual(payload['version'], 1)
        self.assertIs(load_mock.call_args.kwargs['Loader'], yaml.SafeLoader)


if __name__ == '__main__':
    unittest.main()
