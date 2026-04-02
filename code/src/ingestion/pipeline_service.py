import os
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

from data_manager import build_file_snapshot
from data_manager import save_data_manifest


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


class FactorPipelineService:
    def __init__(self, *, config_dir: str = './config', runtime_root: str = './data/runtime/ingestion'):
        self.config_dir = str(config_dir)
        self.runtime_root = os.path.abspath(runtime_root)

    def build_hf_daily(
        self,
        *,
        input_paths: Iterable[str],
        output_path: str,
        manifest_path: str = '',
        dataset_name: str = 'market_bar_1m',
        resample_minutes: Optional[str] = None,
        tail_minutes: Optional[int] = None,
        min_bars: Optional[int] = None,
        skip_raw: bool = False,
        force_suffix: bool = False,
    ) -> Dict[str, str]:
        from build_hf_daily_factors import main as build_hf_daily_main

        output_path = os.path.abspath(output_path)
        if not manifest_path:
            manifest_path = os.path.join(os.path.dirname(output_path), 'data_manifest_hf_daily_factors.json')
        manifest_path = os.path.abspath(manifest_path)

        argv: List[str] = [
            '--pipeline-config-dir',
            self.config_dir,
            '--dataset-name',
            str(dataset_name),
            '--output',
            output_path,
            '--manifest-path',
            manifest_path,
        ]
        for path in input_paths:
            argv.extend(['--input', os.path.abspath(str(path))])
        if str(resample_minutes or '').strip():
            argv.extend(['--resample-minutes', str(resample_minutes)])
        if tail_minutes is not None:
            argv.extend(['--tail-minutes', str(int(tail_minutes))])
        if min_bars is not None:
            argv.extend(['--min-bars', str(int(min_bars))])
        if bool(skip_raw):
            argv.append('--skip-raw')
        if bool(force_suffix):
            argv.append('--force-suffix')

        build_hf_daily_main(argv)
        return {'output': output_path, 'manifest': manifest_path}

    def build_factor_graph(
        self,
        *,
        base_input: str,
        output_path: str,
        manifest_path: str = '',
        feature_set_version: str = 'v1',
        hf_daily_input: str = '',
        hf_minute_input: str = '',
        macro_input: str = '',
        run_id: str = '',
        strict: bool = False,
    ) -> Dict[str, str]:
        from build_factor_graph import main as build_factor_graph_main

        output_path = os.path.abspath(output_path)
        if not manifest_path:
            manifest_path = os.path.join(os.path.dirname(output_path), 'data_manifest_factor_graph.json')
        manifest_path = os.path.abspath(manifest_path)

        argv: List[str] = [
            '--pipeline-config-dir',
            self.config_dir,
            '--feature-set-version',
            str(feature_set_version),
            '--base-input',
            os.path.abspath(base_input),
            '--output',
            output_path,
            '--manifest-path',
            manifest_path,
        ]
        if str(hf_daily_input or '').strip():
            argv.extend(['--hf-daily-input', os.path.abspath(hf_daily_input)])
        if str(hf_minute_input or '').strip():
            argv.extend(['--hf-minute-input', os.path.abspath(hf_minute_input)])
        if str(macro_input or '').strip():
            argv.extend(['--macro-input', os.path.abspath(macro_input)])
        if str(run_id or '').strip():
            argv.extend(['--run-id', str(run_id)])
        if bool(strict):
            argv.append('--strict')

        build_factor_graph_main(argv)
        return {'output': output_path, 'manifest': manifest_path}

    def run_factor_pipeline(
        self,
        *,
        base_input: str,
        output_dir: str,
        feature_set_version: str = 'v1',
        hf_daily_input: str = '',
        hf_minute_input: str = '',
        macro_input: str = '',
        run_id: str = '',
        strict: bool = False,
    ) -> Dict[str, str]:
        run_id = str(run_id or '').strip() or _utc_run_id()
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        built_hf = {'output': '', 'manifest': ''}
        effective_hf_daily_input = str(hf_daily_input or '').strip()
        if str(hf_minute_input or '').strip() and not effective_hf_daily_input:
            built_hf = self.build_hf_daily(
                input_paths=[hf_minute_input],
                output_path=os.path.join(output_dir, 'hf_daily_factors.csv'),
                manifest_path=os.path.join(output_dir, 'data_manifest_hf_daily_factors.json'),
            )
            effective_hf_daily_input = built_hf['output']

        factor_graph = self.build_factor_graph(
            base_input=base_input,
            hf_daily_input=effective_hf_daily_input,
            hf_minute_input=hf_minute_input,
            macro_input=macro_input,
            output_path=os.path.join(output_dir, 'factor_graph.csv'),
            manifest_path=os.path.join(output_dir, 'data_manifest_factor_graph.json'),
            feature_set_version=feature_set_version,
            run_id=run_id,
            strict=bool(strict),
        )

        pipeline_manifest_path = save_data_manifest(
            output_dir,
            {
                'action': 'factor_pipeline',
                'run_id': run_id,
                'feature_set_version': str(feature_set_version),
                'inputs': {
                    'base_input': build_file_snapshot(base_input, inspect_csv=True),
                    'hf_minute_input': build_file_snapshot(hf_minute_input, inspect_csv=True),
                    'hf_daily_input': build_file_snapshot(effective_hf_daily_input, inspect_csv=True),
                    'macro_input': build_file_snapshot(macro_input, inspect_csv=True),
                },
                'outputs': {
                    'hf_daily_output': build_file_snapshot(built_hf.get('output', ''), inspect_csv=True),
                    'hf_daily_manifest': build_file_snapshot(built_hf.get('manifest', '')),
                    'factor_graph_output': build_file_snapshot(factor_graph['output'], inspect_csv=True),
                    'factor_graph_manifest': build_file_snapshot(factor_graph['manifest']),
                },
            },
            filename='data_manifest_factor_pipeline.json',
        )
        return {
            'run_id': run_id,
            'hf_daily_output': built_hf.get('output', effective_hf_daily_input),
            'hf_daily_manifest': built_hf.get('manifest', ''),
            'factor_graph_output': factor_graph['output'],
            'factor_graph_manifest': factor_graph['manifest'],
            'pipeline_manifest': pipeline_manifest_path,
        }
