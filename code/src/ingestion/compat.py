import json
import os
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd

from data_manager import build_file_snapshot
from data_manager import normalize_stock_code_series
from data_manager import resolve_data_root
from data_manager import resolve_dataset_write_targets
from data_manager import save_data_manifest
from ingestion.models import IngestionRequest


LEGACY_STOCK_COLUMNS = [
    '股票代码',
    '日期',
    '开盘',
    '收盘',
    '最高',
    '最低',
    '成交量',
    '成交额',
    '振幅',
    '涨跌额',
    '换手率',
    '涨跌幅',
]


def _format_legacy_date(value: Any) -> str:
    ts = pd.to_datetime(value, errors='coerce')
    if pd.isna(ts):
        return ''
    return f'{ts.year}/{ts.month}/{ts.day}'


def canonical_daily_to_legacy(df: pd.DataFrame) -> pd.DataFrame:
    required = {'instrument_id', 'trade_date', 'open', 'high', 'low', 'close'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'canonical daily frame missing columns: {missing}')

    out = pd.DataFrame(index=df.index)
    out['股票代码'] = normalize_stock_code_series(df['instrument_id']).astype(str)
    out['日期'] = pd.to_datetime(df['trade_date'], errors='coerce').map(_format_legacy_date)
    out['开盘'] = pd.to_numeric(df['open'], errors='coerce')
    out['收盘'] = pd.to_numeric(df['close'], errors='coerce')
    out['最高'] = pd.to_numeric(df['high'], errors='coerce')
    out['最低'] = pd.to_numeric(df['low'], errors='coerce')
    out['成交量'] = pd.to_numeric(df.get('volume', np.nan), errors='coerce')
    out['成交额'] = pd.to_numeric(df.get('amount', np.nan), errors='coerce')

    preclose = pd.to_numeric(df.get('preclose', np.nan), errors='coerce').replace(0, np.nan)
    out['振幅'] = (((out['最高'] - out['最低']) / preclose) * 100.0).fillna(0.0)
    out['涨跌额'] = (out['收盘'] - preclose).fillna(0.0)
    out['换手率'] = pd.to_numeric(df.get('turnover', np.nan), errors='coerce')
    out['涨跌幅'] = pd.to_numeric(df.get('pct_chg', np.nan), errors='coerce')

    out = out.dropna(subset=['股票代码', '日期']).copy()
    out = out.sort_values(['股票代码', '日期']).reset_index(drop=True)
    return out[LEGACY_STOCK_COLUMNS]


def _default_service_builder(*, config_dir: str, runtime_root: str):
    from ingestion.adapters import build_default_adapters
    from ingestion.registry import DatasetRegistry
    from ingestion.service import IngestionService

    registry = DatasetRegistry.from_config_dir(config_dir)
    specs = {spec.dataset: spec for spec in registry.list_datasets()}
    return IngestionService(specs=specs, adapters=build_default_adapters(), runtime_root=runtime_root)


def _load_manifest(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def run_stock_data_bridge(
    args,
    *,
    runtime_root: str = '',
    service_builder: Optional[Callable[..., Any]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if config_dict is None:
        from config import config as default_config

        config_dict = default_config

    config_dir = str(getattr(args, 'pipeline_config_dir', './config') or './config')
    if not runtime_root:
        runtime_root = os.path.join(resolve_data_root(config_dict), 'runtime', 'ingestion')
    runtime_root = os.path.abspath(runtime_root)

    output_arg = str(getattr(args, 'output_path', '') or '').strip()
    if output_arg:
        output_path = str(Path(output_arg).expanduser().resolve())
    else:
        output_path = str(Path(resolve_dataset_write_targets(config_dict, 'stock_data.csv')['primary']).resolve())

    mirror_output_paths = []
    default_targets = resolve_dataset_write_targets(config_dict, 'stock_data.csv')
    if str(Path(output_path).resolve()) == str(Path(default_targets['primary']).resolve()):
        mirror_output_paths = [
            str(Path(path).resolve())
            for path in default_targets.get('mirrors', [])
        ]

    request = IngestionRequest(
        dataset=str(getattr(args, 'dataset_name', 'market_bar_1d') or 'market_bar_1d'),
        start=str(getattr(args, 'start_date', '')).strip(),
        end=str(getattr(args, 'end_date', '')).strip(),
        source='baostock',
        mode='incremental',
        adjustment=str(getattr(args, 'adjustflag', '')).strip() or None,
        extra={
            'index_date': str(getattr(args, 'index_date', '')).strip(),
        },
    )

    builder = service_builder or _default_service_builder
    service = builder(config_dir=config_dir, runtime_root=runtime_root)
    if hasattr(service, 'create_and_run'):
        finished = service.create_and_run(request)
    else:
        job = service.create_job(request)
        finished = service.run_job(job.job_id)

    if hasattr(service, 'load_manifest'):
        source_manifest = service.load_manifest(finished)
    else:
        source_manifest = _load_manifest(getattr(finished, 'manifest_path', ''))

    curated_paths = source_manifest.get('curated_paths', []) if isinstance(source_manifest, dict) else []
    if not curated_paths:
        raise ValueError('source ingestion manifest missing curated_paths')
    curated_path = str(curated_paths[0])
    if not os.path.exists(curated_path):
        raise FileNotFoundError(f'curated output not found: {curated_path}')

    legacy_df = canonical_daily_to_legacy(pd.read_csv(curated_path))
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    legacy_df.to_csv(output_path, index=False, encoding='utf-8')
    for mirror_path in mirror_output_paths:
        os.makedirs(os.path.dirname(mirror_path) or '.', exist_ok=True)
        legacy_df.to_csv(mirror_path, index=False, encoding='utf-8')

    manifest_target = str(getattr(args, 'manifest_path', '') or '').strip()
    if manifest_target:
        manifest_path = str(Path(manifest_target).expanduser().resolve())
    else:
        manifest_path = str((Path(output_path).parent / 'data_manifest_stock_fetch.json').resolve())

    bridge_manifest = {
        'action': 'fetch_stock_data_bridge',
        'job_id': getattr(finished, 'job_id', ''),
        'dataset': request.dataset,
        'parameters': {
            'pipeline_config_dir': config_dir,
            'start_date': request.start,
            'end_date': request.end,
            'index_date': request.extra.get('index_date', ''),
            'adjustflag': request.adjustment or '',
        },
        'source_ingestion_manifest': build_file_snapshot(getattr(finished, 'manifest_path', '')),
        'source_curated_output': build_file_snapshot(curated_path, inspect_csv=True),
        'outputs': {
            'stock_data_csv': build_file_snapshot(output_path, inspect_csv=True),
            'stock_data_csv_mirrors': [build_file_snapshot(path, inspect_csv=True) for path in mirror_output_paths],
        },
        'stats': {
            'rows': int(len(legacy_df)),
            'stock_count': int(legacy_df['股票代码'].nunique()) if not legacy_df.empty else 0,
            'date_min': str(pd.to_datetime(legacy_df['日期']).min().date()) if not legacy_df.empty else '',
            'date_max': str(pd.to_datetime(legacy_df['日期']).max().date()) if not legacy_df.empty else '',
        },
    }
    saved_manifest = save_data_manifest(
        os.path.dirname(manifest_path) or '.',
        bridge_manifest,
        filename=os.path.basename(manifest_path),
    )
    return {
        'job_id': getattr(finished, 'job_id', ''),
        'status': getattr(finished, 'status', ''),
        'output_path': output_path,
        'manifest_path': saved_manifest,
    }
