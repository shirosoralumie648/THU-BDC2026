import argparse
import copy
import json
import os
import sys
from datetime import datetime
from datetime import timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from config import config
from data_manager import build_csv_metadata_from_dataframe
from data_manager import build_file_snapshot
from data_manager import build_stock_industry_index
from data_manager import collect_data_sources
from data_manager import infer_existing_column
from data_manager import load_stock_to_industry_map
from data_manager import normalize_stock_code_series
from data_manager import resolve_data_root
from data_manager import resolve_dataset_path
from data_manager import resolve_dataset_write_targets
from data_manager import resolve_hf_factor_path
from data_manager import save_data_manifest
from pipeline_config import load_pipeline_configs
from pipeline_config import PipelineConfigError
from pipeline_config import render_feature_csv_compat_uri


class FactorManifestError(ValueError):
    """Raised when a factor build manifest cannot be parsed or violates expected shape."""


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _exception_message(exc: Exception) -> str:
    if exc.args:
        first = exc.args[0]
        if isinstance(first, str) and first:
            return first
    return str(exc)


def _cli_error_message(exc: Exception) -> str:
    return _exception_message(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='统一数据管理工具')
    subparsers = parser.add_subparsers(dest='command', required=True)
    default_ingestion_runtime_root = os.path.join(resolve_data_root(config), 'runtime', 'ingestion')

    manifest_parser = subparsers.add_parser('manifest', help='生成数据源清单')
    manifest_parser.add_argument(
        '--output',
        default=os.path.join(config['output_dir'], 'data_manifest_cli.json'),
        help='清单输出路径',
    )
    manifest_parser.add_argument(
        '--include-csv-stats',
        action='store_true',
        help='附带 CSV 级统计信息（行数、列数、日期范围、股票数）',
    )

    validate_parser = subparsers.add_parser('validate', help='校验关键数据文件是否存在')
    validate_parser.add_argument(
        '--mode',
        choices=['train', 'split', 'full'],
        default='train',
        help='校验模式: train=训练输入, split=切分输入, full=全量',
    )

    pipeline_parser = subparsers.add_parser('validate-pipeline-config', help='校验 pipeline YAML 配置')
    pipeline_parser.add_argument(
        '--config-dir',
        default='./config',
        help='配置目录（需包含 datasets.yaml/factors.yaml/storage.yaml）',
    )
    pipeline_parser.add_argument(
        '--strict',
        action='store_true',
        help='严格模式：有 warning 也返回非 0',
    )

    index_parser = subparsers.add_parser('industry-index', help='构建股票-行业索引映射')
    index_parser.add_argument(
        '--input',
        default=resolve_dataset_path(config, 'train.csv'),
        help='输入数据文件路径（用于提取股票池）',
    )
    index_parser.add_argument('--stock-col', default='股票代码', help='股票代码列名（默认 股票代码）')
    index_parser.add_argument(
        '--output-index',
        default=os.path.join(config['output_dir'], 'stock_industry_idx_cli.npy'),
        help='行业索引输出路径（.npy）',
    )
    index_parser.add_argument(
        '--output-vocab',
        default=os.path.join(config['output_dir'], 'industry_vocab_cli.json'),
        help='行业词表输出路径（.json）',
    )
    index_parser.add_argument(
        '--output-manifest',
        default=os.path.join(config['output_dir'], 'data_manifest_industry_index.json'),
        help='行业索引清单输出路径（.json）',
    )

    default_train_target = resolve_dataset_write_targets(config, 'train.csv')['primary']
    default_output_dir = str(Path(default_train_target).parent)
    build_parser = subparsers.add_parser('build-dataset', help='构建训练/测试数据集（可选合并因子）')
    build_parser.add_argument(
        '--base-input',
        default=resolve_dataset_path(config, 'stock_data.csv'),
        help='基础行情输入（通常为 stock_data.csv）',
    )
    build_parser.add_argument(
        '--feature-input',
        default='',
        help='可选宽表因子输入（CSV/Parquet）；留空时将尝试从 factors.yaml 推断 csv_compat_uri',
    )
    build_parser.add_argument(
        '--feature-set-version',
        default='v1',
        help='特征集版本，用于渲染 factors.yaml 中 csv_compat_uri 模板，默认 v1',
    )
    build_parser.add_argument(
        '--factor-fingerprint',
        default='',
        help='可选：当前因子流水线指纹（用于训练阶段一致性校验）',
    )
    build_parser.add_argument(
        '--build-id',
        default='',
        help='可选：构建批次 ID（留空自动生成 UTC 时间戳）',
    )
    build_parser.add_argument(
        '--join-how',
        choices=['left', 'inner'],
        default='left',
        help='底表与因子表合并方式，默认 left',
    )
    build_parser.add_argument('--stock-col', default='股票代码', help='基础输入股票列名')
    build_parser.add_argument('--date-col', default='日期', help='基础输入日期列名')
    build_parser.add_argument('--factor-stock-col', default='', help='因子输入股票列名（留空自动推断）')
    build_parser.add_argument('--factor-date-col', default='', help='因子输入日期列名（留空自动推断）')
    build_parser.add_argument(
        '--pipeline-config-dir',
        default='./config',
        help='多源管道配置目录（含 datasets.yaml/factors.yaml/storage.yaml），默认 ./config',
    )
    build_parser.add_argument(
        '--output-dir',
        default=default_output_dir,
        help=f'输出目录，默认 {default_output_dir}',
    )
    build_parser.add_argument(
        '--manifest-path',
        default='',
        help='数据清单输出路径，默认 <output-dir>/data_manifest_dataset_build.json',
    )
    build_parser.add_argument('--train-start', default='2015-01-01', help='训练集开始日期')
    build_parser.add_argument('--train-end', default='2026-03-06', help='训练集结束日期')
    build_parser.add_argument('--test-start', default='2026-03-09', help='测试集开始日期')
    build_parser.add_argument('--test-end', default='2026-03-13', help='测试集结束日期')

    factor_graph_parser = subparsers.add_parser('build-factor-graph', help='按 factors.yaml DAG 构建宽表因子')
    factor_graph_parser.add_argument(
        '--pipeline-config-dir',
        default='./config',
        help='配置目录（datasets/factors/storage YAML）',
    )
    factor_graph_parser.add_argument('--feature-set-version', default='v1', help='特征版本号，例如 v1')
    factor_graph_parser.add_argument(
        '--base-input',
        default=resolve_dataset_path(config, 'stock_data.csv', for_write=False),
        help='日级行情输入（CSV/Parquet）',
    )
    factor_graph_parser.add_argument(
        '--hf-daily-input',
        default=resolve_hf_factor_path(config),
        help='高频聚合后的日级因子输入（可选）',
    )
    factor_graph_parser.add_argument(
        '--hf-minute-input',
        default='',
        help='分钟级原始行情输入（可选）',
    )
    factor_graph_parser.add_argument(
        '--macro-input',
        default='',
        help='宏观序列输入（可选）',
    )
    factor_graph_parser.add_argument(
        '--output',
        default='',
        help='输出宽表因子路径（留空按 factors.yaml 渲染）',
    )
    factor_graph_parser.add_argument(
        '--manifest-path',
        default='',
        help='构建清单输出路径（留空按 factors.yaml 渲染）',
    )
    factor_graph_parser.add_argument(
        '--strict',
        action='store_true',
        help='严格模式：缺少输入或节点计算失败时直接失败',
    )
    factor_graph_parser.add_argument(
        '--run-id',
        default='',
        help='运行 ID（留空自动生成 UTC 时间戳）',
    )

    ingest_parser = subparsers.add_parser('ingest', help='统一 ingestion 作业管理')
    ingest_subparsers = ingest_parser.add_subparsers(dest='ingest_command', required=True)

    ingest_create = ingest_subparsers.add_parser('create', help='创建 ingestion 作业')
    ingest_create.add_argument('--dataset', required=True, help='dataset 名称')
    ingest_create.add_argument('--start', required=True, help='开始日期')
    ingest_create.add_argument('--end', required=True, help='结束日期')
    ingest_create.add_argument('--source', default='', help='覆盖 source 名称')
    ingest_create.add_argument('--mode', default='incremental', help='作业模式')
    ingest_create.add_argument('--universe', default='', help='股票池')
    ingest_create.add_argument('--adjustment', default='', help='复权方式')
    ingest_create.add_argument('--config-dir', default='./config', help='配置目录')
    ingest_create.add_argument('--runtime-root', default='./temp/ingestion_runtime', help='运行时目录')

    ingest_run = ingest_subparsers.add_parser('run', help='运行已有 ingestion 作业')
    ingest_run.add_argument('--job-id', required=True, help='作业 ID')
    ingest_run.add_argument('--config-dir', default='./config', help='配置目录')
    ingest_run.add_argument('--runtime-root', default='./temp/ingestion_runtime', help='运行时目录')

    ingest_get = ingest_subparsers.add_parser('get', help='查询 ingestion 作业')
    ingest_get.add_argument('--job-id', required=True, help='作业 ID')
    ingest_get.add_argument('--config-dir', default='./config', help='配置目录')
    ingest_get.add_argument('--runtime-root', default='./temp/ingestion_runtime', help='运行时目录')

    ingest_replay = ingest_subparsers.add_parser('replay', help='重放 ingestion 作业')
    ingest_replay.add_argument('--job-id', required=True, help='原作业 ID')
    ingest_replay.add_argument('--config-dir', default='./config', help='配置目录')
    ingest_replay.add_argument('--runtime-root', default='./temp/ingestion_runtime', help='运行时目录')

    ingest_datasets = ingest_subparsers.add_parser('datasets', help='列出支持的 ingestion datasets')
    ingest_datasets.add_argument('--config-dir', default='./config', help='配置目录')
    ingest_datasets.add_argument('--runtime-root', default='./temp/ingestion_runtime', help='运行时目录')
    return parser.parse_args()


def command_manifest(args: argparse.Namespace) -> int:
    manifest = collect_data_sources(config, include_csv_stats=bool(args.include_csv_stats))
    manifest['generated_at_utc'] = utc_timestamp()
    output = os.path.abspath(args.output)
    saved = save_data_manifest(os.path.dirname(output), manifest, filename=os.path.basename(output))
    print(f'已生成数据清单: {saved}')
    return 0


def _required_keys_by_mode(mode: str):
    if mode == 'train':
        return ['train_csv']
    if mode == 'split':
        return ['stock_data_csv']
    return ['train_csv', 'test_csv', 'stock_data_csv']


def command_validate(args: argparse.Namespace) -> int:
    manifest = collect_data_sources(config, include_csv_stats=False)
    missing = []

    for key in _required_keys_by_mode(args.mode):
        meta = manifest.get(key, {})
        if not bool(meta.get('exists', False)):
            missing.append((key, meta.get('path', '')))

    if (
        args.mode in {'train', 'full'}
        and bool(config.get('use_hf_daily_factor_merge', False))
        and bool(config.get('hf_factor_required', False))
    ):
        hf_meta = manifest.get('hf_daily_factor', {})
        if not bool(hf_meta.get('exists', False)):
            missing.append(('hf_daily_factor', hf_meta.get('path', '')))

    if missing:
        print('数据校验失败，缺失文件:')
        for key, path in missing:
            print(f'  - {key}: {path}')
        return 2

    print(f'数据校验通过，模式={args.mode}')
    for key in _required_keys_by_mode(args.mode):
        meta = manifest[key]
        print(f"  - {key}: {meta.get('path', '')}")
    if bool(config.get('use_hf_daily_factor_merge', False)):
        hf_meta = manifest.get('hf_daily_factor', {})
        print(f"  - hf_daily_factor: {hf_meta.get('path', '')} (exists={hf_meta.get('exists', False)})")
    return 0


def command_validate_pipeline_config(args: argparse.Namespace) -> int:
    try:
        _, report = load_pipeline_configs(config_dir=args.config_dir, strict=False)
    except PipelineConfigError as exc:
        print(f'配置加载失败: {exc}')
        return 2

    print(f'pipeline 配置校验: valid={report.valid}')
    if report.errors:
        print('Errors:')
        for msg in report.errors:
            print(f'  - {msg}')
    if report.warnings:
        print('Warnings:')
        for msg in report.warnings:
            print(f'  - {msg}')

    if report.errors:
        return 2
    if bool(args.strict) and report.warnings:
        return 3
    return 0


def command_industry_index(args: argparse.Namespace) -> int:
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'未找到输入数据文件: {input_path}')

    df = pd.read_csv(input_path, dtype=str)
    stock_col = args.stock_col
    if stock_col not in df.columns:
        inferred = infer_existing_column(df, ['股票代码', 'stock_id', 'code', 'ts_code'])
        if inferred is None:
            raise ValueError(f'输入数据缺少股票代码列: {stock_col}')
        stock_col = inferred

    stock_ids = normalize_stock_code_series(df[stock_col]).dropna().unique().tolist()
    stock_ids = sorted(stock_ids)
    if not stock_ids:
        raise ValueError('输入数据中没有可用股票代码')

    stock_to_industry = load_stock_to_industry_map(
        config,
        stock_col_key='prior_graph_stock_col',
        industry_col_key='prior_graph_industry_col',
    )
    stock_industry_idx, industry_vocab, matched = build_stock_industry_index(stock_ids, stock_to_industry)

    output_index = os.path.abspath(args.output_index)
    output_vocab = os.path.abspath(args.output_vocab)
    output_manifest = os.path.abspath(args.output_manifest)

    os.makedirs(os.path.dirname(output_index), exist_ok=True)
    os.makedirs(os.path.dirname(output_vocab), exist_ok=True)
    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)

    np.save(output_index, stock_industry_idx.astype(np.int64))
    with open(output_vocab, 'w', encoding='utf-8') as f:
        json.dump(industry_vocab, f, ensure_ascii=False, indent=2)

    coverage = matched / float(max(1, len(stock_ids)))
    manifest = {
        'action': 'industry_index',
        'generated_at_utc': utc_timestamp(),
        'input': build_file_snapshot(input_path, inspect_csv=True),
        'output': {
            'industry_index_npy': build_file_snapshot(output_index),
            'industry_vocab_json': build_file_snapshot(output_vocab),
        },
        'summary': {
            'num_stocks': int(len(stock_ids)),
            'matched_stocks': int(matched),
            'coverage': float(coverage),
            'industry_count': int(len(industry_vocab)),
        },
    }
    saved_manifest = save_data_manifest(
        os.path.dirname(output_manifest),
        manifest,
        filename=os.path.basename(output_manifest),
    )

    print('行业索引构建完成:')
    print(f'  - stock_count: {len(stock_ids)}')
    print(f'  - matched: {matched}')
    print(f'  - coverage: {coverage:.2%}')
    print(f'  - industry_count: {len(industry_vocab)}')
    print(f'  - index: {output_index}')
    print(f'  - vocab: {output_vocab}')
    print(f'  - manifest: {saved_manifest}')
    return 0


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_str: str) -> str:
    path = Path(str(path_str or '').strip()).expanduser()
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _to_timestamp(date_str: str, name: str) -> pd.Timestamp:
    ts = pd.to_datetime(date_str, errors='coerce')
    if pd.isna(ts):
        raise ValueError(f'参数 {name} 的日期格式无效: {date_str}')
    return ts.normalize()


def _resolve_column(df: pd.DataFrame, preferred: str, candidates) -> str:
    preferred = str(preferred or '').strip()
    if preferred and preferred in df.columns:
        return preferred
    inferred = infer_existing_column(df, candidates)
    if inferred is None:
        raise ValueError(f'无法解析列名，候选={candidates}')
    return inferred


def _load_factor_dataframe(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {'.parquet', '.pq'}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _finalize_output_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['股票代码'] = out['__stock_norm'].astype(str)
    out['日期'] = out['__date_norm'].dt.strftime('%Y-%m-%d')
    out = out.drop(columns=[col for col in ['__stock_norm', '__date_norm'] if col in out.columns])
    ordered = [col for col in ['股票代码', '日期'] if col in out.columns]
    remain = [col for col in out.columns if col not in ordered]
    return out[ordered + remain]


def _file_signature(path: str) -> tuple[str, int, int]:
    abs_path = os.path.abspath(str(path))
    stat = os.stat(abs_path)
    return abs_path, int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=128)
def _load_json_payload_cached(path: str, mtime_ns: int, size_bytes: int):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_json_payload(path: str):
    abs_path, mtime_ns, size_bytes = _file_signature(path)
    payload = _load_json_payload_cached(abs_path, mtime_ns, size_bytes)
    return copy.deepcopy(payload)


def _extract_factor_fingerprint_from_manifest(
    manifest_path: str,
    *,
    feature_input: str,
    feature_set_version: str,
) -> str:
    try:
        payload = _load_json_payload(manifest_path)
    except Exception as exc:
        raise FactorManifestError(
            f'解析 factor build manifest 失败: {manifest_path}'
        ) from exc
    if not isinstance(payload, dict):
        raise FactorManifestError(f'解析 factor build manifest 失败: {manifest_path}')
    if str(payload.get('action', '') or '').strip() not in {'', 'build_factor_graph'}:
        return ''
    manifest_feature_set_version = str(payload.get('feature_set_version', '') or '').strip()
    if feature_set_version and manifest_feature_set_version and manifest_feature_set_version != feature_set_version:
        return ''
    output_paths = payload.get('output_paths', {})
    if not isinstance(output_paths, dict):
        output_paths = {}
    snapshot = output_paths.get('wide_csv_snapshot', {})
    if not isinstance(snapshot, dict):
        snapshot = {}
    feature_candidates = {
        os.path.abspath(str(path))
        for path in [output_paths.get('wide_csv', ''), snapshot.get('path', '')]
        if str(path).strip()
    }
    if feature_candidates and os.path.abspath(feature_input) not in feature_candidates:
        return ''
    return str(payload.get('factor_fingerprint', '') or '').strip()


def _candidate_factor_manifest_paths(
    feature_input: str,
    *,
    feature_set_version: str,
    pipeline_configs: dict,
) -> list[str]:
    feature_path = Path(feature_input).resolve()
    candidates: list[str] = []
    seen = set()

    def _append_candidate(path: Path) -> None:
        path_str = str(path)
        if path_str not in seen:
            seen.add(path_str)
            candidates.append(path_str)

    factors_cfg = pipeline_configs.get('factors', {}) if isinstance(pipeline_configs, dict) else {}
    build_manifest_cfg = factors_cfg.get('build_manifest', {}) if isinstance(factors_cfg, dict) else {}
    manifest_uri = str(build_manifest_cfg.get('output_uri', '') or '').strip()

    if manifest_uri:
        resolved_manifest_uri = manifest_uri
        if feature_set_version:
            resolved_manifest_uri = resolved_manifest_uri.replace('{feature_set_version}', feature_set_version)
        if '{' not in resolved_manifest_uri and '}' not in resolved_manifest_uri:
            manifest_path = Path(_resolve_path(resolved_manifest_uri))
            if manifest_path.exists():
                _append_candidate(manifest_path)
            return candidates

    for path in sorted(feature_path.parent.glob('*.json')):
        _append_candidate(path)

    manifest_roots = []
    if manifest_uri:
        if '{feature_set_version}' in manifest_uri and feature_set_version:
            prefix, _, _ = manifest_uri.partition('{feature_set_version}')
            manifest_roots.append(Path(_resolve_path(prefix)) / feature_set_version)
        else:
            manifest_roots.append(Path(_resolve_path(Path(manifest_uri).parent)))
    if feature_set_version:
        manifest_roots.append(Path(_resolve_path(os.path.join('data', 'manifests', 'factor_build', feature_set_version))))

    for root in manifest_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob('*.json'), key=lambda item: item.stat().st_mtime, reverse=True):
            _append_candidate(path)
    return candidates


def _resolve_factor_fingerprint_from_feature_input(
    feature_input: str,
    *,
    feature_set_version: str,
    pipeline_configs: dict,
) -> str:
    if not feature_input:
        return ''
    for manifest_path in _candidate_factor_manifest_paths(
        feature_input,
        feature_set_version=feature_set_version,
        pipeline_configs=pipeline_configs,
    ):
        fingerprint = _extract_factor_fingerprint_from_manifest(
            manifest_path,
            feature_input=feature_input,
            feature_set_version=feature_set_version,
        )
        if fingerprint:
            return fingerprint
    return ''


def command_build_dataset(args: argparse.Namespace) -> int:
    base_input = _resolve_path(args.base_input)
    if not os.path.exists(base_input):
        raise FileNotFoundError(f'未找到基础输入文件: {base_input}')

    output_dir = _resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    build_id = str(args.build_id or '').strip() or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    factor_fingerprint = str(args.factor_fingerprint or '').strip()

    pipeline_validation = {'valid': False, 'errors': [], 'warnings': []}
    feature_input = str(args.feature_input or '').strip()
    try:
        pipeline_configs, report = load_pipeline_configs(
            config_dir=args.pipeline_config_dir,
            strict=False,
        )
        pipeline_validation = report.to_dict()
        if not feature_input:
            rendered = render_feature_csv_compat_uri(
                pipeline_configs.get('factors', {}),
                feature_set_version=str(args.feature_set_version),
            )
            if rendered:
                candidate = _resolve_path(rendered)
                if os.path.exists(candidate):
                    feature_input = candidate
                else:
                    pipeline_validation['warnings'].append(
                        f'自动推断的 feature-input 不存在: {candidate}'
                    )
    except PipelineConfigError as exc:
        pipeline_validation['errors'].append(str(exc))

    train_start = _to_timestamp(args.train_start, '--train-start')
    train_end = _to_timestamp(args.train_end, '--train-end')
    test_start = _to_timestamp(args.test_start, '--test-start')
    test_end = _to_timestamp(args.test_end, '--test-end')
    if train_start > train_end:
        raise ValueError(f'训练集开始日期晚于结束日期: {train_start.date()} > {train_end.date()}')
    if test_start > test_end:
        raise ValueError(f'测试集开始日期晚于结束日期: {test_start.date()} > {test_end.date()}')

    base_df = pd.read_csv(base_input)
    base_input_csv_meta = (
        build_csv_metadata_from_dataframe(base_df)
        if base_input.lower().endswith('.csv')
        else None
    )
    base_stock_col = _resolve_column(base_df, args.stock_col, ['股票代码', 'stock_id', 'code', 'ts_code'])
    base_date_col = _resolve_column(base_df, args.date_col, ['日期', 'date', 'datetime', 'trade_date'])

    merged_df = base_df.copy()
    merged_df['__stock_norm'] = normalize_stock_code_series(merged_df[base_stock_col])
    merged_df['__date_norm'] = pd.to_datetime(merged_df[base_date_col], errors='coerce').dt.normalize()
    merged_df = merged_df.dropna(subset=['__stock_norm', '__date_norm']).copy()

    factor_merge_meta = {
        'enabled': bool(feature_input),
        'path': _resolve_path(feature_input) if feature_input else '',
        'used': False,
        'factor_columns': [],
        'matched_rows': 0,
        'coverage': 0.0,
    }
    feature_input_csv_meta = None

    if feature_input:
        feature_input = _resolve_path(feature_input)
        if not os.path.exists(feature_input):
            raise FileNotFoundError(f'未找到因子输入文件: {feature_input}')

        if not factor_fingerprint:
            factor_fingerprint = _resolve_factor_fingerprint_from_feature_input(
                feature_input,
                feature_set_version=str(args.feature_set_version),
                pipeline_configs=pipeline_configs,
            )
            if not factor_fingerprint:
                pipeline_validation['warnings'].append(
                    f'未能从 factor build manifest 解析 factor_fingerprint: {feature_input}'
                )

        factor_df = _load_factor_dataframe(feature_input)
        if feature_input.lower().endswith('.csv'):
            feature_input_csv_meta = build_csv_metadata_from_dataframe(factor_df)
        factor_stock_col = _resolve_column(
            factor_df,
            args.factor_stock_col,
            ['股票代码', 'stock_id', 'code', 'ts_code'],
        )
        factor_date_col = _resolve_column(
            factor_df,
            args.factor_date_col,
            ['日期', 'date', 'datetime', 'trade_date'],
        )

        factor_df = factor_df.copy()
        factor_df['__stock_norm'] = normalize_stock_code_series(factor_df[factor_stock_col])
        factor_df['__date_norm'] = pd.to_datetime(factor_df[factor_date_col], errors='coerce').dt.normalize()
        factor_df = factor_df.dropna(subset=['__stock_norm', '__date_norm']).copy()

        factor_cols = [
            col for col in factor_df.columns
            if col not in {factor_stock_col, factor_date_col, '__stock_norm', '__date_norm'}
        ]
        rename_map = {}
        for col in factor_cols:
            if col in merged_df.columns:
                rename_map[col] = f'f_ext_{col}'
        if rename_map:
            factor_df = factor_df.rename(columns=rename_map)
        factor_cols = [rename_map.get(col, col) for col in factor_cols]

        factor_df = factor_df.drop_duplicates(subset=['__stock_norm', '__date_norm'], keep='last')
        merged_df = merged_df.merge(
            factor_df[['__stock_norm', '__date_norm', *factor_cols]],
            on=['__stock_norm', '__date_norm'],
            how=args.join_how,
            validate='many_to_one',
        )
        matched_rows = int(merged_df[factor_cols].notna().any(axis=1).sum()) if factor_cols else 0
        factor_merge_meta.update({
            'used': True,
            'factor_columns': factor_cols,
            'matched_rows': matched_rows,
            'coverage': float(matched_rows / float(max(1, len(merged_df)))),
        })

    train_df = merged_df[
        (merged_df['__date_norm'] >= train_start) & (merged_df['__date_norm'] <= train_end)
    ].copy()
    test_df = merged_df[
        (merged_df['__date_norm'] >= test_start) & (merged_df['__date_norm'] <= test_end)
    ].copy()

    train_df = _finalize_output_frame(train_df)
    test_df = _finalize_output_frame(test_df)

    train_path = Path(output_dir) / 'train.csv'
    test_path = Path(output_dir) / 'test.csv'
    default_train_targets = resolve_dataset_write_targets(config, 'train.csv')
    default_test_targets = resolve_dataset_write_targets(config, 'test.csv')
    train_mirror_paths = []
    test_mirror_paths = []
    if str(train_path.resolve()) == str(Path(default_train_targets['primary']).resolve()):
        train_mirror_paths = [Path(path).resolve() for path in default_train_targets.get('mirrors', [])]
    if str(test_path.resolve()) == str(Path(default_test_targets['primary']).resolve()):
        test_mirror_paths = [Path(path).resolve() for path in default_test_targets.get('mirrors', [])]

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    for mirror in train_mirror_paths:
        mirror.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(mirror, index=False)
    for mirror in test_mirror_paths:
        mirror.parent.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(mirror, index=False)

    train_csv_meta = build_csv_metadata_from_dataframe(train_df)
    test_csv_meta = build_csv_metadata_from_dataframe(test_df)

    manifest_target = (
        _resolve_path(args.manifest_path)
        if str(args.manifest_path).strip()
        else str(Path(output_dir) / 'data_manifest_dataset_build.json')
    )
    manifest = {
        'action': 'build_dataset',
        'build_id': build_id,
        'feature_set_version': str(args.feature_set_version),
        'factor_fingerprint': factor_fingerprint,
        'params': {
            'base_input': base_input,
            'feature_input': feature_input,
            'build_id': build_id,
            'feature_set_version': str(args.feature_set_version),
            'factor_fingerprint': factor_fingerprint,
            'join_how': str(args.join_how),
            'train_start': str(train_start.date()),
            'train_end': str(train_end.date()),
            'test_start': str(test_start.date()),
            'test_end': str(test_end.date()),
            'pipeline_config_dir': str(args.pipeline_config_dir),
        },
        'inputs': {
            'base_input': build_file_snapshot(
                base_input,
                inspect_csv=True,
                csv_metadata=base_input_csv_meta,
            ),
            'feature_input': build_file_snapshot(
                feature_input,
                inspect_csv=True,
                csv_metadata=feature_input_csv_meta,
            ),
        },
        'outputs': {
            'train_csv': build_file_snapshot(
                str(train_path),
                inspect_csv=True,
                csv_metadata=train_csv_meta,
            ),
            'test_csv': build_file_snapshot(
                str(test_path),
                inspect_csv=True,
                csv_metadata=test_csv_meta,
            ),
            'train_csv_mirrors': [
                build_file_snapshot(
                    str(path),
                    inspect_csv=True,
                    csv_metadata=train_csv_meta,
                )
                for path in train_mirror_paths
            ],
            'test_csv_mirrors': [
                build_file_snapshot(
                    str(path),
                    inspect_csv=True,
                    csv_metadata=test_csv_meta,
                )
                for path in test_mirror_paths
            ],
        },
        'stats': {
            'base_rows': int(len(base_df)),
            'merged_rows': int(len(merged_df)),
            'train_rows': int(len(train_df)),
            'test_rows': int(len(test_df)),
            'train_stock_count': int(train_df['股票代码'].nunique()) if not train_df.empty else 0,
            'test_stock_count': int(test_df['股票代码'].nunique()) if not test_df.empty else 0,
        },
        'factor_merge': factor_merge_meta,
        'pipeline_config_validation': pipeline_validation,
    }
    saved_manifest = save_data_manifest(
        str(Path(manifest_target).parent),
        manifest,
        filename=Path(manifest_target).name,
    )

    print('数据集构建完成:')
    print(f'  - train: {train_path} rows={len(train_df)} stocks={train_df["股票代码"].nunique() if not train_df.empty else 0}')
    print(f'  - test: {test_path} rows={len(test_df)} stocks={test_df["股票代码"].nunique() if not test_df.empty else 0}')
    if factor_merge_meta['used']:
        print(
            f'  - factor merge: cols={len(factor_merge_meta["factor_columns"])}, '
            f'coverage={factor_merge_meta["coverage"]:.2%}, '
            f'path={factor_merge_meta["path"]}'
        )
    print(f'  - manifest: {saved_manifest}')
    return 0


def command_build_factor_graph(args: argparse.Namespace) -> int:
    from build_factor_graph import main as build_factor_graph_main

    cli_args = [
        '--pipeline-config-dir',
        str(args.pipeline_config_dir),
        '--feature-set-version',
        str(args.feature_set_version),
        '--base-input',
        str(args.base_input),
    ]

    for key in ['hf_daily_input', 'hf_minute_input', 'macro_input', 'output', 'manifest_path', 'run_id']:
        value = str(getattr(args, key, '') or '').strip()
        if not value:
            continue
        cli_args.extend([f'--{key.replace("_", "-")}', value])

    if bool(args.strict):
        cli_args.append('--strict')

    build_factor_graph_main(cli_args)
    return 0


def _build_ingestion_service(args: argparse.Namespace):
    from ingestion.service import IngestionService

    return IngestionService.from_config_dir(
        str(getattr(args, 'config_dir', './config')),
        runtime_root=_resolve_path(str(getattr(args, 'runtime_root', './temp/ingestion_runtime'))),
        project_root=str(PROJECT_ROOT),
    )


def command_ingest(args: argparse.Namespace) -> int:
    service = _build_ingestion_service(args)
    subcommand = str(args.ingest_command)

    if subcommand == 'datasets':
        print(json.dumps({'datasets': sorted(service.specs.keys())}, ensure_ascii=False, indent=2))
        return 0

    if subcommand == 'create':
        from ingestion.models import IngestionRequest

        job = service.create_job(
            IngestionRequest(
                dataset=str(args.dataset),
                start=str(args.start),
                end=str(args.end),
                source=str(args.source or ''),
                mode=str(args.mode or 'incremental'),
                universe=str(args.universe or '') or None,
                adjustment=str(args.adjustment or '') or None,
            )
        )
        print(json.dumps(service.job_to_payload(job), ensure_ascii=False, indent=2))
        return 0

    if subcommand == 'get':
        job_id = str(args.job_id)
        try:
            job = service.get_job(job_id)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'job not found: {job_id}') from exc
        print(json.dumps(service.job_to_payload(job), ensure_ascii=False, indent=2))
        return 0

    if subcommand == 'run':
        job_id = str(args.job_id)
        try:
            job = service.run_job(job_id)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'job not found: {job_id}') from exc
        print(json.dumps(service.job_to_payload(job), ensure_ascii=False, indent=2))
        return 0

    if subcommand == 'replay':
        job_id = str(args.job_id)
        try:
            job = service.replay_job(job_id)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'job not found: {job_id}') from exc
        print(json.dumps(service.job_to_payload(job), ensure_ascii=False, indent=2))
        return 0

    raise ValueError(f'未知 ingest 子命令: {subcommand}')


def _run_command(args: argparse.Namespace) -> int:
    if args.command == 'manifest':
        return command_manifest(args)
    if args.command == 'validate':
        return command_validate(args)
    if args.command == 'validate-pipeline-config':
        return command_validate_pipeline_config(args)
    if args.command == 'industry-index':
        return command_industry_index(args)
    if args.command == 'build-dataset':
        return command_build_dataset(args)
    if args.command == 'build-factor-graph':
        return command_build_factor_graph(args)
    if args.command == 'ingest':
        return command_ingest(args)
    raise ValueError(f'未知命令: {args.command}')


def main() -> None:
    args = parse_args()
    try:
        raise SystemExit(_run_command(args))
    except (FileNotFoundError, ValueError, PipelineConfigError, KeyError, RuntimeError) as exc:
        print(_cli_error_message(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == '__main__':
    main()
