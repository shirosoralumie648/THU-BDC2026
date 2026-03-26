import argparse
import json
import os
from datetime import datetime
from datetime import timezone

import numpy as np
import pandas as pd

from config import config
from data_manager import build_file_snapshot
from data_manager import build_stock_industry_index
from data_manager import collect_data_sources
from data_manager import infer_existing_column
from data_manager import load_stock_to_industry_map
from data_manager import normalize_stock_code_series
from data_manager import resolve_dataset_path
from data_manager import save_data_manifest


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='统一数据管理工具')
    subparsers = parser.add_subparsers(dest='command', required=True)

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

    if missing:
        print('数据校验失败，缺失文件:')
        for key, path in missing:
            print(f'  - {key}: {path}')
        return 2

    print(f'数据校验通过，模式={args.mode}')
    for key in _required_keys_by_mode(args.mode):
        meta = manifest[key]
        print(f"  - {key}: {meta.get('path', '')}")
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


def main() -> None:
    args = parse_args()
    if args.command == 'manifest':
        raise SystemExit(command_manifest(args))
    if args.command == 'validate':
        raise SystemExit(command_validate(args))
    if args.command == 'industry-index':
        raise SystemExit(command_industry_index(args))
    raise SystemExit(f'未知命令: {args.command}')


if __name__ == '__main__':
    main()
