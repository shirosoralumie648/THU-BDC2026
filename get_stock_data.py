#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_SRC_DIR = SCRIPT_DIR / 'code' / 'src'
if str(CODE_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_SRC_DIR))

from ingestion.compat import run_stock_data_bridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='兼容入口：通过 ingestion service 抓取日线数据')
    parser.add_argument('--pipeline-config-dir', default='./config', help='配置目录')
    parser.add_argument('--dataset-name', default='market_bar_1d', help='dataset 名称')
    parser.add_argument('--start-date', default='2015-01-01', help='开始日期')
    parser.add_argument('--end-date', default='2026-03-31', help='结束日期')
    parser.add_argument('--index-date', default='', help='成分股快照日期（兼容参数）')
    parser.add_argument('--output-path', default='', help='兼容输出 stock_data.csv 路径')
    parser.add_argument('--manifest-path', default='', help='兼容输出 manifest 路径')
    parser.add_argument('--adjustflag', default='1', help='复权方式（BaoStock 兼容参数）')
    parser.add_argument('--frequency', default='d', help='K 线频率（兼容参数，当前桥接固定输出日线）')
    parser.add_argument('--max-retries', type=int, default=3, help='兼容参数：最大重试次数')
    parser.add_argument('--retry-backoff-seconds', type=float, default=1.2, help='兼容参数：重试退避秒数')
    parser.add_argument('--request-interval-seconds', type=float, default=0.05, help='兼容参数：请求间隔秒数')
    parser.add_argument('--limit-stocks', type=int, default=0, help='兼容参数：限制抓取股票数量')
    parser.add_argument('--rebuild', action='store_true', help='兼容参数：全量重建')
    parser.add_argument('--keep-suspended', action='store_true', help='兼容参数：保留停牌记录')
    parser.add_argument('--legacy-direct-fetch', action='store_true', help='兼容参数：保留旧入口标记，当前仍优先走 bridge')
    parser.add_argument('--runtime-root', default='./temp/ingestion_runtime', help='运行时目录')
    return parser.parse_args()


def _resolve_runtime_root(runtime_root: str) -> str:
    if os.path.isabs(str(runtime_root)):
        return str(runtime_root)
    return str((SCRIPT_DIR / str(runtime_root)).resolve())


def main() -> None:
    args = parse_args()
    result = run_stock_data_bridge(args, runtime_root=_resolve_runtime_root(str(args.runtime_root)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
