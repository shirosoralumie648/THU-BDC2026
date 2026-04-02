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

from ingestion.models import IngestionRequest
from ingestion.service import IngestionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='兼容入口：通过 ingestion service 抓取日线数据')
    parser.add_argument('--pipeline-config-dir', default='./config', help='配置目录')
    parser.add_argument('--dataset-name', default='market_bar_1d', help='dataset 名称')
    parser.add_argument('--start-date', default='2015-01-01', help='开始日期')
    parser.add_argument('--end-date', default='2026-03-31', help='结束日期')
    parser.add_argument('--runtime-root', default='./temp/ingestion_runtime', help='运行时目录')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = IngestionService.from_config_dir(
        str(args.pipeline_config_dir),
        runtime_root=str((SCRIPT_DIR / args.runtime_root).resolve()) if not os.path.isabs(str(args.runtime_root)) else str(args.runtime_root),
        project_root=str(SCRIPT_DIR),
    )
    job = service.create_and_run(
        IngestionRequest(
            dataset=str(args.dataset_name),
            start=str(args.start_date),
            end=str(args.end_date),
        )
    )
    print(json.dumps(service.job_to_payload(job), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
