#!/usr/bin/env python3
"""
抓取沪深300成分股历史数据（Baostock）

增强点：
- 支持 index_date / adjustflag / frequency 参数化
- 增加网络重试与请求节流
- 增量更新时只补齐缺失时间段
- 一次性加载旧数据建立覆盖索引，避免每只股票重复读 CSV
- 输出失败明细与数据质量统计到 manifest
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import baostock as bs
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_SRC_DIR = SCRIPT_DIR / 'code' / 'src'
if str(CODE_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_SRC_DIR))

from config import config
from data_manager import build_file_snapshot
from data_manager import resolve_data_root
from data_manager import resolve_dataset_path
from data_manager import save_data_manifest

HISTORY_FIELDS = (
    'date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg'
)
OUTPUT_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    default_output = resolve_dataset_path(config, 'stock_data.csv')

    parser = argparse.ArgumentParser(description='抓取沪深300成分股历史日线数据')
    parser.add_argument('--start-date', default='2015-01-01', help='抓取起始日期，格式 YYYY-MM-DD')
    parser.add_argument('--end-date', default=datetime.today().strftime('%Y-%m-%d'), help='抓取结束日期，格式 YYYY-MM-DD')
    parser.add_argument('--index-date', default='', help='HS300 成分股快照日期，默认使用 end-date')
    parser.add_argument('--output-path', default=default_output, help=f'输出文件路径，默认 {default_output}')
    parser.add_argument('--manifest-path', default='', help='数据清单输出路径，默认 <data_path>/data_manifest_stock_fetch.json')

    parser.add_argument('--frequency', default='d', help='K 线频率，日线用 d（默认 d）')
    parser.add_argument('--adjustflag', choices=['1', '2', '3'], default='1', help='复权方式：1后复权 2前复权 3不复权')

    parser.add_argument('--max-retries', type=int, default=3, help='单次请求最大重试次数，默认 3')
    parser.add_argument('--retry-backoff-seconds', type=float, default=1.2, help='重试退避秒数（线性），默认 1.2')
    parser.add_argument('--request-interval-seconds', type=float, default=0.05, help='每只股票请求后的间隔秒数，默认 0.05')
    parser.add_argument('--limit-stocks', type=int, default=0, help='仅抓取前 N 只成分股（0 表示全量）')

    parser.add_argument('--rebuild', action='store_true', help='忽略已有文件并全量重建')
    parser.add_argument('--keep-suspended', action='store_true', help='保留 tradestatus != 1 的记录（默认过滤停牌）')
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def _normalize_date(date_str: str, arg_name: str) -> str:
    ts = pd.to_datetime(date_str, errors='coerce')
    if pd.isna(ts):
        raise ValueError(f'{arg_name} 日期格式无效: {date_str}')
    return ts.strftime('%Y-%m-%d')


def _normalize_stock_code(code_series: pd.Series) -> pd.Series:
    s = code_series.astype(str).str.strip()
    s = s.str.split('.').str[-1]
    s = s.str.replace(r'[^0-9]', '', regex=True)
    s = s.str[-6:].str.zfill(6)
    return s


def _format_date_slash(ts: pd.Timestamp) -> str:
    return f'{ts.year}/{ts.month}/{ts.day}'


def _result_to_frame(rs) -> pd.DataFrame:
    rows: List[List[str]] = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def login() -> None:
    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError(f'baostock 登录失败: {lg.error_msg}')
    print('baostock 登录成功')


def logout() -> None:
    bs.logout()
    print('baostock 已登出')


def get_hs300_stocks(index_date: str) -> Tuple[pd.DataFrame, str]:
    """获取沪深300成分股，优先使用 index_date，失败则回退最新。"""
    candidates = []
    if index_date:
        candidates.append(index_date)
    candidates.append('')

    last_error = ''
    for date_candidate in candidates:
        rs = bs.query_hs300_stocks(date=date_candidate)
        if rs.error_code != '0':
            last_error = rs.error_msg
            continue
        df = _result_to_frame(rs)
        if df.empty:
            last_error = '返回为空'
            continue

        used_date = date_candidate or 'latest'
        print(f'获取到 HS300 成分股 {len(df)} 只，快照日期: {used_date}')
        return df, used_date

    raise RuntimeError(f'获取 HS300 成分股失败: {last_error}')


def _query_history_raw(
    bs_code: str,
    start_date: str,
    end_date: str,
    frequency: str,
    adjustflag: str,
    max_retries: int,
    retry_backoff_seconds: float,
) -> pd.DataFrame:
    last_error = ''
    for attempt in range(1, max_retries + 1):
        rs = bs.query_history_k_data_plus(
            bs_code,
            HISTORY_FIELDS,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag=adjustflag,
        )
        if rs.error_code == '0':
            return _result_to_frame(rs)

        last_error = rs.error_msg
        if attempt < max_retries:
            sleep_seconds = max(0.0, retry_backoff_seconds) * attempt
            print(f'    请求失败，重试 {attempt}/{max_retries - 1}，{sleep_seconds:.2f}s 后重试: {last_error}')
            time.sleep(sleep_seconds)

    raise RuntimeError(last_error or '未知错误')


def _transform_history_frame(raw_df: pd.DataFrame, keep_suspended: bool) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = raw_df.copy()

    if 'date' not in df.columns or 'code' not in df.columns:
        raise ValueError('Baostock 返回缺少 date/code 列')

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).copy()

    numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg', 'tradestatus']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if (not keep_suspended) and 'tradestatus' in df.columns:
        df = df[df['tradestatus'] == 1].copy()

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    price_denominator = df['preclose'].replace(0, pd.NA)
    df['振幅'] = ((df['high'] - df['low']) / price_denominator * 100).fillna(0.0).round(4)
    df['涨跌额'] = (df['close'] - df['preclose']).fillna(0.0).round(4)

    df['date'] = df['date'].apply(_format_date_slash)
    df['code'] = _normalize_stock_code(df['code'])

    out = df.rename(
        columns={
            'code': '股票代码',
            'date': '日期',
            'open': '开盘',
            'close': '收盘',
            'high': '最高',
            'low': '最低',
            'volume': '成交量',
            'amount': '成交额',
            'turn': '换手率',
            'pctChg': '涨跌幅',
        }
    )
    out = out[OUTPUT_COLUMNS].copy()

    for col in ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率', '涨跌幅', '振幅', '涨跌额']:
        out[col] = pd.to_numeric(out[col], errors='coerce')

    out['日期_dt'] = pd.to_datetime(out['日期'], errors='coerce')
    out = out.dropna(subset=['日期_dt']).sort_values('日期_dt').drop(columns=['日期_dt']).reset_index(drop=True)
    return out


def _filter_window(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = df.copy()
    out['股票代码'] = _normalize_stock_code(out['股票代码'])
    out['日期_dt'] = pd.to_datetime(out['日期'], errors='coerce')
    out = out.dropna(subset=['日期_dt']).copy()

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    out = out[(out['日期_dt'] >= start_dt) & (out['日期_dt'] <= end_dt)].copy()

    if out.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out['日期'] = out['日期_dt'].apply(_format_date_slash)
    out = out.drop(columns=['日期_dt'])

    missing_cols = [c for c in OUTPUT_COLUMNS if c not in out.columns]
    for col in missing_cols:
        out[col] = pd.NA
    out = out[OUTPUT_COLUMNS]
    return out


def load_existing_data(output_path: str, start_date: str, end_date: str, rebuild: bool) -> Tuple[pd.DataFrame, Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]]:
    if rebuild or (not os.path.exists(output_path)):
        return pd.DataFrame(columns=OUTPUT_COLUMNS), {}

    try:
        existing = pd.read_csv(output_path, dtype={'股票代码': str})
    except Exception as exc:
        print(f'警告: 读取已有数据失败，将按空数据处理: {exc}')
        return pd.DataFrame(columns=OUTPUT_COLUMNS), {}

    existing = _filter_window(existing, start_date, end_date)
    if existing.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), {}

    indexed = existing.copy()
    indexed['日期_dt'] = pd.to_datetime(indexed['日期'], errors='coerce')
    indexed = indexed.dropna(subset=['日期_dt'])

    coverage: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    ranges = indexed.groupby('股票代码')['日期_dt'].agg(['min', 'max']).reset_index()
    for row in ranges.itertuples(index=False):
        coverage[str(row.股票代码).zfill(6)] = (row.min, row.max)

    print(
        f'已加载旧数据: rows={len(existing):,}, stocks={existing["股票代码"].nunique()}, '
        f'date={existing["日期"].min()}~{existing["日期"].max()}'
    )
    return existing, coverage


def _build_fetch_ranges(
    start_date: str,
    end_date: str,
    existing_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
) -> List[Tuple[str, str, str]]:
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    if existing_range is None:
        return [(start_date, end_date, '全量')]

    existing_min, existing_max = existing_range
    ranges: List[Tuple[str, str, str]] = []

    if existing_min > start_dt:
        early_end = existing_min - timedelta(days=1)
        if early_end >= start_dt:
            ranges.append((start_dt.strftime('%Y-%m-%d'), early_end.strftime('%Y-%m-%d'), '早期补齐'))

    if existing_max < end_dt:
        late_start = existing_max + timedelta(days=1)
        if late_start <= end_dt:
            ranges.append((late_start.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'), '近期补齐'))

    return ranges


def merge_stock_data(existing_df: pd.DataFrame, new_df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    if new_df is None or new_df.empty:
        return existing_df

    if existing_df is None or existing_df.empty:
        merged = new_df.copy()
    else:
        base = existing_df.copy()
        base['股票代码'] = _normalize_stock_code(base['股票代码'])

        stock_old = base[base['股票代码'] == stock_code].copy()
        stock_new = new_df.copy()
        stock_new['股票代码'] = _normalize_stock_code(stock_new['股票代码'])

        if stock_old.empty:
            stock_merged = stock_new
        else:
            stock_merged = pd.concat([stock_old, stock_new], ignore_index=True)
            stock_merged['日期_dt'] = pd.to_datetime(stock_merged['日期'], errors='coerce')
            stock_merged = (
                stock_merged.dropna(subset=['日期_dt'])
                .sort_values('日期_dt')
                .drop_duplicates(subset=['日期_dt'], keep='last')
                .drop(columns=['日期_dt'])
            )

        other = base[base['股票代码'] != stock_code].copy()
        merged = pd.concat([other, stock_merged], ignore_index=True)

    merged['日期_dt'] = pd.to_datetime(merged['日期'], errors='coerce')
    merged = merged.dropna(subset=['日期_dt']).copy()
    merged['股票代码'] = _normalize_stock_code(merged['股票代码'])
    merged['日期'] = merged['日期_dt'].apply(_format_date_slash)
    merged = (
        merged.sort_values(['股票代码', '日期_dt'])
        .drop_duplicates(subset=['股票代码', '日期_dt'], keep='last')
        .drop(columns=['日期_dt'])
    )

    missing_cols = [c for c in OUTPUT_COLUMNS if c not in merged.columns]
    for col in missing_cols:
        merged[col] = pd.NA
    return merged[OUTPUT_COLUMNS].reset_index(drop=True)


def _count_trade_days(start_date: str, end_date: str) -> Optional[int]:
    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    if rs.error_code != '0':
        print(f'警告: query_trade_dates 失败: {rs.error_msg}')
        return None

    trade_df = _result_to_frame(rs)
    if trade_df.empty or 'is_trading_day' not in trade_df.columns:
        return None

    trade_df['is_trading_day'] = pd.to_numeric(trade_df['is_trading_day'], errors='coerce').fillna(0).astype(int)
    return int((trade_df['is_trading_day'] == 1).sum())


def _save_frame(df: pd.DataFrame, output_path: str) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, encoding='utf-8-sig')


def main() -> None:
    args = parse_args()

    start_date = _normalize_date(args.start_date, '--start-date')
    end_date = _normalize_date(args.end_date, '--end-date')
    if pd.to_datetime(start_date) > pd.to_datetime(end_date):
        raise ValueError(f'开始日期晚于结束日期: {start_date} > {end_date}')

    index_date = _normalize_date(args.index_date, '--index-date') if str(args.index_date).strip() else end_date

    output_path = str(_resolve_path(args.output_path))
    data_root = _resolve_path(resolve_data_root(config))
    save_dir = Path(output_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    manifest_target = (
        _resolve_path(args.manifest_path)
        if str(args.manifest_path).strip()
        else data_root / 'data_manifest_stock_fetch.json'
    )

    print(f'目标时间范围: {start_date} ~ {end_date}')
    print(f'成分股快照日期: {index_date}')
    print(f'输出文件: {output_path}')
    print(f'抓取参数: frequency={args.frequency}, adjustflag={args.adjustflag}')
    print(f'重试参数: max_retries={args.max_retries}, retry_backoff={args.retry_backoff_seconds}s')
    print('=' * 72)

    login()

    hs300_list_path = str(save_dir / 'hs300_stock_list.csv')
    failed_path = ''

    failed_rows: List[Dict[str, str]] = []
    total_new_records = 0
    success_count = 0
    skipped_complete_count = 0
    new_stock_count = 0
    incremental_count = 0

    try:
        hs300_df, hs300_used_date = get_hs300_stocks(index_date)
        if int(args.limit_stocks) > 0:
            hs300_df = hs300_df.head(int(args.limit_stocks)).copy()
            print(f'启用 limit_stocks={int(args.limit_stocks)}，本次仅抓取前 {len(hs300_df)} 只')
        hs300_df.to_csv(hs300_list_path, index=False, encoding='utf-8-sig')

        hs300_df['纯代码'] = _normalize_stock_code(hs300_df['code'])
        total = len(hs300_df)

        existing_df, coverage_map = load_existing_data(output_path, start_date, end_date, rebuild=bool(args.rebuild))

        for idx, row in hs300_df.iterrows():
            bs_code = str(row.get('code', '')).strip()
            stock_name = str(row.get('code_name', '')).strip()
            pure_code = str(row.get('纯代码', '')).zfill(6)

            if not bs_code or len(pure_code) != 6:
                failed_rows.append({'股票代码': bs_code, '股票名称': stock_name, '原因': '股票代码格式非法'})
                continue

            existing_range = coverage_map.get(pure_code)
            fetch_ranges = _build_fetch_ranges(start_date, end_date, existing_range)

            if not fetch_ranges:
                skipped_complete_count += 1
                print(f'[{idx + 1:03d}/{total}] {bs_code} {stock_name} - 数据已完整，跳过')
                continue

            if existing_range is None:
                print(f'[{idx + 1:03d}/{total}] {bs_code} {stock_name} - 全量抓取')
            else:
                prev_min, prev_max = existing_range
                print(
                    f'[{idx + 1:03d}/{total}] {bs_code} {stock_name} - 增量抓取 '
                    f'({prev_min.date()}~{prev_max.date()})'
                )

            new_frames: List[pd.DataFrame] = []
            try:
                for range_start, range_end, range_name in fetch_ranges:
                    print(f'    {range_name}: {range_start} ~ {range_end}')
                    raw = _query_history_raw(
                        bs_code=bs_code,
                        start_date=range_start,
                        end_date=range_end,
                        frequency=args.frequency,
                        adjustflag=args.adjustflag,
                        max_retries=max(1, int(args.max_retries)),
                        retry_backoff_seconds=max(0.0, float(args.retry_backoff_seconds)),
                    )
                    transformed = _transform_history_frame(raw, keep_suspended=bool(args.keep_suspended))
                    if not transformed.empty:
                        new_frames.append(transformed)

                if not new_frames:
                    print('    无新增记录')
                    continue

                new_data = pd.concat(new_frames, ignore_index=True)
                new_data['股票代码'] = _normalize_stock_code(new_data['股票代码'])
                new_data = new_data[new_data['股票代码'] == pure_code].copy()
                if new_data.empty:
                    print('    新数据为空（可能被停牌过滤）')
                    continue

                new_data['日期_dt'] = pd.to_datetime(new_data['日期'], errors='coerce')
                new_data = (
                    new_data.dropna(subset=['日期_dt'])
                    .sort_values('日期_dt')
                    .drop_duplicates(subset=['日期_dt'], keep='last')
                    .drop(columns=['日期_dt'])
                )

                existing_df = merge_stock_data(existing_df, new_data, pure_code)
                _save_frame(existing_df, output_path)

                new_min = pd.to_datetime(new_data['日期'], errors='coerce').min()
                new_max = pd.to_datetime(new_data['日期'], errors='coerce').max()
                if pd.notna(new_min) and pd.notna(new_max):
                    if existing_range is None:
                        coverage_map[pure_code] = (new_min, new_max)
                        new_stock_count += 1
                    else:
                        coverage_map[pure_code] = (min(existing_range[0], new_min), max(existing_range[1], new_max))
                        incremental_count += 1

                success_count += 1
                total_new_records += int(len(new_data))
                print(f'    ✓ 新增 {len(new_data)} 条')

            except Exception as exc:
                print(f'    ✗ 抓取失败: {exc}')
                failed_rows.append({'股票代码': bs_code, '股票名称': stock_name, '原因': str(exc)})

            interval_seconds = max(0.0, float(args.request_interval_seconds))
            if interval_seconds > 0:
                time.sleep(interval_seconds)

        # 最终去重排序并回写
        final_df = _filter_window(existing_df, start_date, end_date)
        final_df['股票代码'] = _normalize_stock_code(final_df['股票代码'])
        final_df['日期_dt'] = pd.to_datetime(final_df['日期'], errors='coerce')
        final_df = (
            final_df.dropna(subset=['日期_dt'])
            .sort_values(['股票代码', '日期_dt'])
            .drop_duplicates(subset=['股票代码', '日期_dt'], keep='last')
            .drop(columns=['日期_dt'])
        )
        final_df = final_df[OUTPUT_COLUMNS].reset_index(drop=True)
        _save_frame(final_df, output_path)

        trade_days = _count_trade_days(start_date, end_date)

        duplicate_rows = 0
        coverage_median = None
        coverage_p10 = None
        coverage_p90 = None
        if not final_df.empty:
            temp = final_df.copy()
            temp['日期_dt'] = pd.to_datetime(temp['日期'], errors='coerce')
            duplicate_rows = int(temp.duplicated(subset=['股票代码', '日期_dt']).sum())

            if trade_days and trade_days > 0:
                day_counts = temp.groupby('股票代码')['日期_dt'].nunique().astype(float)
                ratios = (day_counts / float(trade_days)).clip(lower=0.0)
                coverage_median = float(ratios.median())
                coverage_p10 = float(ratios.quantile(0.1))
                coverage_p90 = float(ratios.quantile(0.9))

        if failed_rows:
            failed_df = pd.DataFrame(failed_rows)
            failed_path = str(save_dir / 'failed_stocks.csv')
            failed_df.to_csv(failed_path, index=False, encoding='utf-8-sig')
            print(f'失败列表已保存: {failed_path}')

        print('\n' + '=' * 72)
        print('抓取完成')
        print(f'  - HS300 股票数: {total}')
        print(f'  - 成功抓取股票: {success_count}')
        print(f'  - 已完整跳过: {skipped_complete_count}')
        print(f'  - 全新写入股票: {new_stock_count}')
        print(f'  - 增量补齐股票: {incremental_count}')
        print(f'  - 失败股票: {len(failed_rows)}')
        print(f'  - 新增记录数: {total_new_records}')

        if os.path.exists(output_path):
            print(f'  - 输出文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB')
        if not final_df.empty:
            print(f'  - 总行数: {len(final_df):,}')
            print(f'  - 股票数量: {final_df["股票代码"].nunique()}')
            print(f'  - 日期范围: {final_df["日期"].min()} ~ {final_df["日期"].max()}')

        manifest = {
            'action': 'fetch_stock_data',
            'parameters': {
                'start_date': start_date,
                'end_date': end_date,
                'index_date': index_date,
                'hs300_used_date': hs300_used_date,
                'frequency': args.frequency,
                'adjustflag': args.adjustflag,
                'max_retries': int(args.max_retries),
                'retry_backoff_seconds': float(args.retry_backoff_seconds),
                'request_interval_seconds': float(args.request_interval_seconds),
                'limit_stocks': int(args.limit_stocks),
                'rebuild': bool(args.rebuild),
                'keep_suspended': bool(args.keep_suspended),
            },
            'stats': {
                'total_hs300_stocks': int(total),
                'success_stocks': int(success_count),
                'skipped_complete_stocks': int(skipped_complete_count),
                'new_stock_count': int(new_stock_count),
                'incremental_stock_count': int(incremental_count),
                'failed_stock_count': int(len(failed_rows)),
                'total_new_records': int(total_new_records),
                'trade_days_in_window': int(trade_days) if trade_days is not None else None,
            },
            'quality': {
                'duplicate_rows_by_stock_date': int(duplicate_rows),
                'median_stock_coverage_vs_trade_days': coverage_median,
                'p10_stock_coverage_vs_trade_days': coverage_p10,
                'p90_stock_coverage_vs_trade_days': coverage_p90,
            },
            'outputs': {
                'stock_data_csv': build_file_snapshot(output_path, inspect_csv=True),
                'hs300_list_csv': build_file_snapshot(hs300_list_path, inspect_csv=True),
                'failed_stocks_csv': build_file_snapshot(failed_path, inspect_csv=True),
            },
        }

        saved_manifest = save_data_manifest(
            str(manifest_target.parent),
            manifest,
            filename=manifest_target.name,
        )
        print(f'数据清单: {saved_manifest}')

    finally:
        logout()


if __name__ == '__main__':
    main()
