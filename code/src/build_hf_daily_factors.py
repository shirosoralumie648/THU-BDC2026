import argparse
import glob
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import config
from data_manager import build_file_snapshot
from data_manager import infer_existing_column
from data_manager import normalize_stock_code_series
from data_manager import resolve_data_root
from data_manager import save_data_manifest
from pipeline_config import derive_hf_builder_defaults
from pipeline_config import load_pipeline_configs
from pipeline_config import PipelineConfigError


def _resolve_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((Path(__file__).resolve().parents[2] / path).resolve())


def _split_csv_values(raw: str) -> List[str]:
    raw = str(raw or '').strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(',') if part.strip()]


def _parse_resample_minutes(raw: str) -> List[int]:
    values = []
    for token in _split_csv_values(raw):
        try:
            minutes = int(token)
        except ValueError as exc:
            raise ValueError(f'非法重采样周期: {token}') from exc
        if minutes <= 0:
            raise ValueError(f'重采样周期必须大于 0: {minutes}')
        values.append(minutes)
    return sorted(set(values))


def _resolve_column(
    df: pd.DataFrame,
    preferred: str,
    candidates: Iterable[str],
    required: bool,
    label: str,
) -> Optional[str]:
    preferred = str(preferred or '').strip()
    if preferred and preferred in df.columns:
        return preferred
    inferred = infer_existing_column(df, candidates)
    if inferred is not None:
        return inferred
    if required:
        raise ValueError(f'缺少必要列 {label}，候选={list(candidates)}')
    return None


def _collect_input_paths(inputs: List[str], input_globs: List[str]) -> List[str]:
    paths = []
    for item in inputs:
        for token in _split_csv_values(item):
            paths.append(_resolve_path(token))

    for pattern in input_globs:
        for token in _split_csv_values(pattern):
            resolved_pattern = _resolve_path(token)
            matched = sorted(glob.glob(resolved_pattern))
            paths.extend([str(Path(path).resolve()) for path in matched])

    dedup = []
    seen = set()
    for path in paths:
        norm = str(Path(path).resolve())
        if norm in seen:
            continue
        seen.add(norm)
        dedup.append(norm)
    return dedup


def parse_args() -> argparse.Namespace:
    default_output = os.path.join(resolve_data_root(config), 'hf_daily_factors.csv')

    parser = argparse.ArgumentParser(description='将高频(日内)数据聚合为日级因子表')
    parser.add_argument(
        '--pipeline-config-dir',
        default='./config',
        help='多源管道配置目录（含 datasets.yaml/factors.yaml/storage.yaml），默认 ./config',
    )
    parser.add_argument(
        '--dataset-name',
        default='market_bar_1m',
        help='datasets.yaml 中的分钟数据集名称，仅用于配置校验与记录，默认 market_bar_1m',
    )
    parser.add_argument(
        '--input',
        action='append',
        default=[],
        help='高频数据文件路径（CSV），可重复传参，也支持逗号分隔',
    )
    parser.add_argument(
        '--input-glob',
        action='append',
        default=[],
        help='高频数据文件 glob 模式，可重复传参，也支持逗号分隔',
    )
    parser.add_argument('--output', default=default_output, help=f'输出日因子路径，默认 {default_output}')
    parser.add_argument('--manifest-path', default='', help='清单输出路径，默认 <output_dir>/data_manifest_hf_daily_factors.json')

    parser.add_argument('--stock-col', default='股票代码', help='股票代码列名（支持自动推断）')
    parser.add_argument('--datetime-col', default='datetime', help='时间戳列名（支持自动推断）')
    parser.add_argument('--price-col', default='', help='价格列名（留空自动推断）')
    parser.add_argument('--volume-col', default='', help='成交量列名（留空自动推断，可选）')
    parser.add_argument('--amount-col', default='', help='成交额列名（留空自动推断，可选）')

    parser.add_argument(
        '--resample-minutes',
        default='',
        help='可选重采样周期（分钟），逗号分隔，如 "5,15,30"；留空表示不额外重采样',
    )
    parser.add_argument('--skip-raw', action='store_true', default=None, help='不输出原始频率版本，仅输出重采样版本')
    parser.add_argument('--force-suffix', action='store_true', default=None, help='即使单版本也给因子列添加后缀')
    parser.add_argument('--tail-minutes', type=int, default=None, help='尾盘窗口分钟数（默认来自 factors.yaml 或 30）')
    parser.add_argument('--min-bars', type=int, default=None, help='单股票单日最小bar数（默认来自 factors.yaml 或 10）')
    args = parser.parse_args()

    hf_defaults: Dict[str, object] = {}
    pipeline_validation = {'valid': False, 'errors': [], 'warnings': []}
    try:
        pipeline_configs, report = load_pipeline_configs(
            config_dir=args.pipeline_config_dir,
            strict=False,
        )
        pipeline_validation = report.to_dict()
        hf_defaults = derive_hf_builder_defaults(pipeline_configs.get('factors', {}))
        datasets_cfg = pipeline_configs.get('datasets', {}).get('datasets', {})
        if args.dataset_name not in datasets_cfg:
            pipeline_validation['warnings'].append(
                f'datasets.yaml 未找到 dataset `{args.dataset_name}`，将继续执行并使用 CLI 输入'
            )
    except PipelineConfigError as exc:
        pipeline_validation['errors'].append(str(exc))

    if args.tail_minutes is None:
        args.tail_minutes = int(hf_defaults.get('tail_minutes') or 30)
    if args.min_bars is None:
        args.min_bars = int(hf_defaults.get('min_bars') or 10)
    if (not str(args.resample_minutes).strip()) and hf_defaults.get('resample_minutes'):
        args.resample_minutes = ','.join(str(v) for v in hf_defaults.get('resample_minutes', []))
    if args.skip_raw is None:
        args.skip_raw = bool(hf_defaults.get('skip_raw', False))
    if args.force_suffix is None:
        args.force_suffix = bool(hf_defaults.get('force_suffix', False))

    args.pipeline_validation = pipeline_validation
    return args


def _normalize_source_data(
    path: str,
    stock_col: str,
    datetime_col: str,
    price_col: str,
    volume_col: str,
    amount_col: str,
) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(path)
    resolved_stock_col = _resolve_column(
        df,
        preferred=stock_col,
        candidates=['股票代码', 'stock_id', 'code', 'ts_code'],
        required=True,
        label='stock_col',
    )
    resolved_datetime_col = _resolve_column(
        df,
        preferred=datetime_col,
        candidates=['datetime', 'time', 'timestamp', '日期时间', 'date'],
        required=True,
        label='datetime_col',
    )
    resolved_price_col = _resolve_column(
        df,
        preferred=price_col,
        candidates=['收盘', 'close', '价格', 'price', '最新价', 'last_price'],
        required=True,
        label='price_col',
    )
    resolved_volume_col = _resolve_column(
        df,
        preferred=volume_col,
        candidates=['成交量', 'volume', 'vol'],
        required=False,
        label='volume_col',
    )
    resolved_amount_col = _resolve_column(
        df,
        preferred=amount_col,
        candidates=['成交额', 'amount', 'turnover'],
        required=False,
        label='amount_col',
    )

    cols = [resolved_stock_col, resolved_datetime_col, resolved_price_col]
    if resolved_volume_col:
        cols.append(resolved_volume_col)
    if resolved_amount_col:
        cols.append(resolved_amount_col)
    data = df[cols].copy()
    data['__stock_norm'] = normalize_stock_code_series(data[resolved_stock_col])
    data['__dt'] = pd.to_datetime(data[resolved_datetime_col], errors='coerce')
    data['__price'] = pd.to_numeric(data[resolved_price_col], errors='coerce')
    if resolved_volume_col:
        data['__volume'] = pd.to_numeric(data[resolved_volume_col], errors='coerce')
    else:
        data['__volume'] = np.nan
    if resolved_amount_col:
        data['__amount'] = pd.to_numeric(data[resolved_amount_col], errors='coerce')
    else:
        data['__amount'] = np.nan
    data = data.dropna(subset=['__stock_norm', '__dt', '__price'])
    data['__date'] = data['__dt'].dt.normalize()
    data['__source'] = Path(path).name

    meta = {
        'path': path,
        'resolved_columns': {
            'stock_col': resolved_stock_col,
            'datetime_col': resolved_datetime_col,
            'price_col': resolved_price_col,
            'volume_col': resolved_volume_col,
            'amount_col': resolved_amount_col,
        },
        'rows': int(len(data)),
        'stocks': int(data['__stock_norm'].nunique()) if not data.empty else 0,
    }
    return data, meta


def _resample_intraday(data: pd.DataFrame, period_minutes: int) -> pd.DataFrame:
    freq = f'{int(period_minutes)}min'
    parts = []
    for (stock, date), group in data.groupby(['__stock_norm', '__date'], sort=False):
        g = group.sort_values('__dt').set_index('__dt')
        agg_dict = {'__price': 'last'}
        if '__volume' in g.columns:
            agg_dict['__volume'] = 'sum'
        if '__amount' in g.columns:
            agg_dict['__amount'] = 'sum'
        rs = g.resample(freq).agg(agg_dict)
        rs = rs.dropna(subset=['__price'])
        if rs.empty:
            continue
        rs = rs.reset_index()
        rs['__stock_norm'] = stock
        rs['__date'] = date
        parts.append(rs[['__stock_norm', '__date', '__dt', '__price', '__volume', '__amount']])
    if not parts:
        return pd.DataFrame(columns=['__stock_norm', '__date', '__dt', '__price', '__volume', '__amount'])
    return pd.concat(parts, axis=0, ignore_index=True)


def _compute_day_features(
    group: pd.DataFrame,
    has_volume: bool,
    has_amount: bool,
    tail_minutes: int,
):
    eps = 1e-12
    group = group.sort_values('__dt').copy()
    prices = pd.to_numeric(group['__price'], errors='coerce')
    prices = prices.replace([np.inf, -np.inf], np.nan)
    dt = group['__dt']

    out = {
        'hf_bar_count': int(prices.notna().sum()),
    }
    valid_prices = prices.dropna()
    if len(valid_prices) < 2:
        out.update({
            'hf_open_close_ret': np.nan,
            'hf_intraday_range_pct': np.nan,
            'hf_logret_std': np.nan,
            'hf_realized_var': np.nan,
            'hf_realized_vol': np.nan,
            'hf_max_abs_logret': np.nan,
            'hf_up_bar_ratio': np.nan,
            'hf_last_tail_ret': np.nan,
        })
        if has_volume:
            out['hf_last_tail_volume_share'] = np.nan
        if has_amount:
            out['hf_last_tail_amount_share'] = np.nan
        return pd.Series(out)

    first_px = float(valid_prices.iloc[0])
    last_px = float(valid_prices.iloc[-1])
    max_px = float(valid_prices.max())
    min_px = float(valid_prices.min())
    out['hf_open_close_ret'] = (last_px - first_px) / (first_px + eps)
    out['hf_intraday_range_pct'] = (max_px - min_px) / (first_px + eps)

    log_ret = np.log(valid_prices + eps).diff().dropna()
    if len(log_ret) > 0:
        out['hf_logret_std'] = float(log_ret.std(ddof=0))
        realized_var = float((log_ret ** 2).sum())
        out['hf_realized_var'] = realized_var
        out['hf_realized_vol'] = float(np.sqrt(realized_var))
        out['hf_max_abs_logret'] = float(log_ret.abs().max())
        out['hf_up_bar_ratio'] = float((log_ret > 0).mean())
    else:
        out['hf_logret_std'] = np.nan
        out['hf_realized_var'] = np.nan
        out['hf_realized_vol'] = np.nan
        out['hf_max_abs_logret'] = np.nan
        out['hf_up_bar_ratio'] = np.nan

    end_ts = dt.max()
    tail_start = end_ts - pd.Timedelta(minutes=max(1, int(tail_minutes)))
    tail_mask = dt >= tail_start
    tail_prices = pd.to_numeric(group.loc[tail_mask, '__price'], errors='coerce').dropna()
    if len(tail_prices) >= 2:
        tail_first = float(tail_prices.iloc[0])
        tail_last = float(tail_prices.iloc[-1])
        out['hf_last_tail_ret'] = (tail_last - tail_first) / (tail_first + eps)
    else:
        out['hf_last_tail_ret'] = np.nan

    if has_volume:
        volumes = pd.to_numeric(group['__volume'], errors='coerce').clip(lower=0.0).fillna(0.0)
        total_volume = float(volumes.sum())
        tail_volume = float(volumes[tail_mask].sum())
        out['hf_last_tail_volume_share'] = tail_volume / (total_volume + eps) if total_volume > 0 else np.nan

    if has_amount:
        amounts = pd.to_numeric(group['__amount'], errors='coerce').clip(lower=0.0).fillna(0.0)
        total_amount = float(amounts.sum())
        tail_amount = float(amounts[tail_mask].sum())
        out['hf_last_tail_amount_share'] = tail_amount / (total_amount + eps) if total_amount > 0 else np.nan

    return pd.Series(out)


def _apply_feature_suffix(daily: pd.DataFrame, suffix: str, force: bool) -> pd.DataFrame:
    key_cols = {'股票代码', '日期'}
    if not force:
        return daily
    rename_map = {}
    for col in daily.columns:
        if col in key_cols:
            continue
        rename_map[col] = f'{col}_{suffix}'
    return daily.rename(columns=rename_map)


def _build_daily_feature_table(
    data: pd.DataFrame,
    tail_minutes: int,
    min_bars: int,
    suffix: str,
    add_suffix: bool,
) -> pd.DataFrame:
    grouped = data.groupby(['__stock_norm', '__date'], sort=True)
    has_volume = bool(data['__volume'].notna().any())
    has_amount = bool(data['__amount'].notna().any())

    try:
        daily = grouped.apply(
            _compute_day_features,
            has_volume=has_volume,
            has_amount=has_amount,
            tail_minutes=tail_minutes,
            include_groups=False,
        ).reset_index()
    except TypeError:
        daily = grouped.apply(
            _compute_day_features,
            has_volume=has_volume,
            has_amount=has_amount,
            tail_minutes=tail_minutes,
        ).reset_index()

    daily = daily.rename(columns={'__stock_norm': '股票代码', '__date': '日期'})
    daily = daily[daily['hf_bar_count'] >= max(1, int(min_bars))].copy()
    daily['日期'] = pd.to_datetime(daily['日期']).dt.strftime('%Y-%m-%d')
    daily = _apply_feature_suffix(daily, suffix=suffix, force=add_suffix)
    return daily


def _merge_daily_tables(daily_tables: List[pd.DataFrame]) -> pd.DataFrame:
    if not daily_tables:
        return pd.DataFrame(columns=['股票代码', '日期'])

    merged = daily_tables[0]
    for daily in daily_tables[1:]:
        merged = merged.merge(daily, on=['股票代码', '日期'], how='outer', validate='one_to_one')
    merged = merged.sort_values(['股票代码', '日期']).reset_index(drop=True)
    return merged


def main() -> None:
    args = parse_args()
    input_paths = _collect_input_paths(args.input, args.input_glob)
    if not input_paths:
        raise ValueError('至少需要通过 --input 或 --input-glob 提供一个输入文件')

    output_path = _resolve_path(args.output)
    if str(args.manifest_path).strip():
        manifest_path = _resolve_path(args.manifest_path)
    else:
        manifest_path = str(Path(output_path).parent / 'data_manifest_hf_daily_factors.json')

    for path in input_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f'未找到输入文件: {path}')

    normalized_sources = []
    source_meta = []
    for path in input_paths:
        data, meta = _normalize_source_data(
            path=path,
            stock_col=args.stock_col,
            datetime_col=args.datetime_col,
            price_col=args.price_col,
            volume_col=args.volume_col,
            amount_col=args.amount_col,
        )
        if data.empty:
            print(f'跳过空输入: {path}')
            continue
        normalized_sources.append(data)
        source_meta.append(meta)

    if not normalized_sources:
        raise ValueError('所有输入文件清洗后均为空，无法生成日因子')

    data = pd.concat(normalized_sources, axis=0, ignore_index=True)
    before = len(data)
    data = data.sort_values('__dt').drop_duplicates(subset=['__stock_norm', '__dt'], keep='last')
    dropped = int(before - len(data))
    if dropped > 0:
        print(f'已去除重复 tick/bar: {dropped}')

    periods = _parse_resample_minutes(args.resample_minutes)
    variant_specs = []
    if not bool(args.skip_raw):
        variant_specs.append({'label': 'raw', 'period': None})
    for period in periods:
        variant_specs.append({'label': f'm{period:02d}', 'period': int(period)})
    if not variant_specs:
        raise ValueError('没有可计算的版本：请关闭 --skip-raw 或设置 --resample-minutes')

    add_suffix = bool(args.force_suffix or len(variant_specs) > 1)
    daily_tables = []
    variant_rows = {}
    for spec in variant_specs:
        label = spec['label']
        period = spec['period']
        if period is None:
            variant_data = data[['__stock_norm', '__date', '__dt', '__price', '__volume', '__amount']].copy()
        else:
            variant_data = _resample_intraday(
                data[['__stock_norm', '__date', '__dt', '__price', '__volume', '__amount']].copy(),
                period_minutes=period,
            )
        daily = _build_daily_feature_table(
            variant_data,
            tail_minutes=max(1, int(args.tail_minutes)),
            min_bars=max(1, int(args.min_bars)),
            suffix=label,
            add_suffix=add_suffix,
        )
        daily_tables.append(daily)
        variant_rows[label] = int(len(daily))
        print(f'版本 {label} 生成完成: rows={len(daily)}')

    merged_daily = _merge_daily_tables(daily_tables)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    merged_daily.to_csv(output_path, index=False, encoding='utf-8')

    summary = {
        'rows': int(len(merged_daily)),
        'stocks': int(merged_daily['股票代码'].nunique()) if not merged_daily.empty else 0,
        'date_min': str(pd.to_datetime(merged_daily['日期']).min().date()) if not merged_daily.empty else '',
        'date_max': str(pd.to_datetime(merged_daily['日期']).max().date()) if not merged_daily.empty else '',
        'factor_cols': [col for col in merged_daily.columns if col not in {'股票代码', '日期'}],
        'variants': [spec['label'] for spec in variant_specs],
        'variant_rows': variant_rows,
        'source_count': int(len(source_meta)),
    }

    manifest = {
        'action': 'build_hf_daily_factors',
        'inputs': [build_file_snapshot(path, inspect_csv=True) for path in input_paths],
        'output': build_file_snapshot(output_path, inspect_csv=True),
        'params': {
            'pipeline_config_dir': str(args.pipeline_config_dir),
            'dataset_name': str(args.dataset_name),
            'stock_col': args.stock_col,
            'datetime_col': args.datetime_col,
            'price_col': args.price_col,
            'volume_col': args.volume_col,
            'amount_col': args.amount_col,
            'tail_minutes': int(args.tail_minutes),
            'min_bars': int(args.min_bars),
            'resample_minutes': periods,
            'skip_raw': bool(args.skip_raw),
            'force_suffix': bool(args.force_suffix),
        },
        'pipeline_config_validation': args.pipeline_validation,
        'sources': source_meta,
        'summary': summary,
    }
    saved_manifest = save_data_manifest(
        os.path.dirname(manifest_path) or '.',
        manifest,
        filename=os.path.basename(manifest_path),
    )

    print(f'高频日因子生成完成: {output_path}')
    print(f'输出行数: {summary["rows"]}, 股票数: {summary["stocks"]}')
    print(f'版本: {summary["variants"]}')
    print(f'清单文件: {saved_manifest}')


if __name__ == '__main__':
    main()
