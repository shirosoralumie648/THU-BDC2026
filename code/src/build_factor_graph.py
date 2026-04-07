import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

from config import config
from data_manager import build_canonical_csv_metadata_from_dataframe
from data_manager import build_csv_metadata_from_dataframe
from data_manager import build_file_snapshot
from data_manager import infer_existing_column
from data_manager import normalize_stock_code_series
from data_manager import resolve_dataset_path
from data_manager import resolve_hf_factor_path
from data_manager import save_data_manifest
from factor_store import apply_factor_expressions
from factor_store import build_factor_execution_plan
from factor_pipeline.intraday import _compute_intraday_nodes_from_minute
from factor_pipeline.macro import _build_macro_cutoff_frame
from factor_pipeline.macro import _compute_macro_series_asof
from pipeline_config import load_pipeline_configs
from pipeline_config import PipelineConfigError
from pipeline_config import render_feature_csv_compat_uri


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _resolve_path(path_str: str) -> str:
    path = Path(str(path_str or '').strip()).expanduser()
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _load_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {'.parquet', '.pq'}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _resolve_column(
    df: pd.DataFrame,
    preferred: str,
    candidates: Iterable[str],
    *,
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


def _canonicalize_daily_base(df: pd.DataFrame) -> pd.DataFrame:
    stock_col = _resolve_column(
        df,
        preferred='instrument_id',
        candidates=['instrument_id', '股票代码', 'stock_id', 'code', 'ts_code'],
        required=True,
        label='instrument_id',
    )
    date_col = _resolve_column(
        df,
        preferred='trade_date',
        candidates=['trade_date', '日期', 'date', 'datetime'],
        required=True,
        label='trade_date',
    )

    close_col = _resolve_column(
        df,
        preferred='close',
        candidates=['close', '收盘', 'latest_price'],
        required=True,
        label='close',
    )
    open_col = _resolve_column(
        df,
        preferred='open',
        candidates=['open', '开盘'],
        required=False,
        label='open',
    )
    high_col = _resolve_column(
        df,
        preferred='high',
        candidates=['high', '最高'],
        required=False,
        label='high',
    )
    low_col = _resolve_column(
        df,
        preferred='low',
        candidates=['low', '最低'],
        required=False,
        label='low',
    )
    preclose_col = _resolve_column(
        df,
        preferred='preclose',
        candidates=['preclose', '昨收', '前收'],
        required=False,
        label='preclose',
    )
    volume_col = _resolve_column(
        df,
        preferred='volume',
        candidates=['volume', '成交量', 'vol'],
        required=False,
        label='volume',
    )
    amount_col = _resolve_column(
        df,
        preferred='amount',
        candidates=['amount', '成交额', 'turnover_amount'],
        required=False,
        label='amount',
    )
    turnover_col = _resolve_column(
        df,
        preferred='turnover',
        candidates=['turnover', '换手率', 'turn'],
        required=False,
        label='turnover',
    )
    pct_col = _resolve_column(
        df,
        preferred='pct_chg',
        candidates=['pct_chg', '涨跌幅', 'pctChg'],
        required=False,
        label='pct_chg',
    )

    out = pd.DataFrame(index=df.index)
    out['instrument_id'] = normalize_stock_code_series(df[stock_col]).astype(str)
    out['trade_date'] = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
    out['close'] = pd.to_numeric(df[close_col], errors='coerce')
    out['open'] = pd.to_numeric(df[open_col], errors='coerce') if open_col else np.nan
    out['high'] = pd.to_numeric(df[high_col], errors='coerce') if high_col else np.nan
    out['low'] = pd.to_numeric(df[low_col], errors='coerce') if low_col else np.nan
    out['preclose'] = pd.to_numeric(df[preclose_col], errors='coerce') if preclose_col else np.nan
    out['volume'] = pd.to_numeric(df[volume_col], errors='coerce') if volume_col else np.nan
    out['amount'] = pd.to_numeric(df[amount_col], errors='coerce') if amount_col else np.nan
    out['turnover'] = pd.to_numeric(df[turnover_col], errors='coerce') if turnover_col else np.nan
    out['pct_chg'] = pd.to_numeric(df[pct_col], errors='coerce') if pct_col else np.nan

    out = out.dropna(subset=['instrument_id', 'trade_date', 'close']).copy()
    out = out.sort_values(['instrument_id', 'trade_date']).drop_duplicates(
        subset=['instrument_id', 'trade_date'],
        keep='last',
    )
    out['股票代码'] = out['instrument_id']
    out['日期'] = out['trade_date']
    return out.reset_index(drop=True)


def _detect_hf_daily_columns(df: pd.DataFrame) -> Dict[str, str]:
    candidates = {
        'f_hf_realized_vol_1d': [
            'f_hf_realized_vol_1d',
            'hf_realized_vol',
            'hf_realized_vol_raw',
            'hf_realized_vol_m05',
            'hf_realized_vol_m15',
            'hf_realized_vol_m30',
        ],
        'f_hf_tail_ret_30m': [
            'f_hf_tail_ret_30m',
            'hf_last_tail_ret',
            'hf_last_tail_ret_raw',
            'hf_last_tail_ret_m05',
            'hf_last_tail_ret_m15',
            'hf_last_tail_ret_m30',
        ],
        'f_hf_tail_amount_share_30m': [
            'f_hf_tail_amount_share_30m',
            'hf_last_tail_amount_share',
            'hf_last_tail_amount_share_raw',
            'hf_last_tail_amount_share_m05',
            'hf_last_tail_amount_share_m15',
            'hf_last_tail_amount_share_m30',
        ],
    }
    mapping = {}
    for target, names in candidates.items():
        for name in names:
            if name in df.columns:
                mapping[target] = name
                break
    return mapping


def _canonicalize_hf_daily_table(df: pd.DataFrame) -> pd.DataFrame:
    stock_col = _resolve_column(
        df,
        preferred='instrument_id',
        candidates=['instrument_id', '股票代码', 'stock_id', 'code', 'ts_code'],
        required=True,
        label='hf.instrument_id',
    )
    date_col = _resolve_column(
        df,
        preferred='trade_date',
        candidates=['trade_date', '日期', 'date', 'datetime'],
        required=True,
        label='hf.trade_date',
    )
    hf_cols = _detect_hf_daily_columns(df)
    if not hf_cols:
        raise ValueError(f'高频日因子文件缺少可识别列: {path}')

    out = pd.DataFrame(index=df.index)
    out['instrument_id'] = normalize_stock_code_series(df[stock_col]).astype(str)
    out['trade_date'] = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
    for target, source in hf_cols.items():
        out[target] = pd.to_numeric(df[source], errors='coerce')
    out = out.dropna(subset=['instrument_id', 'trade_date']).copy()
    out = out.sort_values(['instrument_id', 'trade_date']).drop_duplicates(
        subset=['instrument_id', 'trade_date'],
        keep='last',
    )
    return out.reset_index(drop=True)


def _load_hf_daily_table(path: str) -> pd.DataFrame:
    return _canonicalize_hf_daily_table(_load_table(path))


def _canonicalize_hf_minute_table(df: pd.DataFrame) -> pd.DataFrame:
    stock_col = _resolve_column(
        df,
        preferred='instrument_id',
        candidates=['instrument_id', '股票代码', 'stock_id', 'code', 'ts_code'],
        required=True,
        label='hf_minute.instrument_id',
    )
    ts_col = _resolve_column(
        df,
        preferred='ts',
        candidates=['ts', 'datetime', 'time', 'timestamp', '日期时间', 'date'],
        required=True,
        label='hf_minute.ts',
    )
    close_col = _resolve_column(
        df,
        preferred='close',
        candidates=['close', '收盘', '价格', 'price', '最新价', 'last_price'],
        required=True,
        label='hf_minute.close',
    )
    amount_col = _resolve_column(
        df,
        preferred='amount',
        candidates=['amount', '成交额', 'turnover'],
        required=False,
        label='hf_minute.amount',
    )

    out = pd.DataFrame(index=df.index)
    out['instrument_id'] = normalize_stock_code_series(df[stock_col]).astype(str)
    out['ts'] = pd.to_datetime(df[ts_col], errors='coerce')
    out['close'] = pd.to_numeric(df[close_col], errors='coerce')
    if amount_col:
        out['amount'] = pd.to_numeric(df[amount_col], errors='coerce')
    else:
        out['amount'] = np.nan
    out['trade_date'] = out['ts'].dt.normalize()
    out = out.dropna(subset=['instrument_id', 'ts', 'close', 'trade_date']).copy()
    out = out.sort_values(['instrument_id', 'ts']).drop_duplicates(
        subset=['instrument_id', 'ts'],
        keep='last',
    )
    return out.reset_index(drop=True)


def _load_hf_minute_table(path: str) -> pd.DataFrame:
    return _canonicalize_hf_minute_table(_load_table(path))


def _canonicalize_macro_table(df: pd.DataFrame) -> pd.DataFrame:
    series_col = _resolve_column(
        df,
        preferred='series_id',
        candidates=['series_id', 'macro_id', '指标ID', '指标'],
        required=True,
        label='macro.series_id',
    )
    available_col = _resolve_column(
        df,
        preferred='available_time',
        candidates=['available_time', 'release_time', 'observation_date', '日期', 'date'],
        required=True,
        label='macro.available_time',
    )
    value_col = _resolve_column(
        df,
        preferred='value',
        candidates=['value', '值', '数值'],
        required=True,
        label='macro.value',
    )

    out = pd.DataFrame(index=df.index)
    out['series_id'] = df[series_col].astype(str).str.strip()
    out['available_time'] = pd.to_datetime(df[available_col], errors='coerce')
    out['value'] = pd.to_numeric(df[value_col], errors='coerce')
    out = out.dropna(subset=['series_id', 'available_time']).copy()
    out = out.sort_values(['series_id', 'available_time'])
    return out.reset_index(drop=True)


def _load_macro_table(path: str) -> pd.DataFrame:
    return _canonicalize_macro_table(_load_table(path))


def _build_factor_fingerprint(nodes: List[Dict]) -> str:
    payload = []
    for node in nodes:
        compute = node.get('compute', {}) if isinstance(node, dict) else {}
        output = node.get('output', {}) if isinstance(node, dict) else {}
        payload.append(
            {
                'id': node.get('id'),
                'layer': node.get('layer'),
                'engine': compute.get('engine'),
                'expression': compute.get('expression', ''),
                'series_id': compute.get('series_id', ''),
                'dependencies': node.get('dependencies', []),
                'output': output.get('column', ''),
            }
        )
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _build_expression_specs(nodes: List[Dict]) -> List[Dict]:
    specs = []
    for node in nodes:
        compute = node.get('compute', {})
        output = node.get('output', {})
        engine = str(compute.get('engine', '')).strip()
        if engine not in {'expression', 'meta_expression'}:
            continue
        output_col = str(output.get('column', '')).strip()
        expression = str(compute.get('expression', '')).strip()
        if not output_col or not expression:
            continue
        specs.append(
            {
                'name': output_col,
                'expression': expression,
                'inputs': {},
                'group': str(node.get('layer', '')),
                'enabled': True,
            }
        )
    return specs


def _compute_expression_factors(df: pd.DataFrame, specs: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
    if not specs:
        return df, {'ordered_specs': [], 'time_series_specs': [], 'cross_sectional_specs': []}
    plan = build_factor_execution_plan(specs, error_prefix='factors.yaml')
    ts_specs = plan.get('time_series_specs', [])
    cs_specs = plan.get('cross_sectional_specs', [])

    out = df.copy()
    if ts_specs:
        grouped = []
        for _, g in out.groupby('instrument_id', sort=False):
            g_sorted = g.sort_values('trade_date')
            g_calc = apply_factor_expressions(
                g_sorted,
                ts_specs,
                error_prefix='时序因子',
                date_col='trade_date',
            )
            grouped.append(g_calc)
        out = pd.concat(grouped, axis=0, ignore_index=True) if grouped else out

    out = out.sort_values(['trade_date', 'instrument_id']).reset_index(drop=True)
    if cs_specs:
        out = apply_factor_expressions(
            out,
            cs_specs,
            error_prefix='截面因子',
            date_col='trade_date',
        )
    return out, plan


def _git_code_version() -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return ''
    if result.returncode != 0:
        return ''
    return result.stdout.strip()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='执行 factors.yaml DAG，输出宽表因子')
    parser.add_argument('--pipeline-config-dir', default='./config', help='配置目录（datasets/factors/storage YAML）')
    parser.add_argument('--feature-set-version', default='v1', help='特征版本号，例如 v1')
    parser.add_argument(
        '--base-input',
        default=resolve_dataset_path(config, 'stock_data.csv', for_write=False),
        help='日级行情输入（CSV/Parquet）',
    )
    parser.add_argument(
        '--hf-daily-input',
        default=resolve_hf_factor_path(config),
        help='高频聚合后的日级因子输入（可选）',
    )
    parser.add_argument(
        '--hf-minute-input',
        default='',
        help='分钟级原始行情输入（可选，若提供可直接计算 intraday_aggregate 节点）',
    )
    parser.add_argument(
        '--macro-input',
        default='',
        help='宏观序列输入（可选，需含 series_id/available_time/value）',
    )
    parser.add_argument(
        '--output',
        default='',
        help='输出宽表因子路径（留空按 factors.yaml factor_views.csv_compat_uri 渲染）',
    )
    parser.add_argument(
        '--manifest-path',
        default='',
        help='构建清单输出路径（留空按 factors.yaml build_manifest.output_uri 渲染）',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='严格模式：缺少高频/宏观输入或节点无法计算时直接失败',
    )
    parser.add_argument(
        '--run-id',
        default='',
        help='运行 ID（留空自动生成 UTC 时间戳）',
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run_id = str(args.run_id or '').strip() or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    try:
        pipeline_configs, report = load_pipeline_configs(
            config_dir=args.pipeline_config_dir,
            strict=False,
        )
    except PipelineConfigError as exc:
        raise ValueError(f'加载 pipeline 配置失败: {exc}') from exc

    factors_cfg = pipeline_configs.get('factors', {})
    factor_nodes = factors_cfg.get('factor_nodes', [])
    if not isinstance(factor_nodes, list) or not factor_nodes:
        raise ValueError('factors.yaml 缺少 factor_nodes')

    view_cfg = None
    views = factors_cfg.get('factor_views', [])
    if isinstance(views, list):
        for view in views:
            if isinstance(view, dict) and view.get('layout') == 'wide':
                view_cfg = view
                break
    if view_cfg is None:
        raise ValueError('factors.yaml 未配置 wide factor_view')

    base_input = _resolve_path(args.base_input)
    if not os.path.exists(base_input):
        raise FileNotFoundError(f'未找到 base-input: {base_input}')
    base_source_df = _load_table(base_input)
    base_input_csv_meta = (
        build_csv_metadata_from_dataframe(base_source_df)
        if base_input.lower().endswith('.csv')
        else None
    )
    base_df = _canonicalize_daily_base(base_source_df)

    node_status = []
    warnings = []
    errors = []
    intraday_source_map: Dict[str, str] = {}

    hf_input = str(args.hf_daily_input or '').strip()
    if hf_input:
        hf_input = _resolve_path(hf_input)
    hf_df = None
    hf_input_csv_meta = None
    if hf_input:
        if os.path.exists(hf_input):
            hf_source_df = _load_table(hf_input)
            if hf_input.lower().endswith('.csv'):
                hf_input_csv_meta = build_csv_metadata_from_dataframe(hf_source_df)
            hf_df = _canonicalize_hf_daily_table(hf_source_df)
            base_df = base_df.merge(
                hf_df,
                on=['instrument_id', 'trade_date'],
                how='left',
                validate='many_to_one',
            )
            for col in hf_df.columns:
                if col in {'instrument_id', 'trade_date'}:
                    continue
                intraday_source_map[col] = 'hf_daily_input'
        else:
            msg = f'hf-daily-input 不存在: {hf_input}'
            if args.strict:
                raise FileNotFoundError(msg)
            warnings.append(msg)

    hf_minute_input = str(args.hf_minute_input or '').strip()
    if hf_minute_input:
        hf_minute_input = _resolve_path(hf_minute_input)
    minute_df = None
    hf_minute_input_csv_meta = None
    intraday_issue_map: Dict[str, str] = {}
    if hf_minute_input:
        if os.path.exists(hf_minute_input):
            minute_source_df = _load_table(hf_minute_input)
            if hf_minute_input.lower().endswith('.csv'):
                hf_minute_input_csv_meta = build_csv_metadata_from_dataframe(minute_source_df)
            minute_df = _canonicalize_hf_minute_table(minute_source_df)
            minute_factor_df, minute_source_map, minute_issue_map, minute_warnings, minute_errors = _compute_intraday_nodes_from_minute(
                minute_df,
                factor_nodes,
                strict=bool(args.strict),
            )
            intraday_issue_map.update(minute_issue_map)
            warnings.extend(minute_warnings)
            errors.extend(minute_errors)
            if args.strict and minute_errors:
                raise ValueError('分钟级 intraday_aggregate 计算失败:\n- ' + '\n- '.join(minute_errors))

            if not minute_factor_df.empty:
                minute_cols = [col for col in minute_factor_df.columns if col not in {'instrument_id', 'trade_date'}]
                merged = base_df.merge(
                    minute_factor_df,
                    on=['instrument_id', 'trade_date'],
                    how='left',
                    suffixes=('', '__minute'),
                )
                for col in minute_cols:
                    minute_col = f'{col}__minute'
                    if minute_col in merged.columns:
                        merged[col] = merged[col].where(merged[col].notna(), merged[minute_col])
                        merged = merged.drop(columns=[minute_col])
                base_df = merged

                for col, source in minute_source_map.items():
                    if col in intraday_source_map:
                        intraday_source_map[col] = f'{intraday_source_map[col]}+minute_backfill'
                    else:
                        intraday_source_map[col] = source
        else:
            msg = f'hf-minute-input 不存在: {hf_minute_input}'
            if args.strict:
                raise FileNotFoundError(msg)
            warnings.append(msg)

    macro_input = str(args.macro_input or '').strip()
    if macro_input:
        macro_input = _resolve_path(macro_input)
    macro_df = None
    macro_input_csv_meta = None
    if macro_input:
        if os.path.exists(macro_input):
            macro_source_df = _load_table(macro_input)
            if macro_input.lower().endswith('.csv'):
                macro_input_csv_meta = build_csv_metadata_from_dataframe(macro_source_df)
            macro_df = _canonicalize_macro_table(macro_source_df)
        else:
            msg = f'macro-input 不存在: {macro_input}'
            if args.strict:
                raise FileNotFoundError(msg)
            warnings.append(msg)

    macro_cutoff_frame = None
    macro_join_frame = None
    macro_series_map: Dict[str, pd.DataFrame] = {}
    if macro_df is not None:
        macro_cutoff_frame = _build_macro_cutoff_frame(base_df['trade_date'])
        macro_join_frame = macro_cutoff_frame[['trade_date']].copy()
        macro_series_map = {
            str(series_id): group[['available_time', 'value']].sort_values('available_time').reset_index(drop=True)
            for series_id, group in macro_df.groupby('series_id', sort=False)
        }

    for node in factor_nodes:
        node_id = str(node.get('id', '')).strip()
        compute = node.get('compute', {}) if isinstance(node, dict) else {}
        output = node.get('output', {}) if isinstance(node, dict) else {}
        engine = str(compute.get('engine', '')).strip()
        output_col = str(output.get('column', '')).strip()
        status = {
            'id': node_id,
            'engine': engine,
            'output_column': output_col,
            'status': 'pending',
            'message': '',
        }

        if engine in {'expression', 'meta_expression'}:
            status['status'] = 'deferred'
            node_status.append(status)
            continue

        if engine == 'intraday_aggregate':
            if not output_col:
                status['status'] = 'error'
                status['message'] = '缺少 output.column'
                errors.append(f'节点 {node_id} 缺少 output.column')
            elif output_col in intraday_issue_map:
                issue = intraday_issue_map[output_col]
                if args.strict:
                    status['status'] = 'error'
                    status['message'] = issue
                    if issue not in errors:
                        errors.append(issue)
                else:
                    status['status'] = 'skipped'
                    status['message'] = issue
                    if issue not in warnings:
                        warnings.append(issue)
            elif output_col in base_df.columns:
                status['status'] = 'ok'
                status['message'] = f"source={intraday_source_map.get(output_col, 'available')}"
            else:
                base_df[output_col] = np.nan
                msg = f'节点 {node_id} 需要高频日因子列 `{output_col}`，当前不可用'
                if args.strict:
                    status['status'] = 'error'
                    status['message'] = msg
                    errors.append(msg)
                else:
                    status['status'] = 'skipped'
                    status['message'] = msg
                    warnings.append(msg)
            node_status.append(status)
            continue

        if engine == 'macro_asof_join':
            series_id = str(compute.get('series_id', '')).strip()
            if not output_col:
                status['status'] = 'error'
                status['message'] = '缺少 output.column'
                errors.append(f'节点 {node_id} 缺少 output.column')
            elif macro_df is None:
                base_df[output_col] = np.nan
                msg = f'节点 {node_id} 需要 macro-input'
                if args.strict:
                    status['status'] = 'error'
                    status['message'] = msg
                    errors.append(msg)
                else:
                    status['status'] = 'skipped'
                    status['message'] = msg
                    warnings.append(msg)
            else:
                max_staleness = compute.get('max_staleness_days')
                fill_method = str(compute.get('fill_method', 'forward'))
                macro_series_df = macro_series_map.get(series_id)
                series_asof = _compute_macro_series_asof(
                    macro_cutoff_frame if macro_cutoff_frame is not None else _build_macro_cutoff_frame(base_df['trade_date']),
                    macro_series_df,
                    max_staleness_days=int(max_staleness) if max_staleness is not None else None,
                    fill_method=fill_method,
                )
                if macro_join_frame is not None:
                    macro_join_frame = macro_join_frame.merge(
                        series_asof.rename(columns={'value': output_col}),
                        on='trade_date',
                        how='left',
                        validate='one_to_one',
                    )
                status['status'] = 'ok'
                status['message'] = f'series_id={series_id}'
            node_status.append(status)
            continue

        # 未实现 engine
        msg = f'节点 {node_id} 使用未实现 engine: {engine}'
        if args.strict:
            status['status'] = 'error'
            status['message'] = msg
            errors.append(msg)
        else:
            status['status'] = 'skipped'
            status['message'] = msg
            warnings.append(msg)
        node_status.append(status)

    if macro_join_frame is not None:
        macro_columns = [col for col in macro_join_frame.columns if col != 'trade_date']
        if macro_columns:
            base_df = base_df.merge(
                macro_join_frame,
                on='trade_date',
                how='left',
                validate='many_to_one',
            )

    expression_specs = _build_expression_specs(factor_nodes)
    base_df, plan = _compute_expression_factors(base_df, expression_specs)

    expr_name_to_status = {item['output_column']: item for item in node_status}
    for node in factor_nodes:
        compute = node.get('compute', {}) if isinstance(node, dict) else {}
        engine = str(compute.get('engine', '')).strip()
        if engine not in {'expression', 'meta_expression'}:
            continue
        output_col = str((node.get('output') or {}).get('column', '')).strip()
        node_id = str(node.get('id', '')).strip()
        item = expr_name_to_status.get(output_col, None)
        if item is None:
            item = {'id': node_id, 'engine': engine, 'output_column': output_col}
            node_status.append(item)
        if output_col and output_col in base_df.columns:
            item['status'] = 'ok'
            item['message'] = 'computed'
        else:
            msg = f'节点 {node_id} 未产出列: {output_col}'
            if args.strict:
                item['status'] = 'error'
                item['message'] = msg
                errors.append(msg)
            else:
                item['status'] = 'skipped'
                item['message'] = msg
                warnings.append(msg)

    if args.strict and errors:
        raise ValueError('因子构建失败:\n- ' + '\n- '.join(errors))

    include_factor_columns = view_cfg.get('include_factor_columns', [])
    if not isinstance(include_factor_columns, list):
        include_factor_columns = []
    missing_factor_columns = [col for col in include_factor_columns if col not in base_df.columns]
    for col in missing_factor_columns:
        base_df[col] = np.nan

    null_policy = view_cfg.get('null_policy', {}) if isinstance(view_cfg, dict) else {}
    default_value = float((null_policy or {}).get('default_value', 0.0))
    if include_factor_columns:
        base_df[include_factor_columns] = base_df[include_factor_columns].replace([np.inf, -np.inf], np.nan)
        base_df[include_factor_columns] = base_df[include_factor_columns].fillna(default_value)

    base_df['trade_date'] = pd.to_datetime(base_df['trade_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    base_df['日期'] = base_df['trade_date']
    base_df['股票代码'] = base_df['instrument_id'].astype(str)

    export_cols = ['股票代码', '日期', 'instrument_id', 'trade_date', *include_factor_columns]
    export_cols = [col for col in export_cols if col in base_df.columns]
    output_df = base_df[export_cols].copy()
    output_df = output_df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    rendered_default_output = render_feature_csv_compat_uri(
        factors_cfg,
        feature_set_version=str(args.feature_set_version),
    )
    output_path = str(args.output or '').strip()
    if not output_path:
        output_path = rendered_default_output or f'data/datasets/features/train_features_{args.feature_set_version}.csv'
    output_path = _resolve_path(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding='utf-8')
    output_csv_meta = build_canonical_csv_metadata_from_dataframe(
        output_df,
        date_col='日期',
        stock_col='股票代码',
    )

    build_manifest_cfg = factors_cfg.get('build_manifest', {}) if isinstance(factors_cfg, dict) else {}
    manifest_path = str(args.manifest_path or '').strip()
    if not manifest_path:
        manifest_tpl = str(build_manifest_cfg.get('output_uri', '')).strip()
        if manifest_tpl:
            manifest_path = manifest_tpl.format(
                feature_set_version=str(args.feature_set_version),
                run_date=run_date,
                run_id=run_id,
            )
        else:
            manifest_path = os.path.join(
                os.path.dirname(output_path),
                f'data_manifest_factor_build_{run_id}.json',
            )
    manifest_path = _resolve_path(manifest_path)

    factor_fingerprint = _build_factor_fingerprint(factor_nodes)
    quality_summary = {
        'row_count': int(len(output_df)),
        'stock_count': int(output_df['股票代码'].nunique()) if '股票代码' in output_df.columns else 0,
        'date_min': str(pd.to_datetime(output_df['日期']).min().date()) if not output_df.empty else '',
        'date_max': str(pd.to_datetime(output_df['日期']).max().date()) if not output_df.empty else '',
        'missing_factor_columns': missing_factor_columns,
    }
    if include_factor_columns:
        null_ratios = {}
        for col in include_factor_columns:
            if col in output_df.columns:
                null_ratios[col] = float(output_df[col].isna().mean())
        quality_summary['null_ratios'] = null_ratios

    manifest = {
        'action': 'build_factor_graph',
        'run_id': run_id,
        'feature_set_version': str(args.feature_set_version),
        'factor_fingerprint': factor_fingerprint,
        'generated_at_utc': _utc_now_iso(),
        'input_data_versions': {
            'base_input': build_file_snapshot(
                base_input,
                inspect_csv=True,
                csv_metadata=base_input_csv_meta,
            ),
            'hf_daily_input': build_file_snapshot(
                hf_input,
                inspect_csv=True,
                csv_metadata=hf_input_csv_meta,
            ),
            'hf_minute_input': build_file_snapshot(
                hf_minute_input,
                inspect_csv=True,
                csv_metadata=hf_minute_input_csv_meta,
            ),
            'macro_input': build_file_snapshot(
                macro_input,
                inspect_csv=True,
                csv_metadata=macro_input_csv_meta,
            ),
        },
        'node_status': node_status,
        'row_count': int(len(output_df)),
        'quality_summary': quality_summary,
        'output_paths': {
            'wide_csv': output_path,
            'wide_csv_snapshot': build_file_snapshot(
                output_path,
                inspect_csv=True,
                csv_metadata=output_csv_meta,
            ),
        },
        'code_version': _git_code_version(),
        'pipeline_config_validation': report.to_dict(),
        'warnings': warnings,
        'errors': errors,
        'execution_plan': {
            'total_nodes': int(len(factor_nodes)),
            'expression_nodes': int(len(expression_specs)),
            'ts_expression_nodes': int(len(plan.get('time_series_specs', []))),
            'cs_expression_nodes': int(len(plan.get('cross_sectional_specs', []))),
        },
        'intraday_source_map': intraday_source_map,
    }

    saved_manifest = save_data_manifest(
        os.path.dirname(manifest_path),
        manifest,
        filename=os.path.basename(manifest_path),
    )

    print(f'因子 DAG 构建完成: {output_path}')
    print(f'输出行数: {len(output_df)}, 股票数: {quality_summary["stock_count"]}')
    print(f'因子指纹: {factor_fingerprint}')
    print(f'构建清单: {saved_manifest}')
    if warnings:
        print('Warnings:')
        for msg in warnings:
            print(f'  - {msg}')


if __name__ == '__main__':
    main()
