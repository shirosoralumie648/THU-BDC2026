import copy
import json
import os
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

STRUCTURED_DATA_SUBDIRS = {
    'stock_data.csv': 'raw',
    'train.csv': 'splits',
    'test.csv': 'splits',
}


def _file_signature(path: str) -> Tuple[str, int, int]:
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


def _snapshot_error(code: str, message: str) -> Dict[str, str]:
    return {'code': code, 'message': message}


def infer_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_stock_code_series(series: pd.Series) -> pd.Series:
    s = series.fillna('').astype(str).str.strip().str.upper()
    extracted = s.str.extract(r'(\d{6})', expand=False)
    digits = s.str.replace(r'[^0-9]', '', regex=True)

    normalized = extracted.fillna(digits)
    normalized = normalized.fillna('').str.strip().str[-6:]

    has_digits = normalized.str.len() > 0
    normalized = normalized.where(~has_digits, normalized.str.zfill(6))
    return normalized


def resolve_data_root(config: Dict) -> str:
    return str(config.get('data_path', './data')).strip() or './data'


def resolve_hf_factor_path(config: Dict) -> str:
    raw_path = str(config.get('hf_daily_factor_path', '')).strip()
    if not raw_path:
        return ''
    data_root = os.path.abspath(resolve_data_root(config))
    return _normalize_candidate_path(raw_path, data_root)


def resolve_structured_data_root(config: Dict) -> str:
    data_root = os.path.abspath(resolve_data_root(config))
    rel_root = str(config.get('structured_data_root', 'datasets')).strip()
    if not rel_root:
        return data_root
    return os.path.join(data_root, rel_root)


def _normalize_candidate_path(path: str, data_root: str) -> str:
    path = str(path or '').strip()
    if not path:
        return ''
    if os.path.isabs(path):
        return path
    direct_path = os.path.abspath(path)
    if os.path.exists(direct_path):
        return direct_path
    return os.path.abspath(os.path.join(data_root, path))


def _build_structured_dataset_path(config: Dict, filename: str) -> str:
    subdir = STRUCTURED_DATA_SUBDIRS.get(filename, '')
    structured_root = resolve_structured_data_root(config)
    if subdir:
        return os.path.join(structured_root, subdir, filename)
    return os.path.join(structured_root, filename)


def resolve_dataset_candidates(config: Dict, filename: str) -> List[str]:
    data_root = os.path.abspath(resolve_data_root(config))
    candidates = []

    dataset_paths = config.get('dataset_paths', {})
    if isinstance(dataset_paths, dict):
        override_path = _normalize_candidate_path(dataset_paths.get(filename, ''), data_root)
        if override_path:
            candidates.append(override_path)

    candidates.append(os.path.join(data_root, filename))

    structured_path = _build_structured_dataset_path(config, filename)
    if structured_path:
        candidates.append(os.path.abspath(structured_path))

    dedup = []
    seen = set()
    for path in candidates:
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        dedup.append(norm)
    return dedup


def resolve_dataset_write_targets(config: Dict, filename: str) -> Dict[str, object]:
    candidates = resolve_dataset_candidates(config, filename)
    if not candidates:
        raise ValueError(f'无法为数据集构建路径候选: {filename}')

    data_root = os.path.abspath(resolve_data_root(config))
    legacy_path = os.path.abspath(os.path.join(data_root, filename))
    structured_path = os.path.abspath(_build_structured_dataset_path(config, filename))

    prefer_structured = bool(config.get('prefer_structured_data_layout', False))
    override_path = None
    dataset_paths = config.get('dataset_paths', {})
    if isinstance(dataset_paths, dict):
        override_path = _normalize_candidate_path(dataset_paths.get(filename, ''), data_root)
        if override_path:
            override_path = os.path.abspath(override_path)

    if override_path:
        primary_path = override_path
    elif prefer_structured:
        primary_path = structured_path
    else:
        primary_path = legacy_path

    mirror_enabled = bool(config.get('mirror_legacy_and_structured_data', True))
    mirrors = []
    if mirror_enabled:
        for candidate in [legacy_path, structured_path]:
            if os.path.abspath(candidate) == os.path.abspath(primary_path):
                continue
            mirrors.append(candidate)

    return {
        'primary': os.path.abspath(primary_path),
        'legacy': legacy_path,
        'structured': structured_path,
        'mirrors': mirrors,
        'candidates': candidates,
    }


def resolve_dataset_path(config: Dict, filename: str, *, for_write: bool = False) -> str:
    if for_write:
        targets = resolve_dataset_write_targets(config, filename)
        return str(targets['primary'])

    candidates = resolve_dataset_candidates(config, filename)
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def resolve_dataset_build_manifest_path(config: Dict, *, filename: str = 'train.csv') -> str:
    override_path = str(config.get('dataset_build_manifest_path', '')).strip()
    if override_path:
        return os.path.abspath(override_path)
    dataset_path = resolve_dataset_path(config, filename, for_write=False)
    return os.path.join(
        os.path.dirname(os.path.abspath(dataset_path)),
        'data_manifest_dataset_build.json',
    )


def load_train_dataset_from_build_manifest(
    config: Dict,
    feature_pipeline: Dict,
) -> Tuple[Optional[str], Dict]:
    strict_mode = _is_truthy(config.get('dataset_manifest_strict', False))
    info = {
        'enabled': _is_truthy(config.get('use_dataset_build_manifest', True)),
        'strict': strict_mode,
        'used': False,
        'manifest_path': '',
        'train_path': '',
        'build_id': '',
        'feature_set_version': '',
        'factor_fingerprint': '',
        'warnings': [],
        'errors': [],
    }
    if not info['enabled']:
        return None, info

    manifest_path = resolve_dataset_build_manifest_path(config, filename='train.csv')
    info['manifest_path'] = manifest_path
    if not os.path.exists(manifest_path):
        msg = f'未找到 dataset build manifest: {manifest_path}'
        if strict_mode:
            raise FileNotFoundError(msg)
        info['warnings'].append(msg)
        return None, info

    try:
        payload = _load_json_payload(manifest_path)
    except Exception as exc:
        msg = f'读取 dataset build manifest 失败: {manifest_path} | {exc}'
        if strict_mode:
            raise ValueError(msg) from exc
        info['errors'].append(msg)
        return None, info

    if not isinstance(payload, dict):
        msg = f'dataset build manifest 顶层必须为对象: {manifest_path}'
        if strict_mode:
            raise ValueError(msg)
        info['errors'].append(msg)
        return None, info

    action = str(payload.get('action', '')).strip()
    if action and action != 'build_dataset':
        info['warnings'].append(f'manifest action 不是 build_dataset: {action}')

    params = payload.get('params', {})
    if not isinstance(params, dict):
        params = {}

    outputs = payload.get('outputs', {})
    if not isinstance(outputs, dict):
        outputs = {}
    train_snapshot = outputs.get('train_csv', {})
    if not isinstance(train_snapshot, dict):
        train_snapshot = {}

    train_path = str(train_snapshot.get('path', '')).strip()
    if train_path:
        train_path = os.path.abspath(train_path)
    info['train_path'] = train_path
    info['build_id'] = str(payload.get('build_id', '') or params.get('build_id', '')).strip()
    info['feature_set_version'] = str(
        payload.get('feature_set_version', '') or params.get('feature_set_version', '')
    ).strip()
    info['factor_fingerprint'] = str(
        payload.get('factor_fingerprint', '') or params.get('factor_fingerprint', '')
    ).strip()

    if not train_path:
        msg = f'manifest 缺少 outputs.train_csv.path: {manifest_path}'
        if strict_mode:
            raise ValueError(msg)
        info['errors'].append(msg)
        return None, info
    if not os.path.exists(train_path):
        msg = f'manifest 指向的训练集文件不存在: {train_path}'
        if strict_mode:
            raise FileNotFoundError(msg)
        info['errors'].append(msg)
        return None, info

    expected_feature_set_version = str(config.get('expected_feature_set_version', '')).strip()
    expected_factor_fingerprint = str(config.get('expected_factor_fingerprint', '')).strip()
    require_manifest_fingerprint = _is_truthy(config.get('dataset_manifest_require_factor_fingerprint', False))

    if expected_feature_set_version:
        if not info['feature_set_version']:
            info['errors'].append(
                '配置要求 expected_feature_set_version，但 manifest 未包含 feature_set_version'
            )
        elif info['feature_set_version'] != expected_feature_set_version:
            info['errors'].append(
                f'feature_set_version 不一致: manifest={info["feature_set_version"]}, '
                f'expected={expected_feature_set_version}'
            )

    pipeline_factor_fingerprint = str(feature_pipeline.get('factor_fingerprint', '')).strip()
    if require_manifest_fingerprint and (not info['factor_fingerprint']):
        info['errors'].append('配置要求 manifest 包含 factor_fingerprint，但当前缺失')
    if expected_factor_fingerprint:
        if not info['factor_fingerprint']:
            info['errors'].append('配置要求 expected_factor_fingerprint，但 manifest 未包含 factor_fingerprint')
        elif info['factor_fingerprint'] != expected_factor_fingerprint:
            info['errors'].append(
                f'manifest factor_fingerprint 与 expected_factor_fingerprint 不一致: '
                f'{info["factor_fingerprint"]} != {expected_factor_fingerprint}'
            )
        if pipeline_factor_fingerprint and (pipeline_factor_fingerprint != expected_factor_fingerprint):
            info['errors'].append(
                f'当前激活因子指纹与 expected_factor_fingerprint 不一致: '
                f'{pipeline_factor_fingerprint} != {expected_factor_fingerprint}'
            )

    if info['factor_fingerprint'] and pipeline_factor_fingerprint:
        if info['factor_fingerprint'] != pipeline_factor_fingerprint:
            info['errors'].append(
                f'manifest 因子指纹与当前激活因子指纹不一致: '
                f'{info["factor_fingerprint"]} != {pipeline_factor_fingerprint}'
            )

    if info['errors'] and strict_mode:
        raise ValueError('dataset build manifest 校验失败:\n- ' + '\n- '.join(info['errors']))

    info['used'] = True
    return train_path, info


def log_dataset_manifest_info(info: Dict, *, label: str) -> None:
    if not info.get('enabled', False):
        return

    print(
        f"{label} dataset build manifest: {info.get('manifest_path', '')} "
        f"(strict={info.get('strict', False)})"
    )
    for msg in info.get('warnings', []):
        print(f"[{label}-manifest-warning] {msg}")
    for msg in info.get('errors', []):
        print(f"[{label}-manifest-error] {msg}")
    if info.get('used', False):
        print(
            f"{label} manifest 元信息: "
            f"build_id={info.get('build_id', '')}, "
            f"feature_set_version={info.get('feature_set_version', '')}, "
            f"factor_fingerprint={info.get('factor_fingerprint', '')}"
        )


def _resolve_join_column(df: pd.DataFrame, preferred: str, candidates: Iterable[str]) -> Optional[str]:
    preferred = str(preferred or '').strip()
    if preferred and preferred in df.columns:
        return preferred
    return infer_existing_column(df, candidates)


def merge_hf_daily_factors(base_df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    if not bool(config.get('use_hf_daily_factor_merge', False)):
        return base_df, {'enabled': False, 'reason': 'disabled'}

    hf_path = resolve_hf_factor_path(config)
    if not hf_path:
        if bool(config.get('hf_factor_required', False)):
            raise FileNotFoundError('hf_factor_required=True 但未配置 hf_daily_factor_path')
        return base_df, {'enabled': True, 'path': '', 'used': False, 'reason': 'path_empty'}
    if not os.path.exists(hf_path):
        if bool(config.get('hf_factor_required', False)):
            raise FileNotFoundError(f'未找到高频日因子文件: {hf_path}')
        return base_df, {'enabled': True, 'path': hf_path, 'used': False, 'reason': 'path_not_exists'}

    hf_df = pd.read_csv(hf_path)
    if hf_df.empty:
        return base_df, {'enabled': True, 'path': hf_path, 'used': False, 'reason': 'hf_empty'}

    base_stock_col = _resolve_join_column(
        base_df,
        preferred='股票代码',
        candidates=['股票代码', 'stock_id', 'code', 'ts_code'],
    )
    base_date_col = _resolve_join_column(
        base_df,
        preferred='日期',
        candidates=['日期', 'date', 'datetime', 'trade_date'],
    )
    hf_stock_col = _resolve_join_column(
        hf_df,
        preferred=str(config.get('hf_factor_stock_col', '股票代码')),
        candidates=['股票代码', 'stock_id', 'code', 'ts_code'],
    )
    hf_date_col = _resolve_join_column(
        hf_df,
        preferred=str(config.get('hf_factor_date_col', '日期')),
        candidates=['日期', 'date', 'datetime', 'trade_date'],
    )

    if base_stock_col is None or base_date_col is None:
        raise ValueError('基础数据缺少股票或日期列，无法合并高频因子')
    if hf_stock_col is None or hf_date_col is None:
        if bool(config.get('hf_factor_required', False)):
            raise ValueError('高频因子文件缺少股票或日期列')
        return base_df, {
            'enabled': True,
            'path': hf_path,
            'used': False,
            'reason': 'hf_missing_join_cols',
        }

    base = base_df.copy()
    hf = hf_df.copy()
    base['__join_stock'] = normalize_stock_code_series(base[base_stock_col])
    hf['__join_stock'] = normalize_stock_code_series(hf[hf_stock_col])
    base['__join_date'] = pd.to_datetime(base[base_date_col], errors='coerce').dt.normalize()
    hf['__join_date'] = pd.to_datetime(hf[hf_date_col], errors='coerce').dt.normalize()

    hf = hf.dropna(subset=['__join_stock', '__join_date'])
    if hf.empty:
        return base_df, {'enabled': True, 'path': hf_path, 'used': False, 'reason': 'hf_no_valid_rows'}

    factor_cols = [
        col for col in hf.columns
        if col not in {hf_stock_col, hf_date_col, '__join_stock', '__join_date'}
    ]
    selected_factor_cols = config.get('hf_factor_columns', [])
    if isinstance(selected_factor_cols, (list, tuple)) and selected_factor_cols:
        selected_set = {str(col) for col in selected_factor_cols}
        factor_cols = [col for col in factor_cols if col in selected_set]
    if not factor_cols:
        return base_df, {'enabled': True, 'path': hf_path, 'used': False, 'reason': 'no_factor_columns'}

    prefix = str(config.get('hf_factor_prefix', '')).strip()
    rename_map = {}
    for col in factor_cols:
        target_col = f'{prefix}{col}' if prefix else col
        rename_map[col] = target_col

    collision_cols = [new_col for new_col in rename_map.values() if new_col in base.columns]
    if collision_cols:
        allow_overwrite = bool(config.get('hf_factor_allow_overwrite_columns', False))
        if not allow_overwrite:
            for col, renamed in list(rename_map.items()):
                if renamed in base.columns:
                    rename_map[col] = f'hf_{renamed}'
        else:
            base = base.drop(columns=collision_cols)

    hf_subset = hf[['__join_stock', '__join_date', *factor_cols]].copy()
    hf_subset = hf_subset.rename(columns=rename_map)

    dedup_keep = str(config.get('hf_factor_drop_duplicate_keep', 'last')).lower()
    if dedup_keep not in {'first', 'last'}:
        dedup_keep = 'last'
    before_drop = len(hf_subset)
    hf_subset = hf_subset.drop_duplicates(
        subset=['__join_stock', '__join_date'],
        keep=dedup_keep,
    )
    dropped_duplicates = int(before_drop - len(hf_subset))

    merge_how = str(config.get('hf_factor_merge_how', 'left')).lower()
    if merge_how not in {'left', 'inner'}:
        merge_how = 'left'

    merged = base.merge(
        hf_subset,
        on=['__join_stock', '__join_date'],
        how=merge_how,
        validate='many_to_one',
    )
    merged = merged.drop(columns=['__join_stock', '__join_date'])

    merged_factor_cols = list(rename_map.values())
    matched_rows = int(merged[merged_factor_cols].notna().any(axis=1).sum())
    coverage = matched_rows / float(max(1, len(merged)))

    meta = {
        'enabled': True,
        'path': hf_path,
        'used': True,
        'merge_how': merge_how,
        'factor_columns': merged_factor_cols,
        'factor_count': int(len(merged_factor_cols)),
        'hf_rows': int(len(hf_df)),
        'hf_rows_after_clean': int(len(hf_subset)),
        'dropped_duplicates': dropped_duplicates,
        'matched_rows': matched_rows,
        'coverage': float(coverage),
    }
    print(
        f'高频因子合并完成: path={hf_path}, factors={meta["factor_count"]}, '
        f'coverage={coverage:.2%}, merge_how={merge_how}'
    )
    return merged, meta


def load_market_dataset(
    config: Dict,
    filename: str = 'train.csv',
    *,
    dtype: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, str]:
    dataset_path = resolve_dataset_path(config, filename, for_write=False)
    return load_market_dataset_from_path(config, dataset_path, dtype=dtype)


def load_market_dataset_from_path(
    config: Dict,
    dataset_path: str,
    *,
    dtype: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, str]:
    dataset_path = os.path.abspath(str(dataset_path))
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f'未找到数据文件: {dataset_path}')
    df = pd.read_csv(dataset_path, dtype=dtype)
    merged_df, hf_meta = merge_hf_daily_factors(df, config)
    if hf_meta.get('enabled') and (not hf_meta.get('used', False)):
        reason = hf_meta.get('reason', 'unknown')
        path = hf_meta.get('path', '')
        print(f'高频因子未启用到本次数据: reason={reason}, path={path}')
    return merged_df, dataset_path


def resolve_industry_mapping_path(config: Dict) -> str:
    candidates = [
        str(config.get('prior_graph_industry_map_path', '')).strip(),
        str(config.get('label_industry_map_path', '')).strip(),
        str(config.get('stock_static_feature_path', '')).strip(),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ''


def load_stock_to_industry_map(
    config: Dict,
    *,
    stock_col_key: str = 'prior_graph_stock_col',
    industry_col_key: str = 'prior_graph_industry_col',
    mapping_path: Optional[str] = None,
) -> Dict[str, str]:
    source_path = str(mapping_path or '').strip()
    if not source_path:
        source_path = resolve_industry_mapping_path(config)
    if not source_path or not os.path.exists(source_path):
        return {}

    mapping_df = pd.read_csv(source_path, dtype=str)
    stock_col = str(config.get(stock_col_key, '股票代码')).strip()
    industry_col = str(config.get(industry_col_key, '行业')).strip()

    if stock_col not in mapping_df.columns:
        stock_col = infer_existing_column(mapping_df, ['股票代码', 'stock_id', 'code', 'ts_code'])
        if stock_col is None:
            return {}
    if industry_col not in mapping_df.columns:
        industry_col = infer_existing_column(mapping_df, ['行业', 'industry', 'sw_l1', 'sector'])
        if industry_col is None:
            return {}

    mapping_df = mapping_df[[stock_col, industry_col]].copy()
    mapping_df[stock_col] = normalize_stock_code_series(mapping_df[stock_col])
    mapping_df[industry_col] = mapping_df[industry_col].astype(str).str.strip()
    mapping_df = mapping_df.dropna(subset=[stock_col, industry_col])
    mapping_df = mapping_df[mapping_df[industry_col] != '']
    if mapping_df.empty:
        return {}

    return (
        mapping_df
        .drop_duplicates(subset=[stock_col], keep='last')
        .set_index(stock_col)[industry_col]
        .to_dict()
    )


def build_stock_industry_index(
    stock_ids: List[str],
    stock_to_industry: Dict[str, str],
) -> Tuple[np.ndarray, List[str], int]:
    num_stocks = len(stock_ids)
    stock_industry_idx = np.full(num_stocks, -1, dtype=np.int64)
    if num_stocks <= 0 or not stock_to_industry:
        return stock_industry_idx, [], 0

    normalized_codes = normalize_stock_code_series(pd.Series(stock_ids)).tolist()
    industries = []
    for normalized in normalized_codes:
        industry = stock_to_industry.get(normalized, None)
        if industry:
            industries.append(industry)
    if not industries:
        return stock_industry_idx, [], 0

    industry_vocab = sorted(set(industries))
    industry2idx = {industry: idx for idx, industry in enumerate(industry_vocab)}

    matched = 0
    for idx, normalized in enumerate(normalized_codes):
        industry = stock_to_industry.get(normalized, None)
        if not industry:
            continue
        stock_industry_idx[idx] = int(industry2idx[industry])
        matched += 1
    return stock_industry_idx, industry_vocab, matched


def collect_data_sources(config: Dict, *, include_csv_stats: bool = False) -> Dict:
    data_root = os.path.abspath(resolve_data_root(config))
    structured_root = os.path.abspath(resolve_structured_data_root(config))
    train_csv = resolve_dataset_path(config, 'train.csv', for_write=False)
    test_csv = resolve_dataset_path(config, 'test.csv', for_write=False)
    stock_data_csv = resolve_dataset_path(config, 'stock_data.csv', for_write=False)
    hf_daily_factor_path = resolve_hf_factor_path(config)
    industry_map = resolve_industry_mapping_path(config)
    benchmark_path = str(config.get('label_benchmark_return_path', '')).strip()
    static_feature_path = str(config.get('stock_static_feature_path', '')).strip()

    return {
        'data_root': data_root,
        'structured_data_root': structured_root,
        'dataset_candidates': {
            'train_csv': resolve_dataset_candidates(config, 'train.csv'),
            'test_csv': resolve_dataset_candidates(config, 'test.csv'),
            'stock_data_csv': resolve_dataset_candidates(config, 'stock_data.csv'),
        },
        'train_csv': build_file_snapshot(train_csv, inspect_csv=include_csv_stats),
        'test_csv': build_file_snapshot(test_csv, inspect_csv=include_csv_stats),
        'stock_data_csv': build_file_snapshot(stock_data_csv, inspect_csv=include_csv_stats),
        'hf_daily_factor': build_file_snapshot(hf_daily_factor_path, inspect_csv=include_csv_stats),
        'industry_mapping': build_file_snapshot(industry_map),
        'benchmark': build_file_snapshot(benchmark_path),
        'static_feature': build_file_snapshot(static_feature_path),
    }


def inspect_csv_metadata(
    path: str,
    *,
    date_col_candidates: Optional[Iterable[str]] = None,
    stock_col_candidates: Optional[Iterable[str]] = None,
) -> Dict:
    if not path or not os.path.exists(path):
        return {}

    date_candidates = list(date_col_candidates or ['日期', 'date', 'datetime', 'trade_date'])
    stock_candidates = list(stock_col_candidates or ['股票代码', 'stock_id', 'code', 'ts_code'])
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return {'status': 'error', 'error_code': 'csv_parse_error', 'message': str(exc)}
    except Exception as exc:
        return {'status': 'error', 'error_code': 'csv_read_error', 'message': str(exc)}

    info = {
        'status': 'ok',
        'row_count': int(len(df)),
        'column_count': int(len(df.columns)),
    }

    date_col = infer_existing_column(df, date_candidates)
    if date_col is not None:
        date_values = pd.to_datetime(df[date_col], errors='coerce')
        date_values = date_values.dropna()
        if not date_values.empty:
            info['date_column'] = date_col
            info['date_min'] = str(date_values.min().date())
            info['date_max'] = str(date_values.max().date())

    stock_col = infer_existing_column(df, stock_candidates)
    if stock_col is not None:
        stock_norm = normalize_stock_code_series(df[stock_col]).replace('', np.nan).dropna()
        if not stock_norm.empty:
            info['stock_column'] = stock_col
            info['stock_count'] = int(stock_norm.nunique())

    return info


def build_file_snapshot(
    path: str,
    *,
    inspect_csv: bool = False,
    date_col_candidates: Optional[Iterable[str]] = None,
    stock_col_candidates: Optional[Iterable[str]] = None,
) -> Dict:
    target_path = str(path or '').strip()
    if not target_path:
        return {'path': '', 'exists': False}

    abs_path = os.path.abspath(target_path)
    exists = os.path.exists(abs_path)
    snapshot = {'path': abs_path, 'exists': bool(exists)}
    if not exists:
        return snapshot

    errors = []
    try:
        snapshot['size_bytes'] = int(os.path.getsize(abs_path))
    except Exception as exc:
        errors.append(_snapshot_error('stat_failed', str(exc)))

    if inspect_csv and abs_path.lower().endswith('.csv'):
        csv_meta = inspect_csv_metadata(
            abs_path,
            date_col_candidates=date_col_candidates,
            stock_col_candidates=stock_col_candidates,
        )
        if csv_meta:
            snapshot['csv'] = csv_meta
            if csv_meta.get('status') == 'error':
                errors.append(
                    _snapshot_error(
                        str(csv_meta.get('error_code', 'csv_parse_error')),
                        str(csv_meta.get('message', 'unknown csv parse error')),
                    )
                )

    if errors:
        snapshot['errors'] = errors
    return snapshot


def save_data_manifest(output_dir: str, manifest: Dict, filename: str = 'data_manifest.json') -> str:
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, filename)
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return target_path
