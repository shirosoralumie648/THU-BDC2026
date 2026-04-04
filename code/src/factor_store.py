import ast
import copy
import hashlib
import json
import os
import re
import types
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd

from utils import engineer_features_39, engineer_features_158plus39


FACTOR_STORE_VERSION = 1
DEFAULT_BUILTIN_FACTOR_REGISTRY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'builtin_factors.json')
)

FEATURE_ENGINEER_FUNC_MAP = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39,
}

IDENTIFIER_PATTERN = re.compile(r'[A-Za-z_\u4e00-\u9fff][0-9A-Za-z_\u4e00-\u9fff]*')
SAFE_NUMPY_CALLS = {
    'abs': np.abs,
    'clip': np.clip,
    'exp': np.exp,
    'log': np.log,
    'log1p': np.log1p,
    'maximum': np.maximum,
    'minimum': np.minimum,
    'nan_to_num': np.nan_to_num,
    'power': np.power,
    'sign': np.sign,
    'sqrt': np.sqrt,
    'where': np.where,
}
SAFE_NUMPY_CONSTANTS = {
    'pi': float(np.pi),
    'e': float(np.e),
}
SAFE_NUMPY_NAMESPACE = types.SimpleNamespace(**SAFE_NUMPY_CALLS, **SAFE_NUMPY_CONSTANTS)
SAFE_NUMPY_ATTRS = set(SAFE_NUMPY_CALLS) | set(SAFE_NUMPY_CONSTANTS)
EXPRESSION_RESERVED_NAMES = {'True', 'False', 'None'}
CROSS_SECTIONAL_HELPER_NAMES = {'cs_rank', 'cs_zscore'}
SAFE_OBJECT_METHOD_CALLS = {'astype'}
SAFE_LITERAL_TYPES = {
    'float': float,
    'int': int,
    'bool': bool,
}


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def _normalize_factor_inputs(raw_inputs):
    if raw_inputs is None:
        return {}
    if not isinstance(raw_inputs, dict):
        raise ValueError(f'inputs 必须为对象(dict)，当前类型: {type(raw_inputs).__name__}')
    normalized = {}
    for key, value in raw_inputs.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f'inputs 的 key 必须是非空字符串，当前: {key!r}')
        normalized[key] = value
    return normalized


def _default_feature_set_config():
    return {
        'disabled_builtin_factors': [],
        'builtin_overrides': [],
        'custom_factors': [],
    }


def _default_store():
    return {
        'version': FACTOR_STORE_VERSION,
        'feature_sets': {
            '39': _default_feature_set_config(),
            '158+39': _default_feature_set_config(),
        },
    }


def ensure_factor_store(store_path):
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    if not os.path.exists(store_path):
        save_factor_store(_default_store(), store_path)


@lru_cache(maxsize=None)
def load_builtin_factor_registry(registry_path):
    registry_path = os.path.abspath(registry_path)
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    return registry


def get_builtin_specs_map(registry_path=DEFAULT_BUILTIN_FACTOR_REGISTRY_PATH):
    registry = load_builtin_factor_registry(registry_path)
    feature_sets = registry.get('feature_sets', {})
    for feature_set in FEATURE_ENGINEER_FUNC_MAP:
        if feature_set not in feature_sets:
            raise ValueError(f'内置因子注册表缺少 feature_set: {feature_set}')
    return feature_sets


def get_builtin_specs(feature_set, registry_path=DEFAULT_BUILTIN_FACTOR_REGISTRY_PATH):
    specs_map = get_builtin_specs_map(registry_path)
    return [dict(spec) for spec in specs_map[feature_set]]


def load_factor_store(store_path):
    ensure_factor_store(store_path)
    with open(store_path, 'r', encoding='utf-8') as f:
        store = json.load(f)

    if store.get('version') != FACTOR_STORE_VERSION:
        raise ValueError(
            f'factor_store 版本不匹配: 当前={store.get("version")}, 期望={FACTOR_STORE_VERSION}'
        )

    feature_sets = store.setdefault('feature_sets', {})
    for feature_set in FEATURE_ENGINEER_FUNC_MAP:
        feature_set_config = feature_sets.setdefault(
            feature_set, copy.deepcopy(_default_feature_set_config())
        )
        defaults = _default_feature_set_config()
        for key, value in defaults.items():
            feature_set_config.setdefault(key, copy.deepcopy(value))
    return store


def save_factor_store(store, store_path):
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    with open(store_path, 'w', encoding='utf-8') as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


def _get_feature_set_config(store, feature_set):
    if feature_set not in FEATURE_ENGINEER_FUNC_MAP:
        raise ValueError(f'不支持的 feature_set: {feature_set}')
    return store.setdefault('feature_sets', {}).setdefault(
        feature_set, copy.deepcopy(_default_feature_set_config())
    )


def _build_builtin_specs(feature_set, disabled_builtin_factors):
    disabled_set = set(disabled_builtin_factors)
    builtin_specs = []
    for spec in get_builtin_specs(feature_set):
        current = dict(spec)
        current['enabled'] = current['name'] not in disabled_set
        builtin_specs.append(current)
    return builtin_specs


def _normalize_custom_factor_spec(spec):
    normalized = {
        'name': spec['name'],
        'group': spec.get('group', 'custom'),
        'source': 'custom',
        'enabled': bool(spec.get('enabled', True)),
        'expression': spec['expression'],
        'description': spec.get('description', ''),
        'inputs': _normalize_factor_inputs(spec.get('inputs', {})),
    }
    if spec.get('created_at'):
        normalized['created_at'] = spec['created_at']
    if spec.get('updated_at'):
        normalized['updated_at'] = spec['updated_at']
    if spec.get('author'):
        normalized['author'] = spec['author']
    return normalized


def _validate_custom_factor_specs(custom_specs, builtin_names):
    seen = set(builtin_names)
    for spec in custom_specs:
        if not spec.get('name'):
            raise ValueError('自定义因子缺少 name')
        if not spec.get('expression'):
            raise ValueError(f'自定义因子 {spec["name"]} 缺少 expression')
        _normalize_factor_inputs(spec.get('inputs', {}))
        if spec['name'] in seen:
            raise ValueError(f'因子名称重复: {spec["name"]}')
        seen.add(spec['name'])


def _apply_builtin_overrides(builtin_specs, builtin_overrides):
    overrides_map = {
        spec['name']: dict(spec)
        for spec in builtin_overrides
    }
    resolved_specs = []
    for base_spec in builtin_specs:
        current = dict(base_spec)
        current['source'] = 'builtin'
        current['default_expression'] = base_spec.get('expression', '')
        current['default_description'] = base_spec.get('description', '')
        current['default_group'] = base_spec.get('group', '')
        current['overridden'] = False

        override = overrides_map.get(current['name'])
        if override is not None:
            for field in ('expression', 'description', 'group', 'inputs'):
                if field in override:
                    current[field] = override[field]
            for meta_field in ('created_at', 'updated_at', 'author'):
                if meta_field in override:
                    current[meta_field] = override[meta_field]
            current['overridden'] = True
        resolved_specs.append(current)
    return resolved_specs


def _compute_factor_fingerprint(feature_set, specs):
    serializable_specs = []
    for spec in specs:
        serializable_specs.append({
            'name': spec.get('name'),
            'group': spec.get('group', ''),
            'source': spec.get('source', ''),
            'expression': spec.get('expression', ''),
            'inputs': _normalize_factor_inputs(spec.get('inputs', {})),
            'enabled': bool(spec.get('enabled', True)),
        })
    payload = json.dumps(
        {
            'feature_set': feature_set,
            'specs': serializable_specs,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:16]


def _factor_pipeline_cache_key(store_path, builtin_registry_path):
    abs_store_path = os.path.abspath(store_path)
    ensure_factor_store(abs_store_path)
    store_stat = os.stat(abs_store_path)

    abs_registry_path = os.path.abspath(builtin_registry_path)
    registry_stat = os.stat(abs_registry_path)

    return (
        abs_store_path,
        int(store_stat.st_mtime_ns),
        int(store_stat.st_size),
        abs_registry_path,
        int(registry_stat.st_mtime_ns),
        int(registry_stat.st_size),
    )


@lru_cache(maxsize=64)
def _resolve_factor_pipeline_cached(
    feature_set,
    store_path,
    store_mtime_ns,
    store_size,
    builtin_registry_path,
    registry_mtime_ns,
    registry_size,
):
    if feature_set not in FEATURE_ENGINEER_FUNC_MAP:
        raise ValueError(f'不支持的 feature_set: {feature_set}')

    store = load_factor_store(store_path)
    feature_set_config = _get_feature_set_config(store, feature_set)
    builtin_specs = _apply_builtin_overrides(
        get_builtin_specs(feature_set, builtin_registry_path),
        feature_set_config.get('builtin_overrides', []),
    )
    disabled_set = set(feature_set_config.get('disabled_builtin_factors', []))
    for spec in builtin_specs:
        spec['enabled'] = spec['name'] not in disabled_set
        spec['inputs'] = _normalize_factor_inputs(spec.get('inputs', {}))
    builtin_names = [spec['name'] for spec in builtin_specs]

    custom_specs = [
        _normalize_custom_factor_spec(spec)
        for spec in feature_set_config.get('custom_factors', [])
    ]
    _validate_custom_factor_specs(custom_specs, builtin_names)

    all_specs = builtin_specs + custom_specs
    active_specs = [spec for spec in all_specs if spec.get('enabled', True)]
    execution_plan = build_factor_execution_plan(active_specs, error_prefix='因子')
    active_feature_names = [spec['name'] for spec in active_specs]
    group_counts = {}
    for spec in active_specs:
        group = spec.get('group', 'unknown')
        group_counts[group] = group_counts.get(group, 0) + 1

    return {
        'feature_set': feature_set,
        'store_path': store_path,
        'builtin_registry_path': os.path.abspath(builtin_registry_path),
        'engineer': FEATURE_ENGINEER_FUNC_MAP[feature_set],
        'builtin_specs': builtin_specs,
        'builtin_override_specs': [spec for spec in builtin_specs if spec.get('overridden')],
        'custom_specs': custom_specs,
        'all_specs': all_specs,
        'active_specs': active_specs,
        'ordered_specs': execution_plan['ordered_specs'],
        'time_series_specs': execution_plan['time_series_specs'],
        'cross_sectional_specs': execution_plan['cross_sectional_specs'],
        'dependency_graph': execution_plan['dependency_graph'],
        'active_features': active_feature_names,
        'factor_fingerprint': _compute_factor_fingerprint(feature_set, execution_plan['ordered_specs']),
        'snapshot_meta': {},
        'summary': {
            'builtin_total': len(builtin_specs),
            'builtin_enabled': sum(1 for spec in builtin_specs if spec['enabled']),
            'builtin_overridden': sum(1 for spec in builtin_specs if spec.get('overridden')),
            'custom_total': len(custom_specs),
            'custom_enabled': sum(1 for spec in custom_specs if spec['enabled']),
            'active_total': len(active_feature_names),
            'cross_sectional_total': len(execution_plan['cross_sectional_specs']),
            'group_counts': group_counts,
        },
    }


def resolve_factor_pipeline(feature_set, store_path, builtin_registry_path=DEFAULT_BUILTIN_FACTOR_REGISTRY_PATH):
    (
        abs_store_path,
        store_mtime_ns,
        store_size,
        abs_registry_path,
        registry_mtime_ns,
        registry_size,
    ) = _factor_pipeline_cache_key(store_path, builtin_registry_path)
    pipeline = _resolve_factor_pipeline_cached(
        feature_set,
        abs_store_path,
        store_mtime_ns,
        store_size,
        abs_registry_path,
        registry_mtime_ns,
        registry_size,
    )
    return copy.deepcopy(pipeline)


def _as_series(value, index):
    if isinstance(value, pd.Series):
        return value
    if np.isscalar(value):
        return pd.Series([value] * len(index), index=index)
    if isinstance(value, np.ndarray):
        if value.shape[0] != len(index):
            raise ValueError(f'结果长度不匹配: {value.shape[0]} != {len(index)}')
        return pd.Series(value, index=index)
    return pd.Series(value, index=index)


def _helper_abs(x):
    return x.abs() if isinstance(x, pd.Series) else np.abs(x)


def _helper_log(x):
    return np.log(x)


def _helper_sqrt(x):
    return np.sqrt(x)


def _helper_clip(x, lower=None, upper=None):
    if isinstance(x, pd.Series):
        return x.clip(lower=lower, upper=upper)
    return np.clip(x, lower, upper)


def _helper_shift(x, periods=1):
    return x.shift(int(periods))


def _helper_diff(x, periods=1):
    return x.diff(int(periods))


def _helper_pct_change(x, periods=1):
    return x.pct_change(int(periods), fill_method=None)


def _helper_sma(x, window):
    return x.rolling(int(window)).mean()


def _helper_ema(x, window):
    return x.ewm(span=int(window), adjust=False).mean()


def _helper_rolling_std(x, window):
    return x.rolling(int(window)).std()


def _helper_rolling_min(x, window):
    return x.rolling(int(window)).min()


def _helper_rolling_max(x, window):
    return x.rolling(int(window)).max()


def _helper_rolling_sum(x, window):
    return x.rolling(int(window)).sum()


def _helper_rolling_skew(x, window):
    return x.rolling(int(window)).skew()


def _helper_rank_pct(x, window):
    return x.rolling(int(window)).rank(pct=True)


def _helper_zscore(x, window):
    window = int(window)
    mean = x.rolling(window).mean()
    std = x.rolling(window).std()
    return (x - mean) / (std + 1e-12)


def _helper_where(cond, x, y):
    cond_series = _as_series(cond, cond.index if isinstance(cond, pd.Series) else x.index)
    result = np.where(cond_series, x, y)
    return _as_series(result, cond_series.index)


def _helper_row_max(x, y):
    return pd.concat([_as_series(x, x.index), _as_series(y, y.index)], axis=1).max(axis=1)


def _helper_row_min(x, y):
    return pd.concat([_as_series(x, x.index), _as_series(y, y.index)], axis=1).min(axis=1)


def _helper_vwap(amount, volume):
    return amount / (volume + 1e-12)


def _helper_rolling_quantile(x, window, quantile):
    return x.rolling(int(window)).quantile(float(quantile))


def _helper_argmax_ratio(x, window):
    window = int(window)
    return x.rolling(window).apply(np.argmax, raw=True) / window


def _helper_argmin_ratio(x, window):
    window = int(window)
    return x.rolling(window).apply(np.argmin, raw=True) / window


def _helper_argmax_minus_argmin_ratio(x, y, window):
    window = int(window)
    argmax = x.rolling(window).apply(np.argmax, raw=True)
    argmin = y.rolling(window).apply(np.argmin, raw=True)
    return (argmax - argmin) / window


def _helper_rolling_corr(x, y, window):
    return x.rolling(int(window)).corr(y)


def _helper_linearreg_slope(x, window):
    window = int(window)
    time_index = np.arange(window, dtype=np.float64)
    return x.rolling(window).apply(
        lambda arr: np.polyfit(time_index, arr, 1)[0],
        raw=True,
    )


def _helper_linearreg_rsquare(x, window):
    window = int(window)
    time_series = pd.Series(np.arange(len(x), dtype=np.float64), index=x.index)
    corr = x.rolling(window).corr(time_series)
    return corr ** 2


def _helper_linearreg_residual(x, window):
    window = int(window)
    time_index = np.arange(window, dtype=np.float64)

    def _residual(arr):
        slope, intercept = np.polyfit(time_index, arr, 1)
        predicted = slope * (window - 1) + intercept
        return arr[-1] - predicted

    return x.rolling(window).apply(_residual, raw=True)


def _helper_direction_up_rate(x, window):
    return (x > x.shift(1)).astype(np.float32).rolling(int(window)).mean()


def _helper_direction_down_rate(x, window):
    return (x < x.shift(1)).astype(np.float32).rolling(int(window)).mean()


def _helper_direction_balance_rate(x, window):
    return _helper_direction_up_rate(x, window) - _helper_direction_down_rate(x, window)


def _helper_diff_up_ratio(x, window):
    diff = x - x.shift(1)
    diff_abs = diff.abs().rolling(int(window)).sum()
    diff_up = diff.clip(lower=0).rolling(int(window)).sum()
    return diff_up / (diff_abs + 1e-12)


def _helper_diff_down_ratio(x, window):
    diff = x - x.shift(1)
    diff_abs = diff.abs().rolling(int(window)).sum()
    diff_down = (-diff.clip(upper=0)).rolling(int(window)).sum()
    return diff_down / (diff_abs + 1e-12)


def _helper_diff_balance_ratio(x, window):
    diff = x - x.shift(1)
    diff_abs = diff.abs().rolling(int(window)).sum()
    diff_up = diff.clip(lower=0).rolling(int(window)).sum()
    diff_down = (-diff.clip(upper=0)).rolling(int(window)).sum()
    return (diff_up - diff_down) / (diff_abs + 1e-12)


def _helper_vol_weighted_volatility(close, volume, window):
    vol_weighted_ret = (close / (close.shift(1) + 1e-12) - 1).abs() * volume
    mean = vol_weighted_ret.rolling(int(window)).mean()
    std = vol_weighted_ret.rolling(int(window)).std()
    return std / (mean + 1e-12)


def _import_talib():
    try:
        import talib
    except ImportError as exc:
        raise ImportError('请先安装 TA-Lib 后再使用该因子公式') from exc
    return talib


def _helper_rsi(close, window):
    talib = _import_talib()
    return _as_series(talib.RSI(close.astype(float), timeperiod=int(window)), close.index)


def _helper_macd_line(close, fast, slow, signal):
    talib = _import_talib()
    macd_line, _, _ = talib.MACD(
        close.astype(float),
        fastperiod=int(fast),
        slowperiod=int(slow),
        signalperiod=int(signal),
    )
    return _as_series(macd_line, close.index)


def _helper_macd_signal(close, fast, slow, signal):
    talib = _import_talib()
    _, macd_signal, _ = talib.MACD(
        close.astype(float),
        fastperiod=int(fast),
        slowperiod=int(slow),
        signalperiod=int(signal),
    )
    return _as_series(macd_signal, close.index)


def _helper_obv(close, volume):
    talib = _import_talib()
    return _as_series(talib.OBV(close.astype(float), volume.astype(float)), close.index)


def _helper_atr(high, low, close, window):
    talib = _import_talib()
    values = talib.ATR(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        timeperiod=int(window),
    )
    return _as_series(values, close.index)


def _helper_boll_mid(close, window, nbdev=2):
    talib = _import_talib()
    _, mid, _ = talib.BBANDS(
        close.astype(float),
        timeperiod=int(window),
        nbdevup=float(nbdev),
        nbdevdn=float(nbdev),
        matype=0,
    )
    return _as_series(mid, close.index)


def _helper_boll_std(close, window, nbdev=2):
    talib = _import_talib()
    upper, mid, _ = talib.BBANDS(
        close.astype(float),
        timeperiod=int(window),
        nbdevup=float(nbdev),
        nbdevdn=float(nbdev),
        matype=0,
    )
    return _as_series((upper - mid) / 2, close.index)


def _helper_kdj_k(high, low, close, fastk=9, slowk=3, slowd=3):
    talib = _import_talib()
    k_value, _ = talib.STOCH(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        fastk_period=int(fastk),
        slowk_period=int(slowk),
        slowd_period=int(slowd),
    )
    return _as_series(k_value, close.index)


def _helper_kdj_d(high, low, close, fastk=9, slowk=3, slowd=3):
    talib = _import_talib()
    _, d_value = talib.STOCH(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        fastk_period=int(fastk),
        slowk_period=int(slowk),
        slowd_period=int(slowd),
    )
    return _as_series(d_value, close.index)


def _helper_tema(close, window):
    talib = _import_talib()
    return _as_series(talib.TEMA(close.astype(float), timeperiod=int(window)), close.index)


def _helper_cci(high, low, close, window):
    talib = _import_talib()
    values = talib.CCI(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        timeperiod=int(window),
    )
    return _as_series(values, close.index)


def _helper_mfi(high, low, close, volume, window):
    talib = _import_talib()
    values = talib.MFI(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        volume.astype(float),
        timeperiod=int(window),
    )
    return _as_series(values, close.index)


def _helper_ad_line(high, low, close, volume):
    talib = _import_talib()
    values = talib.AD(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        volume.astype(float),
    )
    return _as_series(values, close.index)


def _helper_chaikin_osc(high, low, close, volume, fast=3, slow=10):
    talib = _import_talib()
    values = talib.ADOSC(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        volume.astype(float),
        fastperiod=int(fast),
        slowperiod=int(slow),
    )
    return _as_series(values, close.index)


def _helper_adx(high, low, close, window):
    talib = _import_talib()
    values = talib.ADX(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        timeperiod=int(window),
    )
    return _as_series(values, close.index)


def _helper_aroon_osc(high, low, window):
    talib = _import_talib()
    values = talib.AROONOSC(
        high.astype(float),
        low.astype(float),
        timeperiod=int(window),
    )
    return _as_series(values, high.index)


def _helper_cmo(close, window):
    talib = _import_talib()
    values = talib.CMO(close.astype(float), timeperiod=int(window))
    return _as_series(values, close.index)


def _helper_psy(close, window):
    return (close > close.shift(1)).astype(np.float32).rolling(int(window)).mean() * 100.0


def _helper_bbi(close):
    return (
        close.rolling(3).mean()
        + close.rolling(6).mean()
        + close.rolling(12).mean()
        + close.rolling(24).mean()
    ) / 4.0


def _helper_pvt(close, volume):
    return ((close.pct_change(fill_method=None).fillna(0.0)) * volume).cumsum()


def _helper_emv(high, low, volume, window):
    midpoint_move = ((high + low) / 2.0).diff()
    box_ratio = volume / (high - low + 1e-12)
    emv_raw = midpoint_move / (box_ratio + 1e-12)
    return emv_raw.rolling(int(window)).mean()


def _helper_imi(open_, close, window):
    intraday_move = close - open_
    up_move = intraday_move.clip(lower=0).rolling(int(window)).sum()
    total_move = intraday_move.abs().rolling(int(window)).sum()
    return up_move / (total_move + 1e-12) * 100.0


def _helper_vhf(high, low, close, window):
    window = int(window)
    trend = high.rolling(window).max() - low.rolling(window).min()
    noise = close.diff().abs().rolling(window).sum()
    return trend / (noise + 1e-12)


def _helper_price_deviation(close, window):
    return close / (close.rolling(int(window)).mean() + 1e-12) - 1.0


def _helper_max_daily_return(close, window):
    return close.pct_change(fill_method=None).rolling(int(window)).max()


def _helper_normalized_ma_momentum(close, *windows):
    deviations = []
    for window in windows:
        deviations.append(close / (close.rolling(int(window)).mean() + 1e-12) - 1.0)
    return sum(deviations) / max(len(deviations), 1)


def _helper_chaikin_volatility(high, low, window):
    window = int(window)
    ema_range = (high - low).ewm(span=window, adjust=False).mean()
    return (ema_range - ema_range.shift(window)) / (ema_range.shift(window) + 1e-12)


EXPRESSION_HELPERS = {
    'abs': _helper_abs,
    'log': _helper_log,
    'sqrt': _helper_sqrt,
    'clip': _helper_clip,
    'shift': _helper_shift,
    'diff': _helper_diff,
    'delta': _helper_diff,
    'pct_change': _helper_pct_change,
    'sma': _helper_sma,
    'ema': _helper_ema,
    'rolling_mean': _helper_sma,
    'rolling_std': _helper_rolling_std,
    'rolling_min': _helper_rolling_min,
    'rolling_max': _helper_rolling_max,
    'rolling_sum': _helper_rolling_sum,
    'rolling_skew': _helper_rolling_skew,
    'rolling_quantile': _helper_rolling_quantile,
    'rank_pct': _helper_rank_pct,
    'zscore': _helper_zscore,
    'where': _helper_where,
    'row_max': _helper_row_max,
    'row_min': _helper_row_min,
    'vwap': _helper_vwap,
    'argmax_ratio': _helper_argmax_ratio,
    'argmin_ratio': _helper_argmin_ratio,
    'argmax_minus_argmin_ratio': _helper_argmax_minus_argmin_ratio,
    'rolling_corr': _helper_rolling_corr,
    'linearreg_slope': _helper_linearreg_slope,
    'linearreg_rsquare': _helper_linearreg_rsquare,
    'linearreg_residual': _helper_linearreg_residual,
    'direction_up_rate': _helper_direction_up_rate,
    'direction_down_rate': _helper_direction_down_rate,
    'direction_balance_rate': _helper_direction_balance_rate,
    'diff_up_ratio': _helper_diff_up_ratio,
    'diff_down_ratio': _helper_diff_down_ratio,
    'diff_balance_ratio': _helper_diff_balance_ratio,
    'vol_weighted_volatility': _helper_vol_weighted_volatility,
    'rsi': _helper_rsi,
    'macd_line': _helper_macd_line,
    'macd_signal': _helper_macd_signal,
    'obv': _helper_obv,
    'atr': _helper_atr,
    'boll_mid': _helper_boll_mid,
    'boll_std': _helper_boll_std,
    'kdj_k': _helper_kdj_k,
    'kdj_d': _helper_kdj_d,
    'tema': _helper_tema,
    'cci': _helper_cci,
    'mfi': _helper_mfi,
    'ad_line': _helper_ad_line,
    'chaikin_osc': _helper_chaikin_osc,
    'adx': _helper_adx,
    'aroon_osc': _helper_aroon_osc,
    'cmo': _helper_cmo,
    'psy': _helper_psy,
    'bbi': _helper_bbi,
    'pvt': _helper_pvt,
    'emv': _helper_emv,
    'imi': _helper_imi,
    'vhf': _helper_vhf,
    'price_deviation': _helper_price_deviation,
    'max_daily_return': _helper_max_daily_return,
    'normalized_ma_momentum': _helper_normalized_ma_momentum,
    'chaikin_volatility': _helper_chaikin_volatility,
    'np': SAFE_NUMPY_NAMESPACE,
}


TIME_SERIES_STATEFUL_HELPERS = {
    'shift',
    'diff',
    'delta',
    'pct_change',
    'sma',
    'ema',
    'rolling_mean',
    'rolling_std',
    'rolling_min',
    'rolling_max',
    'rolling_sum',
    'rolling_skew',
    'rolling_quantile',
    'rank_pct',
    'zscore',
    'argmax_ratio',
    'argmin_ratio',
    'argmax_minus_argmin_ratio',
    'rolling_corr',
    'linearreg_slope',
    'linearreg_rsquare',
    'linearreg_residual',
    'direction_up_rate',
    'direction_down_rate',
    'direction_balance_rate',
    'diff_up_ratio',
    'diff_down_ratio',
    'diff_balance_ratio',
    'vol_weighted_volatility',
    'rsi',
    'macd_line',
    'macd_signal',
    'obv',
    'atr',
    'boll_mid',
    'boll_std',
    'kdj_k',
    'kdj_d',
    'tema',
    'cci',
    'mfi',
    'ad_line',
    'chaikin_osc',
    'adx',
    'aroon_osc',
    'cmo',
    'psy',
    'bbi',
    'pvt',
    'emv',
    'imi',
    'vhf',
    'price_deviation',
    'max_daily_return',
    'normalized_ma_momentum',
    'chaikin_volatility',
}
ALLOWED_FUNCTION_CALLS = (set(EXPRESSION_HELPERS) - {'np'}) | CROSS_SECTIONAL_HELPER_NAMES


class _SafeExpressionValidator(ast.NodeVisitor):
    ALLOWED_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Attribute,
        ast.IfExp,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        ast.Subscript,
        ast.Slice,
        ast.keyword,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Not,
        ast.And,
        ast.Or,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
    )

    def generic_visit(self, node):
        if not isinstance(node, self.ALLOWED_NODES):
            raise ValueError(f'表达式包含不允许的语法节点: {type(node).__name__}')
        return super().generic_visit(node)

    def visit_Name(self, node):
        if node.id.startswith('__'):
            raise ValueError(f'表达式包含不安全名称: {node.id}')
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if (
            not isinstance(node.value, ast.Name)
            or node.value.id != 'np'
            or node.attr.startswith('_')
            or node.attr not in SAFE_NUMPY_ATTRS
        ):
            raise ValueError('仅允许使用白名单 np 函数/常量')
        self.visit(node.value)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id not in ALLOWED_FUNCTION_CALLS:
                raise ValueError(f'表达式调用了未授权函数: {node.func.id}')
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'np':
                if node.func.attr not in SAFE_NUMPY_ATTRS or node.func.attr.startswith('_'):
                    raise ValueError('仅允许使用白名单 np 函数/常量')
                self.visit(node.func.value)
            elif node.func.attr in SAFE_OBJECT_METHOD_CALLS:
                self.visit(node.func.value)
            else:
                raise ValueError('仅允许调用白名单方法')
        else:
            raise ValueError('表达式函数调用格式不受支持')

        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError('不允许 **kwargs 展开')
            self.visit(kw.value)


class _ExpressionSymbolCollector(ast.NodeVisitor):
    def __init__(self):
        self.variable_names = set()
        self.called_functions = set()
        self.called_numpy_functions = set()

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.called_functions.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'np':
                self.called_numpy_functions.add(node.func.attr)
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

    def visit_Name(self, node):
        self.variable_names.add(node.id)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'np':
            return
        self.generic_visit(node)


@lru_cache(maxsize=4096)
def _compile_expression(expression):
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as exc:
        raise ValueError(f'表达式语法错误: {exc}') from exc

    _SafeExpressionValidator().visit(tree)
    collector = _ExpressionSymbolCollector()
    collector.visit(tree)

    regex_identifiers = set(IDENTIFIER_PATTERN.findall(expression))
    variable_names = set(collector.variable_names)
    variable_names.update(regex_identifiers)
    variable_names.difference_update(collector.called_functions)
    variable_names.difference_update(SAFE_OBJECT_METHOD_CALLS)
    variable_names.difference_update(SAFE_NUMPY_ATTRS)
    variable_names.difference_update(EXPRESSION_RESERVED_NAMES)
    variable_names.discard('np')

    code = compile(tree, '<factor_expression>', mode='eval')
    return {
        'code': code,
        'variable_names': frozenset(variable_names),
        'called_functions': frozenset(collector.called_functions),
        'called_numpy_functions': frozenset(collector.called_numpy_functions),
    }


def _as_cache_key(value):
    if isinstance(value, pd.Series):
        return ('series', id(value), value.name)
    if isinstance(value, np.ndarray):
        return ('ndarray', id(value), value.shape)
    if np.isscalar(value):
        return ('scalar', value)
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_as_cache_key(item) for item in value))
    if isinstance(value, dict):
        return ('dict', tuple(sorted((k, _as_cache_key(v)) for k, v in value.items())))
    return ('object', id(value))


def _build_cached_helper(name, helper_fn, middleware_cache):
    if not callable(helper_fn):
        return helper_fn

    def _wrapped(*args, **kwargs):
        key = (
            name,
            tuple(_as_cache_key(arg) for arg in args),
            tuple(sorted((k, _as_cache_key(v)) for k, v in kwargs.items())),
        )
        if key in middleware_cache:
            return middleware_cache[key]
        out = helper_fn(*args, **kwargs)
        middleware_cache[key] = out
        return out

    return _wrapped


def _build_cross_sectional_helpers(df, date_col, middleware_cache):
    if date_col not in df.columns:
        return {}

    date_series = df[date_col]

    def _cs_rank(x):
        series = _as_series(x, df.index)
        key = ('cs_rank', _as_cache_key(series))
        if key in middleware_cache:
            return middleware_cache[key]
        ranked = series.groupby(date_series).rank(pct=True)
        middleware_cache[key] = ranked
        return ranked

    def _cs_zscore(x):
        series = _as_series(x, df.index)
        key = ('cs_zscore', _as_cache_key(series))
        if key in middleware_cache:
            return middleware_cache[key]
        grouped = series.groupby(date_series)
        mean = grouped.transform('mean')
        std = grouped.transform('std').replace(0.0, np.nan)
        z = (series - mean) / (std + 1e-12)
        middleware_cache[key] = z
        return z

    return {
        'cs_rank': _cs_rank,
        'cs_zscore': _cs_zscore,
    }


def _build_runtime_helpers(df, date_col, middleware_cache):
    helper_env = {}
    for name, helper in EXPRESSION_HELPERS.items():
        helper_env[name] = _build_cached_helper(name, helper, middleware_cache)
    helper_env.update(SAFE_LITERAL_TYPES)
    helper_env.update(_build_cross_sectional_helpers(df, date_col, middleware_cache))
    return helper_env


def _extract_input_dependencies(raw_value):
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return set()
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if parsed is not None:
            if isinstance(parsed, str):
                return set()
            return set()
        return set(IDENTIFIER_PATTERN.findall(text))

    if isinstance(raw_value, (list, tuple, set)):
        out = set()
        for item in raw_value:
            out.update(_extract_input_dependencies(item))
        return out

    if isinstance(raw_value, dict):
        out = set()
        for value in raw_value.values():
            out.update(_extract_input_dependencies(value))
        return out

    return set()


def _normalize_execution_spec(spec):
    normalized = dict(spec)
    normalized['inputs'] = _normalize_factor_inputs(normalized.get('inputs', {}))
    return normalized


def build_factor_execution_plan(factor_specs, error_prefix='因子'):
    if not factor_specs:
        return {
            'ordered_specs': [],
            'time_series_specs': [],
            'cross_sectional_specs': [],
            'dependency_graph': {},
        }

    normalized_specs = [_normalize_execution_spec(spec) for spec in factor_specs]
    factor_names = {spec['name'] for spec in normalized_specs}

    prepared_specs = []
    for order_idx, spec in enumerate(normalized_specs):
        if not spec.get('name'):
            raise ValueError(f'{error_prefix} 存在缺少 name 的因子定义')
        if not spec.get('expression'):
            raise ValueError(f'{error_prefix} {spec["name"]} 缺少 expression')
        compile_meta = _compile_expression(spec['expression'])
        variable_names = set(compile_meta['variable_names'])
        inputs = spec.get('inputs', {})

        referenced_symbols = set()
        used_input_aliases = {alias for alias in inputs if alias in variable_names}
        for alias in used_input_aliases:
            referenced_symbols.update(_extract_input_dependencies(inputs[alias]))

        referenced_symbols.update(variable_names - used_input_aliases)
        referenced_symbols.difference_update(EXPRESSION_RESERVED_NAMES)
        referenced_symbols.discard('np')
        referenced_symbols.difference_update(set(EXPRESSION_HELPERS))
        referenced_symbols.difference_update(set(SAFE_LITERAL_TYPES))
        dependencies = sorted((referenced_symbols & factor_names) - {spec['name']})

        called_functions = set(compile_meta['called_functions'])
        uses_cross_sectional = bool(called_functions & CROSS_SECTIONAL_HELPER_NAMES)
        uses_stateful_ts = bool(called_functions & TIME_SERIES_STATEFUL_HELPERS)

        enriched = dict(spec)
        enriched.update({
            '_order': order_idx,
            '_dependencies': dependencies,
            '_called_functions': sorted(called_functions),
            '_uses_cross_sectional_helper': uses_cross_sectional,
            '_uses_stateful_ts_helper': uses_stateful_ts,
        })
        prepared_specs.append(enriched)

    name_to_spec = {spec['name']: spec for spec in prepared_specs}
    in_degree = {spec['name']: len(spec['_dependencies']) for spec in prepared_specs}
    forward_edges = defaultdict(set)
    for spec in prepared_specs:
        for dep_name in spec['_dependencies']:
            forward_edges[dep_name].add(spec['name'])

    ready = deque(
        sorted(
            [name for name, degree in in_degree.items() if degree == 0],
            key=lambda n: name_to_spec[n]['_order'],
        )
    )
    ordered_names = []
    while ready:
        current = ready.popleft()
        ordered_names.append(current)
        next_nodes = sorted(
            forward_edges.get(current, []),
            key=lambda n: name_to_spec[n]['_order'],
        )
        for nxt in next_nodes:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                ready.append(nxt)

    if len(ordered_names) != len(prepared_specs):
        cycle_nodes = sorted([name for name, degree in in_degree.items() if degree > 0])
        raise ValueError(f'{error_prefix} 存在循环依赖: {cycle_nodes}')

    ordered_specs = []
    cs_names = set()
    for exec_idx, name in enumerate(ordered_names):
        spec = dict(name_to_spec[name])
        is_cross_sectional = spec['_uses_cross_sectional_helper'] or any(
            dep in cs_names for dep in spec['_dependencies']
        )

        if is_cross_sectional and spec['_uses_stateful_ts_helper']:
            raise ValueError(
                f'{error_prefix} {spec["name"]} 在截面阶段调用了时序函数，'
                f'请先拆分为时序因子再做 cs_* 计算。'
            )

        if is_cross_sectional:
            cs_names.add(spec['name'])

        spec['execution_order'] = exec_idx
        spec['dependencies'] = list(spec['_dependencies'])
        spec['is_cross_sectional'] = bool(is_cross_sectional)
        spec['called_functions'] = list(spec['_called_functions'])

        for transient_key in (
            '_order',
            '_dependencies',
            '_called_functions',
            '_uses_cross_sectional_helper',
            '_uses_stateful_ts_helper',
        ):
            spec.pop(transient_key, None)
        ordered_specs.append(spec)

    return {
        'ordered_specs': ordered_specs,
        'time_series_specs': [spec for spec in ordered_specs if not spec.get('is_cross_sectional')],
        'cross_sectional_specs': [spec for spec in ordered_specs if spec.get('is_cross_sectional')],
        'dependency_graph': {
            spec['name']: list(spec.get('dependencies', []))
            for spec in ordered_specs
        },
    }


def _resolve_input_value(raw_value, runtime_env):
    if not isinstance(raw_value, str):
        return raw_value

    text = raw_value.strip()
    if text in runtime_env:
        return runtime_env[text]

    try:
        return ast.literal_eval(text)
    except Exception:
        return raw_value


def _build_inputs_env(spec, runtime_env):
    inputs = _normalize_factor_inputs(spec.get('inputs', {}))
    return {
        alias: _resolve_input_value(raw_value, runtime_env)
        for alias, raw_value in inputs.items()
    }


def apply_factor_expressions(df, factor_specs, error_prefix='因子', date_col='日期'):
    if not factor_specs:
        return df

    if any(spec.get('is_cross_sectional') for spec in factor_specs) and date_col not in df.columns:
        raise ValueError(f'{error_prefix} 需要日期列 `{date_col}` 才能计算截面因子')

    if all('execution_order' in spec for spec in factor_specs):
        execution_specs = [_normalize_execution_spec(spec) for spec in factor_specs]
    else:
        execution_specs = build_factor_execution_plan(factor_specs, error_prefix=error_prefix)['ordered_specs']

    df = df.copy()
    runtime_env = {column: df[column] for column in df.columns}
    middleware_cache = {}
    helper_env = _build_runtime_helpers(df, date_col, middleware_cache)
    computed_columns = {}

    for spec in execution_specs:
        local_env = dict(runtime_env)
        local_env.update(helper_env)
        local_env.update(_build_inputs_env(spec, runtime_env))
        compile_meta = _compile_expression(spec['expression'])
        try:
            value = eval(compile_meta['code'], {'__builtins__': {}}, local_env)
        except Exception as exc:
            raise ValueError(
                f'{error_prefix} {spec["name"]} 计算失败: {exc}. expression={spec["expression"]}'
            ) from exc

        series_value = _as_series(value, df.index).astype(np.float32)
        computed_columns[spec['name']] = series_value
        runtime_env[spec['name']] = series_value

    if computed_columns:
        replace_names = [name for name in computed_columns if name in df.columns]
        if replace_names:
            df = df.drop(columns=replace_names)
        df = pd.concat([df, pd.DataFrame(computed_columns, index=df.index)], axis=1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


def apply_builtin_overrides(df, builtin_override_specs):
    override_specs = [spec for spec in builtin_override_specs if spec.get('overridden')]
    return apply_factor_expressions(df, override_specs, error_prefix='内置因子覆盖')


def apply_custom_factors(df, custom_factor_specs):
    return apply_factor_expressions(df, custom_factor_specs, error_prefix='自定义因子')


def engineer_group_features(task):
    if isinstance(task, dict):
        group = task['group']
        feature_set = task['feature_set']
        factor_specs = task.get('factor_specs', [])
    elif len(task) == 3:
        group, feature_set, factor_specs = task
    elif len(task) == 4:
        group, feature_set, builtin_override_specs, custom_factor_specs = task
        legacy_specs = [spec for spec in builtin_override_specs if spec.get('overridden')]
        legacy_specs.extend(custom_factor_specs)
        factor_specs = build_factor_execution_plan(legacy_specs, error_prefix='因子')['time_series_specs']
    else:
        raise ValueError('engineer_group_features 任务参数不合法')

    engineer = FEATURE_ENGINEER_FUNC_MAP[feature_set]
    processed = engineer(group.copy())
    factor_specs = [spec for spec in factor_specs if not spec.get('is_cross_sectional')]
    processed = apply_factor_expressions(
        processed,
        factor_specs,
        error_prefix='时序因子',
        date_col='日期',
    )
    return processed


def get_factor_spec(feature_set, store_path, factor_name):
    pipeline = resolve_factor_pipeline(feature_set, store_path)
    for spec in pipeline['all_specs']:
        if spec['name'] == factor_name:
            return spec
    raise ValueError(f'未找到因子: {factor_name}')


def set_factor_enabled(store_path, feature_set, factor_name, enabled):
    store = load_factor_store(store_path)
    feature_set_config = _get_feature_set_config(store, feature_set)
    builtin_names = {spec['name'] for spec in get_builtin_specs(feature_set)}

    if factor_name in builtin_names:
        disabled = set(feature_set_config.get('disabled_builtin_factors', []))
        if enabled:
            disabled.discard(factor_name)
        else:
            disabled.add(factor_name)
        feature_set_config['disabled_builtin_factors'] = sorted(disabled)
        save_factor_store(store, store_path)
        return

    for spec in feature_set_config.get('custom_factors', []):
        if spec['name'] == factor_name:
            spec['enabled'] = bool(enabled)
            save_factor_store(store, store_path)
            return

    raise ValueError(f'未找到因子: {factor_name}')


def set_factors_enabled(store_path, feature_set, factor_names, enabled, strict=True):
    names = [str(name).strip() for name in factor_names if str(name).strip()]
    dedup_names = list(dict.fromkeys(names))
    if not dedup_names:
        return {
            'updated_total': 0,
            'updated_builtin': 0,
            'updated_custom': 0,
            'missing': [],
        }

    store = load_factor_store(store_path)
    feature_set_config = _get_feature_set_config(store, feature_set)
    builtin_names = {spec['name'] for spec in get_builtin_specs(feature_set)}
    custom_factors = feature_set_config.get('custom_factors', [])
    custom_name_to_spec = {spec['name']: spec for spec in custom_factors}

    existing_names = builtin_names | set(custom_name_to_spec)
    missing = [name for name in dedup_names if name not in existing_names]
    if missing and strict:
        raise ValueError(f'未找到因子: {missing}')

    disabled = set(feature_set_config.get('disabled_builtin_factors', []))
    updated_builtin = 0
    updated_custom = 0

    for name in dedup_names:
        if name in builtin_names:
            before = name in disabled
            if enabled:
                disabled.discard(name)
            else:
                disabled.add(name)
            after = name in disabled
            if before != after:
                updated_builtin += 1
            continue

        spec = custom_name_to_spec.get(name)
        if spec is None:
            continue
        before = bool(spec.get('enabled', True))
        spec['enabled'] = bool(enabled)
        if before != bool(enabled):
            updated_custom += 1

    feature_set_config['disabled_builtin_factors'] = sorted(disabled)
    save_factor_store(store, store_path)

    return {
        'updated_total': int(updated_builtin + updated_custom),
        'updated_builtin': int(updated_builtin),
        'updated_custom': int(updated_custom),
        'missing': missing,
    }


def set_group_enabled(store_path, feature_set, group_name, enabled, source='all'):
    source = str(source or 'all').lower()
    if source not in {'all', 'builtin', 'custom'}:
        raise ValueError(f'不支持的 source: {source}')

    pipeline = resolve_factor_pipeline(feature_set, store_path)
    target_specs = []
    for spec in pipeline['all_specs']:
        if str(spec.get('group', '')) != str(group_name):
            continue
        if source == 'builtin' and spec.get('source') != 'builtin':
            continue
        if source == 'custom' and spec.get('source') != 'custom':
            continue
        target_specs.append(spec)

    if not target_specs:
        raise ValueError(f'分组下无因子: group={group_name}, source={source}')

    factor_names = [spec['name'] for spec in target_specs]
    result = set_factors_enabled(
        store_path,
        feature_set,
        factor_names,
        enabled=enabled,
        strict=False,
    )
    result.update({
        'group': group_name,
        'source': source,
        'matched': len(factor_names),
        'factors': factor_names,
    })
    return result


def activate_only_factors(store_path, feature_set, active_factor_names, strict=True):
    active_names = [str(name).strip() for name in active_factor_names if str(name).strip()]
    active_set = set(active_names)

    store = load_factor_store(store_path)
    feature_set_config = _get_feature_set_config(store, feature_set)
    builtin_names = {spec['name'] for spec in get_builtin_specs(feature_set)}
    custom_factors = feature_set_config.get('custom_factors', [])
    custom_names = {spec['name'] for spec in custom_factors}
    all_names = builtin_names | custom_names

    unknown = sorted(active_set - all_names)
    if unknown and strict:
        raise ValueError(f'activate-only 包含未知因子: {unknown}')

    effective_active = active_set & all_names
    feature_set_config['disabled_builtin_factors'] = sorted(
        name for name in builtin_names if name not in effective_active
    )
    for spec in custom_factors:
        spec['enabled'] = bool(spec['name'] in effective_active)

    save_factor_store(store, store_path)

    return {
        'active_count': int(len(effective_active)),
        'builtin_active': int(sum(1 for name in builtin_names if name in effective_active)),
        'custom_active': int(sum(1 for name in custom_names if name in effective_active)),
        'unknown': unknown,
        'active_factors': sorted(effective_active),
    }


def upsert_custom_factor(
    store_path,
    feature_set,
    factor_name,
    expression,
    group='custom',
    description='',
    enabled=True,
    inputs=None,
    author=None,
):
    store = load_factor_store(store_path)
    feature_set_config = _get_feature_set_config(store, feature_set)
    builtin_names = {spec['name'] for spec in get_builtin_specs(feature_set)}
    if factor_name in builtin_names:
        raise ValueError(f'内置因子不能被覆盖: {factor_name}')

    now = _utc_now_iso()
    resolved_author = (author or os.environ.get('USER') or '').strip()
    normalized_inputs = None if inputs is None else _normalize_factor_inputs(inputs)

    custom_factors = feature_set_config.setdefault('custom_factors', [])
    for spec in custom_factors:
        if spec['name'] == factor_name:
            spec['expression'] = expression
            spec['group'] = group
            spec['description'] = description
            spec['enabled'] = bool(enabled)
            if normalized_inputs is not None:
                spec['inputs'] = normalized_inputs
            else:
                spec.setdefault('inputs', {})
            spec['created_at'] = spec.get('created_at') or now
            spec['updated_at'] = now
            if resolved_author:
                spec['author'] = resolved_author
            elif spec.get('author') is None:
                spec['author'] = ''
            save_factor_store(store, store_path)
            return

    custom_factors.append({
        'name': factor_name,
        'expression': expression,
        'group': group,
        'description': description,
        'enabled': bool(enabled),
        'inputs': normalized_inputs if normalized_inputs is not None else {},
        'created_at': now,
        'updated_at': now,
        'author': resolved_author,
    })
    save_factor_store(store, store_path)


def delete_custom_factor(store_path, feature_set, factor_name):
    store = load_factor_store(store_path)
    feature_set_config = _get_feature_set_config(store, feature_set)
    custom_factors = feature_set_config.setdefault('custom_factors', [])
    new_custom_factors = [spec for spec in custom_factors if spec['name'] != factor_name]
    if len(new_custom_factors) == len(custom_factors):
        raise ValueError(f'未找到自定义因子: {factor_name}')
    feature_set_config['custom_factors'] = new_custom_factors
    save_factor_store(store, store_path)


def upsert_builtin_override(
    store_path,
    feature_set,
    factor_name,
    expression,
    group=None,
    description=None,
    inputs=None,
    author=None,
):
    store = load_factor_store(store_path)
    feature_set_config = _get_feature_set_config(store, feature_set)
    builtin_names = {spec['name'] for spec in get_builtin_specs(feature_set)}
    if factor_name not in builtin_names:
        raise ValueError(f'未找到内置因子: {factor_name}')

    now = _utc_now_iso()
    resolved_author = (author or os.environ.get('USER') or '').strip()
    normalized_inputs = None if inputs is None else _normalize_factor_inputs(inputs)

    builtin_overrides = feature_set_config.setdefault('builtin_overrides', [])
    for spec in builtin_overrides:
        if spec['name'] == factor_name:
            spec['expression'] = expression
            if group is not None:
                spec['group'] = group
            if description is not None:
                spec['description'] = description
            if normalized_inputs is not None:
                spec['inputs'] = normalized_inputs
            else:
                spec.setdefault('inputs', {})
            spec['created_at'] = spec.get('created_at') or now
            spec['updated_at'] = now
            if resolved_author:
                spec['author'] = resolved_author
            elif spec.get('author') is None:
                spec['author'] = ''
            save_factor_store(store, store_path)
            return

    new_spec = {
        'name': factor_name,
        'expression': expression,
        'inputs': normalized_inputs if normalized_inputs is not None else {},
        'created_at': now,
        'updated_at': now,
        'author': resolved_author,
    }
    if group is not None:
        new_spec['group'] = group
    if description is not None:
        new_spec['description'] = description
    builtin_overrides.append(new_spec)
    save_factor_store(store, store_path)


def clear_builtin_override(store_path, feature_set, factor_name):
    store = load_factor_store(store_path)
    feature_set_config = _get_feature_set_config(store, feature_set)
    builtin_overrides = feature_set_config.setdefault('builtin_overrides', [])
    new_builtin_overrides = [spec for spec in builtin_overrides if spec['name'] != factor_name]
    if len(new_builtin_overrides) == len(builtin_overrides):
        raise ValueError(f'未找到内置因子 override: {factor_name}')
    feature_set_config['builtin_overrides'] = new_builtin_overrides
    save_factor_store(store, store_path)


def build_factor_snapshot(pipeline):
    active_specs_for_hash = pipeline.get('ordered_specs') or pipeline.get('active_specs', [])
    factor_fingerprint = pipeline.get('factor_fingerprint') or _compute_factor_fingerprint(
        pipeline['feature_set'],
        active_specs_for_hash,
    )
    snapshot = {
        'feature_set': pipeline['feature_set'],
        'store_path': pipeline['store_path'],
        'builtin_registry_path': pipeline.get('builtin_registry_path', DEFAULT_BUILTIN_FACTOR_REGISTRY_PATH),
        'summary': pipeline['summary'],
        'active_features': pipeline['active_features'],
        'builtin_specs': pipeline['builtin_specs'],
        'custom_specs': pipeline['custom_specs'],
        'dependency_graph': pipeline.get('dependency_graph', {}),
        'factor_fingerprint': factor_fingerprint,
        'snapshot': {
            'created_at': _utc_now_iso(),
            'factor_fingerprint': factor_fingerprint,
        },
    }
    return snapshot


def save_factor_snapshot(pipeline, output_path):
    snapshot = build_factor_snapshot(pipeline)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    return snapshot


def load_factor_snapshot(snapshot_path):
    with open(snapshot_path, 'r', encoding='utf-8') as f:
        snapshot = json.load(f)

    feature_set = snapshot['feature_set']
    if feature_set not in FEATURE_ENGINEER_FUNC_MAP:
        raise ValueError(f'快照中的 feature_set 不支持: {feature_set}')

    builtin_specs = [dict(spec) for spec in snapshot.get('builtin_specs', [])]
    for spec in builtin_specs:
        spec['inputs'] = _normalize_factor_inputs(spec.get('inputs', {}))
    custom_specs = [
        _normalize_custom_factor_spec(spec)
        for spec in snapshot.get('custom_specs', [])
    ]
    all_specs = builtin_specs + custom_specs
    active_specs = [spec for spec in all_specs if spec.get('enabled', True)]
    execution_plan = build_factor_execution_plan(active_specs, error_prefix='快照因子')
    active_features = snapshot.get('active_features', [spec['name'] for spec in active_specs])
    summary = snapshot.get('summary', {
        'builtin_total': len(builtin_specs),
        'builtin_enabled': sum(1 for spec in builtin_specs if spec.get('enabled', True)),
        'builtin_overridden': sum(1 for spec in builtin_specs if spec.get('overridden')),
        'custom_total': len(custom_specs),
        'custom_enabled': sum(1 for spec in custom_specs if spec.get('enabled', True)),
        'active_total': len(active_features),
        'cross_sectional_total': len(execution_plan['cross_sectional_specs']),
        'group_counts': {},
    })
    summary.setdefault('cross_sectional_total', len(execution_plan['cross_sectional_specs']))

    factor_fingerprint = snapshot.get('factor_fingerprint') or _compute_factor_fingerprint(
        feature_set,
        execution_plan['ordered_specs'],
    )

    return {
        'feature_set': feature_set,
        'store_path': snapshot.get('store_path', ''),
        'builtin_registry_path': snapshot.get('builtin_registry_path', DEFAULT_BUILTIN_FACTOR_REGISTRY_PATH),
        'engineer': FEATURE_ENGINEER_FUNC_MAP[feature_set],
        'builtin_specs': builtin_specs,
        'builtin_override_specs': [spec for spec in builtin_specs if spec.get('overridden')],
        'custom_specs': custom_specs,
        'all_specs': all_specs,
        'active_specs': active_specs,
        'ordered_specs': execution_plan['ordered_specs'],
        'time_series_specs': execution_plan['time_series_specs'],
        'cross_sectional_specs': execution_plan['cross_sectional_specs'],
        'dependency_graph': snapshot.get('dependency_graph', execution_plan['dependency_graph']),
        'active_features': active_features,
        'factor_fingerprint': factor_fingerprint,
        'snapshot_meta': snapshot.get('snapshot', {}),
        'summary': summary,
    }
