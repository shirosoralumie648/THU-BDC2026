import copy
import json
import os
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
    for spec in BUILTIN_FACTOR_SPECS_MAP[feature_set]:
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
    }
    return normalized


def _validate_custom_factor_specs(custom_specs, builtin_names):
    seen = set(builtin_names)
    for spec in custom_specs:
        if not spec.get('name'):
            raise ValueError('自定义因子缺少 name')
        if not spec.get('expression'):
            raise ValueError(f'自定义因子 {spec["name"]} 缺少 expression')
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
            for field in ('expression', 'description', 'group'):
                if field in override:
                    current[field] = override[field]
            current['overridden'] = True
        resolved_specs.append(current)
    return resolved_specs


def resolve_factor_pipeline(feature_set, store_path, builtin_registry_path=DEFAULT_BUILTIN_FACTOR_REGISTRY_PATH):
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
    builtin_names = [spec['name'] for spec in builtin_specs]

    custom_specs = [
        _normalize_custom_factor_spec(spec)
        for spec in feature_set_config.get('custom_factors', [])
    ]
    _validate_custom_factor_specs(custom_specs, builtin_names)

    all_specs = builtin_specs + custom_specs
    active_specs = [spec for spec in all_specs if spec.get('enabled', True)]
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
        'active_features': active_feature_names,
        'summary': {
            'builtin_total': len(builtin_specs),
            'builtin_enabled': sum(1 for spec in builtin_specs if spec['enabled']),
            'builtin_overridden': sum(1 for spec in builtin_specs if spec.get('overridden')),
            'custom_total': len(custom_specs),
            'custom_enabled': sum(1 for spec in custom_specs if spec['enabled']),
            'active_total': len(active_feature_names),
            'group_counts': group_counts,
        },
    }


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
    'np': np,
}


def apply_factor_expressions(df, factor_specs, error_prefix='因子'):
    if not factor_specs:
        return df

    df = df.copy()
    for spec in factor_specs:
        local_env = {column: df[column] for column in df.columns}
        local_env.update(EXPRESSION_HELPERS)
        try:
            value = eval(spec['expression'], {'__builtins__': {}}, local_env)
        except Exception as exc:
            raise ValueError(
                f'{error_prefix} {spec["name"]} 计算失败: {exc}. expression={spec["expression"]}'
            ) from exc

        df[spec['name']] = _as_series(value, df.index).astype(np.float32)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


def apply_builtin_overrides(df, builtin_override_specs):
    override_specs = [spec for spec in builtin_override_specs if spec.get('overridden')]
    return apply_factor_expressions(df, override_specs, error_prefix='内置因子覆盖')


def apply_custom_factors(df, custom_factor_specs):
    return apply_factor_expressions(df, custom_factor_specs, error_prefix='自定义因子')


def engineer_group_features(task):
    group, feature_set, builtin_override_specs, custom_factor_specs = task
    engineer = FEATURE_ENGINEER_FUNC_MAP[feature_set]
    processed = engineer(group.copy())
    processed = apply_builtin_overrides(processed, builtin_override_specs)
    processed = apply_custom_factors(processed, custom_factor_specs)
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


def upsert_custom_factor(
    store_path,
    feature_set,
    factor_name,
    expression,
    group='custom',
    description='',
    enabled=True,
):
    store = load_factor_store(store_path)
    feature_set_config = _get_feature_set_config(store, feature_set)
    builtin_names = {spec['name'] for spec in BUILTIN_FACTOR_SPECS_MAP[feature_set]}
    if factor_name in builtin_names:
        raise ValueError(f'内置因子不能被覆盖: {factor_name}')

    custom_factors = feature_set_config.setdefault('custom_factors', [])
    for spec in custom_factors:
        if spec['name'] == factor_name:
            spec['expression'] = expression
            spec['group'] = group
            spec['description'] = description
            spec['enabled'] = bool(enabled)
            save_factor_store(store, store_path)
            return

    custom_factors.append({
        'name': factor_name,
        'expression': expression,
        'group': group,
        'description': description,
        'enabled': bool(enabled),
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
):
    store = load_factor_store(store_path)
    feature_set_config = _get_feature_set_config(store, feature_set)
    builtin_names = {spec['name'] for spec in get_builtin_specs(feature_set)}
    if factor_name not in builtin_names:
        raise ValueError(f'未找到内置因子: {factor_name}')

    builtin_overrides = feature_set_config.setdefault('builtin_overrides', [])
    for spec in builtin_overrides:
        if spec['name'] == factor_name:
            spec['expression'] = expression
            if group is not None:
                spec['group'] = group
            if description is not None:
                spec['description'] = description
            save_factor_store(store, store_path)
            return

    new_spec = {
        'name': factor_name,
        'expression': expression,
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
    return {
        'feature_set': pipeline['feature_set'],
        'store_path': pipeline['store_path'],
        'builtin_registry_path': pipeline.get('builtin_registry_path', DEFAULT_BUILTIN_FACTOR_REGISTRY_PATH),
        'summary': pipeline['summary'],
        'active_features': pipeline['active_features'],
        'builtin_specs': pipeline['builtin_specs'],
        'custom_specs': pipeline['custom_specs'],
    }


def save_factor_snapshot(pipeline, output_path):
    snapshot = build_factor_snapshot(pipeline)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)


def load_factor_snapshot(snapshot_path):
    with open(snapshot_path, 'r', encoding='utf-8') as f:
        snapshot = json.load(f)

    feature_set = snapshot['feature_set']
    if feature_set not in FEATURE_ENGINEER_FUNC_MAP:
        raise ValueError(f'快照中的 feature_set 不支持: {feature_set}')

    builtin_specs = snapshot.get('builtin_specs', [])
    custom_specs = [
        _normalize_custom_factor_spec(spec)
        for spec in snapshot.get('custom_specs', [])
    ]
    all_specs = builtin_specs + custom_specs
    active_specs = [spec for spec in all_specs if spec.get('enabled', True)]
    active_features = snapshot.get('active_features', [spec['name'] for spec in active_specs])
    summary = snapshot.get('summary', {
        'builtin_total': len(builtin_specs),
        'builtin_enabled': sum(1 for spec in builtin_specs if spec.get('enabled', True)),
        'builtin_overridden': sum(1 for spec in builtin_specs if spec.get('overridden')),
        'custom_total': len(custom_specs),
        'custom_enabled': sum(1 for spec in custom_specs if spec.get('enabled', True)),
        'active_total': len(active_features),
        'group_counts': {},
    })

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
        'active_features': active_features,
        'summary': summary,
    }
