import ast
import re
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd


INTRADAY_ALLOWED_FUNCTIONS = {
    'log',
    'diff',
    'sum',
    'mean',
    'std',
    'min',
    'max',
    'first',
    'last',
    'sqrt',
    'abs',
}

INTRADAY_ALLOWED_NAMES = {
    'close',
    'open',
    'high',
    'low',
    'volume',
    'amount',
    'full_day',
}


class _SafeIntradayExpressionValidator(ast.NodeVisitor):
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
    )

    def generic_visit(self, node):
        if not isinstance(node, self.ALLOWED_NODES):
            raise ValueError(f'intraday expression 包含不允许的语法节点: {type(node).__name__}')
        return super().generic_visit(node)

    def visit_Name(self, node):
        if node.id.startswith('__'):
            raise ValueError(f'intraday expression 包含不安全名称: {node.id}')
        if node.id not in INTRADAY_ALLOWED_NAMES and node.id not in INTRADAY_ALLOWED_FUNCTIONS:
            raise ValueError(f'intraday expression 引用了未授权变量: {node.id}')
        return self.generic_visit(node)

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError('intraday expression 仅允许调用白名单函数')
        if node.func.id not in INTRADAY_ALLOWED_FUNCTIONS:
            raise ValueError(f'intraday expression 调用了未授权函数: {node.func.id}')
        return self.generic_visit(node)


def _normalize_intraday_expression(expression: str) -> str:
    text = str(expression or '').strip()
    if not text:
        return text
    text = text.replace('^', '**')
    # 将 DSL 时间窗口 token 30m/15m 转换为字符串字面量
    text = re.sub(r'([,(=]\s*)(\d+)\s*m(?=\s*[),])', r"\1'\2m'", text, flags=re.IGNORECASE)
    # 将 full_day 作为窗口参数时转成字符串
    text = re.sub(r'([,(=]\s*)(full_day)(?=\s*[),])', r"\1'full_day'", text, flags=re.IGNORECASE)
    return text


def _compile_intraday_expression(expression: str) -> Dict:
    normalized = _normalize_intraday_expression(expression)
    parsed = ast.parse(normalized, mode='eval')
    _SafeIntradayExpressionValidator().visit(parsed)
    return {
        'raw_expression': str(expression),
        'normalized_expression': normalized,
        'code': compile(parsed, '<intraday_expression>', 'eval'),
    }


def _parse_window_arg(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {'full_day', 'all', 'day'}:
            return None
        match = re.match(r'^(\d+)\s*m$', text)
        if match:
            return pd.Timedelta(minutes=int(match.group(1)))
        if text.isdigit():
            return int(text)
        raise ValueError(f'不支持的窗口参数: {value}')
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    return None


def _as_intraday_series(value, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        s = pd.to_numeric(value, errors='coerce')
        if not s.index.equals(index):
            s = s.reindex(index)
        return s
    if np.isscalar(value):
        return pd.Series(float(value), index=index, dtype=np.float64)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return pd.Series(float(arr), index=index, dtype=np.float64)
    if len(arr) != len(index):
        raise ValueError(f'向量长度与分钟序列不一致: len(arr)={len(arr)}, len(index)={len(index)}')
    return pd.Series(pd.to_numeric(arr, errors='coerce'), index=index, dtype=np.float64)


def _slice_intraday_window(series: pd.Series, window) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return s
    parsed = _parse_window_arg(window)
    if parsed is None:
        return s
    if isinstance(parsed, pd.Timedelta):
        end_ts = s.index.max()
        start_ts = end_ts - parsed
        return s[s.index >= start_ts]
    if isinstance(parsed, int):
        return s.tail(max(1, int(parsed)))
    return s


def _evaluate_intraday_expression(
    group: pd.DataFrame,
    compile_meta: Dict,
    *,
    min_bars: int,
) -> float:
    eps = 1e-12
    g = group.sort_values('ts').copy()
    g['close'] = pd.to_numeric(g['close'], errors='coerce')
    g['open'] = pd.to_numeric(g.get('open', np.nan), errors='coerce')
    g['high'] = pd.to_numeric(g.get('high', np.nan), errors='coerce')
    g['low'] = pd.to_numeric(g.get('low', np.nan), errors='coerce')
    g['volume'] = pd.to_numeric(g.get('volume', np.nan), errors='coerce')
    g['amount'] = pd.to_numeric(g.get('amount', np.nan), errors='coerce')
    g = g.dropna(subset=['close'])
    if len(g) < max(1, int(min_bars)):
        return float('nan')

    idx = pd.DatetimeIndex(g['ts'])
    close = pd.Series(g['close'].to_numpy(dtype=np.float64), index=idx)
    open_ = pd.Series(g['open'].to_numpy(dtype=np.float64), index=idx)
    high = pd.Series(g['high'].to_numpy(dtype=np.float64), index=idx)
    low = pd.Series(g['low'].to_numpy(dtype=np.float64), index=idx)
    volume = pd.Series(g['volume'].to_numpy(dtype=np.float64), index=idx)
    amount = pd.Series(g['amount'].to_numpy(dtype=np.float64), index=idx)

    def _sum(x, window=None, **kwargs):
        window = kwargs.get('last', window)
        return float(_slice_intraday_window(_as_intraday_series(x, idx), window).sum(skipna=True))

    def _mean(x, window=None, **kwargs):
        window = kwargs.get('last', window)
        s = _slice_intraday_window(_as_intraday_series(x, idx), window)
        return float(s.mean(skipna=True)) if not s.empty else float('nan')

    def _std(x, window=None, **kwargs):
        window = kwargs.get('last', window)
        s = _slice_intraday_window(_as_intraday_series(x, idx), window)
        return float(s.std(ddof=0, skipna=True)) if not s.empty else float('nan')

    def _min(x, window=None, **kwargs):
        window = kwargs.get('last', window)
        s = _slice_intraday_window(_as_intraday_series(x, idx), window)
        return float(s.min(skipna=True)) if not s.empty else float('nan')

    def _max(x, window=None, **kwargs):
        window = kwargs.get('last', window)
        s = _slice_intraday_window(_as_intraday_series(x, idx), window)
        return float(s.max(skipna=True)) if not s.empty else float('nan')

    def _first(x, window=None, **kwargs):
        window = kwargs.get('last', window)
        s = _slice_intraday_window(_as_intraday_series(x, idx), window)
        return float(s.iloc[0]) if not s.empty else float('nan')

    def _last(x, window=None, **kwargs):
        window = kwargs.get('last', window)
        s = _slice_intraday_window(_as_intraday_series(x, idx), window)
        return float(s.iloc[-1]) if not s.empty else float('nan')

    local_env = {
        'close': close,
        'open': open_,
        'high': high,
        'low': low,
        'volume': volume,
        'amount': amount,
        'full_day': 'full_day',
        'log': lambda x: np.log(_as_intraday_series(x, idx) + eps),
        'diff': lambda x: _as_intraday_series(x, idx).diff(),
        'sum': _sum,
        'mean': _mean,
        'std': _std,
        'min': _min,
        'max': _max,
        'first': _first,
        'last': _last,
        'sqrt': np.sqrt,
        'abs': np.abs,
    }

    value = eval(compile_meta['code'], {'__builtins__': {}}, local_env)
    if isinstance(value, pd.Series):
        value = value.dropna().iloc[-1] if not value.dropna().empty else np.nan
    try:
        value = float(value)
    except Exception:
        return float('nan')
    if not np.isfinite(value):
        return float('nan')
    return value


def _format_intraday_node_message(
    *,
    node_id: str,
    engine: str,
    strict: bool,
    dependencies: List[str],
    detail: str,
) -> str:
    return (
        f'节点 {node_id} engine={engine} strict={strict} '
        f'dependencies={dependencies} {detail}'
    )


def _compute_intraday_group_values(
    minute_df: pd.DataFrame,
    *,
    keys: List[str],
    compile_meta: Dict,
    min_bars: int,
) -> pd.DataFrame:
    rows = []
    grouped = minute_df.groupby(keys, sort=False)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {key: group_key[idx] for idx, key in enumerate(keys)}
        row['__value'] = _evaluate_intraday_expression(
            group,
            compile_meta=compile_meta,
            min_bars=min_bars,
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=[*keys, '__value'])


def _compute_intraday_nodes_from_minute(
    minute_df: pd.DataFrame,
    factor_nodes: List[Dict],
    *,
    strict: bool,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str], List[str], List[str]]:
    keys = ['instrument_id', 'trade_date']
    out = minute_df[keys].drop_duplicates().copy()
    source_map: Dict[str, str] = {}
    issue_map: Dict[str, str] = {}
    warnings: List[str] = []
    errors: List[str] = []

    intraday_nodes = [
        node for node in factor_nodes
        if str((node.get('compute') or {}).get('engine', '')).strip() == 'intraday_aggregate'
    ]
    for node in intraday_nodes:
        node_id = str(node.get('id', '')).strip()
        dependencies = [str(item).strip() for item in (node.get('dependencies', []) or [])]
        engine = 'intraday_aggregate'
        compute = node.get('compute', {}) if isinstance(node, dict) else {}
        output_col = str((node.get('output') or {}).get('column', '')).strip()
        expression = str(compute.get('expression', '')).strip()
        min_bars = int(compute.get('min_bars', 1))

        if not output_col:
            msg = _format_intraday_node_message(
                node_id=node_id,
                engine=engine,
                strict=bool(strict),
                dependencies=dependencies,
                detail='缺少 output.column',
            )
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)
            continue

        if not expression:
            msg = _format_intraday_node_message(
                node_id=node_id,
                engine=engine,
                strict=bool(strict),
                dependencies=dependencies,
                detail='intraday_aggregate 缺少 expression',
            )
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)
            issue_map[output_col] = msg
            out[output_col] = np.nan
            continue

        try:
            compile_meta = _compile_intraday_expression(expression)
        except Exception as exc:
            msg = _format_intraday_node_message(
                node_id=node_id,
                engine=engine,
                strict=bool(strict),
                dependencies=dependencies,
                detail=f'intraday_aggregate expression 非法: {expression} | {exc}',
            )
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)
            issue_map[output_col] = msg
            out[output_col] = np.nan
            continue

        try:
            node_df = _compute_intraday_group_values(
                minute_df,
                keys=keys,
                compile_meta=compile_meta,
                min_bars=min_bars,
            )
        except Exception as exc:
            msg = _format_intraday_node_message(
                node_id=node_id,
                engine=engine,
                strict=bool(strict),
                dependencies=dependencies,
                detail=(
                    f'intraday_aggregate 计算失败: {exc} | '
                    f'expression={compile_meta.get("normalized_expression", expression)}'
                ),
            )
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)
            issue_map[output_col] = msg
            out[output_col] = np.nan
            continue

        node_df = node_df.rename(columns={'__value': output_col})
        out = out.merge(node_df, on=keys, how='left', validate='one_to_one')
        source_map[output_col] = (
            'hf_minute_input('
            + compile_meta.get('normalized_expression', expression)
            + ')'
        )

    return out, source_map, issue_map, warnings, errors
