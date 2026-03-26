from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import config
from data_manager import resolve_dataset_path
from factor_store import apply_factor_expressions
from factor_store import clear_builtin_override
from factor_store import delete_custom_factor
from factor_store import engineer_group_features
from factor_store import get_factor_spec
from factor_store import resolve_factor_pipeline
from factor_store import set_factor_enabled
from factor_store import upsert_builtin_override
from factor_store import upsert_custom_factor
from gui_console.common import load_csv_cached
from utils import apply_cross_sectional_normalization


def _factor_table_df(pipeline: Dict) -> pd.DataFrame:
    rows = []
    for spec in pipeline['all_specs']:
        rows.append(
            {
                'name': spec['name'],
                'enabled': bool(spec.get('enabled', True)),
                'source': spec.get('source', 'unknown'),
                'group': spec.get('group', ''),
                'overridden': bool(spec.get('overridden', False)),
                'description': spec.get('description', ''),
                'expression': spec.get('expression', ''),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _load_sample_frame(source_path: str, max_stocks: int, max_rows_per_stock: int) -> pd.DataFrame:
    df = load_csv_cached(source_path)
    if '股票代码' not in df.columns:
        raise ValueError('样本数据缺少 股票代码 列。')

    out = df.copy()
    out['股票代码'] = out['股票代码'].astype(str).str.zfill(6)
    out['日期'] = pd.to_datetime(out.get('日期'), errors='coerce')
    out = out.dropna(subset=['日期'])

    stock_list = sorted(out['股票代码'].unique().tolist())[: max(1, int(max_stocks))]
    out = out[out['股票代码'].isin(stock_list)].copy()
    out = out.sort_values(['股票代码', '日期'])
    out = out.groupby('股票代码', group_keys=False).tail(max(10, int(max_rows_per_stock))).copy()
    out['日期'] = out['日期'].dt.strftime('%Y-%m-%d')
    return out


def _validate_expression(sample_df: pd.DataFrame, expression: str):
    if not expression.strip():
        return False, '表达式不能为空', None

    preview = sample_df.sort_values(['股票代码', '日期']).copy()
    preview = preview.groupby('股票代码', group_keys=False).head(300)
    spec = {'name': '__preview_factor__', 'expression': expression}
    try:
        out = apply_factor_expressions(preview, [spec], error_prefix='表达式校验')
        values = pd.to_numeric(out['__preview_factor__'], errors='coerce').dropna()
        if values.empty:
            return False, '表达式有效，但结果全为空。', None
        stat = {
            'count': int(values.shape[0]),
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max()),
        }
        return True, '表达式校验通过。', stat
    except Exception as exc:
        return False, f'表达式校验失败: {exc}', None


@st.cache_data(show_spinner=False)
def _compute_factor_distribution(
    source_path: str,
    feature_set: str,
    store_path: str,
    factor_name: str,
    max_stocks: int,
    max_rows_per_stock: int,
    norm_method: str,
) -> pd.DataFrame:
    sample = _load_sample_frame(source_path, max_stocks, max_rows_per_stock)
    pipeline = resolve_factor_pipeline(feature_set, store_path, config['builtin_factor_registry_path'])
    target_spec = None
    for spec in pipeline['all_specs']:
        if spec['name'] == factor_name:
            target_spec = spec
            break
    if target_spec is None:
        raise ValueError(f'未找到因子: {factor_name}')

    grouped = []
    for _, g in sample.groupby('股票代码', sort=False):
        local = g.sort_values('日期').copy()
        try:
            local = apply_factor_expressions(local, [target_spec], error_prefix='分布分析')
        except Exception:
            local = engineer_group_features(
                (local, feature_set, pipeline['builtin_override_specs'], pipeline['custom_specs'])
            )

        if factor_name not in local.columns:
            continue
        grouped.append(local[['日期', '股票代码', factor_name]].copy())

    if not grouped:
        raise ValueError('未计算到有效因子值，请检查因子表达式。')

    result = pd.concat(grouped, ignore_index=True)
    result = result.rename(columns={factor_name: 'raw_value'})
    result['raw_value'] = pd.to_numeric(result['raw_value'], errors='coerce')
    result = result.dropna(subset=['raw_value']).copy()
    if result.empty:
        raise ValueError('因子值为空，无法绘图。')

    norm_df = result[['日期', 'raw_value']].copy()
    norm_df = norm_df.rename(columns={'raw_value': 'norm_value'})
    norm_df = apply_cross_sectional_normalization(
        norm_df,
        columns=['norm_value'],
        date_col='日期',
        method=norm_method,
    )
    result['norm_value'] = pd.to_numeric(norm_df['norm_value'], errors='coerce')
    result = result.dropna(subset=['norm_value'])
    return result


def _plot_distribution(df: pd.DataFrame, factor_name: str, norm_method: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df['raw_value'],
            nbinsx=80,
            opacity=0.65,
            name='原始分布',
            marker_color='#5b8ff9',
        )
    )
    fig.add_trace(
        go.Histogram(
            x=df['norm_value'],
            nbinsx=80,
            opacity=0.65,
            name=f'{norm_method} 标准化后',
            marker_color='#f6bd16',
        )
    )
    fig.update_layout(
        barmode='overlay',
        template='plotly_white',
        margin={'l': 20, 'r': 20, 't': 30, 'b': 20},
        title=f'因子分布对比: {factor_name}',
        xaxis_title='值',
        yaxis_title='频次',
        height=380,
    )
    return fig


def render_factor_lab() -> None:
    st.subheader('因子实验室 (Factor Lab)')

    feature_set = st.selectbox('因子集合', ['39', '158+39'], index=1 if config['feature_num'] == '158+39' else 0)
    store_path = st.text_input('因子存储路径', value=config['factor_store_path'])

    try:
        pipeline = resolve_factor_pipeline(feature_set, store_path, config['builtin_factor_registry_path'])
    except Exception as exc:
        st.error(f'加载因子配置失败: {exc}')
        return

    table_df = _factor_table_df(pipeline)
    st.caption(
        f"builtin={pipeline['summary']['builtin_enabled']}/{pipeline['summary']['builtin_total']} | "
        f"custom={pipeline['summary']['custom_enabled']}/{pipeline['summary']['custom_total']} | "
        f"active={pipeline['summary']['active_total']}"
    )

    st.dataframe(
        table_df[['name', 'enabled', 'source', 'group', 'overridden', 'description']],
        width='stretch',
        height=320,
    )

    st.markdown('### 因子启用/禁用')
    selected_names = st.multiselect('选择因子（支持批量）', table_df['name'].tolist(), key='factor_batch_select')
    col_enable, col_disable = st.columns(2)

    if col_enable.button('批量启用'):
        try:
            for name in selected_names:
                set_factor_enabled(store_path, feature_set, name, True)
            st.success(f'已启用 {len(selected_names)} 个因子。')
            st.rerun()
        except Exception as exc:
            st.error(f'启用失败: {exc}')

    if col_disable.button('批量禁用'):
        try:
            for name in selected_names:
                set_factor_enabled(store_path, feature_set, name, False)
            st.success(f'已禁用 {len(selected_names)} 个因子。')
            st.rerun()
        except Exception as exc:
            st.error(f'禁用失败: {exc}')

    st.markdown('### 因子编辑器')
    editor_mode = st.radio('编辑模式', ['编辑现有因子', '新建自定义因子'], horizontal=True)

    source_path = st.text_input('表达式校验数据源', value=resolve_dataset_path(config, 'train.csv'), key='factor_validate_source')
    sample_stocks = st.slider('校验样本股票数', min_value=5, max_value=80, value=30, step=1)
    sample_rows = st.slider('每只股票样本行数', min_value=120, max_value=1000, value=400, step=20)

    try:
        sample_df = _load_sample_frame(source_path, sample_stocks, sample_rows)
    except Exception as exc:
        sample_df = None
        st.warning(f'加载校验样本失败: {exc}')

    if editor_mode == '编辑现有因子':
        factor_name = st.selectbox('选择因子', table_df['name'].tolist(), key='factor_editor_existing')
        spec = get_factor_spec(feature_set, store_path, factor_name)

        edited_expression = st.text_area('表达式', value=spec.get('expression', ''), height=110)
        edited_group = st.text_input('分组', value=spec.get('group', 'custom'))
        edited_desc = st.text_input('描述', value=spec.get('description', ''))

        col_val, col_save, col_reset, col_delete = st.columns(4)
        if col_val.button('校验表达式'):
            if sample_df is None:
                st.error('缺少校验样本。')
            else:
                ok, msg, stat = _validate_expression(sample_df, edited_expression)
                (st.success if ok else st.error)(msg)
                if stat:
                    st.json(stat)

        if col_save.button('保存'):
            try:
                if spec.get('source') == 'custom':
                    upsert_custom_factor(
                        store_path,
                        feature_set,
                        factor_name,
                        edited_expression,
                        group=edited_group,
                        description=edited_desc,
                        enabled=bool(spec.get('enabled', True)),
                    )
                else:
                    upsert_builtin_override(
                        store_path,
                        feature_set,
                        factor_name,
                        edited_expression,
                        group=edited_group,
                        description=edited_desc,
                    )
                st.success('保存成功。')
                st.rerun()
            except Exception as exc:
                st.error(f'保存失败: {exc}')

        if col_reset.button('重置内置覆盖'):
            try:
                clear_builtin_override(store_path, feature_set, factor_name)
                st.success('已恢复内置默认公式。')
                st.rerun()
            except Exception as exc:
                st.error(f'重置失败: {exc}')

        if col_delete.button('删除自定义因子'):
            if spec.get('source') != 'custom':
                st.error('内置因子不能删除。')
            else:
                try:
                    delete_custom_factor(store_path, feature_set, factor_name)
                    st.success('已删除。')
                    st.rerun()
                except Exception as exc:
                    st.error(f'删除失败: {exc}')

    else:
        new_name = st.text_input('新因子名称', value='my_custom_factor')
        new_group = st.text_input('分组', value='custom')
        new_desc = st.text_input('描述', value='')
        new_expr = st.text_area('表达式', value='(开盘 / (shift(收盘, 1) + 1e-12)) - 1', height=110)
        new_enabled = st.checkbox('创建后启用', value=True)

        col_val_new, col_create = st.columns(2)
        if col_val_new.button('校验表达式', key='btn_validate_new_factor'):
            if sample_df is None:
                st.error('缺少校验样本。')
            else:
                ok, msg, stat = _validate_expression(sample_df, new_expr)
                (st.success if ok else st.error)(msg)
                if stat:
                    st.json(stat)

        if col_create.button('创建因子'):
            try:
                upsert_custom_factor(
                    store_path,
                    feature_set,
                    new_name.strip(),
                    new_expr,
                    group=new_group.strip() or 'custom',
                    description=new_desc,
                    enabled=new_enabled,
                )
                st.success(f'已创建因子: {new_name}')
                st.rerun()
            except Exception as exc:
                st.error(f'创建失败: {exc}')

    st.markdown('### 因子分布分析')
    factor_for_dist = st.selectbox('选择分析因子', table_df['name'].tolist(), key='factor_dist_name')
    norm_method = st.selectbox('标准化方式', ['zscore', 'rank'], key='factor_dist_norm')
    dist_sample_stocks = st.slider('分布样本股票数', min_value=5, max_value=120, value=40, step=5)
    dist_rows = st.slider('分布每股样本行数', min_value=120, max_value=1500, value=500, step=20)

    if st.button('计算分布并绘图'):
        try:
            dist_df = _compute_factor_distribution(
                source_path,
                feature_set,
                store_path,
                factor_for_dist,
                dist_sample_stocks,
                dist_rows,
                norm_method,
            )
            fig = _plot_distribution(dist_df, factor_for_dist, norm_method)
            st.plotly_chart(fig, width='stretch')
            st.caption(f'样本量: {len(dist_df):,}')
        except Exception as exc:
            st.error(f'分布分析失败: {exc}')
