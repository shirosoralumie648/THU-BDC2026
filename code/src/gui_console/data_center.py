from datetime import date
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_manager import resolve_data_root
from data_manager import resolve_dataset_path
from gui_console.common import apply_dark_figure_style
from gui_console.common import build_env
from gui_console.common import get_job
from gui_console.common import job_status_text
from gui_console.common import load_csv_cached
from gui_console.common import load_effective_config
from gui_console.common import normalize_stock_code
from gui_console.common import project_path
from gui_console.common import python_cmd
from gui_console.common import render_job_panel
from gui_console.common import render_collapsible_job_panel
from gui_console.common import render_metric_card
from gui_console.common import render_page_hero
from gui_console.common import render_section_header
from gui_console.common import resolve_effective_path
from gui_console.common import start_job
from gui_console.common import stop_job


def _safe_preview_df(preview_path: str) -> pd.DataFrame | None:
    if not Path(preview_path).exists():
        return None
    try:
        return load_csv_cached(preview_path)
    except Exception:
        return None


def _safe_datetime_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors='coerce')
    return parsed


def _resolve_preview_source(choice: str, custom_path: str, effective_config: dict) -> str:
    source_map = {
        'stock_data.csv': resolve_dataset_path(effective_config, 'stock_data.csv'),
        'train.csv': resolve_dataset_path(effective_config, 'train.csv'),
        'test.csv': resolve_dataset_path(effective_config, 'test.csv'),
        '自定义': custom_path,
    }
    return source_map.get(choice, source_map['stock_data.csv'])


def _build_kline_figure(stock_df: pd.DataFrame) -> Optional[go.Figure]:
    required_cols = {'日期', '开盘', '最高', '最低', '收盘'}
    if not required_cols.issubset(stock_df.columns):
        return None

    plot_df = stock_df.copy()
    plot_df['日期_dt'] = pd.to_datetime(plot_df['日期'], errors='coerce')
    plot_df = plot_df.dropna(subset=['日期_dt']).sort_values('日期_dt')
    if plot_df.empty:
        return None

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        specs=[[{'secondary_y': True}], [{'secondary_y': True}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=plot_df['日期_dt'],
            open=plot_df['开盘'].astype(float),
            high=plot_df['最高'].astype(float),
            low=plot_df['最低'].astype(float),
            close=plot_df['收盘'].astype(float),
            name='K线',
            increasing_line_color='#d62728',
            decreasing_line_color='#2ca02c',
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    if '成交量' in plot_df.columns:
        fig.add_trace(
            go.Bar(
                x=plot_df['日期_dt'],
                y=pd.to_numeric(plot_df['成交量'], errors='coerce').fillna(0.0),
                name='成交量',
                marker_color='#5b8ff9',
                opacity=0.75,
            ),
            row=2,
            col=1,
            secondary_y=False,
        )

    if '换手率' in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df['日期_dt'],
                y=pd.to_numeric(plot_df['换手率'], errors='coerce').fillna(0.0),
                name='换手率',
                line={'color': '#f6bd16', 'width': 1.6},
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

    fig = apply_dark_figure_style(fig, height=620)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text='价格', row=1, col=1)
    fig.update_yaxes(title_text='成交量', row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text='换手率', row=2, col=1, secondary_y=True)
    return fig


def _derive_split_dates(source_path: str, mode: str, ratio: float, last_n_days: int):
    df = load_csv_cached(source_path)
    if '日期' not in df.columns:
        raise ValueError('输入数据缺少 日期 列，无法自动推导切分日期。')

    dates = pd.to_datetime(df['日期'], errors='coerce').dropna().dt.normalize().sort_values().unique()
    dates = list(dates)
    if len(dates) < 10:
        raise ValueError('交易日数量过少，无法自动切分。')

    if mode == 'ratio':
        split_idx = int(len(dates) * ratio)
        split_idx = min(max(split_idx, 1), len(dates) - 1)
        train_start = dates[0]
        train_end = dates[split_idx - 1]
        test_start = dates[split_idx]
        test_end = dates[-1]
        return train_start, train_end, test_start, test_end

    n = min(max(1, int(last_n_days)), len(dates) - 1)
    train_start = dates[0]
    train_end = dates[-n - 1]
    test_start = dates[-n]
    test_end = dates[-1]
    return train_start, train_end, test_start, test_end


def render_data_center(config_override_path: Optional[str]) -> None:
    effective_config = load_effective_config(config_override_path)
    render_page_hero(
        'Data Center',
        '数据抓取、切分、预览与行情检查统一工作台。',
        eyebrow='Data Operations',
    )

    default_preview_path = resolve_effective_path(resolve_dataset_path(effective_config, 'stock_data.csv'))
    preview_choice = st.session_state.get('preview_source_choice', 'stock_data.csv')
    preview_custom = st.session_state.get('preview_custom_path', resolve_dataset_path(effective_config, 'stock_data.csv'))
    preview_path = resolve_effective_path(_resolve_preview_source(preview_choice, preview_custom, effective_config))
    preview_df = _safe_preview_df(preview_path)

    preview_rows = '—'
    preview_universe = '—'
    preview_start = '—'
    preview_end = '—'
    if preview_df is not None and not preview_df.empty:
        preview_rows = f'{len(preview_df):,}'
        if '股票代码' in preview_df.columns:
            preview_universe = str(preview_df['股票代码'].nunique())
        if '日期' in preview_df.columns:
            dt_series = _safe_datetime_series(preview_df['日期'])
            if dt_series.notna().any():
                preview_start = str(dt_series.min().date())
                preview_end = str(dt_series.max().date())

    summary_cols = st.columns(4)
    with summary_cols[0]:
        render_metric_card('Preview Rows', preview_rows, '当前预览源总行数')
    with summary_cols[1]:
        render_metric_card('Universe', preview_universe, '股票覆盖数')
    with summary_cols[2]:
        render_metric_card('Fetch Job', job_status_text(get_job('fetch_data')), '抓取任务状态')
    with summary_cols[3]:
        render_metric_card('Split Job', job_status_text(get_job('split_data')), '切分任务状态')

    summary_cols_2 = st.columns(2)
    with summary_cols_2[0]:
        render_metric_card('Start Date', preview_start, preview_choice)
    with summary_cols_2[1]:
        render_metric_card('End Date', preview_end, preview_choice)

    control_col, insight_col = st.columns([1.0, 1.25])

    with control_col:
        render_section_header('数据抓取控制台', '执行行情抓取并跟踪运行日志。')
        today = date.today()
        default_start = today - timedelta(days=365 * 5)
        start_date = st.date_input('开始日期', value=default_start, key='fetch_start_date')
        end_date = st.date_input('结束日期', value=today, key='fetch_end_date')
        index_date = st.date_input('成分股快照日期', value=end_date, key='fetch_index_date')
        output_path = st.text_input('输出文件路径', value=resolve_dataset_path(effective_config, 'stock_data.csv'), key='fetch_output_path')
        manifest_path = st.text_input(
            '抓取清单路径',
            value=str(Path(resolve_data_root(effective_config)) / 'data_manifest_stock_fetch.json'),
            key='fetch_manifest_path',
        )
        freq = st.selectbox('K 线频率', options=['d', 'w', 'm', '60', '30', '15', '5'], index=0, key='fetch_frequency')
        adjustflag_label = st.selectbox('复权方式', options=['后复权(1)', '前复权(2)', '不复权(3)'], index=0, key='fetch_adjustflag')
        adjustflag = {'后复权(1)': '1', '前复权(2)': '2', '不复权(3)': '3'}[adjustflag_label]

        with st.expander('高级抓取参数', expanded=False):
            max_retries = st.number_input('最大重试次数', min_value=1, max_value=12, value=3, step=1, key='fetch_max_retries')
            retry_backoff = st.number_input('重试退避秒数', min_value=0.0, max_value=10.0, value=1.2, step=0.1, key='fetch_retry_backoff')
            request_interval = st.number_input('请求间隔秒数', min_value=0.0, max_value=5.0, value=0.05, step=0.01, key='fetch_request_interval')
            limit_stocks = st.number_input('仅抓取前 N 只（0=全量）', min_value=0, max_value=300, value=0, step=1, key='fetch_limit_stocks')
            rebuild = st.checkbox('全量重建（忽略已有数据）', value=False, key='fetch_rebuild')
            keep_suspended = st.checkbox('保留停牌记录 (tradestatus != 1)', value=False, key='fetch_keep_suspended')

        action_cols = st.columns(2)
        run_fetch = action_cols[0].button('开始抓取', key='btn_run_fetch')
        stop_fetch = action_cols[1].button('停止抓取', key='btn_stop_fetch')

        if run_fetch:
            cmd = python_cmd(
                'get_stock_data.py',
                [
                    '--start-date', str(start_date),
                    '--end-date', str(end_date),
                    '--index-date', str(index_date),
                    '--output-path', output_path,
                    '--manifest-path', manifest_path,
                    '--frequency', str(freq),
                    '--adjustflag', str(adjustflag),
                    '--max-retries', str(int(max_retries)),
                    '--retry-backoff-seconds', str(float(retry_backoff)),
                    '--request-interval-seconds', str(float(request_interval)),
                    '--limit-stocks', str(int(limit_stocks)),
                ],
            )
            if rebuild:
                cmd.append('--rebuild')
            if keep_suspended:
                cmd.append('--keep-suspended')
            try:
                start_job('fetch_data', cmd, cwd=project_path(), env=build_env(config_override_path), replace_existing=True)
                st.success('已启动数据抓取任务。')
            except Exception as exc:
                st.error(f'启动失败: {exc}')

        if stop_fetch:
            stop_job('fetch_data')
            st.warning('已请求停止数据抓取任务。')

        render_collapsible_job_panel('fetch_data', '抓取任务日志', log_lines=160, expanded=False)

        st.markdown('<div class="terminal-divider"></div>', unsafe_allow_html=True)
        render_section_header('数据集切分控制台', '组织训练/测试集区间并生成切分清单。')
        source_path = st.text_input('原始数据文件', value=resolve_dataset_path(effective_config, 'stock_data.csv'), key='split_source_path')
        output_dir = st.text_input('输出目录', value=resolve_data_root(effective_config), key='split_output_dir')
        manifest_path = st.text_input(
            '切分清单路径',
            value=str(Path(resolve_data_root(effective_config)) / 'data_manifest_split.json'),
            key='split_manifest_path',
        )

        split_mode = st.radio('切分方式', options=['按比例', '按最后 N 个交易日', '手动日期'], horizontal=True, key='split_mode')
        train_start = train_end = test_start = test_end = None
        try:
            if split_mode == '按比例':
                ratio = st.slider('训练集比例', min_value=0.5, max_value=0.95, value=0.85, step=0.01, key='split_ratio')
                train_start, train_end, test_start, test_end = _derive_split_dates(source_path, 'ratio', ratio, 5)
            elif split_mode == '按最后 N 个交易日':
                last_n = st.slider('测试集天数', min_value=1, max_value=30, value=5, step=1, key='split_last_n')
                train_start, train_end, test_start, test_end = _derive_split_dates(source_path, 'last_n', 0.85, last_n)
            else:
                train_start = st.date_input('训练开始', value=date(2015, 1, 1), key='manual_train_start')
                train_end = st.date_input('训练结束', value=date(2026, 3, 6), key='manual_train_end')
                test_start = st.date_input('测试开始', value=date(2026, 3, 9), key='manual_test_start')
                test_end = st.date_input('测试结束', value=date(2026, 3, 13), key='manual_test_end')

            st.caption(
                f'训练区间: {pd.to_datetime(train_start).date()} ~ {pd.to_datetime(train_end).date()} | '
                f'测试区间: {pd.to_datetime(test_start).date()} ~ {pd.to_datetime(test_end).date()}'
            )
        except Exception as exc:
            st.error(f'自动推导切分日期失败: {exc}')

        split_action_cols = st.columns(2)
        run_split = split_action_cols[0].button('执行切分', key='btn_run_split')
        stop_split = split_action_cols[1].button('停止切分', key='btn_stop_split')

        if run_split and train_start is not None:
            cmd = python_cmd(
                'data/split_train_test.py',
                [
                    '--input', source_path,
                    '--output-dir', output_dir,
                    '--manifest-path', manifest_path,
                    '--train-start', str(pd.to_datetime(train_start).date()),
                    '--train-end', str(pd.to_datetime(train_end).date()),
                    '--test-start', str(pd.to_datetime(test_start).date()),
                    '--test-end', str(pd.to_datetime(test_end).date()),
                ],
            )
            try:
                start_job('split_data', cmd, cwd=project_path(), env=build_env(config_override_path), replace_existing=True)
                st.success('已启动切分任务。')
            except Exception as exc:
                st.error(f'启动失败: {exc}')

        if stop_split:
            stop_job('split_data')
            st.warning('已请求停止切分任务。')

        render_collapsible_job_panel('split_data', '切分任务日志', log_lines=160, expanded=False)

    with insight_col:
        render_section_header('数据预览与行情视图', '选择数据源、过滤股票并查看 K 线与交易特征。')
        preview_col1, preview_col2 = st.columns([1, 1])
        with preview_col1:
            source_choice = st.selectbox('预览数据源', ['stock_data.csv', 'train.csv', 'test.csv', '自定义'], key='preview_source_choice')
        with preview_col2:
            custom_path = st.text_input('自定义路径', value=default_preview_path, key='preview_custom_path')

        preview_path = resolve_effective_path(_resolve_preview_source(source_choice, custom_path, effective_config))
        if not Path(preview_path).exists():
            st.warning(f'文件不存在: {preview_path}')
            return

        try:
            preview_df = load_csv_cached(preview_path)
        except Exception as exc:
            st.error(f'读取失败: {exc}')
            return

        if preview_df.empty:
            st.warning('数据为空。')
            return

        if '日期' in preview_df.columns:
            preview_df['日期_dt'] = _safe_datetime_series(preview_df['日期'])

        info_cols = st.columns(4)
        info_cols[0].metric('总行数', f'{len(preview_df):,}')
        info_cols[1].metric('股票数', str(preview_df['股票代码'].nunique()) if '股票代码' in preview_df.columns else 'N/A')
        if '日期_dt' in preview_df.columns and preview_df['日期_dt'].notna().any():
            info_cols[2].metric('开始日期', str(preview_df['日期_dt'].min().date()))
            info_cols[3].metric('结束日期', str(preview_df['日期_dt'].max().date()))
        else:
            info_cols[2].metric('开始日期', 'N/A')
            info_cols[3].metric('结束日期', 'N/A')

        if '股票代码' not in preview_df.columns:
            st.dataframe(preview_df.head(200), width='stretch')
            return

        preview_df['股票代码_norm'] = normalize_stock_code(preview_df['股票代码'])
        stock_options = sorted(preview_df['股票代码_norm'].dropna().unique().tolist())
        selected_stock = st.selectbox('股票代码', stock_options, key='preview_stock')

        stock_df = preview_df[preview_df['股票代码_norm'] == selected_stock].copy()
        if '日期_dt' in stock_df.columns and stock_df['日期_dt'].notna().any():
            min_date = stock_df['日期_dt'].min().date()
            max_date = stock_df['日期_dt'].max().date()
            start_dt, end_dt = st.date_input('日期区间', value=(min_date, max_date), key='preview_range')
            stock_df = stock_df[(stock_df['日期_dt'] >= pd.to_datetime(start_dt)) & (stock_df['日期_dt'] <= pd.to_datetime(end_dt))]

        show_cols = [col for col in ['股票代码', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率', '涨跌幅'] if col in stock_df.columns]
        st.dataframe(stock_df[show_cols].sort_values('日期').tail(300), width='stretch', height=320)

        fig = _build_kline_figure(stock_df)
        if fig is None:
            st.info('当前数据缺少 K 线所需列（日期/开高低收）。')
        else:
            st.plotly_chart(fig, width='stretch')
