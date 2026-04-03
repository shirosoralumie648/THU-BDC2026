from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui_console.common import apply_dark_figure_style
from gui_console.common import build_env
from gui_console.common import get_job
from gui_console.common import job_status_text
from gui_console.common import load_effective_config
from gui_console.common import parse_training_log
from gui_console.common import project_path
from gui_console.common import python_cmd
from gui_console.common import read_text_tail
from gui_console.common import render_collapsible_job_panel
from gui_console.common import render_metric_card
from gui_console.common import render_page_hero
from gui_console.common import render_section_header
from gui_console.common import resolve_effective_path
from gui_console.common import start_job
from gui_console.common import stop_job


def _read_log_full(path: str, max_bytes: int = 2_000_000) -> str:
    target = Path(path)
    if not target.exists():
        return ''
    size = target.stat().st_size
    if size <= max_bytes:
        return target.read_text(encoding='utf-8', errors='ignore')

    with open(target, 'rb') as f:
        f.seek(-max_bytes, 2)
        data = f.read(max_bytes)
    return data.decode('utf-8', errors='ignore')


def _plot_metric_curves(metrics_df: pd.DataFrame) -> Optional[go.Figure]:
    if metrics_df.empty or 'epoch' not in metrics_df.columns:
        return None

    fig = go.Figure()

    curve_map = [
        ('train_Loss', 'Train Loss', '#5b8ff9'),
        ('eval_Loss', 'Eval Loss', '#5ad8a6'),
        ('eval_rank_ic_mean', 'RankIC Mean', '#f6bd16'),
        ('eval_rank_ic_ir', 'RankIC IR', '#e8684a'),
        ('strategy_return', '策略收益均值', '#9270ca'),
        ('validation_objective', '验证目标值', '#269a99'),
    ]

    x = metrics_df['epoch']
    for col, label, color in curve_map:
        if col in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=metrics_df[col],
                    mode='lines+markers',
                    name=label,
                    line={'color': color, 'width': 2},
                )
            )

    fig = apply_dark_figure_style(fig, title='训练核心指标', height=430)
    return fig


def _plot_ablation(ablation_df: pd.DataFrame) -> Optional[go.Figure]:
    if ablation_df.empty:
        return None

    latest_epoch = int(ablation_df['epoch'].max())
    latest_df = ablation_df[ablation_df['epoch'] == latest_epoch].sort_values('delta')

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=latest_df['delta'],
            y=latest_df['group'],
            orientation='h',
            marker_color=['#d62728' if val < 0 else '#2ca02c' for val in latest_df['delta']],
            text=[f'{val:.4f}' for val in latest_df['delta']],
            textposition='outside',
        )
    )
    fig = apply_dark_figure_style(fig, title=f'因子消融 Delta (Epoch {latest_epoch})', height=360)
    fig.update_layout(xaxis_title='Return Delta vs Baseline', yaxis_title='Factor Group')
    return fig


def render_training_monitor(config_override_path: Optional[str]) -> None:
    effective_config = load_effective_config(config_override_path)
    render_page_hero(
        'Training Monitor',
        '训练任务、核心指标曲线、早停状态与消融分析指挥台。',
        eyebrow='Experiment Command Center',
    )

    action_cols = st.columns([1.2, 1, 3])
    start_train = action_cols[0].button('启动训练', key='btn_start_train', type='primary')
    stop_train = action_cols[1].button('停止训练', key='btn_stop_train')

    if start_train:
        cmd = python_cmd('code/src/train.py')
        try:
            start_job('train', cmd, cwd=project_path(), env=build_env(config_override_path), replace_existing=True)
            st.success('训练任务已启动。')
        except Exception as exc:
            st.error(f'启动失败: {exc}')

    if stop_train:
        stop_job('train')
        st.warning('已发送停止信号。')

    job = get_job('train')
    if not job:
        render_metric_card('Train Status', '未启动', '先启动训练任务以查看实时监控。')
        return

    log_text = _read_log_full(job['log_path'])
    parsed = parse_training_log(log_text)
    metrics_df = parsed['metrics']
    ablation_df = parsed['ablation']
    early_df = parsed['early_stop']

    latest_epoch = '—'
    best_objective = '—'
    rank_ic_mean = '—'
    if not metrics_df.empty:
        latest_epoch = str(int(metrics_df['epoch'].max()))
        if 'validation_objective' in metrics_df.columns and metrics_df['validation_objective'].notna().any():
            best_objective = f"{metrics_df['validation_objective'].dropna().max():.6f}"
        if 'eval_rank_ic_mean' in metrics_df.columns and metrics_df['eval_rank_ic_mean'].notna().any():
            rank_ic_mean = f"{metrics_df['eval_rank_ic_mean'].dropna().iloc[-1]:.6f}"

    kpi_cols = st.columns([1.15, 1, 1, 1])
    with kpi_cols[0]:
        render_metric_card('任务状态', job_status_text(job), '当前训练状态 / current run state')
    with kpi_cols[1]:
        render_metric_card('Latest Epoch', latest_epoch, '最近解析到的 epoch')
    with kpi_cols[2]:
        render_metric_card('Best Objective', best_objective, '最优验证目标')
    with kpi_cols[3]:
        render_metric_card('RankIC Mean', rank_ic_mean, '最新排名相关性')

    chart_col, side_col = st.columns([1.45, 0.9])
    with chart_col:
        render_section_header('训练主曲线', '损失、目标值与 RankIC 走势。')
        if metrics_df.empty:
            st.info('尚未解析到 Epoch 指标。')
        else:
            curve_fig = _plot_metric_curves(metrics_df)
            if curve_fig is not None:
                st.plotly_chart(curve_fig, width='stretch')

        st.markdown('<div class="terminal-divider"></div>', unsafe_allow_html=True)
        render_section_header('因子消融分析', '按最新 epoch 查看分组因子对收益的边际影响。')
        if ablation_df.empty:
            st.info('当前日志中暂无因子消融记录。')
        else:
            ablation_fig = _plot_ablation(ablation_df)
            if ablation_fig is not None:
                st.plotly_chart(ablation_fig, width='stretch')
            with st.expander('因子消融明细', expanded=False):
                st.dataframe(ablation_df.tail(50), width='stretch')

    with side_col:
        render_section_header('运行观察窗', '状态先读，日志按需展开。')
        render_metric_card('Log Source', Path(job['log_path']).name, 'current log file')
        render_collapsible_job_panel('train', '训练任务日志', log_lines=220, expanded=False)

        render_section_header('早停风险面板', '当前监控指标与停训距离。')
        if not early_df.empty:
            latest_early = early_df.iloc[-1].to_dict()
            bad_epochs = latest_early.get('bad_epochs', 'N/A')
            patience = latest_early.get('patience', 'N/A')
            monitor = latest_early.get('monitor', 'N/A')
            best = latest_early.get('best', None)
            value = latest_early.get('value', None)
            risk_state = '稳定'
            if isinstance(bad_epochs, int) and isinstance(patience, int) and patience:
                if bad_epochs / patience >= 0.7:
                    risk_state = '接近阈值'

            risk_cols = st.columns(2)
            risk_cols[0].metric('Risk State', risk_state)
            risk_cols[1].metric('Bad Epochs', f'{bad_epochs}/{patience}')

            detail_cols = st.columns(2)
            detail_cols[0].metric('Monitor', str(monitor))
            if isinstance(best, float) and isinstance(value, float):
                detail_cols[1].metric('Best vs Current', f'{best:.6f} / {value:.6f}')
            else:
                detail_cols[1].metric('Best vs Current', 'N/A')

            with st.expander('查看早停历史', expanded=False):
                st.dataframe(early_df.tail(30), width='stretch')
        else:
            st.info('日志中暂无早停状态记录。')

    output_dir = Path(resolve_effective_path(str(effective_config.get('output_dir', './model'))))
    tensorboard_log = output_dir / 'log'
    st.caption(f'TensorBoard: {tensorboard_log}')
