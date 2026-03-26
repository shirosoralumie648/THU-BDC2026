from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui_console.common import build_env
from gui_console.common import format_cmd
from gui_console.common import get_job
from gui_console.common import job_status_text
from gui_console.common import load_effective_config
from gui_console.common import parse_training_log
from gui_console.common import project_path
from gui_console.common import python_cmd
from gui_console.common import read_text_tail
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

    fig.update_layout(
        title='训练过程核心指标',
        template='plotly_white',
        margin={'l': 20, 'r': 20, 't': 30, 'b': 20},
        xaxis_title='Epoch',
        yaxis_title='Value',
        height=430,
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1.0},
    )
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
    fig.update_layout(
        title=f'因子消融 Delta (Epoch {latest_epoch})',
        template='plotly_white',
        margin={'l': 20, 'r': 20, 't': 30, 'b': 20},
        xaxis_title='Return Delta vs Baseline',
        yaxis_title='Factor Group',
        height=360,
    )
    return fig


def render_training_monitor(config_override_path: Optional[str]) -> None:
    st.subheader('训练监控站 (Training Monitor)')
    effective_config = load_effective_config(config_override_path)

    col_left, col_right = st.columns(2)
    with col_left:
        start_train = st.button('启动训练', key='btn_start_train')
    with col_right:
        stop_train = st.button('停止训练', key='btn_stop_train')

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
    st.caption(f'任务状态: {job_status_text(job)}')
    if not job:
        st.info('尚未启动训练任务。')
        return

    st.code(format_cmd(job['cmd']), language='bash')
    st.caption(f"日志文件: {job['log_path']}")

    tail_log = read_text_tail(job['log_path'], max_lines=180)
    st.text_area('训练日志 (tail)', value=tail_log, height=240, key='train_tail_log')

    log_text = _read_log_full(job['log_path'])
    parsed = parse_training_log(log_text)
    metrics_df = parsed['metrics']
    ablation_df = parsed['ablation']
    early_df = parsed['early_stop']

    if metrics_df.empty:
        st.info('尚未解析到 Epoch 指标。')
        return

    st.markdown('### 实时指标曲线')
    curve_fig = _plot_metric_curves(metrics_df)
    if curve_fig is not None:
        st.plotly_chart(curve_fig, width='stretch')

    st.markdown('### 早停状态')
    if not early_df.empty:
        latest_early = early_df.iloc[-1].to_dict()
        bad_epochs = latest_early.get('bad_epochs', 'N/A')
        patience = latest_early.get('patience', 'N/A')
        monitor = latest_early.get('monitor', 'N/A')
        best = latest_early.get('best', None)
        value = latest_early.get('value', None)

        cols = st.columns(4)
        cols[0].metric('monitor', str(monitor))
        cols[1].metric('bad_epochs', f'{bad_epochs}/{patience}')
        cols[2].metric('best', f'{best:.6f}' if isinstance(best, float) else str(best))
        cols[3].metric('current', f'{value:.6f}' if isinstance(value, float) else str(value))

        st.dataframe(early_df.tail(30), width='stretch')
    else:
        st.info('日志中暂无早停状态记录。')

    st.markdown('### 因子消融实验视图')
    if ablation_df.empty:
        st.info('当前日志中暂无因子消融记录。')
    else:
        ablation_fig = _plot_ablation(ablation_df)
        if ablation_fig is not None:
            st.plotly_chart(ablation_fig, width='stretch')
        st.dataframe(ablation_df.tail(50), width='stretch')

    output_dir = Path(resolve_effective_path(str(effective_config.get('output_dir', './model'))))
    tensorboard_log = output_dir / 'log'
    st.caption(f'TensorBoard 日志目录: {tensorboard_log}')
