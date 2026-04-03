from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui_console.common import apply_dark_figure_style
from gui_console.common import build_env
from gui_console.common import get_job
from gui_console.common import job_status_text
from gui_console.common import load_csv_cached
from gui_console.common import load_effective_config
from gui_console.common import project_path
from gui_console.common import python_cmd
from gui_console.common import read_json
from gui_console.common import render_collapsible_job_panel
from gui_console.common import render_job_panel
from gui_console.common import render_metric_card
from gui_console.common import render_page_hero
from gui_console.common import render_section_header
from gui_console.common import resolve_effective_path
from gui_console.common import start_job
from gui_console.common import stop_job


def _normalize_code(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.split('.').str[-1]
    s = s.str.replace(r'[^0-9]', '', regex=True)
    s = s.str[-6:].str.zfill(6)
    return s


def _safe_read_dataframe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return load_csv_cached(str(path))
    except Exception:
        return None


def _compute_stock_returns(test_df: pd.DataFrame) -> pd.DataFrame:
    data = test_df.copy()
    data['股票代码'] = _normalize_code(data['股票代码'])
    data['日期'] = pd.to_datetime(data['日期'], errors='coerce')
    data['开盘'] = pd.to_numeric(data['开盘'], errors='coerce')
    data = data.dropna(subset=['日期', '开盘'])

    tail = data.sort_values(['股票代码', '日期']).groupby('股票代码', group_keys=False).tail(5)

    rows = []
    for stock, g in tail.groupby('股票代码'):
        g = g.sort_values('日期')
        if len(g) < 2:
            continue
        start_open = float(g.iloc[0]['开盘'])
        end_open = float(g.iloc[-1]['开盘'])
        if abs(start_open) < 1e-12:
            continue
        rows.append({'stock_id': stock, 'return': (end_open - start_open) / start_open})
    return pd.DataFrame(rows)


def _simulate_strategies(scores_df: pd.DataFrame, returns_df: pd.DataFrame, temperature: float = 1.0) -> pd.DataFrame:
    if scores_df.empty:
        return pd.DataFrame()

    scores = scores_df.copy()
    if 'stock_id' not in scores.columns or 'score' not in scores.columns:
        return pd.DataFrame()

    scores['stock_id'] = _normalize_code(scores['stock_id'])
    scores['score'] = pd.to_numeric(scores['score'], errors='coerce')
    scores = scores.dropna(subset=['score'])
    scores = scores.sort_values('score', ascending=False).reset_index(drop=True)

    rows = []
    for top_k in [1, 2, 3, 4, 5]:
        top = scores.head(top_k).copy()
        if top.empty:
            continue

        for weighting in ['equal', 'softmax']:
            weights = None
            if weighting == 'equal' or len(top) == 1:
                weights = np.full(len(top), 1.0 / len(top), dtype=np.float64)
            else:
                stable = top['score'].to_numpy() - top['score'].max()
                scaled = stable / max(float(temperature), 1e-6)
                exp_vals = np.exp(scaled)
                weights = exp_vals / exp_vals.sum()

            sim = top[['stock_id', 'score']].copy()
            sim['weight'] = weights
            merged = sim.merge(returns_df, on='stock_id', how='left')
            covered = int(merged['return'].notna().sum())
            merged['return'] = merged['return'].fillna(0.0)
            pnl = float((merged['weight'] * merged['return']).sum())

            rows.append(
                {
                    'strategy': f'top{top_k}_{weighting}',
                    'top_k': top_k,
                    'weighting': weighting,
                    'expected_return': pnl,
                    'covered_stocks': covered,
                }
            )

    out = pd.DataFrame(rows).sort_values('expected_return', ascending=False).reset_index(drop=True)
    return out


def _plot_strategy_board(df: pd.DataFrame) -> Optional[go.Figure]:
    if df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df['strategy'],
            y=df['expected_return'],
            marker_color=['#2ca02c' if val >= 0 else '#d62728' for val in df['expected_return']],
            text=[f'{val:.4%}' for val in df['expected_return']],
            textposition='outside',
        )
    )
    fig = apply_dark_figure_style(fig, title='策略模拟收益看板', height=380)
    fig.update_layout(xaxis_title='Strategy', yaxis_title='Weighted Return')
    return fig


def render_backtest_prediction(config_override_path: Optional[str]) -> None:
    effective_config = load_effective_config(config_override_path)
    render_page_hero(
        'Prediction Desk',
        '推理执行、结果持仓、评分看板与策略模拟统一工作台。',
        eyebrow='Portfolio Output',
    )

    result_path = Path(project_path('output', 'result.csv'))
    tmp_score_path = Path(project_path('temp', 'tmp.csv'))
    score_df = _safe_read_dataframe(tmp_score_path)
    result_df = _safe_read_dataframe(result_path)

    summary_cols = st.columns(4)
    with summary_cols[0]:
        render_metric_card('Predict Status', job_status_text(get_job('predict')), '推理任务')
    with summary_cols[1]:
        render_metric_card('Score Status', job_status_text(get_job('score_self')), '评分任务')
    with summary_cols[2]:
        final_score = '—'
        if score_df is not None and 'Final Score' in score_df.columns and not score_df.empty:
            final_score = f"{float(score_df.iloc[-1]['Final Score']):.6f}"
        render_metric_card('Final Score', final_score, 'temp/tmp.csv')
    with summary_cols[3]:
        render_metric_card('Holdings', str(len(result_df)) if result_df is not None else '—', 'output/result.csv')

    control_col, result_col = st.columns([0.95, 1.25])
    with control_col:
        render_section_header('推理控制台', '启动推理、停止任务或运行评分脚本。')
        action_cols = st.columns(3)
        if action_cols[0].button('一键推理', type='primary'):
            cmd = python_cmd('code/src/predict.py')
            try:
                start_job('predict', cmd, cwd=project_path(), env=build_env(config_override_path), replace_existing=True)
                st.success('预测任务已启动。')
            except Exception as exc:
                st.error(f'启动失败: {exc}')

        if action_cols[1].button('停止推理'):
            stop_job('predict')
            st.warning('已发送停止信号。')

        if action_cols[2].button('运行 score_self.py'):
            cmd = python_cmd('test/score_self.py')
            try:
                start_job('score_self', cmd, cwd=project_path(), env=build_env(config_override_path), replace_existing=True)
                st.success('评分任务已启动。')
            except Exception as exc:
                st.error(f'启动失败: {exc}')

        render_job_panel('predict', '推理任务', log_lines=150)
        render_collapsible_job_panel('score_self', '评分任务日志', log_lines=150, expanded=False)

    with result_col:
        render_section_header('结果持仓', '当前 result.csv 输出与持仓规模。')
        if result_df is not None:
            st.dataframe(result_df, width='stretch')
        else:
            st.info('尚未生成 output/result.csv。')

        render_section_header('得分看板', 'score_self 输出的参考分数。')
        if score_df is not None:
            st.dataframe(score_df, width='stretch')
        else:
            st.info('尚未检测到 temp/tmp.csv。')

        render_section_header('策略模拟', '根据 prediction_scores.csv 回放不同 top-k 策略。')
        scores_path = Path(resolve_effective_path(str(effective_config.get('prediction_scores_path', './output/prediction_scores.csv'))))
        test_path = Path(resolve_effective_path(str(effective_config.get('data_path', './data')))) / 'test.csv'
        temperature = st.number_input('Softmax 温度', min_value=0.1, max_value=10.0, value=float(effective_config.get('softmax_temperature', 1.0)), step=0.1)

        if not scores_path.exists():
            st.warning('缺少 output/prediction_scores.csv，请先执行新版推理。')
            return
        if not test_path.exists():
            st.warning(f'缺少测试集文件: {test_path}')
            return

        try:
            scores_df = load_csv_cached(str(scores_path))
            test_df = load_csv_cached(str(test_path))
            returns_df = _compute_stock_returns(test_df)
            sim_df = _simulate_strategies(scores_df, returns_df, temperature=temperature)

            if sim_df.empty:
                st.info('策略模拟结果为空。')
            else:
                fig = _plot_strategy_board(sim_df)
                if fig is not None:
                    st.plotly_chart(fig, width='stretch')
                st.dataframe(sim_df, width='stretch')

            best_strategy_path = Path(resolve_effective_path(str(effective_config.get('output_dir', './model')))) / 'best_strategy.json'
            best_strategy = read_json(str(best_strategy_path))
            if best_strategy:
                st.caption('当前训练保存的最佳策略配置')
                st.json(best_strategy)
        except Exception as exc:
            st.error(f'策略模拟失败: {exc}')
