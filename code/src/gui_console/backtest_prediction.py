from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui_console.common import build_env
from gui_console.common import format_cmd
from gui_console.common import get_job
from gui_console.common import job_status_text
from gui_console.common import load_csv_cached
from gui_console.common import load_effective_config
from gui_console.common import project_path
from gui_console.common import python_cmd
from gui_console.common import read_json
from gui_console.common import read_text_tail
from gui_console.common import resolve_effective_path
from gui_console.common import start_job
from gui_console.common import stop_job


def _normalize_code(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.split('.').str[-1]
    s = s.str.replace(r'[^0-9]', '', regex=True)
    s = s.str[-6:].str.zfill(6)
    return s


def _render_job(job_key: str, title: str) -> None:
    job = get_job(job_key)
    st.caption(f'{title} 状态: {job_status_text(job)}')
    if not job:
        return
    st.code(format_cmd(job['cmd']), language='bash')
    st.caption(f"日志: {job['log_path']}")
    st.text_area(f'{title} 日志 (tail)', value=read_text_tail(job['log_path'], 150), height=180, key=f'{job_key}_log')


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
    fig.update_layout(
        title='策略模拟收益看板',
        template='plotly_white',
        margin={'l': 20, 'r': 20, 't': 30, 'b': 20},
        xaxis_title='Strategy',
        yaxis_title='Weighted Return',
        height=380,
    )
    return fig


def render_backtest_prediction(config_override_path: Optional[str]) -> None:
    st.subheader('策略回测与预测 (Backtest & Prediction)')
    effective_config = load_effective_config(config_override_path)

    col_pred, col_stop, col_score = st.columns(3)

    if col_pred.button('一键推理'):
        cmd = python_cmd('code/src/predict.py')
        try:
            start_job('predict', cmd, cwd=project_path(), env=build_env(config_override_path), replace_existing=True)
            st.success('预测任务已启动。')
        except Exception as exc:
            st.error(f'启动失败: {exc}')

    if col_stop.button('停止推理'):
        stop_job('predict')
        st.warning('已发送停止信号。')

    if col_score.button('运行 score_self.py'):
        cmd = python_cmd('test/score_self.py')
        try:
            start_job('score_self', cmd, cwd=project_path(), env=build_env(config_override_path), replace_existing=True)
            st.success('评分任务已启动。')
        except Exception as exc:
            st.error(f'启动失败: {exc}')

    _render_job('predict', '推理任务')
    _render_job('score_self', '评分任务')

    st.markdown('### 推理结果')
    result_path = Path(project_path('output', 'result.csv'))
    if result_path.exists():
        result_df = load_csv_cached(str(result_path))
        st.dataframe(result_df, width='stretch')
    else:
        st.info('尚未生成 output/result.csv。')

    st.markdown('### 得分看板')
    tmp_score_path = Path(project_path('temp', 'tmp.csv'))
    if tmp_score_path.exists():
        score_df = load_csv_cached(str(tmp_score_path))
        st.dataframe(score_df, width='stretch')
        if 'Final Score' in score_df.columns and not score_df.empty:
            st.metric('参考总分', f"{float(score_df.iloc[-1]['Final Score']):.6f}")
    else:
        st.info('尚未检测到 temp/tmp.csv。')

    st.markdown('### 策略模拟')
    scores_path = Path(resolve_effective_path(str(effective_config.get('prediction_scores_path', './output/prediction_scores.csv'))))
    test_path = Path(resolve_effective_path(str(effective_config.get('data_path', './data')))) / 'test.csv'
    temperature = st.number_input(
        'Softmax 温度',
        min_value=0.1,
        max_value=10.0,
        value=float(effective_config.get('softmax_temperature', 1.0)),
        step=0.1,
    )

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
