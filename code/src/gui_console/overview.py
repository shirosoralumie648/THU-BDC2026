from pathlib import Path

import pandas as pd
import streamlit as st

from gui_console.common import get_job
from gui_console.common import job_status_text
from gui_console.common import load_csv_cached
from gui_console.common import load_effective_config
from gui_console.common import project_path
from gui_console.common import read_json
from gui_console.common import render_metric_card
from gui_console.common import render_page_hero
from gui_console.common import render_section_header
from gui_console.common import resolve_effective_path
from gui_console.common import summarize_jobs

JOB_KEYS = [
    'fetch_data',
    'split_data',
    'train',
    'predict',
    'score_self',
    'uv_sync',
    'docker_build',
    'docker_save',
    'docker_validate',
]


def _safe_load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return load_csv_cached(str(path))
    except Exception:
        return None


def _safe_metric_value(df: pd.DataFrame | None, column: str | None = None) -> str:
    if df is None:
        return '—'
    if column and column in df.columns:
        return f"{df[column].nunique():,}"
    return f'{len(df):,}'


def render_overview(config_override_path: str | None = None) -> None:
    render_page_hero(
        'Overview',
        '研究工作流总览与下一步行动入口。',
        eyebrow='Quant Research Console',
    )

    train_csv = Path(project_path('data', 'train.csv'))
    result_csv = Path(project_path('output', 'result.csv'))
    tmp_score_csv = Path(project_path('temp', 'tmp.csv'))
    effective_config = load_effective_config(config_override_path)
    best_strategy_json = Path(resolve_effective_path(str(effective_config.get('output_dir', './model')))) / 'best_strategy.json'

    train_df = _safe_load_csv(train_csv)
    result_df = _safe_load_csv(result_csv)
    score_df = _safe_load_csv(tmp_score_csv)
    summary = summarize_jobs(JOB_KEYS)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card('Train Rows', _safe_metric_value(train_df), 'data/train.csv')
    with metric_cols[1]:
        render_metric_card('Universe', _safe_metric_value(train_df, '股票代码'), 'unique stocks')
    with metric_cols[2]:
        render_metric_card('Running Jobs', str(summary.get('running', 0)), 'active tasks now')
    with metric_cols[3]:
        render_metric_card('Result Rows', _safe_metric_value(result_df), 'output/result.csv')

    metric_cols_2 = st.columns(4)
    with metric_cols_2[0]:
        render_metric_card('Train Status', job_status_text(get_job('train')), 'model training')
    with metric_cols_2[1]:
        render_metric_card('Predict Status', job_status_text(get_job('predict')), 'inference pipeline')
    with metric_cols_2[2]:
        final_score = '—'
        if score_df is not None and 'Final Score' in score_df.columns and not score_df.empty:
            final_score = f"{float(score_df.iloc[-1]['Final Score']):.6f}"
        render_metric_card('Final Score', final_score, 'temp/tmp.csv')
    with metric_cols_2[3]:
        best_payload = read_json(str(best_strategy_json))
        best_name = best_payload.get('name', '—') if isinstance(best_payload, dict) else '—'
        render_metric_card('Best Strategy', best_name, 'saved strategy')

    render_section_header('Next Recommended Action', '根据当前工件和作业状态推荐下一步。')
    if train_df is None:
        st.warning('下一步：先在 Data Center 准备训练集。')
    elif not best_strategy_json.exists():
        st.warning('下一步：前往 Training Monitor 完成训练并生成最佳策略。')
    elif result_df is None:
        st.warning('下一步：前往 Prediction Desk 运行推理并生成 result.csv。')
    else:
        st.success('核心工件已齐备，可前往 Deployment Desk 做验证。')

    render_section_header('Pipeline Progress', '按关键工件是否存在给出粗略流程进度。')
    progress = 0
    if train_df is not None:
        progress = 45
    if best_strategy_json.exists():
        progress = 75
    if result_df is not None:
        progress = 100
    st.progress(progress)

    left_col, right_col = st.columns([1.15, 1.0])
    with left_col:
        render_section_header('Operational Snapshot', '快速查看任务状态。')
        status_rows = [{'job': job_key, 'status': job_status_text(get_job(job_key))} for job_key in JOB_KEYS]
        st.dataframe(pd.DataFrame(status_rows), width='stretch', hide_index=True)

        render_section_header('Action Shortcuts', '优先跳转到最相关页面。')
        st.markdown('- 数据准备 → **数据中心**\n- 因子与配置 → **因子实验室 / 参数微调**\n- 训练与结果 → **训练监控站 / 回测与预测**\n- 验收交付 → **部署工具箱**')

    with right_col:
        render_section_header('Strategy Snapshot', '当前保存的最优策略。')
        payload = read_json(str(best_strategy_json))
        if payload:
            st.json(payload)
        else:
            st.info('未检测到 best_strategy.json。')

        render_section_header('Artifacts', '关键输出文件存在情况。')
        artifact_rows = pd.DataFrame(
            [
                {'artifact': 'data/train.csv', 'exists': train_csv.exists()},
                {'artifact': 'output/result.csv', 'exists': result_csv.exists()},
                {'artifact': 'temp/tmp.csv', 'exists': tmp_score_csv.exists()},
                {'artifact': 'model/best_strategy.json', 'exists': best_strategy_json.exists()},
            ]
        )
        st.dataframe(artifact_rows, width='stretch', hide_index=True)
