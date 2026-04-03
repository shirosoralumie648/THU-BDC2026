import streamlit as st

from gui_console.backtest_prediction import render_backtest_prediction
from gui_console.common import any_job_running
from gui_console.common import ensure_gui_dirs
from gui_console.common import ensure_gui_state
from gui_console.common import get_job
from gui_console.common import inject_terminal_theme
from gui_console.common import job_status_text
from gui_console.common import render_metric_card
from gui_console.common import render_page_hero
from gui_console.common import summarize_jobs
from gui_console.data_center import render_data_center
from gui_console.deployment import render_deployment
from gui_console.factor_lab import render_factor_lab
from gui_console.hyperparams import DEFAULT_OVERRIDE_PATH
from gui_console.hyperparams import render_hyperparams
from gui_console.overview import render_overview
from gui_console.training_monitor import render_training_monitor


st.set_page_config(
    page_title='THU-BDC2026 控制台',
    page_icon='📈',
    layout='wide',
)

inject_terminal_theme()

ensure_gui_dirs()
ensure_gui_state()

if 'gui_override_path' not in st.session_state:
    st.session_state['gui_override_path'] = DEFAULT_OVERRIDE_PATH

job_keys = ['fetch_data', 'split_data', 'train', 'predict', 'score_self', 'uv_sync', 'docker_build', 'docker_save', 'docker_validate']
job_summary = summarize_jobs(job_keys)
override_path = st.session_state.get('gui_override_path', DEFAULT_OVERRIDE_PATH)

sidebar = st.sidebar
sidebar.markdown('## Quant Terminal')
sidebar.caption('Research workflow / execution / validation')
sidebar.markdown('---')
sidebar.caption(f'Override: {override_path}')

pages = {
    '0. Overview': 'overview',
    '1. 数据中心': 'data',
    '2. 因子实验室': 'factor',
    '3. 参数微调': 'hyper',
    '4. 训练监控站': 'train',
    '5. 回测与预测': 'predict',
    '6. 部署工具箱': 'deploy',
}

selected_page = sidebar.radio('功能模块', list(pages.keys()))

sidebar.markdown('---')
sidebar.markdown('### System State')
status_cols = sidebar.columns(2)
status_cols[0].caption(f"运行 {job_summary.get('running', 0)}")
status_cols[1].caption(f"失败 {job_summary.get('failed', 0)}")
sidebar.caption(f"完成 {job_summary.get('success', 0)}")
sidebar.markdown('---')
for key in ['train', 'predict', 'docker_validate']:
    job = get_job(key)
    sidebar.caption(f'{key}: {job_status_text(job)}')

auto_refresh = sidebar.checkbox('运行中自动刷新(5s)', value=True)
if auto_refresh and any_job_running():
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=5000, limit=None, key='gui_auto_refresh')
    except Exception:
        sidebar.info('未安装 streamlit-autorefresh，自动刷新不可用。')

render_page_hero(
    'THU-BDC2026 Quant Research Terminal',
    '专业量化研究工作台：数据、因子、训练、预测与部署验证统一入口。',
    eyebrow='Streamlit Control Console',
)

hero_cols = st.columns(4)
with hero_cols[0]:
    render_metric_card('Running Jobs', str(job_summary.get('running', 0)), 'active now')
with hero_cols[1]:
    render_metric_card('Completed Jobs', str(job_summary.get('success', 0)), 'successful runs')
with hero_cols[2]:
    render_metric_card('Failed Jobs', str(job_summary.get('failed', 0)), 'attention required')
with hero_cols[3]:
    render_metric_card('Auto Refresh', 'ON' if auto_refresh else 'OFF', '5s when active')

if pages[selected_page] == 'overview':
    render_overview(override_path)
elif pages[selected_page] == 'data':
    render_data_center(override_path)
elif pages[selected_page] == 'factor':
    render_factor_lab()
elif pages[selected_page] == 'hyper':
    override_path = render_hyperparams() or override_path
    st.session_state['gui_override_path'] = override_path
elif pages[selected_page] == 'train':
    render_training_monitor(override_path)
elif pages[selected_page] == 'predict':
    render_backtest_prediction(override_path)
elif pages[selected_page] == 'deploy':
    render_deployment(override_path)
