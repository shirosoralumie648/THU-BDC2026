import streamlit as st

from gui_console.backtest_prediction import render_backtest_prediction
from gui_console.common import any_job_running
from gui_console.common import ensure_gui_dirs
from gui_console.common import ensure_gui_state
from gui_console.common import get_job
from gui_console.common import job_status_text
from gui_console.data_center import render_data_center
from gui_console.deployment import render_deployment
from gui_console.factor_lab import render_factor_lab
from gui_console.hyperparams import DEFAULT_OVERRIDE_PATH
from gui_console.hyperparams import render_hyperparams
from gui_console.training_monitor import render_training_monitor


st.set_page_config(
    page_title='THU-BDC2026 控制台',
    page_icon='📈',
    layout='wide',
)

ensure_gui_dirs()
ensure_gui_state()

if 'gui_override_path' not in st.session_state:
    st.session_state['gui_override_path'] = DEFAULT_OVERRIDE_PATH

st.title('THU-BDC2026 量化训练与部署控制台')
st.caption('数据管理 / 因子实验 / 参数调优 / 训练监控 / 回测预测 / 部署验证')

sidebar = st.sidebar
sidebar.header('导航')

pages = {
    '1. 数据中心': 'data',
    '2. 因子实验室': 'factor',
    '3. 参数微调': 'hyper',
    '4. 训练监控站': 'train',
    '5. 回测与预测': 'predict',
    '6. 部署工具箱': 'deploy',
}

selected_page = sidebar.radio('功能模块', list(pages.keys()))

sidebar.markdown('---')
sidebar.subheader('任务状态')
for key in ['fetch_data', 'split_data', 'train', 'predict', 'score_self', 'uv_sync', 'docker_build', 'docker_save', 'docker_validate']:
    sidebar.caption(f'{key}: {job_status_text(get_job(key))}')

auto_refresh = sidebar.checkbox('运行中自动刷新(5s)', value=True)
if auto_refresh and any_job_running():
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=5000, limit=None, key='gui_auto_refresh')
    except Exception:
        sidebar.info('未安装 streamlit-autorefresh，自动刷新不可用。')

override_path = st.session_state.get('gui_override_path', DEFAULT_OVERRIDE_PATH)

if pages[selected_page] == 'data':
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
