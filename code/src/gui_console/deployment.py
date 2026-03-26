from pathlib import Path
from typing import List
from typing import Optional

import streamlit as st

from gui_console.common import build_env
from gui_console.common import format_cmd
from gui_console.common import get_job
from gui_console.common import job_status_text
from gui_console.common import load_csv_cached
from gui_console.common import project_path
from gui_console.common import python_cmd
from gui_console.common import read_text_tail
from gui_console.common import start_job
from gui_console.common import stop_job


def _render_job(job_key: str, title: str) -> None:
    job = get_job(job_key)
    st.caption(f'{title} 状态: {job_status_text(job)}')
    if not job:
        return
    st.code(format_cmd(job['cmd']), language='bash')
    st.caption(f"日志: {job['log_path']}")
    st.text_area(f'{title} 日志 (tail)', value=read_text_tail(job['log_path'], 120), height=170, key=f'{job_key}_tail')


def _discover_tar_files() -> List[str]:
    tar_dir = Path(project_path('test', 'tars'))
    if not tar_dir.exists():
        return []
    files = sorted([p.name for p in tar_dir.glob('*.tar')])
    return files


def _write_tar_list_file(tar_names: List[str]) -> str:
    path = Path(project_path('test', 'tar_files_list.txt'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for name in tar_names:
            f.write(f'{name}\n')
    return str(path)


def render_deployment(config_override_path: Optional[str]) -> None:
    st.subheader('部署工具箱 (Deployment & System)')

    st.markdown('### 环境同步')
    col_sync_start, col_sync_stop = st.columns(2)
    if col_sync_start.button('执行 uv sync'):
        cmd = ['uv', 'sync']
        try:
            start_job('uv_sync', cmd, cwd=project_path(), env=build_env(config_override_path), replace_existing=True)
            st.success('已启动 uv sync。')
        except Exception as exc:
            st.error(f'启动失败: {exc}')
    if col_sync_stop.button('停止 uv sync'):
        stop_job('uv_sync')
        st.warning('已发送停止信号。')

    _render_job('uv_sync', 'uv sync')

    st.markdown('### Docker 镜像构建与导出')
    platform = st.text_input('构建平台', value='linux/amd64')
    image_name_arg = st.text_input('构建参数 IMAGE_NAME', value='nvidia/cuda')
    image_tag = st.text_input('镜像标签', value='bdc2026')
    tar_output = st.text_input('导出 tar 路径', value=project_path('test', 'tars', 'bdc2026.tar'))

    col_build, col_build_stop = st.columns(2)
    if col_build.button('Docker Build'):
        cmd = [
            'docker',
            'buildx',
            'build',
            '--platform',
            platform,
            '--build-arg',
            f'IMAGE_NAME={image_name_arg}',
            '-t',
            image_tag,
            '.',
        ]
        try:
            start_job('docker_build', cmd, cwd=project_path(), env=build_env(config_override_path), replace_existing=True)
            st.success('已启动 Docker Build。')
        except Exception as exc:
            st.error(f'启动失败: {exc}')

    if col_build_stop.button('停止 Build'):
        stop_job('docker_build')
        st.warning('已发送停止信号。')

    col_save, col_save_stop = st.columns(2)
    if col_save.button('Docker Save'):
        target = Path(tar_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        cmd = ['docker', 'save', '-o', str(target), image_tag]
        try:
            start_job('docker_save', cmd, cwd=project_path(), env=build_env(config_override_path), replace_existing=True)
            st.success('已启动 Docker Save。')
        except Exception as exc:
            st.error(f'启动失败: {exc}')

    if col_save_stop.button('停止 Save'):
        stop_job('docker_save')
        st.warning('已发送停止信号。')

    _render_job('docker_build', 'Docker Build')
    _render_job('docker_save', 'Docker Save')

    st.markdown('### 可运行性校验（赛事方验证）')
    discovered_tars = _discover_tar_files()
    selected_tars = st.multiselect('选择待验证 tar 包（来自 test/tars）', discovered_tars, default=discovered_tars[:1])

    manual_tars = st.text_area(
        '额外 tar 文件名（每行一个，可选）',
        value='',
        help='示例: team_a.tar',
    )
    manual_names = [line.strip() for line in manual_tars.splitlines() if line.strip()]

    merged_tars = []
    seen = set()
    for name in selected_tars + manual_names:
        if name in seen:
            continue
        seen.add(name)
        merged_tars.append(name)

    col_val_start, col_val_stop = st.columns(2)
    if col_val_start.button('运行 test/test.py 验证'):
        if not merged_tars:
            st.error('请至少选择一个 tar 文件。')
        else:
            list_file = _write_tar_list_file(merged_tars)
            st.caption(f'已写入 tar 列表: {list_file}')
            cmd = python_cmd('test/test.py')
            try:
                start_job('docker_validate', cmd, cwd=project_path(), env=build_env(config_override_path), replace_existing=True)
                st.success('验证任务已启动。')
            except Exception as exc:
                st.error(f'启动失败: {exc}')

    if col_val_stop.button('停止验证'):
        stop_job('docker_validate')
        st.warning('已发送停止信号。')

    _render_job('docker_validate', '赛事验证')

    result_path = Path(project_path('test', 'result.csv'))
    if result_path.exists():
        st.markdown('### 验证结果表')
        df = load_csv_cached(str(result_path))
        st.dataframe(df, width='stretch')
