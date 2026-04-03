from pathlib import Path
from typing import List
from typing import Optional

import streamlit as st

from gui_console.common import build_env
from gui_console.common import load_csv_cached
from gui_console.common import project_path
from gui_console.common import python_cmd
from gui_console.common import render_collapsible_job_panel
from gui_console.common import render_job_panel
from gui_console.common import render_metric_card
from gui_console.common import render_page_hero
from gui_console.common import render_section_header
from gui_console.common import start_job
from gui_console.common import stop_job


def _count_existing(paths: list[Path]) -> int:
    return sum(int(path.exists()) for path in paths)


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
    render_page_hero(
        'Deployment Desk',
        '环境同步、镜像构建、赛事验证与结果查看统一操作台。',
        eyebrow='Release Validation',
    )

    discovered_tars = _discover_tar_files()
    result_path = Path(project_path('test', 'result.csv'))

    summary_cols = st.columns(4)
    with summary_cols[0]:
        render_metric_card('Tar Packages', str(len(discovered_tars)), 'test/tars')
    with summary_cols[1]:
        render_metric_card('Validation Result', 'Ready' if result_path.exists() else 'Missing', 'test/result.csv')
    with summary_cols[2]:
        render_metric_card('Ops Stages', str(_count_existing([result_path])), '已产出的关键工件')
    with summary_cols[3]:
        render_metric_card('Compose Target', 'bdc2026', '默认镜像标签')

    ops_col, validation_col = st.columns([1.0, 1.15])

    with ops_col:
        render_section_header('环境同步', '同步 Python/uv 依赖。')
        col_sync_start, col_sync_stop = st.columns(2)
        if col_sync_start.button('执行 uv sync', type='primary'):
            cmd = ['uv', 'sync']
            try:
                start_job('uv_sync', cmd, cwd=project_path(), env=build_env(config_override_path), replace_existing=True)
                st.success('已启动 uv sync。')
            except Exception as exc:
                st.error(f'启动失败: {exc}')
        if col_sync_stop.button('停止 uv sync'):
            stop_job('uv_sync')
            st.warning('已发送停止信号。')

        render_job_panel('uv_sync', 'uv sync', log_lines=120)

        st.markdown('<div class="terminal-divider"></div>', unsafe_allow_html=True)
        render_section_header('镜像构建与导出', '构建、保存并准备提交镜像。')
        platform = st.text_input('构建平台', value='linux/amd64')
        image_name_arg = st.text_input('构建参数 IMAGE_NAME', value='nvidia/cuda')
        image_tag = st.text_input('镜像标签', value='bdc2026')
        tar_output = st.text_input('导出 tar 路径', value=project_path('test', 'tars', 'bdc2026.tar'))

        col_build, col_build_stop = st.columns(2)
        if col_build.button('Docker Build'):
            cmd = ['docker', 'buildx', 'build', '--platform', platform, '--build-arg', f'IMAGE_NAME={image_name_arg}', '-t', image_tag, '.']
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

        render_collapsible_job_panel('docker_build', 'Docker Build 日志', log_lines=120, expanded=False)
        render_collapsible_job_panel('docker_save', 'Docker Save 日志', log_lines=120, expanded=False)

    with validation_col:
        render_section_header('赛事验证', '选择 tar 包并运行 test/test.py 验证流程。')
        selected_tars = st.multiselect('选择待验证 tar 包（来自 test/tars）', discovered_tars, default=discovered_tars[:1])
        manual_tars = st.text_area('额外 tar 文件名（每行一个，可选）', value='', help='示例: team_a.tar')
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

        render_collapsible_job_panel('docker_validate', '赛事验证日志', log_lines=120, expanded=False)

        render_section_header('验证结果', '查看最新的赛事验证结果输出。')
        if result_path.exists():
            df = load_csv_cached(str(result_path))
            st.dataframe(df, width='stretch')
        else:
            st.info('尚未生成 test/result.csv。')
