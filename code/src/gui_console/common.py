import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from html import escape
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from config import config as BASE_CONFIG

ROOT_DIR = Path(__file__).resolve().parents[3]
TEMP_DIR = ROOT_DIR / 'temp'
GUI_LOG_DIR = TEMP_DIR / 'gui_logs'

TERMINAL_TONES = {
    'running': 'running',
    'success': 'success',
    'failed': 'failed',
    'idle': 'idle',
    'warning': 'warning',
}


def inject_terminal_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --terminal-bg: #071018;
            --terminal-panel: rgba(13, 22, 32, 0.88);
            --terminal-panel-alt: rgba(9, 16, 24, 0.92);
            --terminal-border: rgba(76, 201, 240, 0.18);
            --terminal-border-strong: rgba(76, 201, 240, 0.32);
            --terminal-text: #e6edf3;
            --terminal-muted: #7d93a8;
            --terminal-accent: #4cc9f0;
            --terminal-accent-2: #90e0ef;
            --terminal-success: #39d98a;
            --terminal-warning: #ffb020;
            --terminal-danger: #ff6b6b;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(76, 201, 240, 0.10), transparent 24%),
                radial-gradient(circle at top left, rgba(144, 224, 239, 0.08), transparent 20%),
                linear-gradient(180deg, #071018 0%, #09131c 100%);
            color: var(--terminal-text);
        }

        .stApp [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(6, 13, 19, 0.98), rgba(9, 16, 24, 0.98));
            border-right: 1px solid rgba(76, 201, 240, 0.12);
        }

        .terminal-shell {
            padding-bottom: 1.2rem;
        }

        .terminal-hero {
            background: linear-gradient(135deg, rgba(13, 22, 32, 0.95), rgba(9, 16, 24, 0.92));
            border: 1px solid var(--terminal-border-strong);
            border-radius: 20px;
            padding: 1.2rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.24);
        }

        .terminal-eyebrow {
            color: var(--terminal-accent);
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.72rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .terminal-title {
            color: var(--terminal-text);
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.08;
            margin: 0;
        }

        .terminal-subtitle {
            color: var(--terminal-muted);
            font-size: 0.95rem;
            margin-top: 0.45rem;
            margin-bottom: 0;
        }

        .terminal-panel {
            background: linear-gradient(180deg, var(--terminal-panel), var(--terminal-panel-alt));
            border: 1px solid var(--terminal-border);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02), 0 12px 28px rgba(0, 0, 0, 0.18);
            margin-bottom: 0.9rem;
        }

        .terminal-panel.tight {
            padding: 0.8rem 0.95rem;
        }

        .terminal-kpi-label {
            color: var(--terminal-muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .terminal-kpi-value {
            color: var(--terminal-text);
            font-size: 1.55rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .terminal-kpi-hint {
            color: var(--terminal-muted);
            font-size: 0.82rem;
            margin-top: 0.45rem;
        }

        .terminal-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.22rem 0.62rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            border: 1px solid transparent;
        }

        .terminal-badge-running {
            color: var(--terminal-accent);
            background: rgba(76, 201, 240, 0.12);
            border-color: rgba(76, 201, 240, 0.22);
        }

        .terminal-badge-success {
            color: var(--terminal-success);
            background: rgba(57, 217, 138, 0.12);
            border-color: rgba(57, 217, 138, 0.22);
        }

        .terminal-badge-failed {
            color: var(--terminal-danger);
            background: rgba(255, 107, 107, 0.12);
            border-color: rgba(255, 107, 107, 0.22);
        }

        .terminal-badge-idle, .terminal-badge-warning {
            color: var(--terminal-warning);
            background: rgba(255, 176, 32, 0.12);
            border-color: rgba(255, 176, 32, 0.22);
        }

        .terminal-section-title {
            color: var(--terminal-text);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .terminal-help {
            color: var(--terminal-muted);
            font-size: 0.84rem;
            margin-top: 0.2rem;
        }

        .terminal-divider {
            height: 1px;
            background: linear-gradient(90deg, rgba(76, 201, 240, 0.22), rgba(76, 201, 240, 0));
            margin: 0.9rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _tone_for_status_text(status_text: str) -> str:
    if '运行' in status_text:
        return TERMINAL_TONES['running']
    if '完成' in status_text:
        return TERMINAL_TONES['success']
    if '失败' in status_text:
        return TERMINAL_TONES['failed']
    return TERMINAL_TONES['idle']


def render_page_hero(title: str, subtitle: str = '', eyebrow: str = 'Quant Research Terminal') -> None:
    title_html = escape(str(title))
    subtitle_html = f'<p class="terminal-subtitle">{escape(str(subtitle))}</p>' if subtitle else ''
    st.markdown(
        f'''<div class="terminal-hero"><div class="terminal-eyebrow">{escape(str(eyebrow))}</div><h1 class="terminal-title">{title_html}</h1>{subtitle_html}</div>''',
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, hint: str = '') -> None:
    hint_html = f'<div class="terminal-kpi-hint">{escape(str(hint))}</div>' if hint else ''
    st.markdown(
        f'''<div class="terminal-panel tight"><div class="terminal-kpi-label">{escape(str(label))}</div><div class="terminal-kpi-value">{escape(str(value))}</div>{hint_html}</div>''',
        unsafe_allow_html=True,
    )


def render_status_badge(text: str, tone: str) -> None:
    tone_key = TERMINAL_TONES.get(tone, TERMINAL_TONES['idle'])
    st.markdown(
        f'<span class="terminal-badge terminal-badge-{tone_key}">{escape(str(text))}</span>',
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str = '') -> None:
    st.markdown(f'<div class="terminal-section-title">{escape(str(title))}</div>', unsafe_allow_html=True)
    if description:
        st.caption(description)


def render_job_panel(job_key: str, title: str, log_lines: int = 120) -> None:
    job = get_job(job_key)
    status_text = job_status_text(job)
    tone = _tone_for_status_text(status_text)
    render_section_header(title)
    render_status_badge(status_text, tone)
    if not job:
        st.caption('当前没有运行记录。')
        return
    st.code(format_cmd(job['cmd']), language='bash')
    st.caption(f"日志: {job['log_path']}")
    st.text_area(
        f'{title} 日志',
        value=read_text_tail(job['log_path'], log_lines),
        height=180,
        key=f'{job_key}_panel_log',
    )


def render_collapsible_job_panel(job_key: str, title: str, *, log_lines: int = 120, expanded: bool = False) -> None:
    job = get_job(job_key)
    status_text = job_status_text(job)
    tone = _tone_for_status_text(status_text)
    with st.expander(f'{title} · {status_text}', expanded=expanded):
        render_status_badge(status_text, tone)
        if not job:
            st.caption('当前没有运行记录。')
            return
        st.code(format_cmd(job['cmd']), language='bash')
        st.caption(f"日志: {job['log_path']}")
        st.text_area(
            f'{title} 日志',
            value=read_text_tail(job['log_path'], log_lines),
            height=180,
            key=f'{job_key}_collapsible_log',
        )


def apply_dark_figure_style(fig, *, height=None, title=None):
    if title is not None:
        fig.update_layout(title=title)
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e6edf3'},
        margin={'l': 20, 'r': 20, 't': 36, 'b': 20},
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1.0},
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(125,147,168,0.12)', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(125,147,168,0.12)', zeroline=False)
    if height is not None:
        fig.update_layout(height=height)
    return fig


def summarize_jobs(job_keys: List[str]) -> Dict[str, int]:
    summary = {'running': 0, 'success': 0, 'failed': 0, 'idle': 0}
    for job_key in job_keys:
        status_text = job_status_text(get_job(job_key))
        tone = _tone_for_status_text(status_text)
        summary[tone] = summary.get(tone, 0) + 1
    return summary


def ensure_gui_dirs() -> None:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    GUI_LOG_DIR.mkdir(parents=True, exist_ok=True)


def ensure_gui_state() -> None:
    if 'gui_jobs' not in st.session_state:
        st.session_state['gui_jobs'] = {}


def project_path(*parts: str) -> str:
    return str((ROOT_DIR.joinpath(*parts)).resolve())


def project_rel(path_str: str) -> str:
    try:
        return str(Path(path_str).resolve().relative_to(ROOT_DIR))
    except Exception:
        return str(path_str)


def python_cmd(script_rel_path: str, args: Optional[List[str]] = None) -> List[str]:
    command = [sys.executable, project_path(script_rel_path)]
    if args:
        command.extend(args)
    return command


def shell_cmd(command: str) -> List[str]:
    return shlex.split(command)


def build_env(config_override_path: Optional[str] = None) -> Dict[str, str]:
    env = dict(os.environ)
    if config_override_path:
        env['THU_BDC_CONFIG_OVERRIDE_PATH'] = str(Path(config_override_path).resolve())
    return env


def _job_is_running(job: Dict) -> bool:
    proc = job.get('proc')
    if proc is None:
        return False
    try:
        return proc.poll() is None
    except Exception:
        return False


def poll_jobs() -> None:
    ensure_gui_state()
    jobs = st.session_state['gui_jobs']
    for job_key, job in list(jobs.items()):
        proc = job.get('proc')
        if proc is None:
            continue
        return_code = proc.poll()
        if return_code is None:
            continue
        job['running'] = False
        job['return_code'] = int(return_code)
        job['finished_at'] = time.time()
        handle = job.get('log_handle')
        if handle is not None and not handle.closed:
            handle.close()


def get_job(job_key: str) -> Optional[Dict]:
    ensure_gui_state()
    poll_jobs()
    return st.session_state['gui_jobs'].get(job_key)


def is_job_running(job_key: str) -> bool:
    job = get_job(job_key)
    if not job:
        return False
    return bool(job.get('running', False)) and _job_is_running(job)


def any_job_running() -> bool:
    ensure_gui_state()
    poll_jobs()
    for job in st.session_state['gui_jobs'].values():
        if bool(job.get('running', False)) and _job_is_running(job):
            return True
    return False


def start_job(
    job_key: str,
    cmd: List[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    replace_existing: bool = False,
) -> Dict:
    ensure_gui_dirs()
    ensure_gui_state()
    poll_jobs()

    jobs = st.session_state['gui_jobs']
    current = jobs.get(job_key)
    if current and _job_is_running(current):
        if not replace_existing:
            raise RuntimeError(f'任务正在运行中: {job_key}')
        stop_job(job_key)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = GUI_LOG_DIR / f'{job_key}_{timestamp}.log'
    handle = open(log_path, 'w', encoding='utf-8')

    process = subprocess.Popen(
        cmd,
        cwd=str(Path(cwd).resolve()) if cwd else str(ROOT_DIR),
        env=env or dict(os.environ),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    job = {
        'job_key': job_key,
        'cmd': list(cmd),
        'cwd': str(cwd or ROOT_DIR),
        'log_path': str(log_path),
        'start_at': time.time(),
        'finished_at': None,
        'running': True,
        'return_code': None,
        'proc': process,
        'log_handle': handle,
    }
    jobs[job_key] = job
    return job


def stop_job(job_key: str) -> None:
    ensure_gui_state()
    poll_jobs()
    job = st.session_state['gui_jobs'].get(job_key)
    if not job:
        return

    proc = job.get('proc')
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

    for _ in range(20):
        if proc is None or proc.poll() is not None:
            break
        time.sleep(0.1)

    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    poll_jobs()


def read_text_tail(path: str, max_lines: int = 300) -> str:
    target = Path(path)
    if not target.exists():
        return ''
    try:
        with open(target, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return ''
    if len(lines) <= max_lines:
        return ''.join(lines)
    return ''.join(lines[-max_lines:])


def read_json(path: str) -> Optional[Dict]:
    target = Path(path)
    if not target.exists():
        return None
    try:
        with open(target, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: str, payload: Dict) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(target)


def _resolve_rooted_path(path_str: str) -> str:
    path = Path(str(path_str))
    if path.is_absolute():
        return str(path)
    return str((ROOT_DIR / path).resolve())


def load_effective_config(config_override_path: Optional[str] = None) -> Dict:
    cfg = dict(BASE_CONFIG)
    if not config_override_path:
        return cfg

    payload = read_json(config_override_path)
    if not isinstance(payload, dict):
        return cfg

    cfg.update(payload)
    if ('sequence_length' in payload or 'feature_num' in payload) and ('output_dir' not in payload):
        cfg['output_dir'] = f"./model/{cfg['sequence_length']}_{cfg['feature_num']}"
    return cfg


def resolve_effective_path(path_value: str) -> str:
    return _resolve_rooted_path(path_value)


def load_csv(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


@st.cache_data(show_spinner=False)
def load_csv_cached(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    return load_csv(path, nrows=nrows)


def normalize_stock_code(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.split('.').str[-1]
    s = s.str.replace(r'[^0-9]', '', regex=True)
    s = s.str[-6:].str.zfill(6)
    return s


def parse_training_log(log_text: str) -> Dict[str, pd.DataFrame]:
    epoch_records: Dict[int, Dict] = {}
    ablation_rows: List[Dict] = []
    early_stop_rows: List[Dict] = []

    current_epoch = None

    epoch_pattern = re.compile(r'^=== Epoch\s+(\d+)/(\d+)\s+===')
    scalar_pattern = re.compile(r'^(Train|Eval)\s+([A-Za-z0-9_\-]+):\s+([-+]?\d+(?:\.\d+)?)')
    best_strategy_pattern = re.compile(
        r'^当前最优持仓策略:\s*([^|]+)\|\s*验证目标值:\s*([-+]?\d+(?:\.\d+)?)\s*\|\s*策略收益均值:\s*([-+]?\d+(?:\.\d+)?)\s*\|\s*RankIC:\s*([-+]?\d+(?:\.\d+)?)'
    )
    early_status_pattern = re.compile(
        r'^早停状态:\s*monitor=([^,]+),\s*value=([-+]?\d+(?:\.\d+)?),\s*best=([-+]?\d+(?:\.\d+)?),\s*bad_epochs=(\d+)/(\d+)'
    )
    early_trigger_pattern = re.compile(
        r'^触发早停:\s*monitor=([^,]+),\s*mode=([^,]+),\s*patience=(\d+),\s*best=([-+]?\d+(?:\.\d+)?)'
    )
    ablation_pattern = re.compile(
        r'^\s*-\s*([^:]+):\s*features=(\d+),\s*return=([-+]?\d+(?:\.\d+)?),\s*delta=([-+]?\d+(?:\.\d+)?)'
    )

    for raw_line in log_text.splitlines():
        line = raw_line.strip('\n')
        m_epoch = epoch_pattern.match(line)
        if m_epoch:
            current_epoch = int(m_epoch.group(1))
            total_epoch = int(m_epoch.group(2))
            epoch_records.setdefault(current_epoch, {'epoch': current_epoch, 'total_epoch': total_epoch})
            continue

        if current_epoch is None:
            continue

        record = epoch_records.setdefault(current_epoch, {'epoch': current_epoch})

        m_scalar = scalar_pattern.match(line)
        if m_scalar:
            split_name = m_scalar.group(1).lower()
            metric_name = m_scalar.group(2)
            value = float(m_scalar.group(3))
            record[f'{split_name}_{metric_name}'] = value
            continue

        m_best = best_strategy_pattern.match(line)
        if m_best:
            record['best_strategy_name'] = m_best.group(1).strip()
            record['validation_objective'] = float(m_best.group(2))
            record['strategy_return'] = float(m_best.group(3))
            record['best_strategy_rank_ic'] = float(m_best.group(4))
            continue

        m_early_status = early_status_pattern.match(line)
        if m_early_status:
            item = {
                'epoch': current_epoch,
                'monitor': m_early_status.group(1).strip(),
                'value': float(m_early_status.group(2)),
                'best': float(m_early_status.group(3)),
                'bad_epochs': int(m_early_status.group(4)),
                'patience': int(m_early_status.group(5)),
            }
            early_stop_rows.append(item)
            record['early_stop_bad_epochs'] = item['bad_epochs']
            record['early_stop_patience'] = item['patience']
            continue

        m_trigger = early_trigger_pattern.match(line)
        if m_trigger:
            item = {
                'epoch': current_epoch,
                'monitor': m_trigger.group(1).strip(),
                'mode': m_trigger.group(2).strip(),
                'patience': int(m_trigger.group(3)),
                'best': float(m_trigger.group(4)),
                'triggered': True,
            }
            early_stop_rows.append(item)
            record['early_stop_triggered'] = True
            continue

        m_ablation = ablation_pattern.match(line)
        if m_ablation:
            ablation_rows.append(
                {
                    'epoch': current_epoch,
                    'group': m_ablation.group(1).strip(),
                    'num_features': int(m_ablation.group(2)),
                    'return': float(m_ablation.group(3)),
                    'delta': float(m_ablation.group(4)),
                }
            )

    metrics_df = pd.DataFrame(sorted(epoch_records.values(), key=lambda x: x.get('epoch', 0)))
    ablation_df = pd.DataFrame(ablation_rows)
    early_df = pd.DataFrame(early_stop_rows)
    return {
        'metrics': metrics_df,
        'ablation': ablation_df,
        'early_stop': early_df,
    }


def format_cmd(cmd: List[str]) -> str:
    return ' '.join(shlex.quote(str(item)) for item in cmd)


def job_status_text(job: Optional[Dict]) -> str:
    if not job:
        return '未启动'
    if bool(job.get('running')) and _job_is_running(job):
        return '运行中'
    rc = job.get('return_code')
    if rc is None:
        return '未启动'
    if int(rc) == 0:
        return '已完成 (0)'
    return f'失败 ({rc})'
