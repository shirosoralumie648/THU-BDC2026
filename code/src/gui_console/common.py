import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from config import config as BASE_CONFIG

ROOT_DIR = Path(__file__).resolve().parents[3]
TEMP_DIR = ROOT_DIR / 'temp'
GUI_LOG_DIR = TEMP_DIR / 'gui_logs'


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
