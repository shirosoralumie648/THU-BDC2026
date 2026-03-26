import json
from pathlib import Path
from typing import Dict
from typing import Optional

import streamlit as st

from config import config
from gui_console.common import project_path
from gui_console.common import read_json
from gui_console.common import write_json

DEFAULT_OVERRIDE_PATH = project_path('config', 'gui_config_override.json')


def _as_int(value, minimum=1, maximum=10000):
    value = int(value)
    value = max(minimum, value)
    value = min(maximum, value)
    return value


def _collect_widget_values() -> Dict:
    values = {}

    values['feature_num'] = st.selectbox(
        '特征集合',
        ['39', '158+39'],
        index=1 if str(config.get('feature_num', '158+39')) == '158+39' else 0,
        key='hp_feature_num',
    )
    values['sequence_length'] = _as_int(
        st.number_input(
            '序列长度',
            min_value=10,
            max_value=240,
            value=int(config.get('sequence_length', 60)),
            step=1,
            key='hp_sequence_length',
        )
    )
    values['d_model'] = _as_int(
        st.number_input(
            'd_model',
            min_value=32,
            max_value=1024,
            value=int(config.get('d_model', 256)),
            step=16,
            key='hp_d_model',
        )
    )
    values['nhead'] = _as_int(
        st.number_input(
            'nhead',
            min_value=1,
            max_value=16,
            value=int(config.get('nhead', 4)),
            step=1,
            key='hp_nhead',
        )
    )
    values['num_layers'] = _as_int(
        st.number_input(
            'num_layers',
            min_value=1,
            max_value=12,
            value=int(config.get('num_layers', 3)),
            step=1,
            key='hp_num_layers',
        )
    )
    values['dim_feedforward'] = _as_int(
        st.number_input(
            'dim_feedforward',
            min_value=64,
            max_value=4096,
            value=int(config.get('dim_feedforward', 512)),
            step=64,
            key='hp_dim_feedforward',
        )
    )

    values['learning_rate'] = float(
        st.number_input(
            'learning_rate',
            min_value=1e-7,
            max_value=1e-2,
            value=float(config.get('learning_rate', 1e-5)),
            format='%.7f',
            key='hp_lr',
        )
    )
    values['batch_size'] = _as_int(
        st.number_input(
            'batch_size',
            min_value=1,
            max_value=128,
            value=int(config.get('batch_size', 4)),
            step=1,
            key='hp_batch_size',
        )
    )
    values['num_epochs'] = _as_int(
        st.number_input(
            'num_epochs',
            min_value=1,
            max_value=400,
            value=int(config.get('num_epochs', 50)),
            step=1,
            key='hp_num_epochs',
        )
    )
    values['weight_decay'] = float(
        st.number_input(
            'weight_decay',
            min_value=0.0,
            max_value=1.0,
            value=float(config.get('weight_decay', 1e-5)),
            format='%.7f',
            key='hp_weight_decay',
        )
    )

    values['use_market_gating'] = st.checkbox(
        '市场状态引导门控 (Market Gating)',
        value=bool(config.get('use_market_gating', True)),
        key='hp_use_market_gating',
    )
    values['use_market_gating_macro_context'] = st.checkbox(
        '宏观情绪 Gate 输入',
        value=bool(config.get('use_market_gating_macro_context', True)),
        key='hp_use_market_gating_macro',
    )
    values['use_multi_scale_temporal'] = st.checkbox(
        '多尺度时序编码',
        value=bool(config.get('use_multi_scale_temporal', True)),
        key='hp_use_multi_scale',
    )
    values['use_temporal_cross_stock_attention'] = st.checkbox(
        '时间步级跨股交互',
        value=bool(config.get('use_temporal_cross_stock_attention', True)),
        key='hp_use_temporal_cross_stock',
    )
    values['use_industry_virtual_stock'] = st.checkbox(
        '行业虚拟股交互',
        value=bool(config.get('use_industry_virtual_stock', True)),
        key='hp_use_industry_virtual',
    )
    values['use_multitask_volatility'] = st.checkbox(
        '多任务波动率辅助头',
        value=bool(config.get('use_multitask_volatility', True)),
        key='hp_use_multitask_vol',
    )

    values['feature_cs_norm_method'] = st.selectbox(
        '截面标准化方法',
        ['zscore', 'rank'],
        index=0 if str(config.get('feature_cs_norm_method', 'zscore')) == 'zscore' else 1,
        key='hp_feature_cs_method',
    )

    values['listnet_weight'] = float(
        st.number_input(
            'ListNet 权重',
            min_value=0.0,
            max_value=10.0,
            value=float(config.get('listnet_weight', 1.0)),
            step=0.1,
            key='hp_listnet',
        )
    )
    values['pairwise_weight'] = float(
        st.number_input(
            'Pairwise 权重',
            min_value=0.0,
            max_value=10.0,
            value=float(config.get('pairwise_weight', 1.0)),
            step=0.1,
            key='hp_pairwise',
        )
    )
    values['lambda_ndcg_weight'] = float(
        st.number_input(
            'LambdaNDCG 权重',
            min_value=0.0,
            max_value=10.0,
            value=float(config.get('lambda_ndcg_weight', 0.8)),
            step=0.1,
            key='hp_lambdandcg',
        )
    )

    values['output_dir'] = st.text_input(
        '输出目录',
        value=str(config.get('output_dir', './model/custom_gui_run')),
        key='hp_output_dir',
    )
    values['data_path'] = st.text_input(
        '数据目录',
        value=str(config.get('data_path', './data')),
        key='hp_data_path',
    )
    return values


def _prefill_widgets(base_cfg: Dict) -> None:
    st.session_state['hp_feature_num'] = base_cfg.get('feature_num', config['feature_num'])
    st.session_state['hp_sequence_length'] = int(base_cfg.get('sequence_length', config['sequence_length']))
    st.session_state['hp_d_model'] = int(base_cfg.get('d_model', config['d_model']))
    st.session_state['hp_nhead'] = int(base_cfg.get('nhead', config['nhead']))
    st.session_state['hp_num_layers'] = int(base_cfg.get('num_layers', config['num_layers']))
    st.session_state['hp_dim_feedforward'] = int(base_cfg.get('dim_feedforward', config['dim_feedforward']))

    st.session_state['hp_lr'] = float(base_cfg.get('learning_rate', config['learning_rate']))
    st.session_state['hp_batch_size'] = int(base_cfg.get('batch_size', config['batch_size']))
    st.session_state['hp_num_epochs'] = int(base_cfg.get('num_epochs', config['num_epochs']))
    st.session_state['hp_weight_decay'] = float(base_cfg.get('weight_decay', config.get('weight_decay', 1e-5)))

    st.session_state['hp_use_market_gating'] = bool(base_cfg.get('use_market_gating', config.get('use_market_gating', True)))
    st.session_state['hp_use_market_gating_macro'] = bool(base_cfg.get('use_market_gating_macro_context', config.get('use_market_gating_macro_context', True)))
    st.session_state['hp_use_multi_scale'] = bool(base_cfg.get('use_multi_scale_temporal', config.get('use_multi_scale_temporal', True)))
    st.session_state['hp_use_temporal_cross_stock'] = bool(base_cfg.get('use_temporal_cross_stock_attention', config.get('use_temporal_cross_stock_attention', True)))
    st.session_state['hp_use_industry_virtual'] = bool(base_cfg.get('use_industry_virtual_stock', config.get('use_industry_virtual_stock', True)))
    st.session_state['hp_use_multitask_vol'] = bool(base_cfg.get('use_multitask_volatility', config.get('use_multitask_volatility', True)))

    st.session_state['hp_feature_cs_method'] = base_cfg.get('feature_cs_norm_method', config.get('feature_cs_norm_method', 'zscore'))

    st.session_state['hp_listnet'] = float(base_cfg.get('listnet_weight', config.get('listnet_weight', 1.0)))
    st.session_state['hp_pairwise'] = float(base_cfg.get('pairwise_weight', config.get('pairwise_weight', 1.0)))
    st.session_state['hp_lambdandcg'] = float(base_cfg.get('lambda_ndcg_weight', config.get('lambda_ndcg_weight', 0.8)))

    st.session_state['hp_output_dir'] = base_cfg.get('output_dir', config.get('output_dir', './model/gui'))
    st.session_state['hp_data_path'] = base_cfg.get('data_path', config.get('data_path', './data'))


def render_hyperparams() -> Optional[str]:
    st.subheader('模型配置与参数微调 (Hyperparameter Tuning)')

    override_path = st.text_input('覆盖配置文件路径', value=st.session_state.get('gui_override_path', DEFAULT_OVERRIDE_PATH))
    st.session_state['gui_override_path'] = override_path

    existing_override = read_json(override_path) or {}
    merged = dict(config)
    merged.update(existing_override)

    col_load, col_clear = st.columns(2)
    if col_load.button('从覆盖文件回填控件'):
        _prefill_widgets(merged)
        st.success('已从覆盖文件回填。')
        st.rerun()

    if col_clear.button('清空覆盖配置'):
        target = Path(override_path)
        if target.exists():
            target.unlink()
        st.success('覆盖配置已清空。')
        st.rerun()

    st.markdown('### 可交互参数')

    with st.container(border=True):
        st.markdown('#### 架构参数')
        st.caption('Transformer 层数、头数、序列长度等。')
        values = _collect_widget_values()

    with st.container(border=True):
        st.markdown('#### 优化版专项开关')
        st.caption('6 项增强能力 + 截面标准化 + 混合损失权重。')
        enabled_count = sum(
            int(values[key])
            for key in [
                'use_market_gating',
                'use_market_gating_macro_context',
                'use_multi_scale_temporal',
                'use_temporal_cross_stock_attention',
                'use_industry_virtual_stock',
                'use_multitask_volatility',
            ]
        )
        st.metric('已开启增强项', f'{enabled_count}/6')

    auto_output = st.checkbox('根据 sequence_length/feature_num 自动重建 output_dir', value=True)

    col_save, col_preview = st.columns(2)

    if col_save.button('保存覆盖配置'):
        payload = dict(values)
        if auto_output:
            payload['output_dir'] = f"./model/{payload['sequence_length']}_{payload['feature_num']}"

        overrides = {
            key: value
            for key, value in payload.items()
            if config.get(key) != value
        }

        write_json(override_path, overrides)
        st.session_state['gui_override_path'] = override_path
        st.success(f'已保存覆盖配置: {override_path}')

    if col_preview.button('刷新预览'):
        st.rerun()

    st.markdown('### 覆盖配置预览 (JSON)')
    latest = read_json(override_path) or {}
    st.code(json.dumps(latest, ensure_ascii=False, indent=2), language='json')

    return override_path
