from __future__ import annotations

import pandas as pd


def build_rolling_validation_folds(df, sequence_length, runtime_config):
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    unique_dates = [pd.Timestamp(d).normalize() for d in sorted(df['日期'].unique())]
    label_horizon = max(int(runtime_config.get('label_horizon', 5)), 1)
    label_ready_dates = unique_dates[:-label_horizon]

    num_folds = int(runtime_config.get('rolling_val_num_folds', 4))
    window_size = int(runtime_config.get('rolling_val_window_size', 20))
    step_size = int(runtime_config.get('rolling_val_step_size', window_size))
    purge_days = max(int(runtime_config.get('rolling_val_purge_days', 0) or 0), 0)
    embargo_days = max(int(runtime_config.get('rolling_val_embargo_days', 0) or 0), 0)
    effective_step_size = max(step_size, window_size + embargo_days) if embargo_days > 0 else step_size

    if num_folds <= 0:
        raise ValueError('rolling_val_num_folds 必须大于 0')
    if window_size <= 0 or step_size <= 0:
        raise ValueError('rolling_val_window_size 和 rolling_val_step_size 必须大于 0')

    required_dates = window_size + (num_folds - 1) * effective_step_size
    if len(label_ready_dates) < required_dates:
        raise ValueError(
            f'可用于滚动验证的交易日不足: 需要至少 {required_dates} 天，当前仅有 {len(label_ready_dates)} 天'
        )

    reverse_bounds = []
    last_end_idx = len(label_ready_dates) - 1
    for offset in range(num_folds):
        end_idx = last_end_idx - offset * effective_step_size
        start_idx = end_idx - window_size + 1
        if start_idx < 0:
            raise ValueError('滚动验证窗口越界，请减小折数或窗口大小')
        reverse_bounds.append((start_idx, end_idx))

    reverse_bounds.reverse()
    folds = []
    for fold_idx, (start_idx, end_idx) in enumerate(reverse_bounds, start=1):
        start_date = label_ready_dates[start_idx]
        end_date = label_ready_dates[end_idx]
        folds.append({
            'name': f'fold_{fold_idx}',
            'start_date': start_date,
            'end_date': end_date,
            'purge_days': purge_days,
            'embargo_days': embargo_days,
            'label_horizon': label_horizon,
        })

    earliest_start = folds[0]['start_date']
    earliest_start_idx = unique_dates.index(earliest_start)
    train_cutoff_idx = earliest_start_idx - purge_days
    if train_cutoff_idx <= 0:
        raise ValueError('滚动验证起点过早，purge 后没有可用于训练的历史数据')

    context_start_idx = max(0, earliest_start_idx - (sequence_length - 1))
    val_context_start = unique_dates[context_start_idx]
    train_cutoff_date = unique_dates[train_cutoff_idx]

    train_df = df[df['日期'] < train_cutoff_date].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {df['日期'].max().date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"滚动验证实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")
    print(
        '滚动验证参数: '
        f'folds={num_folds}, window_size={window_size}, step_size={step_size}, '
        f'purge_days={purge_days}, embargo_days={embargo_days}, label_horizon={label_horizon}'
    )
    print('滚动验证折:')
    for fold in folds:
        print(f"  - {fold['name']}: {fold['start_date'].date()} 到 {fold['end_date'].date()}")

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')
    return train_df, val_df, folds
