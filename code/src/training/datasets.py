import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config


class RankingDataset(torch.utils.data.Dataset):
    """排序数据集类"""

    def __init__(self, sequences, targets, relevance_scores, stock_indices, vol_targets=None):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices
        self.vol_targets = vol_targets if vol_targets is not None else targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequences': torch.FloatTensor(np.array(self.sequences[idx])),
            'targets': torch.FloatTensor(np.array(self.targets[idx])),
            'relevance': torch.LongTensor(np.array(self.relevance_scores[idx])),
            'stock_indices': torch.LongTensor(np.array(self.stock_indices[idx])),
            'vol_targets': torch.FloatTensor(np.array(self.vol_targets[idx])),
        }


class LazyRankingDataset(torch.utils.data.Dataset):
    """懒加载排序数据集，避免一次性将全部窗口序列展开到内存。"""

    def __init__(self, stock_cache, day_entries, sequence_length):
        self.stock_cache = stock_cache
        self.day_entries = day_entries
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.day_entries)

    def __getitem__(self, idx):
        entry = self.day_entries[idx]
        day_sequences = []
        day_targets = []
        day_vol_targets = []
        day_stock_indices = []

        for stock_idx, end_idx in entry['entries']:
            stock_data = self.stock_cache[stock_idx]
            if end_idx >= len(stock_data['labels']):
                raise IndexError(
                    f"懒加载索引越界: date={entry['date']}, stock_idx={stock_idx}, "
                    f"end_idx={end_idx}, labels_len={len(stock_data['labels'])}"
                )
            seq = stock_data['features'][end_idx - self.sequence_length + 1:end_idx + 1]
            target = stock_data['labels'][end_idx]
            vol_target = stock_data['vol_labels'][end_idx]
            day_sequences.append(seq)
            day_targets.append(target)
            day_vol_targets.append(vol_target)
            day_stock_indices.append(stock_idx)

        day_targets = np.asarray(day_targets, dtype=np.float32)
        day_vol_targets = np.asarray(day_vol_targets, dtype=np.float32)
        threshold_2pct = np.quantile(day_targets, 0.98)
        relevance = (day_targets >= threshold_2pct).astype(np.float32)

        return {
            'sequences': torch.FloatTensor(np.asarray(day_sequences, dtype=np.float32)),
            'targets': torch.FloatTensor(day_targets),
            'vol_targets': torch.FloatTensor(day_vol_targets),
            'relevance': torch.LongTensor(relevance.astype(np.int64)),
            'stock_indices': torch.LongTensor(np.asarray(day_stock_indices, dtype=np.int64)),
        }


def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    vol_targets = [item.get('vol_targets', item['targets']) for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]

    max_stocks = max(seq.size(0) for seq in sequences)

    padded_sequences = []
    padded_targets = []
    padded_vol_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []

    for seq, tgt, vol_tgt, rel, stock_idx in zip(sequences, targets, vol_targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)

        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            vol_tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size, dtype=torch.long)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)

            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            vol_tgt = torch.cat([vol_tgt, vol_tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)

        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0

        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_vol_targets.append(vol_tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)

    return {
        'sequences': torch.stack(padded_sequences),
        'targets': torch.stack(padded_targets),
        'vol_targets': torch.stack(padded_vol_targets),
        'relevance': torch.stack(padded_relevance),
        'stock_indices': torch.stack(padded_stock_indices),
        'masks': torch.stack(masks),
    }


def build_lazy_ranking_index(data, features, sequence_length, min_window_end_date=None, max_window_end_date=None):
    """构建懒加载训练索引，仅保存按股票缓存和按日期索引，不保存完整窗口内容。"""
    print("正在创建排序数据集索引（懒加载版本）...")
    indexed = data.copy()
    indexed = indexed.rename(columns={'日期': 'datetime'})
    indexed['datetime'] = pd.to_datetime(indexed['datetime'])
    indexed = indexed.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    required_cols = ['label']
    if bool(config.get('use_multitask_volatility', False)):
        required_cols.append('vol_label')
    indexed = indexed.dropna(subset=required_cols)

    if min_window_end_date is not None:
        min_window_end_date = pd.to_datetime(min_window_end_date)
    if max_window_end_date is not None:
        max_window_end_date = pd.to_datetime(max_window_end_date)

    stock_cache = {}
    date_to_entries = {}

    grouped = indexed.groupby('instrument', sort=False)
    for stock_idx, group in tqdm(grouped, desc="Indexing stocks"):
        group = group.reset_index(drop=True)
        if len(group) < sequence_length:
            continue

        feature_values = group[features].to_numpy(dtype=np.float32, copy=True)
        labels = group['label'].to_numpy(dtype=np.float32, copy=True)
        if 'vol_label' in group.columns:
            vol_labels = group['vol_label'].to_numpy(dtype=np.float32, copy=True)
        else:
            vol_labels = labels.copy()
        dates = pd.to_datetime(group['datetime']).to_numpy()

        stock_idx = int(stock_idx)
        stock_cache[stock_idx] = {
            'features': feature_values,
            'labels': labels,
            'vol_labels': vol_labels,
        }

        for end_idx in range(sequence_length - 1, len(group)):
            end_date = pd.Timestamp(dates[end_idx]).normalize()
            if min_window_end_date is not None and end_date < min_window_end_date:
                continue
            if max_window_end_date is not None and end_date > max_window_end_date:
                continue
            date_to_entries.setdefault(end_date, []).append((stock_idx, int(end_idx)))

    day_entries = []
    for date in sorted(date_to_entries):
        entries = date_to_entries[date]
        if len(entries) < 10:
            continue
        day_entries.append({
            'date': date,
            'entries': entries,
        })

    print(f"成功创建 {len(day_entries)} 个训练索引样本")
    if day_entries:
        avg_stocks = np.mean([len(entry['entries']) for entry in day_entries])
        print(f"每个训练样本平均包含 {avg_stocks:.1f} 只股票")

    return stock_cache, day_entries
