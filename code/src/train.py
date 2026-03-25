import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from factor_store import engineer_group_features
from factor_store import resolve_factor_pipeline
from factor_store import save_factor_snapshot
from model import StockTransformer
from utils import create_ranking_dataset_vectorized
import joblib
import os
import json
import multiprocessing as mp
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def _build_label_and_clean(processed, drop_small_open=True):
    """统一构建标签并清洗无效样本。"""
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    # 过滤无效开盘价，避免收益率极端爆炸
    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label'])

    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


def _preprocess_common(df, stockid2idx, desc, feature_pipeline, drop_small_open=True):
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_columns = feature_pipeline['active_features']
    custom_factor_specs = feature_pipeline['custom_specs']
    builtin_override_specs = feature_pipeline.get('builtin_override_specs', [])

    # 保证时序正确，避免 shift 标签错位
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(int(config.get('feature_engineer_processes', 4)), mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        tasks = [
            (group, feature_pipeline['feature_set'], builtin_override_specs, custom_factor_specs)
            for group in groups
        ]
        processed_list = list(tqdm(pool.imap(engineer_group_features, tasks), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    # 映射股票索引，并剔除映射失败样本
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


# 数据预处理函数
def preprocess_data(df, feature_pipeline, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", feature_pipeline=feature_pipeline, drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", feature_pipeline=feature_pipeline, drop_small_open=True)


def preprocess_val_data(df, feature_pipeline, stockid2idx=None):
    # 验证集与训练集保持同口径，避免 label 分布漂移
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", feature_pipeline=feature_pipeline, drop_small_open=True)


# 加权的组合收益排序损失函数
class PortfolioOptimizationLoss(nn.Module):
    """
    针对量化排序直接优化投资组合收益的损失函数。
    结合 ListNet (全局分布对齐) 和 加权 RankNet (强化头部收益)，直接最大化强个股得分。
    """
    def __init__(self, temperature=10.0, pairwise_weight=1.0):
        super(PortfolioOptimizationLoss, self).__init__()
        self.temperature = temperature
        self.pairwise_weight = pairwise_weight

    def forward(self, y_pred, y_true):
        """
        y_pred: [1, num_items]
        y_true: [1, num_items] (实际收益率)
        """
        # 1. ListNet Loss (全局分布对齐)
        # 放大极端正收益影响，构建期望的收益概率分布
        P_true = F.softmax(y_true * self.temperature, dim=1)
        # 用 log 分布对齐 y_pred
        P_pred = F.log_softmax(y_pred, dim=1)
        listnet_loss = -torch.sum(P_true * P_pred, dim=1).mean()
        
        # 2. Top-K Pairwise RankNet Loss (极大化右尾极值)
        # 剥离多余维度 [N]
        y_pred = y_pred.squeeze(0)
        y_true = y_true.squeeze(0)
        
        # 为了提高效率并且专注于挑出最好的股票，我们只选取真实收益最高的前 10%
        # 将它们与所有其他股票进行配对比较
        k = max(1, int(len(y_true) * 0.1))
        _, top_true_indices = torch.topk(y_true, k)
        
        y_pred_top = y_pred[top_true_indices]
        y_true_top = y_true[top_true_indices]
        
        # 构建预测差值和真实差值矩阵 [k, N]
        pred_diff = y_pred_top.unsqueeze(1) - y_pred.unsqueeze(0)
        true_diff = y_true_top.unsqueeze(1) - y_true.unsqueeze(0)
        
        # 只取真实的收益差 (好 vs 坏)
        mask = (true_diff > 0).float()
        
        # Pairwise Loss: 分数必须对应真实收益差；收益差距越大的 pair，惩罚权重越大
        pairwise_loss = (F.softplus(-pred_diff) * mask * true_diff).sum() / (mask.sum() + 1e-8)
        
        return listnet_loss + self.pairwise_weight * pairwise_loss


def build_strategy_candidates():
    top_k_candidates = sorted({int(k) for k in config.get('prediction_top_k_candidates', [5]) if 1 <= int(k) <= 5})
    weighting_candidates = list(dict.fromkeys(config.get('prediction_weighting_candidates', ['equal'])))

    candidates = []
    for top_k in top_k_candidates:
        for weighting in weighting_candidates:
            if weighting not in {'equal', 'softmax'}:
                continue
            if weighting == 'softmax' and top_k == 1:
                continue
            candidates.append({
                'name': f'{weighting}_top{top_k}',
                'top_k': top_k,
                'weighting': weighting,
            })

    if not candidates:
        candidates = [{'name': 'equal_top5', 'top_k': 5, 'weighting': 'equal'}]
    return candidates


def build_portfolio_weights(scores, top_k, weighting='equal', temperature=1.0):
    top_k = min(int(top_k), scores.numel())
    top_scores, top_indices = torch.topk(scores, top_k)

    if weighting == 'equal' or top_k == 1:
        weights = torch.full(
            (top_k,),
            1.0 / top_k,
            dtype=top_scores.dtype,
            device=top_scores.device
        )
    elif weighting == 'softmax':
        temperature = max(float(temperature), 1e-6)
        weights = torch.softmax(top_scores / temperature, dim=0)
    else:
        raise ValueError(f'不支持的权重方式: {weighting}')

    return top_indices, weights


def calculate_ranking_metrics(y_pred, y_true, masks, strategy_candidates=None, temperature=1.0):
    """按候选持仓策略计算验证收益率，直接服务于最终总收益目标。"""
    batch_size = y_pred.size(0)

    if strategy_candidates is None:
        strategy_candidates = [{'name': 'equal_top5', 'top_k': 5, 'weighting': 'equal'}]

    metrics_lists = {f'return_{candidate["name"]}': [] for candidate in strategy_candidates}
    max_top_k = max(candidate['top_k'] for candidate in strategy_candidates)
    oracle_return_list = []

    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()

        if valid_indices.numel() < max_top_k:
            continue

        valid_pred = y_pred[i][valid_indices]
        valid_true_return = y_true[i][valid_indices]

        for candidate in strategy_candidates:
            metric_name = f'return_{candidate["name"]}'
            pred_indices, weights = build_portfolio_weights(
                valid_pred,
                top_k=candidate['top_k'],
                weighting=candidate['weighting'],
                temperature=temperature,
            )
            pred_top_returns = valid_true_return[pred_indices]
            portfolio_return = torch.sum(pred_top_returns * weights).item()
            metrics_lists[metric_name].append(portfolio_return)

        _, true_indices = torch.topk(valid_true_return, 5)
        true_top_returns = valid_true_return[true_indices]
        oracle_return_list.append(true_top_returns.mean().item())

    metrics = {name: (np.mean(values) if values else 0.0) for name, values in metrics_lists.items()}
    metrics['oracle_top5_equal'] = np.mean(oracle_return_list) if oracle_return_list else 0.0

    return metrics


def choose_best_strategy(eval_metrics, strategy_candidates):
    selection_metric = config.get('selection_metric', 'auto')

    if selection_metric != 'auto':
        metric_value = eval_metrics.get(selection_metric, float('-inf'))
        for candidate in strategy_candidates:
            if f'return_{candidate["name"]}' == selection_metric:
                return candidate, metric_value
        raise ValueError(f'未找到 selection_metric 对应的策略: {selection_metric}')

    best_candidate = None
    best_score = -float('inf')

    for candidate in strategy_candidates:
        metric_name = f'return_{candidate["name"]}'
        metric_value = eval_metrics.get(metric_name, -float('inf'))
        if metric_value > best_score:
            best_score = metric_value
            best_candidate = candidate

    if best_candidate is None:
        raise ValueError('验证指标为空，无法选择最优持仓策略')

    return best_candidate, best_score


def format_strategy_metric_summary(metrics, strategy_candidates):
    """将候选持仓策略收益整理成便于打印的一行文本。"""
    parts = []
    for candidate in strategy_candidates:
        metric_name = f'return_{candidate["name"]}'
        if metric_name in metrics:
            parts.append(f'{candidate["name"]}={metrics[metric_name]:.4f}')
    return ', '.join(parts)


def format_factor_summary(feature_pipeline):
    summary = feature_pipeline['summary']
    group_parts = [
        f'{group}={count}'
        for group, count in sorted(summary['group_counts'].items())
    ]
    return (
        f"feature_set={feature_pipeline['feature_set']}, "
        f"active={summary['active_total']}, "
        f"builtin={summary['builtin_enabled']}/{summary['builtin_total']}, "
        f"builtin_overridden={summary.get('builtin_overridden', 0)}, "
        f"custom={summary['custom_enabled']}/{summary['custom_total']}, "
        f"groups=({', '.join(group_parts)})"
    )


def print_active_factors(feature_pipeline):
    grouped_specs = {}
    for spec in feature_pipeline['active_specs']:
        group = spec.get('group', 'unknown')
        label = spec['name']
        if spec.get('source') == 'custom':
            label = f'{label} [custom]'
        elif spec.get('overridden'):
            label = f'{label} [override]'
        grouped_specs.setdefault(group, []).append(label)

    print("当前启用因子明细:")
    for group, factor_names in sorted(grouped_specs.items()):
        print(f"  - {group} ({len(factor_names)}):")
        print("    " + ", ".join(factor_names))


def _build_factor_markdown(feature_pipeline):
    summary = feature_pipeline['summary']
    lines = [
        f"- feature_set: `{feature_pipeline['feature_set']}`",
        f"- factor_store: `{feature_pipeline['store_path']}`",
        f"- builtin_registry: `{feature_pipeline.get('builtin_registry_path', '')}`",
        f"- active_total: `{summary['active_total']}`",
        f"- builtin_enabled: `{summary['builtin_enabled']}/{summary['builtin_total']}`",
        f"- builtin_overridden: `{summary.get('builtin_overridden', 0)}`",
        f"- custom_enabled: `{summary['custom_enabled']}/{summary['custom_total']}`",
        f"- groups: `{json.dumps(summary['group_counts'], ensure_ascii=False)}`",
        "",
        "Active factors:",
        ", ".join(feature_pipeline['active_features']),
    ]
    if feature_pipeline['custom_specs']:
        lines.extend([
            "",
            "Custom factors:",
            json.dumps(feature_pipeline['custom_specs'], ensure_ascii=False, indent=2),
        ])
    return "\n".join(lines)


def log_factor_dashboard(writer, feature_pipeline, raw_hist_frame, scaled_hist_frame):
    if writer is None:
        return

    summary = feature_pipeline['summary']
    writer.add_text('factors/overview', _build_factor_markdown(feature_pipeline), global_step=0)
    writer.add_scalar('factors/active_total', summary['active_total'], global_step=0)
    writer.add_scalar('factors/builtin_enabled', summary['builtin_enabled'], global_step=0)
    writer.add_scalar('factors/builtin_overridden', summary.get('builtin_overridden', 0), global_step=0)
    writer.add_scalar('factors/custom_enabled', summary['custom_enabled'], global_step=0)

    for group, count in sorted(summary['group_counts'].items()):
        writer.add_scalar(f'factors/group_count/{group}', count, global_step=0)

    max_histograms = max(0, int(config.get('factor_histogram_max_features', 0)))
    if raw_hist_frame is None or scaled_hist_frame is None:
        return

    for feature_name in raw_hist_frame.columns[:max_histograms]:
        raw_values = raw_hist_frame[feature_name].to_numpy(dtype=np.float32, copy=True)
        scaled_values = scaled_hist_frame[feature_name].to_numpy(dtype=np.float32, copy=True)
        writer.add_histogram(f'factors/raw/{feature_name}', raw_values, global_step=0)
        writer.add_histogram(f'factors/scaled/{feature_name}', scaled_values, global_step=0)

class RankingDataset(torch.utils.data.Dataset):
    """排序数据集类"""
    def __init__(self, sequences, targets, relevance_scores, stock_indices):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': torch.FloatTensor(np.array(self.sequences[idx])),  # [num_stocks, seq_len, features]
            'targets': torch.FloatTensor(np.array(self.targets[idx])),      # [num_stocks] 真实涨跌幅
            'relevance': torch.LongTensor(np.array(self.relevance_scores[idx])),  # [num_stocks] 排序标签
            'stock_indices': torch.LongTensor(np.array(self.stock_indices[idx]))  # [num_stocks] 股票索引
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
            day_sequences.append(seq)
            day_targets.append(target)
            day_stock_indices.append(stock_idx)

        day_targets = np.asarray(day_targets, dtype=np.float32)
        threshold_2pct = np.quantile(day_targets, 0.98)
        relevance = (day_targets >= threshold_2pct).astype(np.float32)

        return {
            'sequences': torch.FloatTensor(np.asarray(day_sequences, dtype=np.float32)),
            'targets': torch.FloatTensor(day_targets),
            'relevance': torch.LongTensor(relevance.astype(np.int64)),
            'stock_indices': torch.LongTensor(np.asarray(day_stock_indices, dtype=np.int64)),
        }

def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]
    
    # 找到最大股票数量
    max_stocks = max(seq.size(0) for seq in sequences)
    
    # Padding到相同长度
    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []
    
    for seq, tgt, rel, stock_idx in zip(sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)
        
        # 创建padding
        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size, dtype=torch.long)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)
            
            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)
        
        # 创建mask标记有效位置
        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0
        
        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)
    
    return {
        'sequences': torch.stack(padded_sequences),      # [batch, max_stocks, seq_len, features]
        'targets': torch.stack(padded_targets),          # [batch, max_stocks]
        'relevance': torch.stack(padded_relevance),      # [batch, max_stocks]
        'stock_indices': torch.stack(padded_stock_indices),  # [batch, max_stocks]
        'masks': torch.stack(masks)                      # [batch, max_stocks]
    }


def build_lazy_ranking_index(data, features, sequence_length, min_window_end_date=None, max_window_end_date=None):
    """构建懒加载训练索引，仅保存按股票缓存和按日期索引，不保存完整窗口内容。"""
    print("正在创建排序数据集索引（懒加载版本）...")
    indexed = data.copy()
    indexed = indexed.rename(columns={'日期': 'datetime'})
    indexed['datetime'] = pd.to_datetime(indexed['datetime'])
    indexed = indexed.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    indexed = indexed.dropna(subset=['label'])

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
        dates = pd.to_datetime(group['datetime']).to_numpy()

        stock_idx = int(stock_idx)
        stock_cache[stock_idx] = {
            'features': feature_values,
            'labels': labels,
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

# 排序训练函数
def train_ranking_model(model, dataloader, criterion, optimizer, device, epoch, writer, strategy_candidates):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0
    
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        sequences = batch['sequences'].to(device)    # [batch, max_stocks, seq_len, features]
        targets = batch['targets'].to(device)        # [batch, max_stocks] 真实涨跌幅
        relevance = batch['relevance'].to(device)    # [batch, max_stocks] 预处理的相关性得分
        masks = batch['masks'].to(device)            # [batch, max_stocks] 有效位置mask
        
        optimizer.zero_grad()
        
        # 模型预测
        outputs = model(sequences)  # [batch, max_stocks] 预测分数
        
        # 应用mask，只考虑有效股票
        masked_outputs = outputs * masks + (1 - masks) * (-1e9)  # 无效位置设为很小的值
        masked_targets = targets * masks
        masked_relevance = relevance.float() * masks  # 使用预处理好的相关性得分
        
        # 计算损失（只对有效股票计算）
        batch_loss = None
        batch_size = sequences.size(0)
        
        for i in range(batch_size):
            mask = masks[i]
            valid_indices = mask.nonzero().squeeze()
            
            if valid_indices.numel() == 0:
                continue
                
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
            
            # 获取有效股票的预测值和真实收益率
            valid_pred = masked_outputs[i][valid_indices]
            # [重要] 这里我们改为直接向模型强制灌输真实的未来收益率（masked_targets）
            valid_target = masked_targets[i][valid_indices]
            
            if len(valid_pred) > 1:
                # 使用 PortfolioOptimizationLoss 直接对实际收益率进行平滑优化
                loss = criterion(valid_pred.unsqueeze(0), valid_target.unsqueeze(0))
                batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss
        
        if batch_loss is not None:
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            if not config.get('drop_clip', True):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                if writer:
                    writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch*len(dataloader)+local_step)
            optimizer.step()
            
            total_loss += batch_loss.item()
            
            # 计算评估指标
            with torch.no_grad():
                metrics = calculate_ranking_metrics(
                    masked_outputs,
                    masked_targets,
                    masks,
                    strategy_candidates=strategy_candidates,
                    temperature=config.get('softmax_temperature', 1.0),
                )
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v
            
            local_step += 1
            if writer:
                writer.add_scalar('train/loss', batch_loss.item(), global_step=epoch*len(dataloader)+local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step=epoch*len(dataloader)+local_step)
    
    # 计算平均指标
    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0, total_metrics

def evaluate_ranking_model(
    model,
    dataloader,
    criterion,
    device,
    writer,
    epoch,
    strategy_candidates,
    ablation_feature_indices=None,
):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)

            if ablation_feature_indices:
                sequences = sequences.clone()
                sequences[:, :, :, ablation_feature_indices] = 0
            
            # 模型预测
            outputs = model(sequences)
            
            # 应用mask
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks
            
            # 计算损失
            batch_loss = None
            batch_size = sequences.size(0)
            
            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()
                
                if valid_indices.numel() == 0:
                    continue
                    
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                
                valid_pred = masked_outputs[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]
                
                if len(valid_pred) > 1:
                    # 使用实际收益率进行 loss 验证
                    loss = criterion(valid_pred.unsqueeze(0), valid_true.unsqueeze(0))
                    batch_loss = batch_loss + loss if batch_loss is not None else loss
            
            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()
            
            # 计算评估指标
            metrics = calculate_ranking_metrics(
                masked_outputs,
                masked_targets,
                masks,
                strategy_candidates=strategy_candidates,
                temperature=config.get('softmax_temperature', 1.0),
            )
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v
            
            num_batches += 1
    
    # 计算平均指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches
    
    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)
    
    return avg_loss, total_metrics


def evaluate_ranking_folds(
    model,
    fold_loaders,
    criterion,
    device,
    writer,
    epoch,
    strategy_candidates,
    ablation_feature_indices=None,
):
    """在多个滚动验证折上评估，并返回折均值指标。"""
    fold_results = []
    total_loss = 0.0
    total_metrics = {}

    for fold in fold_loaders:
        fold_loss, fold_metrics = evaluate_ranking_model(
            model,
            fold['loader'],
            criterion,
            device,
            writer=None,
            epoch=epoch,
            strategy_candidates=strategy_candidates,
            ablation_feature_indices=ablation_feature_indices,
        )

        fold_result = {
            'name': fold['name'],
            'start_date': fold['start_date'],
            'end_date': fold['end_date'],
            'num_samples': fold['num_samples'],
            'loss': fold_loss,
            'metrics': fold_metrics,
        }
        fold_results.append(fold_result)

        total_loss += fold_loss
        for key, value in fold_metrics.items():
            total_metrics[key] = total_metrics.get(key, 0.0) + value

    num_folds = len(fold_results)
    avg_loss = total_loss / num_folds if num_folds > 0 else 0.0
    avg_metrics = {
        key: value / num_folds
        for key, value in total_metrics.items()
    } if num_folds > 0 else {}

    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for key, value in avg_metrics.items():
            writer.add_scalar(f'eval/{key}', value, global_step=epoch)

        for fold_result in fold_results:
            fold_prefix = f"eval_{fold_result['name']}"
            writer.add_scalar(f'{fold_prefix}/loss', fold_result['loss'], global_step=epoch)
            for key, value in fold_result['metrics'].items():
                writer.add_scalar(f'{fold_prefix}/{key}', value, global_step=epoch)

    return avg_loss, avg_metrics, fold_results


def build_factor_group_indices(feature_pipeline):
    group_indices = {}
    for feature_idx, spec in enumerate(feature_pipeline['active_specs']):
        group = spec.get('group', 'unknown')
        group_indices.setdefault(group, []).append(feature_idx)
    return group_indices


def evaluate_factor_group_ablation(
    model,
    fold_loaders,
    criterion,
    device,
    epoch,
    strategy_candidates,
    feature_pipeline,
    baseline_candidate,
):
    group_indices = build_factor_group_indices(feature_pipeline)
    ablation_results = []
    baseline_metric_name = f'return_{baseline_candidate["name"]}'

    for group_name, feature_indices in sorted(group_indices.items()):
        ablation_loss, ablation_metrics, _ = evaluate_ranking_folds(
            model,
            fold_loaders,
            criterion,
            device,
            writer=None,
            epoch=epoch,
            strategy_candidates=strategy_candidates,
            ablation_feature_indices=feature_indices,
        )

        ablated_return = ablation_metrics.get(baseline_metric_name, 0.0)
        ablation_results.append({
            'group': group_name,
            'num_features': len(feature_indices),
            'loss': ablation_loss,
            'return': ablated_return,
            'metrics': ablation_metrics,
        })

    return ablation_results


def log_factor_ablation(writer, epoch, baseline_return, ablation_results):
    if writer is None:
        return

    writer.add_scalar('factors/ablation/baseline_return', baseline_return, global_step=epoch)
    for result in ablation_results:
        group = result['group']
        delta = result['return'] - baseline_return
        writer.add_scalar(f'factors/ablation/{group}/return', result['return'], global_step=epoch)
        writer.add_scalar(f'factors/ablation/{group}/delta', delta, global_step=epoch)
        writer.add_scalar(f'factors/ablation/{group}/num_features', result['num_features'], global_step=epoch)


def predict_top_stocks(model, data, features, sequence_length, scaler, stockid2idx, device, top_k=5):
    """
    预测某一天涨幅前top_k的股票
    """
    model.eval()
    
    # 获取最后一天的数据作为预测基础
    latest_date = data['日期'].max()
    
    # 准备预测数据
    day_sequences = []
    day_stock_codes = []
    day_stock_indices = []
    
    for stock_code in data['股票代码'].unique():
        # 获取该股票历史sequence_length天的数据
        stock_history = data[
            (data['股票代码'] == stock_code) & 
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)
        
        if len(stock_history) == sequence_length:
            seq = stock_history[features].values
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)
            day_stock_indices.append(stockid2idx[stock_code])
    
    if len(day_sequences) == 0:
        return []
    
    # 转换为tensor
    sequences = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)  # [1, num_stocks, seq_len, features]
    
    with torch.no_grad():
        # 模型预测
        outputs = model(sequences)  # [1, num_stocks]
        scores = outputs.squeeze().cpu().numpy()  # [num_stocks]
        
        # 获取排名前top_k的股票
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        top_stocks = []
        for idx in top_indices:
            top_stocks.append({
                'stock_code': day_stock_codes[idx],
                'predicted_score': scores[idx],
                'rank': len(top_stocks) + 1
            })
    
    return top_stocks

def save_predictions(top_stocks, output_path):
    """保存预测结果"""
    results = []
    for stock in top_stocks:
        results.append({
            '排名': stock['rank'],
            '股票代码': stock['stock_code'],
            '预测分数': stock['predicted_score']
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {output_path}")


def split_train_val_by_last_month(df, sequence_length):
    """按最后一个月做验证集划分，并为验证集补充序列上下文。"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=2)).normalize()

    # 验证集需要保留前 sequence_length-1 个交易日作为序列上下文，
    # 这样第一个验证样本的窗口结束日就可以落在 val_start。
    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围(最后一个月): {val_start.date()} 到 {last_date.date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    # 恢复为字符串，保持与原流程一致
    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start


def build_rolling_validation_folds(df, sequence_length):
    """构建滚动验证折，并保证所有验证折都不被训练集未来数据污染。"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    unique_dates = [pd.Timestamp(d).normalize() for d in sorted(df['日期'].unique())]
    label_ready_dates = unique_dates[:-5]

    num_folds = int(config.get('rolling_val_num_folds', 4))
    window_size = int(config.get('rolling_val_window_size', 20))
    step_size = int(config.get('rolling_val_step_size', window_size))

    if num_folds <= 0:
        raise ValueError("rolling_val_num_folds 必须大于 0")
    if window_size <= 0 or step_size <= 0:
        raise ValueError("rolling_val_window_size 和 rolling_val_step_size 必须大于 0")

    required_dates = window_size + (num_folds - 1) * step_size
    if len(label_ready_dates) < required_dates:
        raise ValueError(
            f"可用于滚动验证的交易日不足: 需要至少 {required_dates} 天，当前仅有 {len(label_ready_dates)} 天"
        )

    reverse_bounds = []
    last_end_idx = len(label_ready_dates) - 1
    for offset in range(num_folds):
        end_idx = last_end_idx - offset * step_size
        start_idx = end_idx - window_size + 1
        if start_idx < 0:
            raise ValueError("滚动验证窗口越界，请减小折数或窗口大小")
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
        })

    earliest_start = folds[0]['start_date']
    earliest_start_idx = unique_dates.index(earliest_start)
    if earliest_start_idx <= 0:
        raise ValueError("滚动验证起点过早，没有可用于训练的历史数据")

    context_start_idx = max(0, earliest_start_idx - (sequence_length - 1))
    val_context_start = unique_dates[context_start_idx]

    train_df = df[df['日期'] < earliest_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {df['日期'].max().date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"滚动验证实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")
    print(
        "滚动验证参数: "
        f"folds={num_folds}, window_size={window_size}, step_size={step_size}"
    )
    print("滚动验证折:")
    for fold in folds:
        print(f"  - {fold['name']}: {fold['start_date'].date()} 到 {fold['end_date'].date()}")

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')
    return train_df, val_df, folds


def build_validation_fold_loaders(val_data, features, val_folds):
    """为每个滚动验证折构建独立的数据集与 DataLoader。"""
    fold_loaders = []
    total_samples = 0

    for fold in val_folds:
        sequences, targets, relevance, stock_indices = create_ranking_dataset_vectorized(
            val_data,
            features,
            config['sequence_length'],
            ranking_data_path=None,
            min_window_end_date=fold['start_date'].strftime('%Y-%m-%d'),
            max_window_end_date=fold['end_date'].strftime('%Y-%m-%d'),
        )

        if len(sequences) == 0:
            raise ValueError(
                f"{fold['name']} ({fold['start_date'].date()} ~ {fold['end_date'].date()}) 未生成任何验证样本"
            )

        dataset = RankingDataset(sequences, targets, relevance, stock_indices)
        loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
        )

        fold_loaders.append({
            'name': fold['name'],
            'start_date': fold['start_date'],
            'end_date': fold['end_date'],
            'num_samples': len(sequences),
            'loader': loader,
        })
        total_samples += len(sequences)

        print(
            f"验证折 {fold['name']} 样本数: {len(sequences)} "
            f"({fold['start_date'].date()} ~ {fold['end_date'].date()})"
        )

    print(f"滚动验证总样本数: {total_samples}")
    return fold_loaders

# 主程序
def main():
    set_seed(config.get('seed', 42))
    output_dir = config['output_dir']
    os.makedirs(output_dir,exist_ok=True)
    # 保存在output_dir中保存当前的配置文件，以便复现
    data_path = config['data_path']
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    is_train = True
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log')) if is_train else None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if device.type == 'cuda':
        device_msg = f"当前训练设备: cuda ({torch.cuda.get_device_name(device)})"
    elif device.type == 'mps':
        device_msg = "当前训练设备: mps (Apple Silicon)"
    else:
        device_msg = "当前训练设备: cpu"
    print(device_msg)
    
    # 1. 数据加载
    data_file = os.path.join(data_path, 'train.csv')
    full_df = pd.read_csv(data_file)
    factor_pipeline = resolve_factor_pipeline(
        config['feature_num'],
        config['factor_store_path'],
        config['builtin_factor_registry_path'],
    )
    save_factor_snapshot(factor_pipeline, os.path.join(output_dir, 'active_factors.json'))
    print("当前因子配置:", format_factor_summary(factor_pipeline))
    print_active_factors(factor_pipeline)
    validation_mode = config.get('validation_mode', 'rolling')
    if validation_mode == 'rolling':
        train_df, val_df, val_folds = build_rolling_validation_folds(full_df, config['sequence_length'])
    else:
        train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'])
        val_folds = [{
            'name': 'holdout',
            'start_date': pd.Timestamp(val_start).normalize(),
            'end_date': pd.to_datetime(val_df['日期']).max().normalize(),
        }]
    
    # 获取所有股票ID，建立映射
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)
    
    # 2. 特征工程与预处理
    train_data, features = preprocess_data(train_df, factor_pipeline, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, factor_pipeline, stockid2idx=stockid2idx)
    
    # 3. 标准化
    scaler = StandardScaler()

    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    # 丢弃nan数据
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    # 然后再缩放
    histogram_features = features[:max(0, int(config.get('factor_histogram_max_features', 0)))]
    raw_train_hist_frame = train_data[histogram_features].copy() if histogram_features else None
    train_data[features] = scaler.fit_transform(train_data[features]).astype(np.float32)
    val_data[features] = scaler.transform(val_data[features]).astype(np.float32)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    scaled_train_hist_frame = train_data[histogram_features] if histogram_features else None
    log_factor_dashboard(writer, factor_pipeline, raw_train_hist_frame, scaled_train_hist_frame)

    
    # 4. 创建排序数据集
    train_stock_cache, train_day_entries = build_lazy_ranking_index(
        train_data,
        features,
        config['sequence_length'],
    )
    print(f"训练集样本数: {len(train_day_entries)}")
    val_fold_loaders = build_validation_fold_loaders(val_data, features, val_folds)
    
    # 5. 创建排序数据集和数据加载器
    train_dataset = LazyRankingDataset(train_stock_cache, train_day_entries, config['sequence_length'])
    del train_data
    del val_data
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # 减少worker数量避免内存问题
        pin_memory=False
    )
    
    # 6. 模型初始化
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    strategy_candidates = build_strategy_candidates()
    print("候选持仓策略:", ", ".join(candidate['name'] for candidate in strategy_candidates))
    
    # 7. 损失函数和优化器
    criterion = PortfolioOptimizationLoss(temperature=10.0, pairwise_weight=config.get('pairwise_weight', 1.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.2, total_iters=config['num_epochs'])
    
    # 8. 排序模型训练
    if is_train:
        best_score = -float('inf')
        best_epoch = -1
        
        for epoch in range(config['num_epochs']):
            print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")
            
            # 训练
            train_loss, train_metrics = train_ranking_model(
                model, train_loader, criterion, optimizer, device, epoch, writer, strategy_candidates
            )
            
            print(f"Train Loss: {train_loss:.4f}")
            for k, v in train_metrics.items():
                print(f"Train {k}: {v:.4f}")
            
            # 验证
            eval_loss, eval_metrics, fold_results = evaluate_ranking_folds(
                model, val_fold_loaders, criterion, device, writer, epoch, strategy_candidates
            )
            
            print(f"Eval Loss: {eval_loss:.4f}")
            for k, v in eval_metrics.items():
                print(f"Eval {k}: {v:.4f}")
            print(
                "Eval 策略收益汇总: "
                + format_strategy_metric_summary(eval_metrics, strategy_candidates)
            )
            for fold_result in fold_results:
                fold_best_candidate, fold_best_score = choose_best_strategy(
                    fold_result['metrics'], strategy_candidates
                )
                print(
                    f"Eval {fold_result['name']} "
                    f"({fold_result['start_date'].date()} ~ {fold_result['end_date'].date()}) "
                    f"样本数: {fold_result['num_samples']} | "
                    f"Loss: {fold_result['loss']:.4f} | "
                    f"best={fold_best_candidate['name']}:{fold_best_score:.4f}"
                )
                print(
                    "  策略收益: "
                    + format_strategy_metric_summary(fold_result['metrics'], strategy_candidates)
                )
            
            # 学习率调度
            scheduler.step()
            if writer:
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step=epoch)
            

            best_candidate, current_final_score = choose_best_strategy(eval_metrics, strategy_candidates)
            print(f"当前最优持仓策略: {best_candidate['name']} | 验证收益: {current_final_score:.4f}")

            if config.get('factor_ablation_enabled', True):
                ablation_results = evaluate_factor_group_ablation(
                    model,
                    val_fold_loaders,
                    criterion,
                    device,
                    epoch,
                    strategy_candidates,
                    factor_pipeline,
                    best_candidate,
                )
                log_factor_ablation(writer, epoch, current_final_score, ablation_results)
                print("因子分组消融:")
                for result in ablation_results:
                    delta = result['return'] - current_final_score
                    print(
                        f"  - {result['group']}: "
                        f"features={result['num_features']}, "
                        f"return={result['return']:.4f}, "
                        f"delta={delta:.4f}"
                    )

            if current_final_score > best_score:
                best_score = current_final_score
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                with open(os.path.join(output_dir, 'best_strategy.json'), 'w') as f:
                    json.dump({
                        'name': best_candidate['name'],
                        'top_k': best_candidate['top_k'],
                        'weighting': best_candidate['weighting'],
                        'temperature': config.get('softmax_temperature', 1.0),
                        'validation_return': current_final_score,
                        'validation_mode': validation_mode,
                        'validation_folds': [
                            {
                                'name': fold['name'],
                                'start_date': fold['start_date'].strftime('%Y-%m-%d'),
                                'end_date': fold['end_date'].strftime('%Y-%m-%d'),
                            }
                            for fold in val_folds
                        ],
                        'best_epoch': best_epoch,
                    }, f, indent=4, ensure_ascii=False)
                print(f"保存最佳模型 - final score: {best_score:.4f}")
        print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 final score: {best_score:.4f}")
        with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
            f.write(f"Best epoch: {best_epoch}\nBest final_score: {best_score:.6f}\n")

        if writer:
            writer.close()

        return best_score

if __name__ == "__main__":
    # 多进程保护
    mp.set_start_method('spawn', force=True)
    best_score = main()
    print(f"\n########## 训练完成！最佳 final score: {best_score:.4f} ##########")
