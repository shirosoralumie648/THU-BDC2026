import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def _normalize_scale_windows(windows, sequence_length, default_windows):
    if windows is None:
        windows = default_windows
    if isinstance(windows, (int, float, str)):
        windows = [windows]

    normalized = []
    for w in windows:
        try:
            w_int = int(w)
        except (TypeError, ValueError):
            continue
        w_int = max(1, min(int(sequence_length), w_int))
        if w_int not in normalized:
            normalized.append(w_int)

    if not normalized:
        fallback = []
        for w in default_windows:
            w_int = max(1, min(int(sequence_length), int(w)))
            if w_int not in fallback:
                fallback.append(w_int)
        normalized = fallback if fallback else [max(1, int(sequence_length))]

    return normalized

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
class CrossStockAttention(nn.Module):
    """股票间交互注意力模块"""
    def __init__(self, d_model, nhead, dropout=0.1, config=None):
        super(CrossStockAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.nhead = int(nhead)

        config = config or {}
        self.use_sparse_mask = bool(config.get('use_cross_stock_attention_mask', True))
        self.mask_mode = str(config.get('cross_stock_mask_mode', 'similarity')).lower()
        self.similarity_topk = max(1, int(config.get('cross_stock_similarity_topk', 40)))
        self.prior_similarity_combine = str(
            config.get('prior_similarity_combine', 'intersection')
        ).lower()
        self.use_industry_virtual_stock = bool(config.get('use_industry_virtual_stock', False))
        self.industry_virtual_connect_mode = str(
            config.get('industry_virtual_connect_mode', 'same')
        ).lower()
        if self.industry_virtual_connect_mode not in {'same', 'all'}:
            raise ValueError(
                f'不支持的 industry_virtual_connect_mode: {self.industry_virtual_connect_mode}'
            )
        self.industry_virtual_min_members = max(
            1,
            int(config.get('industry_virtual_min_members', 1)),
        )
        self.register_buffer(
            'prior_graph_adj',
            torch.empty(0, 0, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            'stock_to_industry_idx',
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )

    def set_prior_graph(self, prior_graph_adj):
        """设置股票先验关系图邻接矩阵，形状 [num_stocks, num_stocks]。"""
        if prior_graph_adj is None:
            self.prior_graph_adj = torch.empty(
                0,
                0,
                dtype=torch.bool,
                device=self.prior_graph_adj.device,
            )
            return

        if not isinstance(prior_graph_adj, torch.Tensor):
            prior_graph_adj = torch.as_tensor(prior_graph_adj)
        if prior_graph_adj.dim() != 2 or prior_graph_adj.size(0) != prior_graph_adj.size(1):
            raise ValueError(
                f'prior_graph_adj 形状非法: {tuple(prior_graph_adj.shape)}，期望 [N, N]'
            )
        self.prior_graph_adj = prior_graph_adj.to(
            device=self.prior_graph_adj.device,
            dtype=torch.bool,
        )

    def set_stock_industry_index(self, stock_to_industry_idx):
        """
        设置股票索引 -> 行业索引映射，形状 [num_stocks]，未知行业用 -1。
        """
        if stock_to_industry_idx is None:
            self.stock_to_industry_idx = torch.empty(
                0,
                dtype=torch.long,
                device=self.stock_to_industry_idx.device,
            )
            return

        if not isinstance(stock_to_industry_idx, torch.Tensor):
            stock_to_industry_idx = torch.as_tensor(stock_to_industry_idx)
        if stock_to_industry_idx.dim() != 1:
            raise ValueError(
                f'stock_to_industry_idx 形状非法: {tuple(stock_to_industry_idx.shape)}，期望 [N]'
            )
        self.stock_to_industry_idx = stock_to_industry_idx.to(
            device=self.stock_to_industry_idx.device,
            dtype=torch.long,
        )

    def _expand_multihead_blocked_mask(self, blocked):
        batch_size, num_stocks, _ = blocked.size()
        blocked = blocked.unsqueeze(1).expand(-1, self.nhead, -1, -1)
        return blocked.reshape(batch_size * self.nhead, num_stocks, num_stocks)

    def _stabilize_padding_queries(self, allowed, stock_valid_mask):
        """
        对于 padding query，放开到所有有效 key，避免该行全部 -inf 造成数值问题。
        """
        batch_size = allowed.size(0)
        for b in range(batch_size):
            valid_key_row = stock_valid_mask[b]
            if valid_key_row.any():
                allowed[b, ~stock_valid_mask[b], :] = valid_key_row
            else:
                allowed[b, ~stock_valid_mask[b], :] = True
        return allowed

    def _build_similarity_sparse_mask(self, stock_features, stock_valid_mask):
        """
        构造基于同日股票表征余弦相似度的稀疏注意力掩码。
        返回 MultiheadAttention 需要的 bool 掩码:
        True = 禁止关注，False = 允许关注。
        形状: [batch * nhead, num_stocks, num_stocks]
        """
        batch_size, num_stocks, _ = stock_features.size()
        if num_stocks <= 1:
            return None

        normalized = F.normalize(stock_features, p=2, dim=-1, eps=1e-12)
        similarity = torch.matmul(normalized, normalized.transpose(1, 2))  # [B, N, N]

        valid_keys = stock_valid_mask.unsqueeze(1).expand(-1, num_stocks, -1)  # [B, N, N]
        similarity = similarity.masked_fill(~valid_keys, float('-inf'))

        topk = min(self.similarity_topk, num_stocks)
        topk_indices = torch.topk(similarity, k=topk, dim=-1).indices  # [B, N, topk]

        allowed = torch.zeros(batch_size, num_stocks, num_stocks, dtype=torch.bool, device=stock_features.device)
        allowed.scatter_(dim=2, index=topk_indices, value=True)

        eye = torch.eye(num_stocks, dtype=torch.bool, device=stock_features.device).unsqueeze(0)
        valid_pairs = stock_valid_mask.unsqueeze(1) & stock_valid_mask.unsqueeze(2)
        allowed = (allowed | (eye & valid_pairs)) & valid_keys

        allowed = self._stabilize_padding_queries(allowed, stock_valid_mask)

        blocked = ~allowed  # True 表示禁止关注
        return self._expand_multihead_blocked_mask(blocked)

    def _build_prior_sparse_mask(self, stock_indices, stock_valid_mask):
        if stock_indices is None:
            return None
        if self.prior_graph_adj.numel() == 0:
            return None

        batch_size, num_stocks = stock_indices.size()
        graph_size = self.prior_graph_adj.size(0)
        if graph_size <= 0:
            return None

        safe_indices = stock_indices.long().clamp(min=0, max=graph_size - 1)
        prior_adj = self.prior_graph_adj[
            safe_indices.unsqueeze(-1),
            safe_indices.unsqueeze(-2),
        ]  # [B, N, N]

        valid_keys = stock_valid_mask.unsqueeze(1).expand(-1, num_stocks, -1)
        valid_pairs = stock_valid_mask.unsqueeze(1) & stock_valid_mask.unsqueeze(2)
        allowed = (prior_adj & valid_pairs) & valid_keys

        eye = torch.eye(num_stocks, dtype=torch.bool, device=stock_indices.device).unsqueeze(0)
        allowed = allowed | (eye & valid_pairs)
        allowed = self._stabilize_padding_queries(allowed, stock_valid_mask)

        blocked = ~allowed
        return self._expand_multihead_blocked_mask(blocked)

    def _combine_blocked_masks(self, blocked_a, blocked_b):
        if blocked_a is None:
            return blocked_b
        if blocked_b is None:
            return blocked_a

        if self.prior_similarity_combine in {'intersection', 'and'}:
            # 允许边取交集（更稀疏）：blocked = blocked_a OR blocked_b
            return blocked_a | blocked_b
        if self.prior_similarity_combine in {'union', 'or'}:
            # 允许边取并集（更宽松）：blocked = blocked_a AND blocked_b
            return blocked_a & blocked_b
        raise ValueError(
            f'不支持的 prior_similarity_combine: {self.prior_similarity_combine}'
        )

    def _build_stock_blocked_mask(self, stock_features, stock_indices=None, stock_valid_mask=None):
        blocked = self._build_attention_mask(
            stock_features,
            stock_indices=stock_indices,
            stock_valid_mask=stock_valid_mask,
        )
        if blocked is None:
            return None
        batch_size, num_stocks, _ = stock_features.size()
        return blocked.view(batch_size, self.nhead, num_stocks, num_stocks)[:, 0, :, :]

    def _lookup_batch_industry_ids(self, stock_indices, stock_valid_mask):
        if stock_indices is None or self.stock_to_industry_idx.numel() == 0:
            return None

        graph_size = int(self.stock_to_industry_idx.size(0))
        if graph_size <= 0:
            return None

        safe_indices = stock_indices.long().clamp(min=0, max=graph_size - 1)
        industry_ids = self.stock_to_industry_idx[safe_indices]

        invalid_index = (stock_indices < 0) | (stock_indices >= graph_size)
        industry_ids = industry_ids.masked_fill(invalid_index, -1)
        industry_ids = industry_ids.masked_fill(~stock_valid_mask.bool(), -1)
        return industry_ids

    def _build_industry_virtual_tokens(self, stock_features, stock_indices, stock_valid_mask):
        industry_ids = self._lookup_batch_industry_ids(stock_indices, stock_valid_mask)
        if industry_ids is None:
            return None, None, None, None

        batch_size, _, d_model = stock_features.size()
        max_tokens = 0
        token_features_by_batch = []
        token_ids_by_batch = []

        for b in range(batch_size):
            ids = industry_ids[b]
            valid_ids = ids[ids >= 0]
            if valid_ids.numel() == 0:
                token_features_by_batch.append(stock_features.new_zeros((0, d_model)))
                token_ids_by_batch.append(torch.empty(0, dtype=torch.long, device=stock_features.device))
                continue

            unique_ids = torch.unique(valid_ids, sorted=True)
            cur_features = []
            cur_ids = []
            for industry_id in unique_ids.tolist():
                member_mask = ids == int(industry_id)
                member_count = int(member_mask.sum().item())
                if member_count < self.industry_virtual_min_members:
                    continue
                cur_features.append(stock_features[b, member_mask, :].mean(dim=0))
                cur_ids.append(int(industry_id))

            if cur_features:
                token_features = torch.stack(cur_features, dim=0)
                token_ids = torch.as_tensor(cur_ids, dtype=torch.long, device=stock_features.device)
            else:
                token_features = stock_features.new_zeros((0, d_model))
                token_ids = torch.empty(0, dtype=torch.long, device=stock_features.device)

            token_features_by_batch.append(token_features)
            token_ids_by_batch.append(token_ids)
            max_tokens = max(max_tokens, int(token_features.size(0)))

        if max_tokens <= 0:
            return None, None, None, None

        industry_tokens = stock_features.new_zeros((batch_size, max_tokens, d_model))
        industry_valid_mask = torch.zeros(
            (batch_size, max_tokens),
            dtype=torch.bool,
            device=stock_features.device,
        )
        industry_token_ids = torch.full(
            (batch_size, max_tokens),
            -1,
            dtype=torch.long,
            device=stock_features.device,
        )

        for b, (token_features, token_ids) in enumerate(zip(token_features_by_batch, token_ids_by_batch)):
            token_count = int(token_features.size(0))
            if token_count <= 0:
                continue
            industry_tokens[b, :token_count, :] = token_features
            industry_valid_mask[b, :token_count] = True
            industry_token_ids[b, :token_count] = token_ids

        return industry_tokens, industry_valid_mask, industry_token_ids, industry_ids

    def _build_augmented_blocked_mask(
        self,
        stock_blocked,
        stock_valid_mask,
        industry_valid_mask,
        industry_ids,
        industry_token_ids,
    ):
        batch_size, num_stocks = stock_valid_mask.size()
        num_industry_tokens = industry_valid_mask.size(1)
        total_tokens = num_stocks + num_industry_tokens

        blocked = torch.ones(
            (batch_size, total_tokens, total_tokens),
            dtype=torch.bool,
            device=stock_valid_mask.device,
        )

        for b in range(batch_size):
            valid_stocks = stock_valid_mask[b].bool()
            valid_industry_tokens = industry_valid_mask[b].bool()

            if stock_blocked is None:
                stock_allowed = valid_stocks.unsqueeze(1) & valid_stocks.unsqueeze(0)
                stock_allowed = stock_allowed | torch.eye(
                    num_stocks,
                    dtype=torch.bool,
                    device=stock_valid_mask.device,
                )
                blocked[b, :num_stocks, :num_stocks] = ~stock_allowed
            else:
                blocked[b, :num_stocks, :num_stocks] = stock_blocked[b]

            # 行业查询默认仅保留自环（行业 token 主要作为 K/V 给股票使用）
            blocked[b, num_stocks:, :] = True
            industry_positions = torch.nonzero(valid_industry_tokens, as_tuple=False).flatten()
            for pos in industry_positions.tolist():
                row = num_stocks + int(pos)
                blocked[b, row, row] = False

            if not valid_industry_tokens.any():
                continue

            if self.industry_virtual_connect_mode == 'all':
                blocked[b, :num_stocks, num_stocks:] = (~valid_industry_tokens).unsqueeze(0).expand(
                    num_stocks, -1
                )
                continue

            blocked[b, :num_stocks, num_stocks:] = True
            token_ids = industry_token_ids[b]
            for stock_pos in torch.nonzero(valid_stocks, as_tuple=False).flatten().tolist():
                stock_industry = int(industry_ids[b, stock_pos].item())
                if stock_industry < 0:
                    continue
                allowed = valid_industry_tokens & (token_ids == stock_industry)
                if allowed.any():
                    blocked[b, stock_pos, num_stocks:] = ~allowed

        return blocked

    def _build_attention_inputs_with_industry_virtual(
        self,
        stock_features,
        stock_indices=None,
        stock_valid_mask=None,
    ):
        if not self.use_industry_virtual_stock:
            return stock_features, stock_valid_mask, self._build_attention_mask(
                stock_features,
                stock_indices=stock_indices,
                stock_valid_mask=stock_valid_mask,
            )

        industry_tokens, industry_valid_mask, industry_token_ids, industry_ids = self._build_industry_virtual_tokens(
            stock_features,
            stock_indices,
            stock_valid_mask,
        )
        if industry_tokens is None:
            return stock_features, stock_valid_mask, self._build_attention_mask(
                stock_features,
                stock_indices=stock_indices,
                stock_valid_mask=stock_valid_mask,
            )

        stock_blocked = self._build_stock_blocked_mask(
            stock_features,
            stock_indices=stock_indices,
            stock_valid_mask=stock_valid_mask,
        )
        blocked = self._build_augmented_blocked_mask(
            stock_blocked,
            stock_valid_mask.bool(),
            industry_valid_mask,
            industry_ids,
            industry_token_ids,
        )

        augmented_features = torch.cat([stock_features, industry_tokens], dim=1)
        augmented_valid_mask = torch.cat([stock_valid_mask.bool(), industry_valid_mask], dim=1)

        allowed = ~blocked
        allowed = self._stabilize_padding_queries(allowed, augmented_valid_mask)
        blocked = ~allowed
        return augmented_features, augmented_valid_mask, self._expand_multihead_blocked_mask(blocked)

    def _build_attention_mask(self, stock_features, stock_indices=None, stock_valid_mask=None):
        if (not self.use_sparse_mask) or self.mask_mode == 'full':
            return None

        if stock_valid_mask is None:
            stock_valid_mask = torch.ones(
                stock_features.size(0),
                stock_features.size(1),
                dtype=torch.bool,
                device=stock_features.device,
            )
        else:
            stock_valid_mask = stock_valid_mask.bool()

        if self.mask_mode == 'similarity':
            return self._build_similarity_sparse_mask(stock_features, stock_valid_mask)
        if self.mask_mode == 'prior':
            prior_mask = self._build_prior_sparse_mask(stock_indices, stock_valid_mask)
            if prior_mask is not None:
                return prior_mask
            return self._build_similarity_sparse_mask(stock_features, stock_valid_mask)
        if self.mask_mode == 'prior_similarity':
            prior_mask = self._build_prior_sparse_mask(stock_indices, stock_valid_mask)
            similarity_mask = self._build_similarity_sparse_mask(stock_features, stock_valid_mask)
            return self._combine_blocked_masks(prior_mask, similarity_mask)

        return None

    def forward(self, stock_features, stock_indices=None, stock_valid_mask=None):
        # stock_features: [batch, num_stocks, d_model]
        # 股票间交互：默认相似度稀疏注意力，抑制全连接噪声扩散。
        if stock_valid_mask is None:
            stock_valid_mask = torch.ones(
                stock_features.size(0),
                stock_features.size(1),
                dtype=torch.bool,
                device=stock_features.device,
            )
        else:
            stock_valid_mask = stock_valid_mask.bool()

        attn_features, attn_valid_mask, attn_mask = self._build_attention_inputs_with_industry_virtual(
            stock_features,
            stock_indices=stock_indices,
            stock_valid_mask=stock_valid_mask,
        )
        attended, _ = self.cross_attention(
            attn_features,
            attn_features,
            attn_features,
            key_padding_mask=~attn_valid_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        attended = attended[:, :stock_features.size(1), :]
        output = self.norm(stock_features + self.dropout(attended))
        return output

class FeatureAttention(nn.Module):
    """特征注意力模块"""
    def __init__(self, d_model, dropout=0.1):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch*num_stocks, seq_len, d_model]
        attention_weights = self.attention(x)  # [batch*num_stocks, seq_len, 1]
        attended = torch.sum(x * attention_weights, dim=1)  # [batch*num_stocks, d_model]
        return self.dropout(attended)

class StockTransformer(nn.Module):
    def __init__(self, input_dim, config, num_stocks, emb_dim=16):
        super(StockTransformer, self).__init__()
        self.model_type = 'RankingTransformer'
        self.config = config
        self.num_stocks = num_stocks
        self.sequence_length = int(config['sequence_length'])
        self.use_market_gating = bool(config.get('use_market_gating', True))
        self.market_gate_residual = float(config.get('market_gate_residual', 0.5))
        self.use_market_gating_macro_context = bool(
            config.get('use_market_gating_macro_context', True)
        )
        self.market_gate_macro_weight = float(config.get('market_gate_macro_weight', 0.3))
        market_context_feature_names = config.get(
            'market_gating_context_feature_names',
            [
                'market_median_return',
                'market_total_turnover_log',
                'market_limit_up_count_log',
                'market_limit_up_ratio',
            ],
        )
        self.market_context_dim = max(1, len(market_context_feature_names))
        self.use_multitask_volatility = bool(config.get('use_multitask_volatility', False))
        self.use_multi_scale_temporal = bool(config.get('use_multi_scale_temporal', True))
        self.use_ultra_short_branch = bool(config.get('use_ultra_short_branch', True))
        self.use_temporal_cross_stock_attention = bool(
            config.get('use_temporal_cross_stock_attention', True)
        )
        self.industry_virtual_on_temporal_cross_stock = bool(
            config.get('industry_virtual_on_temporal_cross_stock', False)
        )
        self.temporal_cross_stock_nhead = int(
            config.get('temporal_cross_stock_nhead', config['nhead'])
        )
        if config['d_model'] % self.temporal_cross_stock_nhead != 0:
            raise ValueError(
                f"d_model({config['d_model']}) 不能被 temporal_cross_stock_nhead({self.temporal_cross_stock_nhead}) 整除"
            )
        self.ultra_short_windows = _normalize_scale_windows(
            config.get('multi_scale_ultra_short_windows', [1, 2, 3]),
            self.sequence_length,
            default_windows=[1, 2, 3],
        )
        self.short_windows = _normalize_scale_windows(
            config.get('multi_scale_short_windows', [5, 10]),
            self.sequence_length,
            default_windows=[5, 10],
        )
        self.long_windows = _normalize_scale_windows(
            config.get('multi_scale_long_windows', [20, 40, 60]),
            self.sequence_length,
            default_windows=[20, 40, 60],
        )
        self.multi_scale_window_reduce = str(config.get('multi_scale_window_reduce', 'mean')).lower()
        if self.multi_scale_window_reduce not in {'mean', 'last'}:
            raise ValueError(f"不支持的 multi_scale_window_reduce: {self.multi_scale_window_reduce}")
        self.multi_scale_fusion = str(config.get('multi_scale_fusion', 'gated')).lower()
        if self.multi_scale_fusion not in {'gated', 'weighted_sum'}:
            raise ValueError(f"不支持的 multi_scale_fusion: {self.multi_scale_fusion}")
        self.register_buffer(
            'prior_graph_adj',
            torch.empty(0, 0, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            'market_context_feature_indices',
            torch.full((self.market_context_dim,), -1, dtype=torch.long),
            persistent=False,
        )

        if self.use_market_gating:
            gate_hidden_dim = int(config.get('market_gate_hidden_dim', max(32, input_dim // 2)))
            gate_hidden_dim = max(8, gate_hidden_dim)
            self.market_gate = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(config['dropout'] * 0.5),
                nn.Linear(gate_hidden_dim, input_dim),
                nn.Sigmoid(),
            )
            self.market_macro_proj = None
            if self.use_market_gating_macro_context:
                macro_hidden_dim = int(
                    config.get('market_gate_macro_hidden_dim', max(16, input_dim // 4))
                )
                macro_hidden_dim = max(8, macro_hidden_dim)
                self.market_macro_proj = nn.Sequential(
                    nn.LayerNorm(self.market_context_dim),
                    nn.Linear(self.market_context_dim, macro_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config['dropout'] * 0.5),
                    nn.Linear(macro_hidden_dim, input_dim),
                )
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, config['d_model'])
        self.pos_encoder = PositionalEncoding(config['d_model'], config['dropout'], self.sequence_length)

        def _build_temporal_encoder():
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config['d_model'],
                nhead=config['nhead'],
                dim_feedforward=config['dim_feedforward'],
                dropout=config['dropout'],
                batch_first=True
            )
            return nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])

        # 基础时序分支
        self.temporal_encoder = _build_temporal_encoder()
        
        # 基础时序聚合
        self.feature_attention = FeatureAttention(config['d_model'], config['dropout'])

        # 时间步级跨股交互（文献对齐：先做同一时间切片的股票关系，再做时序建模）
        self.temporal_cross_stock_attention = None
        if self.use_temporal_cross_stock_attention:
            temporal_mask_cfg = {
                'use_cross_stock_attention_mask': bool(
                    config.get(
                        'use_temporal_cross_stock_attention_mask',
                        config.get('use_cross_stock_attention_mask', True),
                    )
                ),
                'cross_stock_mask_mode': str(
                    config.get(
                        'temporal_cross_stock_mask_mode',
                        config.get('cross_stock_mask_mode', 'similarity'),
                    )
                ),
                'cross_stock_similarity_topk': max(
                    1,
                    int(
                        config.get(
                            'temporal_cross_stock_similarity_topk',
                            config.get('cross_stock_similarity_topk', 40),
                        )
                    ),
                ),
                'prior_similarity_combine': str(
                    config.get(
                        'temporal_prior_similarity_combine',
                        config.get('prior_similarity_combine', 'intersection'),
                    )
                ),
                'use_industry_virtual_stock': bool(
                    config.get('industry_virtual_on_temporal_cross_stock', False)
                ),
                'industry_virtual_connect_mode': str(
                    config.get(
                        'industry_virtual_connect_mode',
                        'same',
                    )
                ),
                'industry_virtual_min_members': int(
                    config.get('industry_virtual_min_members', 1)
                ),
            }
            self.temporal_cross_stock_attention = CrossStockAttention(
                config['d_model'],
                self.temporal_cross_stock_nhead,
                config['dropout'],
                config=temporal_mask_cfg,
            )

        # 多尺度时序分支（短周期 + 长周期）
        self.ultra_short_temporal_encoder = None
        self.ultra_short_feature_attention = None
        self.short_temporal_encoder = None
        self.long_temporal_encoder = None
        self.short_feature_attention = None
        self.long_feature_attention = None
        self.short_horizon_fusion_gate = None
        self.short_horizon_norm = None
        self.short_horizon_dropout = None
        self.multi_scale_fusion_gate = None
        self.multi_scale_branch_norm = None
        self.multi_scale_branch_dropout = None
        self.multi_scale_branch_logits = None
        if self.use_multi_scale_temporal:
            if self.use_ultra_short_branch:
                self.ultra_short_temporal_encoder = _build_temporal_encoder()
                self.ultra_short_feature_attention = FeatureAttention(config['d_model'], config['dropout'])
            self.short_temporal_encoder = _build_temporal_encoder()
            self.long_temporal_encoder = _build_temporal_encoder()
            self.short_feature_attention = FeatureAttention(config['d_model'], config['dropout'])
            self.long_feature_attention = FeatureAttention(config['d_model'], config['dropout'])
            if self.use_ultra_short_branch:
                self.short_horizon_fusion_gate = nn.Sequential(
                    nn.LayerNorm(config['d_model'] * 2),
                    nn.Linear(config['d_model'] * 2, config['d_model']),
                    nn.GELU(),
                    nn.Dropout(config['dropout']),
                    nn.Linear(config['d_model'], 2),
                )
                self.short_horizon_norm = nn.LayerNorm(config['d_model'])
                self.short_horizon_dropout = nn.Dropout(config['dropout'])
            self.multi_scale_branch_norm = nn.LayerNorm(config['d_model'])
            self.multi_scale_branch_dropout = nn.Dropout(config['dropout'])
            self.multi_scale_branch_logits = nn.Parameter(torch.zeros(3))
            if self.multi_scale_fusion == 'gated':
                self.multi_scale_fusion_gate = nn.Sequential(
                    nn.LayerNorm(config['d_model'] * 3),
                    nn.Linear(config['d_model'] * 3, config['d_model']),
                    nn.GELU(),
                    nn.Dropout(config['dropout']),
                    nn.Linear(config['d_model'], 3),
                )
        
        # 股票间交互注意力
        self.cross_stock_attention = CrossStockAttention(
            config['d_model'],
            config['nhead'],
            config['dropout'],
            config=config,
        )
        
        # 排序特异性层
        self.ranking_layers = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.LayerNorm(config['d_model'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        
        # 最终排序分数输出
        self.score_head = nn.Sequential(
            nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['d_model'] // 4, 1)
        )

        self.volatility_head = None
        if self.use_multitask_volatility:
            self.volatility_head = nn.Sequential(
                nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
                nn.ReLU(),
                nn.Dropout(config['dropout'] * 0.5),
                nn.Linear(config['d_model'] // 4, 1),
            )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _apply_temporal_cross_stock_attention(
        self,
        projected_src,
        stock_indices=None,
        stock_valid_mask=None,
    ):
        """
        projected_src: [B, N, L, D]
        返回同形状张量，按时间步对股票集合做交互。
        """
        if (not self.use_temporal_cross_stock_attention) or self.temporal_cross_stock_attention is None:
            return projected_src

        batch_size, num_stocks, seq_len, d_model = projected_src.size()
        if num_stocks <= 1:
            return projected_src

        step_tokens = projected_src.permute(0, 2, 1, 3).contiguous().reshape(
            batch_size * seq_len,
            num_stocks,
            d_model,
        )  # [B*L, N, D]

        step_stock_indices = None
        if stock_indices is not None:
            step_stock_indices = (
                stock_indices
                .unsqueeze(1)
                .expand(-1, seq_len, -1)
                .reshape(batch_size * seq_len, num_stocks)
            )

        step_valid_mask = None
        if stock_valid_mask is not None:
            step_valid_mask = (
                stock_valid_mask.bool()
                .unsqueeze(1)
                .expand(-1, seq_len, -1)
                .reshape(batch_size * seq_len, num_stocks)
            )

        step_tokens = self.temporal_cross_stock_attention(
            step_tokens,
            stock_indices=step_stock_indices,
            stock_valid_mask=step_valid_mask,
        )

        return step_tokens.reshape(batch_size, seq_len, num_stocks, d_model).permute(0, 2, 1, 3).contiguous()

    def _extract_temporal_features(
        self,
        branch_src,
        temporal_encoder,
        stock_indices=None,
        stock_valid_mask=None,
    ):
        """
        branch_src: [B, N, L, F]
        返回: [B*N, L, D]
        """
        batch_size, num_stocks, seq_len, _ = branch_src.size()
        projected = self.input_proj(branch_src)
        projected = self._apply_temporal_cross_stock_attention(
            projected,
            stock_indices=stock_indices,
            stock_valid_mask=stock_valid_mask,
        )
        projected = projected.reshape(batch_size * num_stocks, seq_len, -1)
        projected = self.pos_encoder(projected)
        return temporal_encoder(projected)

    def _run_temporal_branch(
        self,
        branch_src,
        temporal_encoder,
        feature_attention,
        stock_indices=None,
        stock_valid_mask=None,
    ):
        temporal_features = self._extract_temporal_features(
            branch_src,
            temporal_encoder,
            stock_indices=stock_indices,
            stock_valid_mask=stock_valid_mask,
        )
        return feature_attention(temporal_features)

    def _encode_multi_scale_branch(
        self,
        src,
        windows,
        temporal_encoder,
        feature_attention,
        stock_indices=None,
        stock_valid_mask=None,
    ):
        seq_len = src.size(2)
        effective_windows = [max(1, min(seq_len, int(window))) for window in windows]
        max_window = max(effective_windows)
        branch_src = src[:, :, -max_window:, :]
        branch_temporal = self._extract_temporal_features(
            branch_src,
            temporal_encoder,
            stock_indices=stock_indices,
            stock_valid_mask=stock_valid_mask,
        )

        if len(effective_windows) == 1 or self.multi_scale_window_reduce == 'last':
            return feature_attention(branch_temporal[:, -effective_windows[-1]:, :])

        branch_features = [
            feature_attention(branch_temporal[:, -window:, :])
            for window in effective_windows
        ]
        stacked = torch.stack(branch_features, dim=0)  # [num_windows, batch*num_stocks, d_model]
        return torch.mean(stacked, dim=0)

    def _fuse_short_horizon_features(self, ultra_short_features, short_features):
        if not self.use_ultra_short_branch:
            return short_features
        if self.short_horizon_fusion_gate is None:
            return 0.5 * (ultra_short_features + short_features)

        stacked = torch.stack([ultra_short_features, short_features], dim=1)  # [B*N, 2, D]
        gate_input = torch.cat([ultra_short_features, short_features], dim=-1)
        logits = self.short_horizon_fusion_gate(gate_input)  # [B*N, 2]
        weights = torch.softmax(logits, dim=-1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return self.short_horizon_norm(short_features + self.short_horizon_dropout(fused))

    def _fuse_multi_scale_features(self, base_features, short_features, long_features):
        stacked = torch.stack([base_features, short_features, long_features], dim=1)  # [B*N, 3, D]
        if self.multi_scale_fusion == 'gated' and self.multi_scale_fusion_gate is not None:
            gate_input = torch.cat([base_features, short_features, long_features], dim=-1)
            logits = self.multi_scale_fusion_gate(gate_input)  # [B*N, 3]
        else:
            logits = self.multi_scale_branch_logits.unsqueeze(0).expand(base_features.size(0), -1)

        weights = torch.softmax(logits, dim=-1)  # [B*N, 3]
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return self.multi_scale_branch_norm(base_features + self.multi_scale_branch_dropout(fused))

    def set_market_context_feature_indices(self, feature_indices):
        if feature_indices is None:
            self.market_context_feature_indices = torch.full(
                (self.market_context_dim,),
                -1,
                dtype=torch.long,
                device=self.market_context_feature_indices.device,
            )
            return

        if not isinstance(feature_indices, torch.Tensor):
            feature_indices = torch.as_tensor(feature_indices)
        if feature_indices.dim() != 1:
            raise ValueError(
                f'feature_indices 形状非法: {tuple(feature_indices.shape)}，期望 [K]'
            )

        indices = feature_indices.to(
            device=self.market_context_feature_indices.device,
            dtype=torch.long,
        )
        if int(indices.numel()) != int(self.market_context_dim):
            target = torch.full(
                (self.market_context_dim,),
                -1,
                dtype=torch.long,
                device=self.market_context_feature_indices.device,
            )
            limit = min(int(indices.numel()), int(self.market_context_dim))
            if limit > 0:
                target[:limit] = indices[:limit]
            indices = target
        self.market_context_feature_indices = indices

    def _masked_market_mean_std(self, src, stock_valid_mask):
        valid = stock_valid_mask.float().unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
        denom = valid.sum(dim=1).clamp(min=1.0)  # [B, 1, 1]
        mean_per_time = (src * valid).sum(dim=1) / denom  # [B, L, F]
        diff = src - mean_per_time.unsqueeze(1)
        var_per_time = (diff * diff * valid).sum(dim=1) / denom
        std_per_time = torch.sqrt(var_per_time + 1e-12)
        market_mean = mean_per_time.mean(dim=1)  # [B, F]
        market_vol = std_per_time.mean(dim=1)    # [B, F]
        return market_mean, market_vol

    def _extract_market_context_vector(self, src, stock_valid_mask):
        batch_size, _, _, feature_dim = src.size()
        idx_tensor = self.market_context_feature_indices
        if idx_tensor.numel() == 0:
            return src.new_zeros((batch_size, self.market_context_dim))

        latest = src[:, :, -1, :]  # [B, N, F]
        valid = stock_valid_mask.float()  # [B, N]
        denom = valid.sum(dim=1).clamp(min=1.0)

        context = src.new_zeros((batch_size, int(idx_tensor.numel())))
        for i, idx in enumerate(idx_tensor.tolist()):
            if idx < 0 or idx >= feature_dim:
                continue
            values = latest[:, :, int(idx)]
            context[:, i] = (values * valid).sum(dim=1) / denom
        return context

    def set_stock_industry_index(self, stock_to_industry_idx):
        self.cross_stock_attention.set_stock_industry_index(stock_to_industry_idx)
        if self.temporal_cross_stock_attention is not None:
            self.temporal_cross_stock_attention.set_stock_industry_index(stock_to_industry_idx)

    def set_prior_graph(self, prior_graph_adj):
        if prior_graph_adj is None:
            self.prior_graph_adj = torch.empty(
                0,
                0,
                dtype=torch.bool,
                device=self.prior_graph_adj.device,
            )
            self.cross_stock_attention.set_prior_graph(None)
            if self.temporal_cross_stock_attention is not None:
                self.temporal_cross_stock_attention.set_prior_graph(None)
            return

        if not isinstance(prior_graph_adj, torch.Tensor):
            prior_graph_adj = torch.as_tensor(prior_graph_adj)
        if prior_graph_adj.dim() != 2 or prior_graph_adj.size(0) != prior_graph_adj.size(1):
            raise ValueError(
                f'prior_graph_adj 形状非法: {tuple(prior_graph_adj.shape)}，期望 [N, N]'
            )

        prior_graph_adj = prior_graph_adj.to(
            device=self.prior_graph_adj.device,
            dtype=torch.bool,
        )
        self.prior_graph_adj = prior_graph_adj
        self.cross_stock_attention.set_prior_graph(self.prior_graph_adj)
        if self.temporal_cross_stock_attention is not None:
            self.temporal_cross_stock_attention.set_prior_graph(self.prior_graph_adj)
    
    def forward(self, src, stock_indices=None, stock_valid_mask=None, return_aux=False):
        # src: [batch, num_stocks, seq_len, feature_dim]
        batch_size, num_stocks, seq_len, feature_dim = src.size()
        if stock_valid_mask is None:
            stock_valid_mask = torch.ones(
                batch_size, num_stocks, dtype=torch.bool, device=src.device
            )
        else:
            stock_valid_mask = stock_valid_mask.bool()

        # 市场状态引导门控：用全市场均值+波动提取当前市场状态，对特征维做动态缩放。
        if self.use_market_gating:
            market_mean, market_vol = self._masked_market_mean_std(src, stock_valid_mask)
            market_state = market_mean + market_vol
            if self.use_market_gating_macro_context and self.market_macro_proj is not None:
                macro_context = self._extract_market_context_vector(src, stock_valid_mask)
                macro_delta = self.market_macro_proj(macro_context)
                market_state = market_state + (self.market_gate_macro_weight * macro_delta)
            gates = self.market_gate(market_state).unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, feature_dim]
            residual = self.market_gate_residual
            src = src * (residual + (1.0 - residual) * gates)
        
        # 基础时间尺度聚合
        aggregated_features = self._run_temporal_branch(
            src,
            self.temporal_encoder,
            self.feature_attention,
            stock_indices=stock_indices,
            stock_valid_mask=stock_valid_mask,
        )

        if self.use_multi_scale_temporal:
            base_short_features = self._encode_multi_scale_branch(
                src,
                self.short_windows,
                self.short_temporal_encoder,
                self.short_feature_attention,
                stock_indices=stock_indices,
                stock_valid_mask=stock_valid_mask,
            )
            if self.use_ultra_short_branch:
                ultra_short_features = self._encode_multi_scale_branch(
                    src,
                    self.ultra_short_windows,
                    self.ultra_short_temporal_encoder,
                    self.ultra_short_feature_attention,
                    stock_indices=stock_indices,
                    stock_valid_mask=stock_valid_mask,
                )
                short_features = self._fuse_short_horizon_features(
                    ultra_short_features,
                    base_short_features,
                )
            else:
                short_features = base_short_features

            long_features = self._encode_multi_scale_branch(
                src,
                self.long_windows,
                self.long_temporal_encoder,
                self.long_feature_attention,
                stock_indices=stock_indices,
                stock_valid_mask=stock_valid_mask,
            )
            aggregated_features = self._fuse_multi_scale_features(
                aggregated_features,
                short_features,
                long_features,
            )
        
        # 重塑回股票维度用于股票间交互
        stock_features = aggregated_features.view(batch_size, num_stocks, -1)  # [batch, num_stocks, d_model]
        
        # 股票间交互注意力
        interactive_features = self.cross_stock_attention(
            stock_features,
            stock_indices=stock_indices,
            stock_valid_mask=stock_valid_mask,
        )  # [batch, num_stocks, d_model]
        
        # 重塑回原形状
        interactive_features = interactive_features.view(batch_size * num_stocks, -1)
        
        # 排序特异性变换
        ranking_features = self.ranking_layers(interactive_features)  # [batch*num_stocks, d_model//2]
        
        # 生成排序分数
        scores = self.score_head(ranking_features)  # [batch*num_stocks, 1]
        
        # 重塑为最终输出格式
        output = scores.view(batch_size, num_stocks)  # [batch, num_stocks]

        if return_aux and self.use_multitask_volatility and self.volatility_head is not None:
            vol_scores = self.volatility_head(ranking_features).view(batch_size, num_stocks)
            return output, vol_scores

        return output
