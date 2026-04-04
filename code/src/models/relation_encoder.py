from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CrossStockAttention']


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
        batch_size = allowed.size(0)
        for b in range(batch_size):
            valid_key_row = stock_valid_mask[b]
            if valid_key_row.any():
                allowed[b, ~stock_valid_mask[b], :] = valid_key_row
            else:
                allowed[b, ~stock_valid_mask[b], :] = True
        return allowed

    def _build_similarity_sparse_mask(self, stock_features, stock_valid_mask):
        batch_size, num_stocks, _ = stock_features.size()
        if num_stocks <= 1:
            return None

        normalized = F.normalize(stock_features, p=2, dim=-1, eps=1e-12)
        similarity = torch.matmul(normalized, normalized.transpose(1, 2))

        valid_keys = stock_valid_mask.unsqueeze(1).expand(-1, num_stocks, -1)
        similarity = similarity.masked_fill(~valid_keys, float('-inf'))

        topk = min(self.similarity_topk, num_stocks)
        topk_indices = torch.topk(similarity, k=topk, dim=-1).indices

        allowed = torch.zeros(batch_size, num_stocks, num_stocks, dtype=torch.bool, device=stock_features.device)
        allowed.scatter_(dim=2, index=topk_indices, value=True)

        eye = torch.eye(num_stocks, dtype=torch.bool, device=stock_features.device).unsqueeze(0)
        valid_pairs = stock_valid_mask.unsqueeze(1) & stock_valid_mask.unsqueeze(2)
        allowed = (allowed | (eye & valid_pairs)) & valid_keys

        allowed = self._stabilize_padding_queries(allowed, stock_valid_mask)
        blocked = ~allowed
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
        ]

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
            return blocked_a | blocked_b
        if self.prior_similarity_combine in {'union', 'or'}:
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
