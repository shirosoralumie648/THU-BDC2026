from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.daily_encoder import FeatureAttention
from models.daily_encoder import PositionalEncoding
from models.daily_encoder import _normalize_scale_windows
from models.daily_encoder import build_temporal_encoder
from models.intraday_encoder import build_multi_scale_fusion_gate
from models.intraday_encoder import build_short_horizon_fusion_gate
from models.regime_router import build_market_gate
from models.regime_router import build_market_macro_proj
from models.relation_encoder import CrossStockAttention
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
            self.market_gate = build_market_gate(
                input_dim=input_dim,
                hidden_dim=gate_hidden_dim,
                dropout=config['dropout'],
            )
            self.market_macro_proj = None
            if self.use_market_gating_macro_context:
                macro_hidden_dim = int(
                    config.get('market_gate_macro_hidden_dim', max(16, input_dim // 4))
                )
                macro_hidden_dim = max(8, macro_hidden_dim)
                self.market_macro_proj = build_market_macro_proj(
                    market_context_dim=self.market_context_dim,
                    hidden_dim=macro_hidden_dim,
                    input_dim=input_dim,
                    dropout=config['dropout'],
                )

        # 输入投影层
        self.input_proj = nn.Linear(input_dim, config['d_model'])
        self.pos_encoder = PositionalEncoding(config['d_model'], config['dropout'], self.sequence_length)

        # 基础时序分支
        self.temporal_encoder = build_temporal_encoder(config)

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
                self.ultra_short_temporal_encoder = build_temporal_encoder(config)
                self.ultra_short_feature_attention = FeatureAttention(config['d_model'], config['dropout'])
            self.short_temporal_encoder = build_temporal_encoder(config)
            self.long_temporal_encoder = build_temporal_encoder(config)
            self.short_feature_attention = FeatureAttention(config['d_model'], config['dropout'])
            self.long_feature_attention = FeatureAttention(config['d_model'], config['dropout'])
            if self.use_ultra_short_branch:
                self.short_horizon_fusion_gate = build_short_horizon_fusion_gate(
                    d_model=config['d_model'],
                    dropout=config['dropout'],
                )
                self.short_horizon_norm = nn.LayerNorm(config['d_model'])
                self.short_horizon_dropout = nn.Dropout(config['dropout'])
            self.multi_scale_branch_norm = nn.LayerNorm(config['d_model'])
            self.multi_scale_branch_dropout = nn.Dropout(config['dropout'])
            self.multi_scale_branch_logits = nn.Parameter(torch.zeros(3))
            if self.multi_scale_fusion == 'gated':
                self.multi_scale_fusion_gate = build_multi_scale_fusion_gate(
                    d_model=config['d_model'],
                    dropout=config['dropout'],
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
