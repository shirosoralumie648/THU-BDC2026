import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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

        # 对于 padding query，放开到所有有效 key，避免该行全部 -inf 导致数值问题。
        for b in range(batch_size):
            valid_key_row = stock_valid_mask[b]
            if valid_key_row.any():
                allowed[b, ~stock_valid_mask[b], :] = valid_key_row
            else:
                allowed[b, ~stock_valid_mask[b], :] = True

        blocked = ~allowed  # True 表示禁止关注
        blocked = blocked.unsqueeze(1).expand(-1, self.nhead, -1, -1)
        return blocked.reshape(batch_size * self.nhead, num_stocks, num_stocks)

    def _build_attention_mask(self, stock_features, stock_indices=None, stock_valid_mask=None):
        # 预留 stock_indices 入口，便于后续接入“行业先验掩码”。
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

        return None

    def forward(self, stock_features, stock_indices=None, stock_valid_mask=None):
        # stock_features: [batch, num_stocks, d_model]
        # 股票间交互：默认相似度稀疏注意力，抑制全连接噪声扩散。
        key_padding_mask = None
        if stock_valid_mask is not None:
            key_padding_mask = ~stock_valid_mask.bool()

        attn_mask = self._build_attention_mask(
            stock_features,
            stock_indices=stock_indices,
            stock_valid_mask=stock_valid_mask,
        )
        attended, _ = self.cross_attention(
            stock_features,
            stock_features,
            stock_features,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
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
        self.use_market_gating = bool(config.get('use_market_gating', True))
        self.market_gate_residual = float(config.get('market_gate_residual', 0.5))

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
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, config['d_model'])
        self.pos_encoder = PositionalEncoding(config['d_model'], config['dropout'], config['sequence_length'])
        
        # 时序特征提取
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        
        # 特征注意力
        self.feature_attention = FeatureAttention(config['d_model'], config['dropout'])
        
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
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, src, stock_indices=None, stock_valid_mask=None):
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
            market_mean = src.mean(dim=1).mean(dim=1)  # [batch, feature_dim]
            market_vol = src.std(dim=1, unbiased=False).mean(dim=1)  # [batch, feature_dim]
            market_state = market_mean + market_vol
            gates = self.market_gate(market_state).unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, feature_dim]
            residual = self.market_gate_residual
            src = src * (residual + (1.0 - residual) * gates)
        
        # 重塑为 [batch*num_stocks, seq_len, feature_dim]
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        
        # 输入投影和位置编码
        src_proj = self.input_proj(src_reshaped)  # [batch*num_stocks, seq_len, d_model]
        src_proj = self.pos_encoder(src_proj)
        
        # 时序特征提取
        temporal_features = self.temporal_encoder(src_proj)  # [batch*num_stocks, seq_len, d_model]
        
        # 特征注意力聚合
        aggregated_features = self.feature_attention(temporal_features)  # [batch*num_stocks, d_model]
        
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
        
        return output
