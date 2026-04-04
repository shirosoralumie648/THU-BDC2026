from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config

__all__ = ['PortfolioOptimizationLoss']


class PortfolioOptimizationLoss(nn.Module):
    def __init__(
        self,
        temperature=10.0,
        listnet_weight=1.0,
        pairwise_weight=1.0,
        lambda_ndcg_weight=1.0,
        lambda_ndcg_topk=50,
        ic_weight=0.0,
        ic_mode='pearson',
        runtime_config=None,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.listnet_weight = float(listnet_weight)
        self.pairwise_weight = float(pairwise_weight)
        self.lambda_ndcg_weight = float(lambda_ndcg_weight)
        self.lambda_ndcg_topk = int(lambda_ndcg_topk)
        self.ic_weight = float(ic_weight)
        self.ic_mode = str(ic_mode).lower()
        self.runtime_config = runtime_config if runtime_config is not None else config

    def _zero(self, values):
        return values.sum() * 0.0

    def _listnet_loss(self, predictions, returns):
        p_true = F.softmax(returns * self.temperature, dim=0)
        p_pred = F.log_softmax(predictions, dim=0)
        return -(p_true * p_pred).sum()

    def _pairwise_ranknet_loss(self, predictions, returns):
        n = returns.numel()
        if n <= 1:
            return self._zero(predictions)

        top_fraction = float(self.runtime_config.get('pairwise_top_fraction', 0.1))
        k = max(1, int(n * top_fraction))
        _, top_true_indices = torch.topk(returns, k)

        pred_top = predictions[top_true_indices]
        true_top = returns[top_true_indices]

        pred_diff = pred_top.unsqueeze(1) - predictions.unsqueeze(0)
        true_diff = true_top.unsqueeze(1) - returns.unsqueeze(0)
        mask = (true_diff > 0).float()
        if mask.sum() == 0:
            return self._zero(predictions)
        return (F.softplus(-pred_diff) * mask * true_diff.abs()).sum() / (mask.sum() + 1e-8)

    def _lambda_ndcg_loss(self, predictions, returns):
        n = returns.numel()
        if n <= 1:
            return self._zero(predictions)

        gains = returns - torch.min(returns)
        gains = gains / (torch.max(gains) + 1e-8)

        ideal_order = torch.argsort(gains, descending=True)
        discounts = 1.0 / torch.log2(
            torch.arange(n, device=returns.device, dtype=torch.float32) + 2.0
        )
        ideal_dcg = torch.sum(gains[ideal_order] * discounts) + 1e-8

        pred_order = torch.argsort(predictions, descending=True)
        pred_pos = torch.empty_like(pred_order)
        pred_pos[pred_order] = torch.arange(n, device=returns.device)
        pred_discounts = 1.0 / torch.log2(pred_pos.float() + 2.0)

        gain_diff = gains.unsqueeze(1) - gains.unsqueeze(0)
        discount_diff = pred_discounts.unsqueeze(1) - pred_discounts.unsqueeze(0)
        delta_ndcg = torch.abs(gain_diff * discount_diff) / ideal_dcg

        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)
        true_diff = returns.unsqueeze(1) - returns.unsqueeze(0)
        pair_mask = true_diff > 0

        if self.lambda_ndcg_topk > 0:
            topk = min(self.lambda_ndcg_topk, n)
            _, top_idx = torch.topk(returns, topk)
            top_mask = torch.zeros(n, dtype=torch.bool, device=returns.device)
            top_mask[top_idx] = True
            pair_mask = pair_mask & top_mask.unsqueeze(1)

        if pair_mask.sum() == 0:
            return self._zero(predictions)

        weighted_pair_loss = F.softplus(-pred_diff) * delta_ndcg
        return weighted_pair_loss[pair_mask].mean()

    def _pearson_corr(self, x, y):
        if x.numel() <= 1:
            return self._zero(x)
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        x_std = torch.sqrt((x_centered ** 2).mean() + 1e-12)
        y_std = torch.sqrt((y_centered ** 2).mean() + 1e-12)
        corr = (x_centered * y_centered).mean() / (x_std * y_std + 1e-12)
        return torch.clamp(corr, min=-1.0, max=1.0)

    def _rankize(self, values):
        n = values.numel()
        if n <= 1:
            return torch.zeros_like(values)
        order = torch.argsort(values)
        ranks = torch.empty_like(values, dtype=torch.float32)
        ranks[order] = torch.arange(n, device=values.device, dtype=torch.float32)
        return ranks

    def _ic_regularization_loss(self, predictions, returns):
        if self.ic_weight <= 0.0:
            return self._zero(predictions)
        if self.ic_mode == 'pearson':
            corr = self._pearson_corr(predictions, returns)
        elif self.ic_mode in {'spearman', 'rank'}:
            corr = self._pearson_corr(self._rankize(predictions), self._rankize(returns))
        else:
            raise ValueError(f'不支持的 ic_mode: {self.ic_mode}')
        return -corr

    def forward(self, predictions, returns, relevance=None, volatility_targets=None, volatility_pred=None):
        valid_mask = ~torch.isnan(returns)
        if not valid_mask.any():
            return self._zero(predictions)

        valid_pred = predictions[valid_mask]
        valid_returns = returns[valid_mask]
        if valid_pred.numel() <= 1:
            return self._zero(valid_pred)

        listnet_loss = self._listnet_loss(valid_pred, valid_returns)
        pairwise_loss = self._pairwise_ranknet_loss(valid_pred, valid_returns)
        lambda_ndcg_loss = self._lambda_ndcg_loss(valid_pred, valid_returns)
        ic_loss = self._ic_regularization_loss(valid_pred, valid_returns)

        total_loss = (
            self.listnet_weight * listnet_loss
            + self.pairwise_weight * pairwise_loss
            + self.lambda_ndcg_weight * lambda_ndcg_loss
            + self.ic_weight * ic_loss
        )

        if volatility_targets is not None and volatility_pred is not None:
            vol_mask = ~torch.isnan(volatility_targets)
            if vol_mask.any():
                total_loss = total_loss + F.huber_loss(
                    volatility_pred[vol_mask],
                    volatility_targets[vol_mask],
                )

        return total_loss
