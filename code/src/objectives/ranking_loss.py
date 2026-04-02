from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PortfolioOptimizationLoss(nn.Module):
    def __init__(
        self,
        temperature=10.0,
        listnet_weight=1.0,
        pairwise_weight=1.0,
        lambda_ndcg_weight=0.8,
        lambda_ndcg_topk=60,
        ic_weight=0.2,
        ic_mode='pearson',
    ):
        super().__init__()
        self.temperature = temperature
        self.listnet_weight = listnet_weight
        self.pairwise_weight = pairwise_weight
        self.lambda_ndcg_weight = lambda_ndcg_weight
        self.lambda_ndcg_topk = lambda_ndcg_topk
        self.ic_weight = ic_weight
        self.ic_mode = ic_mode

    def forward(self, predictions, returns, relevance=None, volatility_targets=None, volatility_pred=None):
        valid_mask = ~torch.isnan(returns)
        if valid_mask.sum() == 0:
            zero = predictions.sum() * 0.0
            return zero, {
                'listnet_loss': 0.0,
                'pairwise_loss': 0.0,
                'lambda_ndcg_loss': 0.0,
                'ic_loss': 0.0,
                'volatility_loss': 0.0,
            }

        valid_pred = predictions[valid_mask]
        valid_returns = returns[valid_mask]
        listnet_loss = self._listnet_loss(valid_pred, valid_returns)
        pairwise_loss = self._pairwise_loss(valid_pred, valid_returns)
        lambda_ndcg_loss = self._lambda_ndcg_loss(valid_pred, valid_returns)
        ic_loss = self._ic_loss(valid_pred, valid_returns)

        total_loss = (
            self.listnet_weight * listnet_loss
            + self.pairwise_weight * pairwise_loss
            + self.lambda_ndcg_weight * lambda_ndcg_loss
            + self.ic_weight * ic_loss
        )

        vol_loss = predictions.sum() * 0.0
        if volatility_targets is not None and volatility_pred is not None:
            vol_mask = ~torch.isnan(volatility_targets)
            if vol_mask.any():
                vol_loss = F.huber_loss(volatility_pred[vol_mask], volatility_targets[vol_mask])
                total_loss = total_loss + vol_loss

        metrics = {
            'listnet_loss': float(listnet_loss.detach().item()),
            'pairwise_loss': float(pairwise_loss.detach().item()),
            'lambda_ndcg_loss': float(lambda_ndcg_loss.detach().item()),
            'ic_loss': float(ic_loss.detach().item()),
            'volatility_loss': float(vol_loss.detach().item()),
        }
        return total_loss, metrics

    def _listnet_loss(self, predictions, returns):
        pred_dist = torch.softmax(predictions / self.temperature, dim=0)
        true_dist = torch.softmax(returns / self.temperature, dim=0)
        return -(true_dist * torch.log(pred_dist + 1e-12)).sum()

    def _pairwise_loss(self, predictions, returns):
        n = predictions.numel()
        if n <= 1:
            return predictions.sum() * 0.0
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)
        true_diff = returns.unsqueeze(1) - returns.unsqueeze(0)
        target = (true_diff > 0).float()
        valid_pairs = true_diff != 0
        if not valid_pairs.any():
            return predictions.sum() * 0.0
        return F.binary_cross_entropy_with_logits(pred_diff[valid_pairs], target[valid_pairs])

    def _lambda_ndcg_loss(self, predictions, returns):
        n = predictions.numel()
        if n <= 1:
            return predictions.sum() * 0.0
        topk = min(self.lambda_ndcg_topk, n)
        _, idx = torch.topk(returns, topk)
        top_pred = predictions[idx]
        top_ret = returns[idx]
        pred_diff = top_pred.unsqueeze(1) - top_pred.unsqueeze(0)
        true_diff = top_ret.unsqueeze(1) - top_ret.unsqueeze(0)
        target = (true_diff > 0).float()
        valid_pairs = true_diff != 0
        if not valid_pairs.any():
            return predictions.sum() * 0.0
        return F.binary_cross_entropy_with_logits(pred_diff[valid_pairs], target[valid_pairs])

    def _ic_loss(self, predictions, returns):
        if predictions.numel() <= 1:
            return predictions.sum() * 0.0
        pred = predictions - predictions.mean()
        truth = returns - returns.mean()
        denom = torch.sqrt((pred ** 2).sum() * (truth ** 2).sum()) + 1e-12
        corr = (pred * truth).sum() / denom
        return 1.0 - corr
