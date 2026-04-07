import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import config
from experiments.metrics import calculate_ranking_metrics
from objectives.target_transforms import transform_targets_for_loss


def _compute_volatility_aux_loss(vol_pred, vol_target, stock_valid_mask, runtime_config=None):
    runtime_config = runtime_config or config
    if vol_pred is None or vol_target is None:
        return None, None
    valid_mask = stock_valid_mask.bool()
    if valid_mask.numel() == 0 or (not bool(valid_mask.any())):
        return None, None

    pred = vol_pred[valid_mask]
    true = vol_target[valid_mask]
    if pred.numel() == 0:
        return None, None

    loss_type = str(runtime_config.get('volatility_loss_type', 'huber')).lower()
    if loss_type == 'mse':
        loss = F.mse_loss(pred, true)
    elif loss_type in {'l1', 'mae'}:
        loss = F.l1_loss(pred, true)
    elif loss_type in {'huber', 'smooth_l1'}:
        loss = F.smooth_l1_loss(pred, true)
    else:
        raise ValueError(f'不支持的 volatility_loss_type: {loss_type}')

    mae = torch.mean(torch.abs(pred - true))
    return loss, mae


def train_ranking_model(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    writer,
    strategy_candidates,
    *,
    use_amp=False,
    scaler=None,
    runtime_config=None,
):
    runtime_config = runtime_config or config
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0
    use_multitask_volatility = bool(runtime_config.get('use_multitask_volatility', False))
    volatility_loss_weight = float(runtime_config.get('volatility_loss_weight', 0.2))
    use_amp = bool(use_amp) and (device.type == 'cuda')
    max_train_batches = int(runtime_config.get('max_train_batches_per_epoch', 0) or 0)
    zero_grad_set_to_none = bool(runtime_config.get('train_zero_grad_set_to_none', True))

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}"), start=1):
        if max_train_batches > 0 and batch_idx > max_train_batches:
            break
        sequences = batch['sequences'].to(device)
        targets = batch['targets'].to(device)
        vol_targets = batch.get('vol_targets', None)
        if vol_targets is not None:
            vol_targets = vol_targets.to(device)
        stock_indices = batch['stock_indices'].to(device)
        masks = batch['masks'].to(device)
        stock_valid_mask = masks > 0.5

        optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            outputs = model(
                sequences,
                stock_indices=stock_indices,
                stock_valid_mask=stock_valid_mask,
                return_aux=use_multitask_volatility,
            )
            if use_multitask_volatility:
                outputs, vol_outputs = outputs
            else:
                vol_outputs = None

            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks

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
                valid_target = masked_targets[i][valid_indices]

                valid_pred_for_loss, valid_target_for_loss = transform_targets_for_loss(valid_pred, valid_target)

                if len(valid_pred_for_loss) > 1:
                    loss = criterion(valid_pred_for_loss.unsqueeze(0), valid_target_for_loss.unsqueeze(0))
                    batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss

        if batch_loss is not None:
            rank_loss = batch_loss / batch_size
            total_batch_loss = rank_loss

            vol_aux_loss = None
            vol_aux_mae = None
            if use_multitask_volatility and vol_outputs is not None and vol_targets is not None:
                vol_aux_loss, vol_aux_mae = _compute_volatility_aux_loss(
                    vol_outputs,
                    vol_targets,
                    stock_valid_mask=stock_valid_mask,
                    runtime_config=runtime_config,
                )
                if vol_aux_loss is not None:
                    total_batch_loss = total_batch_loss + volatility_loss_weight * vol_aux_loss

            use_grad_clip = bool(runtime_config.get('enable_grad_clip', True))
            max_grad_norm = float(runtime_config.get('max_grad_norm', 0.0) or 0.0)
            if use_amp:
                if scaler is None:
                    scaler = torch.amp.GradScaler('cuda', enabled=True)
                scaler.scale(total_batch_loss).backward()
                if use_grad_clip and max_grad_norm > 0.0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if writer:
                        writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch * len(dataloader) + local_step)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_batch_loss.backward()
                if use_grad_clip and max_grad_norm > 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if writer:
                        writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch * len(dataloader) + local_step)
                optimizer.step()

            total_loss += float(total_batch_loss.detach().item())

            with torch.no_grad():
                metrics = calculate_ranking_metrics(
                    masked_outputs,
                    masked_targets,
                    masks,
                    strategy_candidates=strategy_candidates,
                    temperature=runtime_config.get('softmax_temperature', 1.0),
                    runtime_config=runtime_config,
                )
                metrics['rank_loss'] = float(rank_loss.item())
                if vol_aux_loss is not None and vol_aux_mae is not None:
                    metrics['vol_aux_loss'] = float(vol_aux_loss.item())
                    metrics['vol_mae'] = float(vol_aux_mae.item())
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v

            local_step += 1
            if writer:
                writer.add_scalar('train/loss', total_batch_loss.item(), global_step=epoch * len(dataloader) + local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step=epoch * len(dataloader) + local_step)

    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step

    return total_loss / local_step if local_step > 0 else 0, total_metrics


def evaluate_ranking_model(
    model,
    dataloader,
    criterion,
    device,
    writer,
    epoch,
    strategy_candidates,
    ablation_feature_indices=None,
    runtime_config=None,
):
    runtime_config = runtime_config or config
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0
    use_multitask_volatility = bool(runtime_config.get('use_multitask_volatility', False))
    volatility_loss_weight = float(runtime_config.get('volatility_loss_weight', 0.2))
    use_amp_eval = bool(runtime_config.get('use_amp_eval', runtime_config.get('use_amp', True))) and (device.type == 'cuda')
    max_eval_batches = int(runtime_config.get('max_eval_batches_per_fold', 0) or 0)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"), start=1):
            if max_eval_batches > 0 and batch_idx > max_eval_batches:
                break
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            vol_targets = batch.get('vol_targets', None)
            if vol_targets is not None:
                vol_targets = vol_targets.to(device)
            stock_indices = batch['stock_indices'].to(device)
            masks = batch['masks'].to(device)
            stock_valid_mask = masks > 0.5

            if ablation_feature_indices:
                sequences = sequences.clone()
                sequences[:, :, :, ablation_feature_indices] = 0

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp_eval):
                outputs = model(
                    sequences,
                    stock_indices=stock_indices,
                    stock_valid_mask=stock_valid_mask,
                    return_aux=use_multitask_volatility,
                )
                if use_multitask_volatility:
                    outputs, vol_outputs = outputs
                else:
                    vol_outputs = None

                masked_outputs = outputs * masks + (1 - masks) * (-1e9)
                masked_targets = targets * masks

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

                    valid_pred_for_loss, valid_target_for_loss = transform_targets_for_loss(valid_pred, valid_true)
                    if len(valid_pred_for_loss) > 1:
                        loss = criterion(valid_pred_for_loss.unsqueeze(0), valid_target_for_loss.unsqueeze(0))
                        batch_loss = batch_loss + loss if batch_loss is not None else loss

            rank_loss = None
            if batch_loss is not None:
                rank_loss = batch_loss / batch_size
                total_batch_loss = rank_loss

                vol_aux_loss = None
                vol_aux_mae = None
                if use_multitask_volatility and vol_outputs is not None and vol_targets is not None:
                    vol_aux_loss, vol_aux_mae = _compute_volatility_aux_loss(
                        vol_outputs,
                        vol_targets,
                        stock_valid_mask=stock_valid_mask,
                        runtime_config=runtime_config,
                    )
                    if vol_aux_loss is not None:
                        total_batch_loss = total_batch_loss + volatility_loss_weight * vol_aux_loss

                total_loss += total_batch_loss.item()
            else:
                vol_aux_loss = None
                vol_aux_mae = None

            metrics = calculate_ranking_metrics(
                masked_outputs,
                masked_targets,
                masks,
                strategy_candidates=strategy_candidates,
                temperature=runtime_config.get('softmax_temperature', 1.0),
                runtime_config=runtime_config,
            )
            if rank_loss is not None:
                metrics['rank_loss'] = float(rank_loss.item())
            if vol_aux_loss is not None and vol_aux_mae is not None:
                metrics['vol_aux_loss'] = float(vol_aux_loss.item())
                metrics['vol_mae'] = float(vol_aux_mae.item())
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v

            num_batches += 1

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
    runtime_config=None,
):
    """在多个滚动验证折上评估，并返回折均值指标。"""
    runtime_config = runtime_config or config
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
            runtime_config=runtime_config,
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
