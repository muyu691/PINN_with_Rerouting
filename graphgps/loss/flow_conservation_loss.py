"""
Phase 4: ST-PINN physics-informed loss.

The supervised term stays in normalized space, but it can now be split into
retained-edge and new-edge components:

    L_data = lambda_old * L_old + lambda_new(epoch) * L_new

The equilibrium penalty is unchanged:

    L_total = L_data + lambda_eq * L_eq

Backward compatibility:
  - If ``batch.new_edge_mask`` is missing, the code falls back to the previous
    unified supervised loss.
  - If ``batch.rho_v_final`` is missing, the equilibrium term is skipped.
"""

import torch
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg


def _align_pred_true_shapes(
    pred: torch.Tensor,
    true: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match singleton dimensions before applying subset masks."""
    if pred.ndim == 1 and true.ndim == 2 and true.shape[1] == 1:
        pred = pred.view(-1, 1)
    elif true.ndim == 1 and pred.ndim == 2 and pred.shape[1] == 1:
        true = true.view(-1, 1)
    return pred, true


def _compute_data_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Dispatch the supervised term according to cfg.model.loss_fun."""
    loss_fun = str(cfg.model.loss_fun).lower()
    if loss_fun == 'l1':
        return F.l1_loss(pred, true)
    if loss_fun == 'smoothl1':
        return F.smooth_l1_loss(pred, true)
    if loss_fun == 'mse':
        return F.mse_loss(pred, true)
    raise ValueError(f"Unsupported cfg.model.loss_fun for PINN loss: {cfg.model.loss_fun}")


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def _compute_subset_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the base loss to one subset only, keeping mean reduction local."""
    mask = mask.bool().view(-1)
    if mask.numel() != pred.shape[0] or mask.numel() != true.shape[0]:
        raise ValueError(
            f"new_edge_mask has {mask.numel()} entries, but pred/true have "
            f"{pred.shape[0]} / {true.shape[0]} rows."
        )
    if not torch.any(mask):
        return _zero_loss(pred)
    return _compute_data_loss(pred[mask], true[mask])


def _get_lambda_new_current() -> float:
    """Linearly ramp the new-edge weight to avoid early optimization shocks."""
    schedule = str(getattr(cfg.model, 'lambda_new_schedule', 'linear')).lower()
    lambda_start = float(getattr(cfg.model, 'lambda_new_start', 1.0))
    lambda_final = float(getattr(cfg.model, 'lambda_new_final', 6.0))
    warmup_epochs = max(int(getattr(cfg.model, 'lambda_new_warmup_epochs', 40)), 0)
    current_epoch = max(
        float(getattr(cfg.train, 'current_epoch', getattr(cfg.optim, 'max_epoch', 0))),
        0.0,
    )

    if schedule in ('constant', 'none') or warmup_epochs == 0:
        return lambda_final
    if schedule != 'linear':
        raise ValueError(f"Unsupported lambda_new schedule: {cfg.model.lambda_new_schedule}")

    progress = min(max(current_epoch / warmup_epochs, 0.0), 1.0)
    return lambda_start + progress * (lambda_final - lambda_start)


def compute_pinn_loss(
    pred: torch.Tensor,
    batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ST-PINN loss for normalized flow supervision plus equilibrium regularization.

    Args:
        pred: [E_new, 1] predicted equilibrium flows in normalized space.
        batch: PyG batch carrying ``y`` and optionally ``new_edge_mask`` /
            ``rho_v_final``.

    Returns:
        total_loss: scalar loss for optimization.
        pred: unchanged prediction tensor for downstream metric logging.
    """
    true = batch.y
    pred, true = _align_pred_true_shapes(pred, true)

    lambda_old = float(getattr(cfg.model, 'lambda_old', 1.0))
    lambda_new_current = _get_lambda_new_current()

    # When the edge split is available, supervise old/new edges separately so
    # each subset keeps its own normalization. Otherwise keep legacy behavior.
    if hasattr(batch, 'new_edge_mask'):
        mask_new = batch.new_edge_mask.bool().view(-1)
        mask_old = ~mask_new
        loss_old = _compute_subset_loss(pred, true, mask_old)
        loss_new = _compute_subset_loss(pred, true, mask_new)
        loss_data = lambda_old * loss_old + lambda_new_current * loss_new
    else:
        loss_old = _compute_data_loss(pred, true)
        loss_new = _zero_loss(pred)
        lambda_new_current = 0.0
        loss_data = loss_old

    rho_v_final = getattr(batch, 'rho_v_final', None)
    if rho_v_final is not None:
        flow_std = cfg.dataset.flow_std
        rho_scaled = rho_v_final / flow_std
        loss_eq = F.mse_loss(rho_scaled, torch.zeros_like(rho_scaled))
        lambda_eq = float(getattr(cfg.model, 'lambda_eq', 1.0))
        total_loss = loss_data + lambda_eq * loss_eq
    else:
        loss_eq = _zero_loss(pred)
        total_loss = loss_data

    batch.loss_old = loss_old.detach()
    batch.loss_new = loss_new.detach()
    batch.lambda_new_current = pred.new_tensor(float(lambda_new_current)).detach()
    batch.loss_data = loss_data.detach()
    batch.loss_eq = loss_eq.detach()

    return total_loss, pred
