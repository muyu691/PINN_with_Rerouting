"""
Phase 4: ST-PINN physics-informed loss.

The supervised term stays in normalized space, but it can now be split into
retained-edge and new-edge components:

    L_data = lambda_old * L_old + lambda_new(epoch) * L_new

The equilibrium penalty is unchanged:

    L_total = L_data + lambda_eq * L_eq

Phase 3 adds an OD-free reduced-cost surrogate on final-step real-space
quantities already exposed by the model:

    L_total = L_data + lambda_eq * L_eq + lambda_rc * L_rc

Backward compatibility:
  - If ``batch.new_edge_mask`` is missing, the code falls back to the previous
    unified supervised loss.
  - If ``batch.rho_v_final`` is missing, the equilibrium term is skipped.
  - The reduced-cost branch is only active when ``cfg.model.lambda_rc > 0``.
    For those runs, missing physics tensors fail loudly with a clear error.
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


def _get_lambda_rc_current() -> float:
    """Linearly warm up the reduced-cost weight after flow fitting starts."""
    lambda_rc_final = float(getattr(cfg.model, 'lambda_rc', 0.05))
    warmup_epochs = max(int(getattr(cfg.model, 'lambda_rc_warmup_epochs', 40)), 0)
    current_epoch = max(
        float(getattr(cfg.train, 'current_epoch', getattr(cfg.optim, 'max_epoch', 0))),
        0.0,
    )

    if lambda_rc_final <= 0.0:
        return 0.0
    if warmup_epochs == 0:
        return lambda_rc_final

    progress = min(max(current_epoch / warmup_epochs, 0.0), 1.0)
    return lambda_rc_final * progress


def _get_flow_ref(reference: torch.Tensor) -> torch.Tensor:
    """Shared global flow scale for both L_eq and the flow part of L_comp."""
    return reference.new_tensor(max(float(cfg.dataset.flow_std), 1e-6)).detach()


def _get_cost_ref(batch, reference: torch.Tensor) -> torch.Tensor:
    """Detached global t0_ref for cost-like quantities only."""
    _ = batch
    free_flow_time_ref = float(getattr(cfg.dataset, 'free_flow_time_ref', 0.0))
    if free_flow_time_ref <= 0.0:
        raise RuntimeError(
            "Reduced-cost surrogate requires a global training-set t0_ref in "
            "cfg.dataset.free_flow_time_ref, but it is missing or invalid. "
            "Please regenerate the processed dataset metadata or let the loader "
            "compute the fallback from the train split."
        )
    return reference.new_tensor(max(free_flow_time_ref, 1e-3)).detach()


def _require_batch_tensor(batch, name: str) -> torch.Tensor:
    """Read a mandatory tensor from batch with a clear error message."""
    value = getattr(batch, name, None)
    if value is None:
        raise RuntimeError(
            f"Reduced-cost branch requires batch.{name}, but it is missing. "
            f"Please ensure topology_model.forward() attaches the final-step "
            f"real-space reduced-cost tensors before compute_pinn_loss()."
        )
    return value


def _require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    """Fail loudly when reduced-cost tensors become numerically invalid."""
    if not torch.isfinite(tensor).all():
        bad = int((~torch.isfinite(tensor)).sum().item())
        raise RuntimeError(
            f"Reduced-cost surrogate produced non-finite values in {name} "
            f"({bad} invalid entries). Please check real-space flow recovery, "
            f"BPR parameters, and node-potential outputs."
        )


def _get_rc_branch_tensors(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the mandatory final-step tensors for the OD-free RC surrogate."""
    reduced_cost_final = _require_batch_tensor(batch, 'reduced_cost_final')
    phi_v_final = _require_batch_tensor(batch, 'phi_v_final')
    f_active_final = _require_batch_tensor(batch, 'f_active_final')

    _require_finite_tensor('batch.reduced_cost_final', reduced_cost_final)
    _require_finite_tensor('batch.phi_v_final', phi_v_final)
    _require_finite_tensor('batch.f_active_final', f_active_final)
    return reduced_cost_final, phi_v_final, f_active_final


def _attach_rc_debug_stats(
    batch,
    reduced_cost_final: torch.Tensor,
    phi_v_final: torch.Tensor,
    f_active_final: torch.Tensor,
    r_bar: torch.Tensor,
    rho_bar: torch.Tensor,
    flow_ref: torch.Tensor,
    cost_ref: torch.Tensor,
) -> None:
    """Expose compact RC diagnostics on batch for epoch logger aggregation."""
    batch.reduced_cost_mean = reduced_cost_final.mean().detach()
    batch.reduced_cost_min = reduced_cost_final.min().detach()
    batch.reduced_cost_max = reduced_cost_final.max().detach()
    batch.phi_mean = phi_v_final.mean().detach()
    batch.phi_std = phi_v_final.std(unbiased=False).detach()
    batch.f_active_mean = f_active_final.mean().detach()
    batch.f_active_max = f_active_final.max().detach()
    batch.flow_ref_rc = flow_ref.detach()
    batch.cost_ref_rc = cost_ref.detach()
    batch.r_bar_mean = r_bar.mean().detach()
    batch.r_bar_min = r_bar.min().detach()
    batch.r_bar_max = r_bar.max().detach()
    batch.rho_bar_mean = rho_bar.mean().detach()


def _compute_reduced_cost_loss(batch, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the OD-free node-potential surrogate loss on final-step tensors.

    This is not an exact Wardrop / Beckmann path gap because no OD or path set
    is available at train time. Instead, it follows the KKT structure of an
    OD-free convex transshipment surrogate with node demand ``Af = D`` and
    separable edge cost ``c_e(f_e)``, using node potentials ``phi`` as the
    dual variables of node conservation.
    """
    lambda_rc_final = float(getattr(cfg.model, 'lambda_rc', 0.05))
    if lambda_rc_final <= 0.0:
        zero = _zero_loss(reference)
        return zero, zero, zero, zero

    reduced_cost_final, phi_v_final, f_active_final = _get_rc_branch_tensors(batch)
    flow_ref = _get_flow_ref(reference)
    cost_ref = _get_cost_ref(batch, reference)
    r_bar = reduced_cost_final / cost_ref
    phi_bar = phi_v_final / cost_ref
    f_active_bar = f_active_final / flow_ref

    lambda_rc_nonneg = float(getattr(cfg.model, 'lambda_rc_nonneg', 1.0))
    lambda_rc_comp = float(getattr(cfg.model, 'lambda_rc_comp', 1.0))
    lambda_rc_gauge = float(getattr(cfg.model, 'lambda_rc_gauge', 0.1))

    # Physical quantities are computed in real units first, then normalized for
    # the loss. flow_ref is shared by L_eq and the flow part of L_comp, while
    # cost_ref is used only for cost-like quantities (r and phi).
    loss_nonneg = torch.mean(torch.relu(-r_bar).pow(2))
    loss_comp = torch.mean(f_active_bar * torch.relu(r_bar).pow(2))
    loss_gauge = torch.mean(phi_bar).pow(2)
    loss_rc = (
        lambda_rc_nonneg * loss_nonneg
        + lambda_rc_comp * loss_comp
        + lambda_rc_gauge * loss_gauge
    )
    return loss_nonneg, loss_comp, loss_gauge, loss_rc


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
    lambda_eq = float(getattr(cfg.model, 'lambda_eq', 1.0))
    flow_ref = _get_flow_ref(pred)
    if rho_v_final is not None:
        rho_bar = rho_v_final / flow_ref
        loss_eq = F.mse_loss(rho_bar, torch.zeros_like(rho_bar))
    else:
        rho_bar = pred.new_zeros((1,))
        loss_eq = _zero_loss(pred)

    loss_nonneg, loss_comp, loss_gauge, loss_rc = _compute_reduced_cost_loss(
        batch=batch,
        reference=pred,
    )
    lambda_rc = _get_lambda_rc_current()
    total_loss = loss_data + lambda_eq * loss_eq + lambda_rc * loss_rc

    if float(getattr(cfg.model, 'lambda_rc', 0.05)) > 0.0:
        reduced_cost_final, phi_v_final, f_active_final = _get_rc_branch_tensors(batch)
        cost_ref = _get_cost_ref(batch, pred)
        r_bar = reduced_cost_final / cost_ref
        _attach_rc_debug_stats(
            batch,
            reduced_cost_final,
            phi_v_final,
            f_active_final,
            r_bar,
            rho_bar,
            flow_ref,
            cost_ref,
        )

    batch.loss_old = loss_old.detach()
    batch.loss_new = loss_new.detach()
    batch.lambda_new_current = pred.new_tensor(float(lambda_new_current)).detach()
    batch.lambda_rc_current = pred.new_tensor(float(lambda_rc)).detach()
    batch.loss_data = loss_data.detach()
    batch.loss_eq = loss_eq.detach()
    batch.loss_nonneg = loss_nonneg.detach()
    batch.loss_comp = loss_comp.detach()
    batch.loss_gauge = loss_gauge.detach()
    batch.loss_rc_nonneg = loss_nonneg.detach()
    batch.loss_rc_comp = loss_comp.detach()
    batch.loss_rc_gauge = loss_gauge.detach()
    batch.loss_rc = loss_rc.detach()

    return total_loss, pred
