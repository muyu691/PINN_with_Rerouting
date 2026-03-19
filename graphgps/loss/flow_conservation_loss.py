"""
Phase 4: ST-PINN Physics-Informed Loss
=======================================

Two-part loss function for the Pseudo-Spatiotemporal Diffusion Model:

  L_total = L_Data + λ_eq · L_Eq

  ┌─────────────────────────────────────────────────────────────────┐
  │ L_Data = MSE(f_scaled^(K), y_scaled)                            │
  │   Standard data-fitting loss in normalized space.               │
  ├─────────────────────────────────────────────────────────────────┤
  │ L_Eq = MSE(ρ_v^(K) / σ, 0)                                     │
  │   Terminal equilibrium constraint.  Forces the final pressure   │
  │   field to zero, ensuring predicted flows satisfy implicit OD   │
  │   demand D_v — without ever needing the OD matrix.              │
  │                                                                 │
  │   Division by σ (flow_std) normalizes the physical-space        │
  │   pressure to dimensionless O(1), matching gradient scale of    │
  │   L_Data to prevent gradient imbalance.                         │
  └─────────────────────────────────────────────────────────────────┘

Key design:
  The forward pass (Phase 3) pre-computes ρ_v^(K) via the Corrected
  Discrete LWR equation and attaches it to ``batch.rho_v_final``.
  This loss function simply reads the pre-computed value — no
  redundant scatter_add recomputation needed.  Gradients flow
  naturally through rho_v_final back into the diffusion loop (BPTT).

Backward compatibility:
  If ``batch.rho_v_final`` is not present (e.g., running a baseline
  model without the diffusion loop), L_Eq is skipped and only L_Data
  is returned.

Configuration dependencies:
  cfg.dataset.flow_std  : float  - StandardScaler σ (for L_Eq normalization)
  cfg.model.lambda_eq   : float  - Equilibrium loss weight (default 1.0)
"""

import torch
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg


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


def compute_pinn_loss(
    pred: torch.Tensor,
    batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ST-PINN loss: L_total = L_Data + λ_eq · L_Eq.

    Args:
        pred  : [E_new, 1]  predicted equilibrium flows (normalized space)
        batch : PyG Batch object containing:
                  batch.y             : [E_new, 1]  ground truth (normalized)
                  batch.rho_v_final   : [N, 1]      terminal pressure ρ^(K)
                                                     (real space, veh/hr)
                                        — attached by Phase 3 forward pass

    Returns:
        total_loss : scalar        L_Data + λ_eq · L_Eq (for backward)
        pred       : [E_new, 1]    predictions (for logger metrics)
    """
    true = batch.y  # [E_new, 1]

    # ── L_Data: data-fitting loss (normalized space) ──────────────────
    loss_data = _compute_data_loss(pred, true)

    # ── L_Eq: terminal equilibrium constraint ─────────────────────────
    rho_v_final = getattr(batch, 'rho_v_final', None)

    if rho_v_final is not None:
        flow_std = cfg.dataset.flow_std             # σ (scalar, veh/hr)
        rho_scaled = rho_v_final / flow_std          # [N, 1] → dimensionless
        loss_eq = F.mse_loss(rho_scaled, torch.zeros_like(rho_scaled))

        lambda_eq = cfg.model.lambda_eq              # default 1.0
        total_loss = loss_data + lambda_eq * loss_eq
    else:
        loss_eq = torch.tensor(0.0, device=pred.device)
        total_loss = loss_data

    # ── Attach individual losses to batch for optional logging ────────
    batch.loss_data = loss_data.detach()
    batch.loss_eq = loss_eq.detach()

    return total_loss, pred
