from torch_geometric.graphgym.register import register_config


@register_config('overwrite_defaults')
def overwrite_defaults_cfg(cfg):
    """Overwrite core GraphGym defaults only."""

    # Training (and validation) pipeline mode.
    cfg.train.mode = 'custom'

    # Overwrite default dataset name.
    cfg.dataset.name = 'none'

    # Overwrite default rounding precision.
    cfg.round = 5


@register_config('extended_cfg')
def extended_cfg(cfg):
    """General extended config options."""

    # Additional name tag used in run_dir and wandb auto generation.
    cfg.name_tag = ""

    # If True, keep checkpointing only the current best validation model.
    cfg.train.ckpt_best = False

    # Physics-informed equilibrium penalty.
    cfg.model.lambda_eq = 1.0

    # Split supervised loss between retained edges and newly added edges.
    cfg.model.lambda_old = 1.0
    cfg.model.lambda_new_start = 1.0
    cfg.model.lambda_new_final = 6.0
    cfg.model.lambda_new_warmup_epochs = 40
    cfg.model.lambda_new_schedule = 'linear'
    cfg.model.enable_node_potential_head = False
    cfg.model.lambda_rc = 0.05
    cfg.model.lambda_rc_warmup_epochs = 40
    cfg.model.lambda_rc_nonneg = 1.0
    cfg.model.lambda_rc_comp = 1.0
    cfg.model.lambda_rc_gauge = 0.1
    cfg.model.flow_softplus_scale = 5.0
    cfg.model.rc_bpr_alpha = 0.15
    cfg.model.rc_bpr_beta = 4.0

    # Runtime-only field injected by the custom train loop so the loss can read
    # the current epoch without changing the loss function signature.
    cfg.train.current_epoch = 0

    # Legacy conservation knobs kept for backward compatibility.
    cfg.model.lambda_cons = 0.1
    cfg.model.cons_norm = 'l2'
