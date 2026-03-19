from torch_geometric.graphgym.register import register_config


@register_config('overwrite_defaults')
def overwrite_defaults_cfg(cfg):
    """Overwrite the default config values that are first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg

    WARNING: At the time of writing, the order in which custom config-setting
    functions like this one are executed is random; see the referenced `set_cfg`
    Therefore never reset here config options that are custom added, only change
    those that exist in core GraphGym.
    """

    # Training (and validation) pipeline mode
    cfg.train.mode = 'custom'  # 'standard' uses PyTorch-Lightning since PyG 2.1

    # Overwrite default dataset name
    cfg.dataset.name = 'none'

    # Overwrite default rounding precision
    cfg.round = 5


@register_config('extended_cfg')
def extended_cfg(cfg):
    """General extended config options.
    """

    # Additional name tag used in `run_dir` and `wandb_name` auto generation.
    cfg.name_tag = ""

    # In training, if True (and also cfg.train.enable_ckpt is True) then
    # always checkpoint the current best model based on validation performance,
    # instead, when False, follow cfg.train.eval_period checkpointing frequency.
    cfg.train.ckpt_best = False

    # ── Phase 4：ST-PINN 物理损失超参数 ────────────────────────────────────
    # 平衡损失权重 λ_eq：L_total = L_Data + λ_eq * L_Eq
    #   L_Eq = MSE(ρ_v^(K) / σ, 0) 终端压力平衡约束
    # λ_eq = 1.0 使数据损失和物理约束具有相同权重；
    # 由于 L_Eq 已经过 /σ 归一化，梯度尺度天然匹配 L_Data。
    cfg.model.lambda_eq = 1.0

    # (Legacy) 旧守恒损失参数，保留用于基线模型向后兼容
    cfg.model.lambda_cons = 0.1
    cfg.model.cons_norm = 'l2'
