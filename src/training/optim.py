"""
AdamW optimizer with 3 param groups (trunk / fc / proxy) — Table 5.
"""
import torch
from torch.optim import AdamW


def build_optimizer(
    embedder: torch.nn.Module,
    proxy_head: torch.nn.Module,
    cfg: dict,
) -> AdamW:
    """
    cfg keys (from base_vit.yaml):
      trunk_lr, fc_lr, proxy_lr
      trunk_wd, fc_wd, proxy_wd
      trunk_beta2, fc_beta2, proxy_beta2
      trunk_eps, fc_eps, proxy_eps
    """
    c = cfg["optimizer"]

    # Separate trunk vs fc parameters
    fc_ids = {id(p) for p in embedder.fc.parameters()}
    trunk_params = [p for p in embedder.parameters() if id(p) not in fc_ids]
    fc_params = list(embedder.fc.parameters())
    proxy_params = list(proxy_head.parameters())

    param_groups = [
        {
            "params": trunk_params,
            "lr": c["trunk_lr"],
            "weight_decay": c["trunk_wd"],
            "betas": (0.9, c["trunk_beta2"]),
            "eps": c["trunk_eps"],
        },
        {
            "params": fc_params,
            "lr": c["fc_lr"],
            "weight_decay": c["fc_wd"],
            "betas": (0.9, c["fc_beta2"]),
            "eps": c["fc_eps"],
        },
        {
            "params": proxy_params,
            "lr": c["proxy_lr"],
            "weight_decay": c["proxy_wd"],
            "betas": (0.9, c["proxy_beta2"]),
            "eps": c["proxy_eps"],
        },
    ]
    return AdamW(param_groups)
