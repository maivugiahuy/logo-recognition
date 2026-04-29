"""
ProxyNCAHN++ loss — paper eq (5).
Extends ProxyNCA++ denominator with image-embedding terms for hard-negative
class samples present in the batch (Sec 3.2.2).

P_i^HN ∝ g(f(xi), zi) / [Σ_{z∈P} g(f(xi), z) + Σ_{xj∈HN(i)} g(f(xi), f(xj))]

HN images in the batch are identified by their class label being in h(yi).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.proxynca_pp import ProxyNCAPPLoss, squared_l2_distance


class ProxyNCAHNPPLoss(ProxyNCAPPLoss):
    """
    ProxyNCAHN++ — ProxyNCA++ with in-batch hard-negative image terms.
    hn_map: {class_idx: [hn_class_idx, ...]} — built by mine_hn.py.
    """

    def __init__(self, sigma: float = 0.06, hn_map: dict[int, list[int]] | None = None):
        super().__init__(sigma=sigma)
        self.hn_map = hn_map or {}

    def forward(
        self,
        embeddings: torch.Tensor,   # (B, D)
        labels: torch.Tensor,       # (B,) int
        proxies: torch.Tensor,      # (C, D)
    ) -> torch.Tensor:
        proxies_n = F.normalize(proxies, dim=-1)
        emb_n = F.normalize(embeddings, dim=-1)

        dist_proxy = squared_l2_distance(emb_n, proxies_n)  # (B, C)
        g_proxy = torch.exp(-dist_proxy / self.sigma)  # (B, C)

        labels_list = labels.tolist()
        label_set = set(labels_list)

        losses = []
        for i, yi in enumerate(labels_list):
            pos_g = g_proxy[i, yi]
            denom = g_proxy[i].sum()

            # Add image-embedding terms for HN classes present in batch
            hn_classes = self.hn_map.get(yi, [])
            for hn_cls in hn_classes:
                if hn_cls not in label_set:
                    continue
                hn_indices = [j for j, lj in enumerate(labels_list) if lj == hn_cls]
                if not hn_indices:
                    continue
                hn_embs = emb_n[hn_indices]  # (K, D)
                dist_hn = squared_l2_distance(emb_n[i:i+1], hn_embs)  # (1, K)
                g_hn = torch.exp(-dist_hn / self.sigma).sum()
                denom = denom + g_hn

            losses.append(-torch.log(pos_g / denom.clamp(min=1e-12)))

        return torch.stack(losses).mean()
