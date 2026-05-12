"""
Sub-center ArcFace loss.

Standard ArcFace (K=1):
  L = CrossEntropy(s * [cos(theta_y + m), cos(theta_j for j!=y)])

Sub-center (K>1):
  class_cos_c = max_k (emb · proxy_c_k)  — best sub-proxy per class
  Apply ArcFace margin on the best sub-proxy of the true class.

Reference: "Sub-center ArcFace" (Deng et al., 2020).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubCenterArcFaceLoss(nn.Module):
    def __init__(self, scale: float = 30.0, margin: float = 0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # ArcFace safe-margin: when theta + m > pi, use linear approximation
        self._threshold = -self.cos_m                # cos(pi - m)
        self._mm = self.sin_m * margin               # sin(pi - m) * m ≈ sin(m) * m

    def forward(
        self,
        embeddings: torch.Tensor,   # (B, D) L2-normalized
        labels: torch.Tensor,       # (B,) int class indices
        proxies: torch.Tensor,      # (C, K, D) or (C, D)
    ) -> torch.Tensor:
        if proxies.dim() == 2:
            proxies = proxies.unsqueeze(1)  # (C, 1, D) — standard ArcFace

        C, K, D = proxies.shape
        B = embeddings.shape[0]

        emb_n = F.normalize(embeddings, dim=-1)                     # (B, D)
        prox_n = F.normalize(proxies.view(C * K, D), dim=-1)        # (C*K, D)

        cos_all = (emb_n @ prox_n.T).view(B, C, K)                  # (B, C, K)

        # Class logit = best sub-proxy cosine similarity
        class_cos, _ = cos_all.max(dim=-1)                           # (B, C)

        # --- ArcFace margin on the true class ---
        cos_y = class_cos[torch.arange(B), labels].clamp(-1 + 1e-7, 1 - 1e-7)
        sin_y = (1.0 - cos_y * cos_y).clamp(min=0.0).sqrt()
        cos_theta_m = cos_y * self.cos_m - sin_y * self.sin_m       # cos(theta + m)
        # Stability: if theta > pi - m fall back to linear
        cos_theta_m = torch.where(cos_y > self._threshold, cos_theta_m, cos_y - self._mm)

        # Scatter margin delta into true-class positions (safe with AMP — no in-place on grad tensor)
        delta = (cos_theta_m - class_cos[torch.arange(B), labels]).unsqueeze(1)  # (B, 1)
        margin_mask = torch.zeros_like(class_cos).scatter_(1, labels.unsqueeze(1), delta)
        logits = (class_cos + margin_mask) * self.scale                          # (B, C)

        return F.cross_entropy(logits, labels)
