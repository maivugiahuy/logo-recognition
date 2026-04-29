"""
ProxyNCA++ loss — paper eqs (1)–(3).
Squared L2 distance between L2-normalized vectors, temperature σ=0.06.

L = -log( g(f(xi), z_i) / Σ_{z∈P} g(f(xi), z) )
where g(z1, z2) = exp(-d(z1,z2) / σ)
      d(z1,z2) = ||z1/||z1|| - z2/||z2||||^2  (squared L2 of normalized vecs)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def squared_l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: (N, D) — already L2 normalized
    b: (M, D) — already L2 normalized
    returns: (N, M) squared L2 distances
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b = 2 - 2*a·b  (unit vectors)
    return 2.0 - 2.0 * (a @ b.T)


class ProxyNCAPPLoss(nn.Module):
    """
    ProxyNCA++ loss (Teh et al., 2020) with squared-L2 and temperature scaling.
    Paper eq (3): L = -log(g(f(xi), zi) / Σ_{z∈P} g(f(xi), z))
    """

    def __init__(self, sigma: float = 0.06):
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        embeddings: torch.Tensor,   # (B, D) — L2 normalized
        labels: torch.Tensor,       # (B,) int class indices
        proxies: torch.Tensor,      # (C, D) — NOT yet normalized (normalized inside)
    ) -> torch.Tensor:
        proxies_n = F.normalize(proxies, dim=-1)  # (C, D)
        embeddings_n = F.normalize(embeddings, dim=-1)  # (B, D)

        # (B, C) squared distances
        dist = squared_l2_distance(embeddings_n, proxies_n)

        # g values: exp(-dist / σ)
        g = torch.exp(-dist / self.sigma)  # (B, C)

        # Numerator: g(f(xi), z_i) — proxy of the positive class
        pos_g = g[torch.arange(len(labels)), labels]  # (B,)

        # Denominator: Σ_{z∈P} g(f(xi), z)
        denom = g.sum(dim=1)  # (B,)

        loss = -torch.log(pos_g / denom.clamp(min=1e-12))
        return loss.mean()
