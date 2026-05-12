"""
Learnable proxy vectors — one per class.
Initialized from per-class mean embeddings of the pretrained model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class ProxyHead(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 128):
        super().__init__()
        self.proxies = nn.Parameter(torch.zeros(num_classes, embed_dim))

    def init_from_embeddings(
        self,
        embedder: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> None:
        """Compute per-class mean embedding and set as initial proxy values."""
        num_classes = self.proxies.shape[0]
        embed_dim = self.proxies.shape[1]

        sums   = torch.zeros(num_classes, embed_dim, device=device)
        counts = torch.zeros(num_classes, device=device)

        embedder.eval()
        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc="Proxy init"):
                imgs   = imgs.to(device)
                labels = labels.to(device)
                embs   = embedder(imgs)                        # (B, D)
                sums.scatter_add_(0, labels.unsqueeze(1).expand_as(embs), embs)
                counts.scatter_add_(0, labels, torch.ones(len(labels), device=device))

        mask = counts > 0
        mean_embs = sums[mask] / counts[mask].unsqueeze(1)    # (C, D)
        mean_embs = F.normalize(mean_embs, dim=-1)

        with torch.no_grad():
            self.proxies.data[mask] = mean_embs.to(self.proxies.device)

        print(f"Proxies initialized for {int(mask.sum())} classes.")


class SubCenterProxyHead(nn.Module):
    """K sub-proxies per class for Sub-center ArcFace.

    proxies shape: (num_classes, K, embed_dim).
    K=1 degrades to standard ArcFace.
    """

    def __init__(self, num_classes: int, embed_dim: int = 128, K: int = 3):
        super().__init__()
        self.K = K
        self.proxies = nn.Parameter(torch.zeros(num_classes, K, embed_dim))

    def init_from_embeddings(
        self,
        embedder: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> None:
        num_classes, K, embed_dim = self.proxies.shape

        sums   = torch.zeros(num_classes, embed_dim, device=device)
        counts = torch.zeros(num_classes, device=device)

        embedder.eval()
        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc="Proxy init"):
                imgs   = imgs.to(device)
                labels = labels.to(device)
                embs   = embedder(imgs)
                sums.scatter_add_(0, labels.unsqueeze(1).expand_as(embs), embs)
                counts.scatter_add_(0, labels, torch.ones(len(labels), device=device))

        mask = counts > 0
        mean_embs = F.normalize(sums[mask] / counts[mask].unsqueeze(1), dim=-1)  # (C', D)

        with torch.no_grad():
            # First sub-proxy = mean; remaining = mean + small noise (so they can diverge)
            self.proxies.data[mask, 0] = mean_embs
            for k in range(1, K):
                noise = torch.randn_like(mean_embs) * 0.02
                self.proxies.data[mask, k] = F.normalize(mean_embs + noise, dim=-1)

        print(f"Sub-center proxies (K={K}) initialized for {int(mask.sum())} classes.")
