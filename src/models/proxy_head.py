"""
Learnable proxy vectors — one per class (paper Sec 3.1, eq 3).
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
        embedder.eval()
        class_sums: dict[int, torch.Tensor] = {}
        class_counts: dict[int, int] = {}

        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc="Proxy init"):
                imgs = imgs.to(device)
                embs = embedder(imgs)  # (B, D), L2 normalized
                for emb, lbl in zip(embs, labels.tolist()):
                    if lbl not in class_sums:
                        class_sums[lbl] = torch.zeros_like(emb)  # on GPU
                        class_counts[lbl] = 0
                    class_sums[lbl] += emb  # GPU + GPU
                    class_counts[lbl] += 1

        with torch.no_grad():
            for lbl, total in class_sums.items():
                mean_emb = total / class_counts[lbl]
                self.proxies.data[lbl] = F.normalize(mean_emb, dim=-1).cpu()

        print(f"Proxies initialized for {len(class_sums)} classes.")
