"""
Build row-normalized confusion matrix on validation set.
Used as input to hard-negative mining (Sec 3.2.3).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import OLG3K47Dataset
from src.data.transforms import val_transforms
from src.models.embedder_vit import build_vit_embedder
from src.models.proxy_head import ProxyHead

ANN = Path("data/processed/openlogodet3k47/annotations.parquet")
SPLITS = Path("data/processed/openlogodet3k47/splits")


def build_confusion_matrix(
    ckpt_path: str | Path,
    embed_dim: int = 128,
    input_size: int = 160,
    batch_size: int = 256,
) -> tuple[np.ndarray, list[str]]:
    """
    Returns:
      C: (num_classes, num_classes) row-normalized confusion matrix
      class_names: list indexed by class_idx
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use closed-set val split (all seen classes)
    val_ds = OLG3K47Dataset.from_split(
        ANN, SPLITS / "closed_val.json",
        transform=val_transforms(input_size), mode="closed_set",
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    class_names = [k for k, v in sorted(val_ds.class_to_idx.items(), key=lambda x: x[1])]
    num_classes = len(class_names)

    embedder = build_vit_embedder(embed_dim, input_size).to(device)
    state = torch.load(ckpt_path, map_location=device)
    embedder.load_state_dict(state["embedder"])

    proxy_head = ProxyHead(num_classes, embed_dim).to(device)
    proxy_head.load_state_dict(state["proxy"])

    proxies_n = F.normalize(proxy_head.proxies.detach(), dim=-1)  # (C, D)

    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    embedder.eval()

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Confusion matrix"):
            imgs = imgs.to(device)
            embs = embedder(imgs)  # (B, D) L2-normalized
            # Similarity to all proxies → predict nearest proxy
            sims = embs @ proxies_n.T  # (B, C)
            preds = sims.argmax(dim=1).cpu().numpy()
            for true_lbl, pred_lbl in zip(labels.numpy(), preds):
                confusion[true_lbl, pred_lbl] += 1

    # Row-normalize
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    C = confusion / row_sums

    return C, class_names
