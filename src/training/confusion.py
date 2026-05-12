"""Build row-normalized confusion matrix on validation set — used as input to hard-negative mining."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LogoDataset
from src.data.transforms import val_transforms, val_transforms_dinov2
from src.models.embedder_dinov2 import build_dinov2_embedder
from src.models.embedder_dinov3 import build_dinov3_embedder
from src.models.embedder_vit import build_vit_embedder
from src.models.proxy_head import ProxyHead, SubCenterProxyHead

ANN = Path("data/processed/openlogodet3k/annotations.parquet")
SPLITS = Path("data/processed/openlogodet3k/splits")

_DINOV2_BACKBONES = {"dinov2_vitb14", "dinov2"}
_IMAGENET_BACKBONES = _DINOV2_BACKBONES | {"dinov3_vitb16"}


def build_confusion_matrix(
    ckpt_path: str | Path,
    backbone: str = "vit_b32_openai",
    embed_dim: int = 128,
    input_size: int | None = None,
    batch_size: int = 256,
    freeze_blocks: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """
    Returns:
      C: (num_classes, num_classes) row-normalized confusion matrix
      class_names: list indexed by class_idx
    """
    is_dinov2 = backbone in _DINOV2_BACKBONES
    if input_size is None:
        input_size = 168 if is_dinov2 else 160
    is_imagenet = backbone in _IMAGENET_BACKBONES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint first to get the exact number of proxy classes
    state = torch.load(ckpt_path, map_location=device)
    num_classes = state["proxy"]["proxies"].shape[0]

    transform = val_transforms_dinov2(input_size) if is_imagenet else val_transforms(input_size)
    train_ds = LogoDataset.from_split(
        ANN, SPLITS / "open_train.json",
        transform=transform, mode="open_set",
    )
    val_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    class_names = [k for k, v in sorted(train_ds.class_to_idx.items(), key=lambda x: x[1])]
    assert len(class_names) == num_classes, (
        f"Class count mismatch: checkpoint has {num_classes} proxies "
        f"but split has {len(class_names)} classes"
    )

    if is_dinov2:
        embedder = build_dinov2_embedder(embed_dim, input_size, freeze_blocks=freeze_blocks).to(device)
    elif backbone in {"dinov3_vitb16"}:
        embedder = build_dinov3_embedder(embed_dim, input_size, freeze_blocks=freeze_blocks).to(device)
    else:
        embedder = build_vit_embedder(embed_dim, input_size, freeze_blocks=freeze_blocks).to(device)
    embedder.load_state_dict(state["embedder"])

    proxy_shape = state["proxy"]["proxies"].shape
    if len(proxy_shape) == 3:  # SubCenterProxyHead: (C, K, D)
        K = proxy_shape[1]
        proxy_head = SubCenterProxyHead(num_classes, embed_dim, K=K).to(device)
        proxy_head.load_state_dict(state["proxy"])
        # class representative = best sub-proxy (max cosine per class used at inference)
        prox_flat = F.normalize(proxy_head.proxies.detach().view(num_classes * K, embed_dim), dim=-1)
        proxies_n = prox_flat.view(num_classes, K, embed_dim)  # (C, K, D)
    else:
        proxy_head = ProxyHead(num_classes, embed_dim).to(device)
        proxy_head.load_state_dict(state["proxy"])
        proxies_n = F.normalize(proxy_head.proxies.detach(), dim=-1).unsqueeze(1)  # (C, 1, D)

    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    embedder.eval()

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Confusion matrix"):
            imgs = imgs.to(device)
            embs = embedder(imgs)  # (B, D) L2-normalized
            # Similarity to all proxies → max over sub-proxies per class → predict nearest class
            # proxies_n: (C, K, D) — works for K=1 (standard) and K>1 (sub-center)
            embs_n = F.normalize(embs, dim=-1)
            sims = torch.einsum("bd,ckd->bck", embs_n, proxies_n).max(dim=-1).values  # (B, C)
            preds = sims.argmax(dim=1).cpu().numpy()
            for true_lbl, pred_lbl in zip(labels.numpy(), preds):
                confusion[true_lbl, pred_lbl] += 1

    # Row-normalize
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    C = confusion / row_sums

    return C, class_names
