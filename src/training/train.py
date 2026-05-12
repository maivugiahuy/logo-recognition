"""
Training loop — open-set (Phase A) and closed-set (Phase C).
25 epochs, AdamW, ReduceLROnPlateau (patience 4, factor 0.25).
Speed opts: AMP mixed precision, torch.compile, num_workers=8.
"""
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="open_clip")
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LogoDataset
from src.data.samplers import (
    HardNegativeBatchSampler,
    MPerClassSampler,
    load_hn_map,
)
from src.data.transforms import (
    train_transforms, val_transforms,
    train_transforms_dinov2, val_transforms_dinov2,
)
from src.losses.arcface import SubCenterArcFaceLoss
from src.losses.proxynca_hn_pp import ProxyNCAHNPPLoss
from src.losses.proxynca_pp import ProxyNCAPPLoss
from src.models.embedder_dinov2 import build_dinov2_embedder
from src.models.embedder_dinov3 import build_dinov3_embedder
from src.models.embedder_vit import build_vit_embedder
from src.models.proxy_head import ProxyHead, SubCenterProxyHead
from src.training.optim import build_optimizer

ANN = Path("data/processed/openlogodet3k/annotations.parquet")
SPLITS = Path("data/processed/openlogodet3k/splits")
CKPT = Path("checkpoints")

_DINOV2_BACKBONES = {"dinov2_vitb14", "dinov2"}
_DINOV3_BACKBONES = {"dinov3_vitb16"}
_IMAGENET_BACKBONES = _DINOV2_BACKBONES | _DINOV3_BACKBONES


def _build_embedder(cfg: dict, device: torch.device) -> nn.Module:
    backbone = cfg.get("backbone", "vit_b16_openai")
    freeze_blocks = cfg.get("freeze_blocks", 0)
    embed_dim = cfg["embed_dim"]
    input_size = cfg["input_size"]
    if backbone in _DINOV2_BACKBONES:
        return build_dinov2_embedder(embed_dim, input_size, freeze_blocks).to(device)
    if backbone in _DINOV3_BACKBONES:
        return build_dinov3_embedder(embed_dim, input_size, freeze_blocks).to(device)
    return build_vit_embedder(embed_dim, input_size, freeze_blocks=freeze_blocks).to(device)


def _get_transforms(cfg: dict):
    backbone = cfg.get("backbone", "vit_b16_openai")
    size = cfg["input_size"]
    if backbone in _IMAGENET_BACKBONES:
        return train_transforms_dinov2(size), val_transforms_dinov2(size)
    return train_transforms(size), val_transforms(size)


def recall_at_1(embedder: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Quick in-epoch recall@1 estimate using proxy-nearest-neighbor."""
    embedder.eval()
    all_embs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            embs = embedder(imgs.to(device))
            all_embs.append(embs.cpu())
            all_labels.extend(labels.tolist())
    embs = torch.cat(all_embs)  # (N, D)
    labels = torch.tensor(all_labels)
    # For each query, find nearest (excluding self)
    sim = embs @ embs.T  # (N, N) — embs are L2 normalized
    sim.fill_diagonal_(-1e9)
    nn_labels = labels[sim.argmax(dim=1)]
    recall = (nn_labels == labels).float().mean().item()
    return recall


def train(cfg_path: str, ckpt_name: str = "vit_base.pt") -> None:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TF32 — free ~10% speedup on Ampere/Ada (RTX 30xx/40xx).
    # No accuracy impact since loss/eval still use bfloat16/float32.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    mode = cfg.get("mode", "open_set")
    split_prefix = "open" if mode == "open_set" else "closed"

    # ── Datasets ──────────────────────────────────────────────────────────
    t_train, t_val = _get_transforms(cfg)
    train_ds = LogoDataset.from_split(
        ANN, SPLITS / f"{split_prefix}_train.json",
        transform=t_train, mode=mode,
    )
    val_ds = LogoDataset.from_split(
        ANN, SPLITS / f"{split_prefix}_val.json",
        transform=t_val, mode=mode,
    )

    k = cfg["training"]["k"]
    m = cfg["training"]["m"]

    # ── Sampler ───────────────────────────────────────────────────────────
    use_hn = "hn_mining" in cfg and Path(cfg["hn_mining"]["map_path"]).exists()
    if use_hn:
        hn_map = load_hn_map(cfg["hn_mining"]["map_path"], train_ds.class_to_idx)
        sampler = HardNegativeBatchSampler(train_ds.labels, hn_map, k=k, m=m)
    else:
        sampler = MPerClassSampler(train_ds.labels, k=k, m=m)

    n_workers = cfg.get("num_workers", 4)
    persistent = cfg.get("persistent_workers", True) and n_workers > 0
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler, num_workers=n_workers,
        pin_memory=True, persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False, num_workers=n_workers,
        pin_memory=True, persistent_workers=persistent,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    # freeze_blocks=0 = full fine-tune of all 12 blocks; any value > 0 degrades accuracy.
    embedder = _build_embedder(cfg, device)

    if "init_from" in cfg and Path(cfg["init_from"]).exists():
        state = torch.load(cfg["init_from"], map_location=device)
        embedder.load_state_dict(state["embedder"], strict=False)
        print(f"Loaded weights from {cfg['init_from']}")

    loss_type = cfg["loss"].get("type", "proxynca")
    K = cfg["loss"].get("K", 1)

    if loss_type == "arcface":
        proxy_head = SubCenterProxyHead(train_ds.num_classes, cfg["embed_dim"], K=K).to(device)
    else:
        proxy_head = ProxyHead(train_ds.num_classes, cfg["embed_dim"]).to(device)

    # proxy init must use full training data, not train_loader with batch_sampler (~m imgs/class).
    init_loader = DataLoader(
        train_ds, batch_size=256, shuffle=False,
        num_workers=n_workers, pin_memory=True,
    )
    proxy_head.init_from_embeddings(embedder, init_loader, device)
    torch.cuda.empty_cache()  # free proxy-init VRAM before training loop
    embedder.train()

    # ── Loss ──────────────────────────────────────────────────────────────
    if loss_type == "arcface":
        criterion = SubCenterArcFaceLoss(
            scale=cfg["loss"].get("scale", 30.0),
            margin=cfg["loss"].get("margin", 0.5),
        )
    elif use_hn:
        criterion = ProxyNCAHNPPLoss(sigma=cfg["loss"]["sigma"], hn_map=hn_map)
    else:
        criterion = ProxyNCAPPLoss(sigma=cfg["loss"]["sigma"])

    # ── Optimizer ─────────────────────────────────────────────────────────
    # build_optimizer BEFORE torch.compile to avoid unstable id(param)
    # when OptimizedModule wraps the original model.
    optimizer = build_optimizer(embedder, proxy_head, cfg)

    # torch.compile — free ~10-20% on Ampere/Ada GPUs (skip on CPU)
    # Must come AFTER optimizer so param groups are bound to the correct tensor objects.
    import platform
    use_compile = (
        device.type == "cuda"
        and cfg.get("compile", True)
        and platform.system() != "Windows"  # Triton not supported on Windows
    )
    if use_compile:
        embedder = torch.compile(embedder)
        print("torch.compile enabled")
    else:
        print(f"torch.compile disabled (Windows or CPU)")
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=cfg["scheduler"]["patience"],
        factor=cfg["scheduler"]["factor"],
        mode="max",
    )

    # AMP scaler — bfloat16 on Ampere+ (CC>=8.0), float16 on Volta/Turing (CC<8.0)
    cc = torch.cuda.get_device_capability()
    amp_dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
    use_amp = device.type == "cuda" and cfg.get("amp", True)
    scaler = GradScaler(enabled=use_amp)
    print(f"AMP: {'enabled (' + str(amp_dtype) + ')' if use_amp else 'disabled'}")

    # Early stopping — stop when val recall shows no improvement for es_patience epochs
    es_patience = cfg.get("early_stopping_patience", 6)
    epochs_no_improve = 0
    print(f"Early stopping patience: {es_patience} epochs")

    # ── Training loop ─────────────────────────────────────────────────────
    best_recall = 0.0
    for epoch in range(cfg["training"]["epochs"]):
        embedder.train()
        proxy_head.train()
        total_loss = 0.0
        n_batches = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                embs = embedder(imgs)
                loss = criterion(embs, labels, proxy_head.proxies)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_recall = recall_at_1(embedder, val_loader, device)
        scheduler.step(val_recall)

        print(f"Epoch {epoch+1:3d} | loss {avg_loss:.4f} | val recall@1 {val_recall:.4f}")

        if val_recall > best_recall:
            best_recall = val_recall
            epochs_no_improve = 0
            CKPT.mkdir(exist_ok=True)
            torch.save(
                {"embedder": embedder.state_dict(), "proxy": proxy_head.state_dict()},
                CKPT / ckpt_name,
            )
            print(f"  → saved {ckpt_name} (recall@1={best_recall:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= es_patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(no improvement for {es_patience} epochs)")
                break

    print(f"\nBest val recall@1: {best_recall:.4f}")
