"""
Training loop — open-set (Phase A) and closed-set (Phase C).
25 epochs, AdamW, ReduceLROnPlateau (patience 4, factor 0.25).
Speed opts: AMP mixed precision, torch.compile, num_workers=8.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import OLG3K47Dataset
from src.data.samplers import (
    HardNegativeBatchSampler,
    MPerClassSampler,
    load_hn_map,
)
from src.data.transforms import train_transforms, val_transforms
from src.losses.proxynca_hn_pp import ProxyNCAHNPPLoss
from src.losses.proxynca_pp import ProxyNCAPPLoss
from src.models.embedder_vit import build_vit_embedder
from src.models.proxy_head import ProxyHead
from src.training.optim import build_optimizer

ANN = Path("data/processed/openlogodet3k47/annotations.parquet")
SPLITS = Path("data/processed/openlogodet3k47/splits")
CKPT = Path("checkpoints")


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
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TF32 — miễn phí ~10% speedup trên Ampere/Ada (RTX 30xx/40xx).
    # Không ảnh hưởng accuracy vì loss/eval vẫn dùng bfloat16/float32.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    mode = cfg.get("mode", "open_set")
    split_prefix = "open" if mode == "open_set" else "closed"

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = OLG3K47Dataset.from_split(
        ANN, SPLITS / f"{split_prefix}_train.json",
        transform=train_transforms(cfg["input_size"]), mode=mode,
    )
    val_ds = OLG3K47Dataset.from_split(
        ANN, SPLITS / f"{split_prefix}_val.json",
        transform=val_transforms(cfg["input_size"]), mode=mode,
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

    n_workers = cfg.get("num_workers", 8)
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler, num_workers=n_workers, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=n_workers)

    # ── Model ─────────────────────────────────────────────────────────────
    # FIX: freeze_blocks=0 để fine-tune toàn bộ 12 transformer blocks đúng paper.
    # freeze_blocks=8 (default cũ) là speedup không có trong paper → accuracy thấp hơn.
    freeze_blocks = cfg.get("freeze_blocks", 0)
    embedder = build_vit_embedder(
        cfg["embed_dim"], cfg["input_size"], freeze_blocks=freeze_blocks
    ).to(device)

    if "init_from" in cfg and Path(cfg["init_from"]).exists():
        state = torch.load(cfg["init_from"], map_location=device)
        embedder.load_state_dict(state["embedder"], strict=False)
        print(f"Loaded weights from {cfg['init_from']}")

    proxy_head = ProxyHead(train_ds.num_classes, cfg["embed_dim"]).to(device)
    # FIX: proxy init phải dùng toàn bộ training data (paper Sec 4.2),
    # không phải train_loader có batch_sampler (chỉ lấy ~m ảnh/class).
    init_loader = DataLoader(
        train_ds, batch_size=256, shuffle=False,
        num_workers=n_workers, pin_memory=True,
    )
    proxy_head.init_from_embeddings(embedder, init_loader, device)
    embedder.train()

    # ── Loss ──────────────────────────────────────────────────────────────
    sigma = cfg["loss"]["sigma"]
    if use_hn:
        criterion = ProxyNCAHNPPLoss(sigma=sigma, hn_map=hn_map)
    else:
        criterion = ProxyNCAPPLoss(sigma=sigma)

    # ── Optimizer ─────────────────────────────────────────────────────────
    # FIX: build_optimizer TRƯỚC torch.compile để tránh id(param) không ổn định
    # khi OptimizedModule wrap model gốc.
    optimizer = build_optimizer(embedder, proxy_head, cfg)

    # torch.compile — free ~10-20% on Ampere/Ada GPUs (skip on CPU)
    # Đặt SAU optimizer để param groups đã được gắn vào đúng tensor objects.
    use_compile = device.type == "cuda" and cfg.get("compile", True)
    if use_compile:
        embedder = torch.compile(embedder)
        print("torch.compile enabled")
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=cfg["scheduler"]["patience"],
        factor=cfg["scheduler"]["factor"],
        mode="max",
    )

    # AMP scaler — mixed precision (bfloat16 on Ampere, float16 elsewhere)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_amp = device.type == "cuda" and cfg.get("amp", True)
    scaler = GradScaler(enabled=use_amp)
    print(f"AMP: {'enabled (' + str(amp_dtype) + ')' if use_amp else 'disabled'}")

    # Early stopping — dừng khi val recall không cải thiện sau `es_patience` epochs
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
                print(f"\nEarly stopping tại epoch {epoch+1} "
                      f"(không cải thiện sau {es_patience} epochs)")
                break

    print(f"\nBest val recall@1: {best_recall:.4f}")
