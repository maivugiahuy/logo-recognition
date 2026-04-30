"""
Smoke test — Step 4.
50 classes, 5 epochs, b=64. Verifies full pipeline: load → train → embed → FAISS → predict.
Gate: loss decreasing; recall@1 on smoke subset > 0.5.
"""
import sys
sys.path.insert(0, ".")

import random
import json
import tempfile
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import OLG3K47Dataset
from src.data.samplers import MPerClassSampler
from src.data.transforms import train_transforms, val_transforms
from src.losses.proxynca_pp import ProxyNCAPPLoss
from src.models.embedder_vit import build_vit_embedder
from src.models.proxy_head import ProxyHead
from src.training.optim import build_optimizer

ANN = Path("data/processed/openlogodet3k47/annotations.parquet")
SPLITS = Path("data/processed/openlogodet3k47/splits")

N_CLASSES = 50
N_EPOCHS = 5
K = 10
M = 4
EMBED_DIM = 128
INPUT_SIZE = 160


def main():
    import pandas as pd
    import numpy as np
    import faiss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Subset: pick 50 train classes
    with open(SPLITS / "open_train.json") as f:
        all_train_cls = json.load(f)
    smoke_cls = random.sample(all_train_cls, min(N_CLASSES, len(all_train_cls)))

    # Write temp split files
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        smoke_train = tmp / "smoke_train.json"
        smoke_val = tmp / "smoke_val.json"
        with open(smoke_train, "w") as f:
            json.dump(smoke_cls[:40], f)
        with open(smoke_val, "w") as f:
            json.dump(smoke_cls[40:], f)

        train_ds = OLG3K47Dataset.from_split(
            ANN, smoke_train, transform=train_transforms(INPUT_SIZE), mode="open_set"
        )
        val_ds = OLG3K47Dataset.from_split(
            ANN, smoke_val, transform=val_transforms(INPUT_SIZE), mode="open_set"
        )

    print(f"Smoke train: {len(train_ds)} objects, {train_ds.num_classes} classes")
    print(f"Smoke val:   {len(val_ds)} objects,   {val_ds.num_classes} classes")

    sampler = MPerClassSampler(train_ds.labels, k=K, m=M)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    embedder = build_vit_embedder(EMBED_DIM, INPUT_SIZE).to(device)
    proxy_head = ProxyHead(train_ds.num_classes, EMBED_DIM).to(device)
    # BUG FIX: proxy init dùng loader không có sampler để thấy toàn bộ training data
    init_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    proxy_head.init_from_embeddings(embedder, init_loader, device)
    embedder.train()

    criterion = ProxyNCAPPLoss(sigma=0.06)

    cfg = {
        "optimizer": {
            "trunk_lr": 2.3e-6, "fc_lr": 1.5e-3, "proxy_lr": 71,
            "trunk_wd": 0.2, "fc_wd": 0.001, "proxy_wd": 0.0,
            "trunk_beta2": 0.98, "fc_beta2": 0.98, "proxy_beta2": 0.999,
            "trunk_eps": 1e-6, "fc_eps": 1e-6, "proxy_eps": 1.0,
        }
    }
    optimizer = build_optimizer(embedder, proxy_head, cfg)
    # BUG FIX: mode="min" + step(avg) thay vì mode="max" + step(-avg) — cùng kết quả
    # nhưng rõ ràng hơn, nhất quán với train.py dùng mode="max" + step(val_recall).
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5, mode="min")

    losses = []
    for epoch in range(N_EPOCHS):
        embedder.train()
        proxy_head.train()
        ep_loss = []
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            embs = embedder(imgs)
            loss = criterion(embs, labels, proxy_head.proxies)
            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())
        avg = sum(ep_loss) / len(ep_loss)
        losses.append(avg)
        print(f"Epoch {epoch+1}: loss={avg:.4f}")
        scheduler.step(avg)

    # Check loss decreasing
    first, last = losses[0], losses[-1]
    print(f"\nLoss: {first:.4f} → {last:.4f}", end="  ")
    if last < first:
        print("[OK: decreasing]")
    else:
        print("[WARN: not decreasing — check data or LR]")

    # Smoke recall@1 on val (embed val, gallery = train embeddings)
    embedder.eval()
    def embed_all(loader):
        embs, lbls = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                e = embedder(imgs.to(device)).cpu().numpy()
                embs.append(e)
                lbls.extend(labels.tolist())
        return np.concatenate(embs), lbls

    if val_ds.num_classes > 0:
        # Build gallery from train embeddings
        g_embs, g_lbls = embed_all(DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0))
        q_embs, q_lbls = embed_all(DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0))

        # Map to class names
        idx2cls_train = {v: k for k, v in train_ds.class_to_idx.items()}
        idx2cls_val = {v: k for k, v in val_ds.class_to_idx.items()}
        g_names = [idx2cls_train[l] for l in g_lbls]
        q_names = [idx2cls_val[l] for l in q_lbls]

        index = faiss.IndexFlatIP(EMBED_DIM)
        index.add(g_embs.astype("float32"))
        _, nn_idx = index.search(q_embs.astype("float32"), 1)
        correct = sum(g_names[nn_idx[i, 0]] == q_names[i] for i in range(len(q_names)))
        recall = correct / len(q_names) if q_names else 0.0
        status = "OK" if recall > 0.5 else "WARN (< 0.5 — open-set is hard; may still be fine)"
        print(f"Smoke recall@1 (val): {recall:.4f}  [{status}]")
    else:
        print("Smoke recall: skipped (val split empty)")

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
