"""Step 11: Precompute ensemble teacher embeddings for distillation.

Embeds all closed-set training crops with ViT-HN + DINOv2-HN, computes weighted
average embedding, saves to data/processed/:
  teacher_embs.npy          — (N, 128) float32
  teacher_train.parquet     — same N rows as embs (image_path, class_name, bbox, ...)

Usage:
  python scripts/11_precompute_teacher.py
  python scripts/11_precompute_teacher.py --vit_weight 0.6
"""
import argparse
import sys
sys.path.insert(0, ".")
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.transforms import val_transforms, val_transforms_dinov2
from src.models.embedder_dinov2 import build_dinov2_embedder
from src.models.embedder_vit import build_vit_embedder
from src.utils.logging_utils import setup_logging

ANN = Path("data/processed/openlogodet3k/annotations.parquet")
SPLITS = Path("data/processed/openlogodet3k/splits")
OUT_EMBS = Path("data/processed/teacher_embs.npy")
OUT_PARQUET = Path("data/processed/teacher_train.parquet")

VIT_CKPT = "checkpoints/vit_hn.pt"
DINO_CKPT = "checkpoints/dinov2_hn.pt"
VIT_INPUT_SIZE = 160
DINO_INPUT_SIZE = 168
EMBED_DIM = 128


class _CropDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        from PIL import Image
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self._Image = Image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._Image.open(row["image_path"]).convert("RGB")
        w, h = img.size
        x1, y1 = max(0, int(row["x1"])), max(0, int(row["y1"]))
        x2, y2 = min(w, int(row["x2"])), min(h, int(row["y2"]))
        if x2 > x1 and y2 > y1:
            img = img.crop((x1, y1, x2, y2))
        return self.transform(img)


def _embed_all(df, embedder, transform, device, batch_size=256):
    ds = _CropDataset(df, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=6)
    all_embs = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="Embedding", leave=False):
            embs = embedder(imgs.to(device)).cpu().numpy()
            all_embs.append(embs)
    return np.concatenate(all_embs).astype("float32")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vit_ckpt", default=VIT_CKPT)
    parser.add_argument("--dinov2_ckpt", default=DINO_CKPT)
    parser.add_argument("--vit_weight", type=float, default=0.5,
                        help="Weight for ViT embeddings in teacher (default: 0.5)")
    parser.add_argument("--split", default="closed_train",
                        help="Split json name under splits/ (default: closed_train)")
    args = parser.parse_args()

    setup_logging(__file__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training annotations
    df = pd.read_parquet(ANN)
    split_json = SPLITS / f"{args.split}.json"
    if split_json.exists():
        with open(split_json) as f:
            split = json.load(f)  # {class_name: [image_paths]}
        keep = []
        for cls, imgs in split.items():
            img_set = set(imgs)
            rows = df[(df["class_name"] == cls) & (df["image_path"].isin(img_set))]
            keep.append(rows)
        df = pd.concat(keep, ignore_index=True) if keep else df
    else:
        print(f"[WARN] {split_json} not found — using all annotations")
    print(f"Training set: {df['class_name'].nunique()} classes, {len(df)} crops")

    # ViT embedder
    print("\n[1/2] Embedding with ViT-HN...")
    vit_emb = build_vit_embedder(EMBED_DIM, VIT_INPUT_SIZE, freeze_blocks=0).to(device)
    state = torch.load(args.vit_ckpt, map_location=device)
    vit_emb.load_state_dict(state["embedder"])
    vit_emb.eval()
    vit_embs = _embed_all(df, vit_emb, val_transforms(VIT_INPUT_SIZE), device)
    del vit_emb
    torch.cuda.empty_cache()

    # DINOv2 embedder
    print("\n[2/2] Embedding with DINOv2-HN...")
    dino_emb = build_dinov2_embedder(EMBED_DIM, DINO_INPUT_SIZE, freeze_blocks=0).to(device)
    state = torch.load(args.dinov2_ckpt, map_location=device)
    dino_emb.load_state_dict(state["embedder"])
    dino_emb.eval()
    dino_embs = _embed_all(df, dino_emb, val_transforms_dinov2(DINO_INPUT_SIZE), device)
    del dino_emb
    torch.cuda.empty_cache()

    # Weighted average + L2-normalize
    dino_weight = 1.0 - args.vit_weight
    combined = args.vit_weight * vit_embs + dino_weight * dino_embs
    norms = np.linalg.norm(combined, axis=1, keepdims=True).clip(1e-8)
    teacher_embs = (combined / norms).astype("float32")

    # Save
    OUT_EMBS.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_EMBS, teacher_embs)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nSaved → {OUT_EMBS}  shape={teacher_embs.shape}")
    print(f"Saved → {OUT_PARQUET}  rows={len(df)}")
