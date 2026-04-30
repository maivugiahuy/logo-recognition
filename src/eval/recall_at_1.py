"""
Recall@1 evaluation — paper Sec 4.1.
Modes:
  query_vs_gallery  — 10 random imgs/class as query, rest as gallery (Sec 4.1)
  all_vs_all        — entire set vs itself, 2nd nearest if query in gallery
Subsets:
  text              — classes containing "_text" in name
  small             — min(crop w, crop h) < 70px
  large             — min(crop w, crop h) ≥ 70px
"""
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import OLG3K47Dataset
from src.data.transforms import val_transforms
from src.models.embedder_vit import build_vit_embedder

SMALL_THRESHOLD = 70  # paper: 40th percentile ~70px (Sec 4.1.4)
N_QUERY_PER_CLASS = 10
SEED = 42


def _embed_dataset(
    ds: OLG3K47Dataset,
    embedder: torch.nn.Module,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, list[str], list[float]]:
    """Returns (embs, class_names, min_sides)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    all_embs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Embedding", leave=False):
            embs = embedder(imgs.to(device)).cpu().numpy()
            all_embs.append(embs)
            all_labels.extend(labels.tolist())
    embs = np.concatenate(all_embs).astype("float32")
    # Map idx→name
    idx_to_cls = {v: k for k, v in ds.class_to_idx.items()}
    class_names = [idx_to_cls[l] for l in all_labels]

    # Compute min_side from df
    min_sides = []
    for _, row in ds.df.iterrows():
        w = row["x2"] - row["x1"]
        h = row["y2"] - row["y1"]
        min_sides.append(min(w, h))
    return embs, class_names, min_sides


def query_vs_gallery(
    embs: np.ndarray,
    class_names: list[str],
    n_query: int = N_QUERY_PER_CLASS,
    seed: int = SEED,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Split into disjoint query (n_query per class) and gallery sets."""
    rng = random.Random(seed)
    class_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, cls in enumerate(class_names):
        class_to_indices[cls].append(i)

    query_idx, gallery_idx = [], []
    for cls, indices in class_to_indices.items():
        rng.shuffle(indices)
        # Adaptive: dùng tối đa 1/3 làm query, ít nhất 1 gallery còn lại
        # Tránh trường hợp class nhỏ có 0 ảnh trong gallery
        n_q = min(n_query, max(1, len(indices) - 1), len(indices) // 3 + 1)
        query_idx.extend(indices[:n_q])
        gallery_idx.extend(indices[n_q:])

    q_embs = embs[query_idx]
    g_embs = embs[gallery_idx]
    q_labels = [class_names[i] for i in query_idx]
    g_labels = [class_names[i] for i in gallery_idx]
    return q_embs, q_labels, g_embs, g_labels


def compute_recall_at_1(
    q_embs: np.ndarray,
    q_labels: list[str],
    g_embs: np.ndarray,
    g_labels: list[str],
    all_vs_all: bool = False,
) -> float:
    """
    Compute recall@1.
    If all_vs_all=True, q==g and we take 2nd-nearest neighbor.
    """
    index = faiss.IndexFlatIP(q_embs.shape[1])
    index.add(g_embs)
    k = 2 if all_vs_all else 1
    _, nn_indices = index.search(q_embs, k)  # (Q, k)
    nn_idx = nn_indices[:, -1]  # last column: 2nd-NN when all_vs_all, else 1st
    correct = sum(
        g_labels[nn_idx[i]] == q_labels[i]
        for i in range(len(q_labels))
    )
    return correct / len(q_labels)


def evaluate(
    ckpt_path: str | Path,
    ann_parquet: str | Path,
    split_json: str | Path,
    mode: str = "open_set",
    embed_dim: int = 128,
    input_size: int = 160,
) -> dict[str, float]:
    """Full evaluation returning recall@1 for all subsets."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = build_vit_embedder(embed_dim, input_size, freeze_blocks=0).to(device)
    state = torch.load(ckpt_path, map_location=device)
    embedder.load_state_dict(state["embedder"])
    embedder.eval()

    ds = OLG3K47Dataset.from_split(
        ann_parquet, split_json, transform=val_transforms(input_size), mode=mode
    )
    embs, class_names, min_sides = _embed_dataset(ds, embedder, device)

    # all-vs-all
    ava = compute_recall_at_1(embs, class_names, embs, class_names, all_vs_all=True)

    # query-vs-gallery
    q_embs, q_labels, g_embs, g_labels = query_vs_gallery(embs, class_names)
    qvg = compute_recall_at_1(q_embs, q_labels, g_embs, g_labels)

    # text subset (Q-vs-G)
    text_mask = [cls.endswith("_text") for cls in q_labels]
    if any(text_mask):
        t_embs = q_embs[[i for i, m in enumerate(text_mask) if m]]
        t_labels = [l for l, m in zip(q_labels, text_mask) if m]
        text_r = compute_recall_at_1(t_embs, t_labels, g_embs, g_labels)
    else:
        text_r = float("nan")

    # small / large subsets — need per-query min_side
    # Recompute min_sides for query indices
    rng = random.Random(SEED)
    class_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, cls in enumerate(class_names):
        class_to_indices[cls].append(i)
    query_idx = []
    for cls, indices in class_to_indices.items():
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_q = min(N_QUERY_PER_CLASS, max(1, len(shuffled) - 1), len(shuffled) // 3 + 1)
        query_idx.extend(shuffled[:n_q])

    q_min_sides = [min_sides[i] for i in query_idx]
    small_mask = [s < SMALL_THRESHOLD for s in q_min_sides]
    large_mask = [not m for m in small_mask]

    def subset_recall(mask):
        idx = [i for i, m in enumerate(mask) if m]
        if not idx:
            return float("nan")
        se = q_embs[idx]
        sl = [q_labels[i] for i in idx]
        return compute_recall_at_1(se, sl, g_embs, g_labels)

    small_r = subset_recall(small_mask)
    large_r = subset_recall(large_mask)

    results = {
        "all_vs_all": ava,
        "qvg": qvg,
        "text_qvg": text_r,
        "small_qvg": small_r,
        "large_qvg": large_r,
    }
    for k, v in results.items():
        print(f"  {k:15s}: {v:.4f}")
    return results
