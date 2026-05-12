"""
Recall@1 evaluation.
Modes:
  query_vs_gallery  — 10 random imgs/class as query, rest as gallery
  all_vs_all        — entire set vs itself, 2nd nearest if query in gallery
Subsets:
  text              — classes containing "_text" in name
  small             — min(crop w, crop h) < 70px
  large             — min(crop w, crop h) ≥ 70px
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LogoDataset
from src.data.transforms import val_transforms, val_transforms_dinov2
from src.models.embedder_dinov2 import build_dinov2_embedder
from src.models.embedder_dinov3 import build_dinov3_embedder
from src.models.embedder_vit import build_vit_embedder

_DINOV2_BACKBONES = {"dinov2_vitb14", "dinov2"}
_DINOV3_BACKBONES = {"dinov3_vitb16"}
_IMAGENET_BACKBONES = _DINOV2_BACKBONES | _DINOV3_BACKBONES

SMALL_THRESHOLD = 70  # 40th percentile of crop min-side ≈ 70px
N_QUERY_PER_CLASS = 10
SEED = 42


def _embed_dataset(
    ds: LogoDataset,
    embedder: torch.nn.Module,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, list[str], list[float]]:
    """Returns (embs, class_names, min_sides)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    all_embs, all_labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Embedding", leave=False):
            embs = embedder(imgs.to(device)).cpu().numpy()
            all_embs.append(embs)
            all_labels.extend(labels.tolist())
    embs = np.concatenate(all_embs).astype("float32")
    elapsed = time.time() - t0
    n = len(embs)
    print(f"  {'embed':15s}: {n} crops  {elapsed:.1f}s  ({1000*elapsed/n:.2f} ms/crop)")
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
        # Adaptive: use at most 1/3 as query, keep at least 1 image in gallery
        # Avoids small classes ending up with 0 gallery images
        n_q = min(n_query, max(1, len(indices) - 1), len(indices) // 3 + 1)
        query_idx.extend(indices[:n_q])
        gallery_idx.extend(indices[n_q:])

    q_embs = embs[query_idx]
    g_embs = embs[gallery_idx]
    q_labels = [class_names[i] for i in query_idx]
    g_labels = [class_names[i] for i in gallery_idx]
    return q_embs, q_labels, g_embs, g_labels



def _rerank_k_reciprocal(
    top_scores: np.ndarray,
    top_indices: np.ndarray,
    g_embs: np.ndarray,
    g_index: faiss.Index,
    k1: int = 20,
    lambda_val: float = 0.3,
) -> np.ndarray:
    """Local k-reciprocal re-ranking (Zhong et al. 2017). Returns best gallery index per query."""
    Q, K = top_indices.shape
    k1 = min(k1, K, g_index.ntotal)
    unique_cands = np.unique(top_indices)
    _, g2g_idx = g_index.search(g_embs[unique_cands], min(k1, g_index.ntotal))
    cand_to_pos = {int(c): i for i, c in enumerate(unique_cands)}

    best_indices = np.empty(Q, dtype=np.int64)
    for qi in range(Q):
        R_q = set(top_indices[qi, :k1].tolist())
        R_q_star = set(R_q)
        for p in R_q:
            pos = cand_to_pos.get(int(p))
            if pos is None:
                continue
            R_p = set(g2g_idx[pos].tolist())
            if len(R_q & R_p) >= (2.0 / 3.0) * len(R_p):
                R_q_star |= R_p
        best_score, best_idx = -1.0, int(top_indices[qi, 0])
        for s, g_idx in zip(top_scores[qi], top_indices[qi]):
            g_idx = int(g_idx)
            pos = cand_to_pos.get(g_idx)
            R_g = set(g2g_idx[pos].tolist()) if pos is not None else set()
            jaccard = len(R_q_star & R_g) / max(len(R_q_star | R_g), 1)
            fused = (1 - lambda_val) * float(s) + lambda_val * jaccard
            if fused > best_score:
                best_score, best_idx = fused, g_idx
        best_indices[qi] = best_idx
    return best_indices


def compute_recall_at_1(
    q_embs: np.ndarray,
    q_labels: list[str],
    g_embs: np.ndarray,
    g_labels: list[str],
    all_vs_all: bool = False,
    rerank: bool = False,
    rerank_k: int = 50,
    rerank_k1: int = 20,
    rerank_lambda: float = 0.3,
    _label: str = "",
) -> float:
    """
    Compute recall@1.
    If all_vs_all=True, q==g and we take 2nd-nearest neighbor.
    If rerank=True, apply k-reciprocal re-ranking (Zhong et al. 2017).
    """
    index = faiss.IndexFlatIP(q_embs.shape[1])
    index.add(g_embs)
    nq = len(q_labels)

    if rerank and not all_vs_all:
        k_search = min(rerank_k, index.ntotal)
        t0 = time.time()
        scores_mat, indices_mat = index.search(q_embs, k_search)
        nn_idx = _rerank_k_reciprocal(scores_mat, indices_mat, g_embs, index,
                                      k1=rerank_k1, lambda_val=rerank_lambda)
        elapsed = time.time() - t0
        tag = f"rerank{' ' + _label if _label else ''}"
        print(f"  {tag:15s}: {nq} queries  {elapsed:.1f}s  ({1000*elapsed/nq:.2f} ms/query)")
        correct = sum(g_labels[nn_idx[i]] == q_labels[i] for i in range(nq))
        return correct / nq

    k = 2 if all_vs_all else 1
    t0 = time.time()
    _, nn_indices = index.search(q_embs, k)  # (Q, k)
    elapsed = time.time() - t0
    tag = f"retrieval{' ' + _label if _label else ''}"
    print(f"  {tag:15s}: {nq} queries  {elapsed:.1f}s  ({1000*elapsed/nq:.2f} ms/query)")
    nn_idx = nn_indices[:, -1]  # last column: 2nd-NN when all_vs_all, else 1st
    correct = sum(
        g_labels[nn_idx[i]] == q_labels[i]
        for i in range(len(q_labels))
    )
    return correct / nq


def evaluate(
    ckpt_path: str | Path,
    ann_parquet: str | Path,
    split_json: str | Path,
    mode: str = "open_set",
    backbone: str = "vit_b32_openai",
    embed_dim: int = 128,
    input_size: int | None = None,
    rerank: bool = False,
    rerank_k: int = 50,
    rerank_k1: int = 20,
    rerank_lambda: float = 0.3,
) -> dict[str, float]:
    """Full evaluation returning recall@1 for all subsets."""
    is_dinov2 = backbone in _DINOV2_BACKBONES
    is_dinov3 = backbone in _DINOV3_BACKBONES
    if input_size is None:
        input_size = 224 if is_dinov2 else 160

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_dinov2:
        embedder = build_dinov2_embedder(embed_dim, input_size, freeze_blocks=0).to(device)
    elif is_dinov3:
        embedder = build_dinov3_embedder(embed_dim, input_size, freeze_blocks=0).to(device)
    else:
        embedder = build_vit_embedder(embed_dim, input_size, freeze_blocks=0).to(device)
    state = torch.load(ckpt_path, map_location=device)
    embedder.load_state_dict(state["embedder"])
    embedder.eval()

    transform = val_transforms_dinov2(input_size) if backbone in _IMAGENET_BACKBONES else val_transforms(input_size)
    ds = LogoDataset.from_split(
        ann_parquet, split_json, transform=transform, mode=mode
    )
    embs, class_names, min_sides = _embed_dataset(ds, embedder, device)

    # Compute query indices for small/large subsets
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

    rerank_args = dict(rerank=rerank, rerank_k=rerank_k, rerank_k1=rerank_k1,
                       rerank_lambda=rerank_lambda)

    # all-vs-all (rerank skipped — self-exclusion incompatible)
    ava = compute_recall_at_1(embs, class_names, embs, class_names, all_vs_all=True,
                              _label="all_vs_all")

    # query-vs-gallery
    q_embs, q_labels, g_embs, g_labels = query_vs_gallery(embs, class_names)
    qvg = compute_recall_at_1(q_embs, q_labels, g_embs, g_labels, **rerank_args, _label="qvg")

    # text subset (Q-vs-G)
    text_mask = [cls.endswith("_text") for cls in q_labels]
    if any(text_mask):
        t_idx = [i for i, m in enumerate(text_mask) if m]
        t_embs = q_embs[t_idx]
        t_labels = [q_labels[i] for i in t_idx]
        text_r = compute_recall_at_1(t_embs, t_labels, g_embs, g_labels,
                                     **rerank_args, _label="text")
    else:
        text_r = float("nan")

    # small / large subsets
    q_min_sides = [min_sides[i] for i in query_idx]
    small_mask = [s < SMALL_THRESHOLD for s in q_min_sides]
    large_mask = [not m for m in small_mask]

    def subset_recall(mask, label):
        idx = [i for i, m in enumerate(mask) if m]
        if not idx:
            return float("nan")
        se = q_embs[idx]
        sl = [q_labels[i] for i in idx]
        return compute_recall_at_1(se, sl, g_embs, g_labels, **rerank_args, _label=label)

    small_r = subset_recall(small_mask, "small")
    large_r = subset_recall(large_mask, "large")

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


def evaluate_ensemble(
    vit_ckpt: str | Path,
    dino_ckpt: str | Path,
    ann_parquet: str | Path,
    split_json: str | Path = None,
    mode: str = "open_set",
    embed_dim: int = 128,
    vit_input_size: int = 160,
    dino_input_size: int | None = None,
    dino_backbone: str = "dinov3_vitb16",
    vit_weight: float = 0.5,
    ensemble_top_k: int = 20,
) -> dict[str, float]:
    """Ensemble recall@1: fuse ViT + DINO scores per label, pick best."""
    import numpy as np

    if dino_input_size is None:
        dino_input_size = 168 if dino_backbone in _DINOV2_BACKBONES else 160

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit_embedder = build_vit_embedder(embed_dim, vit_input_size, freeze_blocks=0).to(device)
    state = torch.load(vit_ckpt, map_location=device)
    vit_embedder.load_state_dict(state["embedder"])
    vit_embedder.eval()

    if dino_backbone in _DINOV2_BACKBONES:
        dino_embedder = build_dinov2_embedder(embed_dim, dino_input_size, freeze_blocks=0).to(device)
    else:
        dino_embedder = build_dinov3_embedder(embed_dim, dino_input_size, freeze_blocks=0).to(device)
    state = torch.load(dino_ckpt, map_location=device)
    dino_embedder.load_state_dict(state["embedder"])
    dino_embedder.eval()

    vit_ds = LogoDataset.from_split(
        ann_parquet, split_json, transform=val_transforms(vit_input_size), mode=mode
    )
    dino_ds = LogoDataset.from_split(
        ann_parquet, split_json, transform=val_transforms_dinov2(dino_input_size), mode=mode
    )

    vit_embs, class_names, min_sides = _embed_dataset(vit_ds, vit_embedder, device)
    dino_embs, _, _ = _embed_dataset(dino_ds, dino_embedder, device)

    # Shared query/gallery split (same seed → same indices for both backbones)
    rng = random.Random(SEED)
    class_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, cls in enumerate(class_names):
        class_to_indices[cls].append(i)
    query_idx, gallery_idx = [], []
    for cls, indices in class_to_indices.items():
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_q = min(N_QUERY_PER_CLASS, max(1, len(shuffled) - 1), len(shuffled) // 3 + 1)
        query_idx.extend(shuffled[:n_q])
        gallery_idx.extend(shuffled[n_q:])

    q_vit = vit_embs[query_idx]
    g_vit = vit_embs[gallery_idx]
    q_dino = dino_embs[query_idx]
    g_dino = dino_embs[gallery_idx]
    q_labels = [class_names[i] for i in query_idx]
    g_labels = [class_names[i] for i in gallery_idx]
    q_min_sides = [min_sides[i] for i in query_idx]

    # All-vs-all: full-set indexes, skip self during fusion
    def _ensemble_ava_recall() -> float:
        vit_all = faiss.IndexFlatIP(embed_dim)
        vit_all.add(vit_embs)
        dino_all = faiss.IndexFlatIP(embed_dim)
        dino_all.add(dino_embs)
        k = min(ensemble_top_k + 1, vit_all.ntotal)
        t0 = time.time()
        vit_s_a, vit_i_a = vit_all.search(vit_embs, k)
        dino_s_a, dino_i_a = dino_all.search(dino_embs, k)
        dinov2_w = 1.0 - vit_weight
        N = len(class_names)
        correct = 0
        for qi in range(N):
            vit_lbl: dict[str, float] = {}
            for s, idx in zip(vit_s_a[qi], vit_i_a[qi]):
                if idx == qi:
                    continue
                lbl = class_names[idx]
                vit_lbl[lbl] = max(vit_lbl.get(lbl, 0.0), float(s))
            dino_lbl: dict[str, float] = {}
            for s, idx in zip(dino_s_a[qi], dino_i_a[qi]):
                if idx == qi:
                    continue
                lbl = class_names[idx]
                dino_lbl[lbl] = max(dino_lbl.get(lbl, 0.0), float(s))
            all_lbls = set(vit_lbl) | set(dino_lbl)
            if not all_lbls:
                continue
            fused = {
                l: vit_weight * vit_lbl.get(l, 0.0) + dinov2_w * dino_lbl.get(l, 0.0)
                for l in all_lbls
            }
            correct += max(fused, key=fused.__getitem__) == class_names[qi]
        elapsed = time.time() - t0
        print(f"  {'all_vs_all':15s}: {N} queries  {elapsed:.1f}s  ({1000*elapsed/N:.2f} ms/query)")
        return correct / N

    ava = _ensemble_ava_recall()

    vit_index = faiss.IndexFlatIP(embed_dim)
    vit_index.add(g_vit)
    dino_index = faiss.IndexFlatIP(embed_dim)
    dino_index.add(g_dino)

    def _ensemble_recall(q_mask: list[bool] | None = None) -> float:
        if q_mask is not None:
            qi_list = [i for i, m in enumerate(q_mask) if m]
            if not qi_list:
                return float("nan")
            qv = q_vit[qi_list]
            qd = q_dino[qi_list]
            ql = [q_labels[i] for i in qi_list]
        else:
            qv, qd, ql = q_vit, q_dino, q_labels

        k = min(ensemble_top_k, vit_index.ntotal)
        vit_s, vit_i = vit_index.search(qv, k)
        dino_s, dino_i = dino_index.search(qd, k)
        dinov2_weight = 1.0 - vit_weight

        correct = 0
        for qi in range(len(ql)):
            vit_lbl: dict[str, float] = {}
            for s, idx in zip(vit_s[qi], vit_i[qi]):
                lbl = g_labels[idx]
                vit_lbl[lbl] = max(vit_lbl.get(lbl, 0.0), float(s))
            dino_lbl: dict[str, float] = {}
            for s, idx in zip(dino_s[qi], dino_i[qi]):
                lbl = g_labels[idx]
                dino_lbl[lbl] = max(dino_lbl.get(lbl, 0.0), float(s))
            all_lbls = set(vit_lbl) | set(dino_lbl)
            fused = {
                l: vit_weight * vit_lbl.get(l, 0.0) + dinov2_weight * dino_lbl.get(l, 0.0)
                for l in all_lbls
            }
            best = max(fused, key=fused.__getitem__)
            correct += best == ql[qi]
        return correct / len(ql)

    qvg = _ensemble_recall()

    text_mask = [cls.endswith("_text") for cls in q_labels]
    text_r = _ensemble_recall(text_mask) if any(text_mask) else float("nan")

    small_mask = [s < SMALL_THRESHOLD for s in q_min_sides]
    large_mask = [not m for m in small_mask]
    small_r = _ensemble_recall(small_mask)
    large_r = _ensemble_recall(large_mask)

    results = {
        "all_vs_all": ava,
        "qvg": qvg,
        "text_qvg": text_r,
        "small_qvg": small_r,
        "large_qvg": large_r,
    }
    for k, v in results.items():
        print(f"  {k:15s}: {v:.4f}" if not (v != v) else f"  {k:15s}: n/a")
    return results
