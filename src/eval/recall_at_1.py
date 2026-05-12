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
from PIL import Image as PILImage
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LogoDataset
from src.data.transforms import val_transforms, val_transforms_dinov2
from src.models.embedder_dinov2 import build_dinov2_embedder
from src.models.embedder_vit import build_vit_embedder
from src.models.embedder_vit_s import build_vit_s_embedder

_DINOV2_BACKBONES = {"dinov2_vitb14", "dinov2"}
_VIT_S_BACKBONES = {"vit_s16"}

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


def _ocr_dataset(ds: LogoDataset, gpu: bool = True) -> list[str]:
    """Run EasyOCR on every crop in ds (same order as _embed_dataset). Returns list of strings."""
    from src.retrieval.ocr import run_ocr
    texts = []
    for _, row in tqdm(ds.df.iterrows(), total=len(ds.df), desc="OCR crops"):
        try:
            img = PILImage.open(row["image_path"]).convert("RGB")
            w, h = img.size
            crop = img.crop((
                max(0, int(row["x1"])), max(0, int(row["y1"])),
                min(w, int(row["x2"])), min(h, int(row["y2"])),
            ))
            text = run_ocr(crop, gpu=gpu)
        except Exception:
            text = ""
        texts.append(text)
    return texts


def compute_recall_at_1(
    q_embs: np.ndarray,
    q_labels: list[str],
    g_embs: np.ndarray,
    g_labels: list[str],
    all_vs_all: bool = False,
    ocr_enabled: bool = False,
    q_ocr_texts: list[str] | None = None,
    ocr_weight: float = 0.3,
    ocr_rerank_k: int = 10,
    _label: str = "",
) -> float:
    """
    Compute recall@1.
    If all_vs_all=True, q==g and we take 2nd-nearest neighbor.
    If ocr_enabled=True (and not all_vs_all), rerank top-k with OCR text fusion.
    """
    index = faiss.IndexFlatIP(q_embs.shape[1])
    index.add(g_embs)
    nq = len(q_labels)

    if ocr_enabled and q_ocr_texts and not all_vs_all:
        from src.retrieval.ocr import text_similarity
        rerank_k = min(ocr_rerank_k, index.ntotal)
        t0 = time.time()
        scores_mat, indices_mat = index.search(q_embs, rerank_k)
        correct = 0
        for qi in range(nq):
            ocr_text = q_ocr_texts[qi]
            best_score, best_idx = -1.0, int(indices_mat[qi, 0])
            for vis_score, g_idx in zip(scores_mat[qi], indices_mat[qi]):
                tsim = text_similarity(ocr_text, g_labels[g_idx])
                fused = (
                    (1 - ocr_weight) * float(vis_score) + ocr_weight * tsim
                    if ocr_text else float(vis_score)
                )
                if fused > best_score:
                    best_score, best_idx = fused, int(g_idx)
            correct += g_labels[best_idx] == q_labels[qi]
        elapsed = time.time() - t0
        tag = f"retrieval{' ' + _label if _label else ''}"
        print(f"  {tag:15s}: {nq} queries  {elapsed:.1f}s  ({1000*elapsed/nq:.2f} ms/query)")
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
    ocr_enabled: bool = False,
    ocr_weight: float = 0.3,
    ocr_rerank_k: int = 10,
) -> dict[str, float]:
    """Full evaluation returning recall@1 for all subsets."""
    is_dinov2 = backbone in _DINOV2_BACKBONES
    is_vit_s = backbone in _VIT_S_BACKBONES
    if input_size is None:
        input_size = 224 if is_dinov2 else 160

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_dinov2:
        embedder = build_dinov2_embedder(embed_dim, input_size, freeze_blocks=0).to(device)
    elif is_vit_s:
        embedder = build_vit_s_embedder(embed_dim, input_size).to(device)
    else:
        embedder = build_vit_embedder(embed_dim, input_size, freeze_blocks=0).to(device)
    state = torch.load(ckpt_path, map_location=device)
    embedder.load_state_dict(state["embedder"])
    embedder.eval()

    # ViT-S uses ImageNet normalization (same as DINOv2)
    transform = val_transforms(input_size) if not (is_dinov2 or is_vit_s) else val_transforms_dinov2(input_size)
    ds = LogoDataset.from_split(
        ann_parquet, split_json, transform=transform, mode=mode
    )
    embs, class_names, min_sides = _embed_dataset(ds, embedder, device)

    # Compute query indices once — shared by OCR, small, large subsets
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

    # OCR: run on all crops, extract query subset
    if ocr_enabled:
        all_ocr_texts = _ocr_dataset(ds, gpu=device.type == "cuda")
        q_ocr_texts = [all_ocr_texts[i] for i in query_idx]
    else:
        q_ocr_texts = None

    ocr_args = dict(ocr_enabled=ocr_enabled, q_ocr_texts=q_ocr_texts,
                    ocr_weight=ocr_weight, ocr_rerank_k=ocr_rerank_k)

    # all-vs-all (OCR skipped — self-exclusion incompatible with reranking)
    ava = compute_recall_at_1(embs, class_names, embs, class_names, all_vs_all=True,
                              _label="all_vs_all")

    # query-vs-gallery
    q_embs, q_labels, g_embs, g_labels = query_vs_gallery(embs, class_names)
    qvg = compute_recall_at_1(q_embs, q_labels, g_embs, g_labels, **ocr_args, _label="qvg")

    # text subset (Q-vs-G)
    text_mask = [cls.endswith("_text") for cls in q_labels]
    if any(text_mask):
        t_idx = [i for i, m in enumerate(text_mask) if m]
        t_embs = q_embs[t_idx]
        t_labels = [q_labels[i] for i in t_idx]
        t_ocr = [q_ocr_texts[i] for i in t_idx] if q_ocr_texts else None
        text_r = compute_recall_at_1(t_embs, t_labels, g_embs, g_labels,
                                     **{**ocr_args, "q_ocr_texts": t_ocr}, _label="text")
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
        so = [q_ocr_texts[i] for i in idx] if q_ocr_texts else None
        return compute_recall_at_1(se, sl, g_embs, g_labels,
                                   **{**ocr_args, "q_ocr_texts": so}, _label=label)

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
    dinov2_ckpt: str | Path,
    ann_parquet: str | Path,
    split_json: str | Path = None,
    mode: str = "open_set",
    embed_dim: int = 128,
    vit_input_size: int = 160,
    dinov2_input_size: int = 168,
    vit_weight: float = 0.5,
    ensemble_top_k: int = 20,
) -> dict[str, float]:
    """Ensemble recall@1: fuse ViT + DINOv2 scores per label, pick best."""
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit_embedder = build_vit_embedder(embed_dim, vit_input_size, freeze_blocks=0).to(device)
    state = torch.load(vit_ckpt, map_location=device)
    vit_embedder.load_state_dict(state["embedder"])
    vit_embedder.eval()

    dino_embedder = build_dinov2_embedder(embed_dim, dinov2_input_size, freeze_blocks=0).to(device)
    state = torch.load(dinov2_ckpt, map_location=device)
    dino_embedder.load_state_dict(state["embedder"])
    dino_embedder.eval()

    vit_ds = LogoDataset.from_split(
        ann_parquet, split_json, transform=val_transforms(vit_input_size), mode=mode
    )
    dino_ds = LogoDataset.from_split(
        ann_parquet, split_json, transform=val_transforms_dinov2(dinov2_input_size), mode=mode
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
        "all_vs_all": float("nan"),  # not supported for ensemble
        "qvg": qvg,
        "text_qvg": text_r,
        "small_qvg": small_r,
        "large_qvg": large_r,
    }
    for k, v in results.items():
        print(f"  {k:15s}: {v:.4f}" if not (v != v) else f"  {k:15s}: n/a")
    return results
