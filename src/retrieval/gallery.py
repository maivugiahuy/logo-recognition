"""
Build FAISS IndexFlatIP gallery for a dataset.
Embeds all reference images with vit_hn.pt embedder → 128-d L2-normalized vectors.
"""
import json
from pathlib import Path

try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss not found. Install via:\n"
        "  conda: conda install -c pytorch -c nvidia faiss-gpu=1.9.0\n"
        "  pip CUDA12: pip install faiss-gpu-cu12\n"
        "  CPU only:   pip install faiss-cpu"
    )
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.transforms import val_transforms, val_transforms_dinov2
from src.models.embedder_dinov2 import build_dinov2_embedder
from src.models.embedder_vit import build_vit_embedder
from src.models.embedder_vit_s import build_vit_s_embedder

GALLERY_DIR = Path("data/galleries")
CKPT = Path("checkpoints/vit_hn.pt")
_DINOV2_BACKBONES = {"dinov2_vitb14", "dinov2"}
_VIT_S_BACKBONES = {"vit_s16"}


def _load_embedder(backbone: str, embed_dim: int, input_size: int,
                   ckpt_path: str | Path, device: torch.device):
    if backbone in _DINOV2_BACKBONES:
        embedder = build_dinov2_embedder(embed_dim, input_size, freeze_blocks=0).to(device)
        transform = val_transforms_dinov2(input_size)
    elif backbone in _VIT_S_BACKBONES:
        embedder = build_vit_s_embedder(embed_dim, input_size).to(device)
        transform = val_transforms_dinov2(input_size)  # ImageNet norm
    else:
        embedder = build_vit_embedder(embed_dim, input_size, freeze_blocks=0).to(device)
        transform = val_transforms(input_size)
    state = torch.load(ckpt_path, map_location=device)
    embedder.load_state_dict(state["embedder"])
    embedder.eval()
    return embedder, transform


class CroppedLogoDataset(Dataset):
    """Wraps a list of (image_path, x1, y1, x2, y2, class_name) rows."""

    def __init__(self, rows: list[dict], transform=None):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        w, h = img.size
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            img = img.crop((x1, y1, x2, y2))
        if self.transform:
            img = self.transform(img)
        return img, row["class_name"]


def build_gallery(
    ann_parquet: str | Path,
    dataset_name: str,
    ckpt_path: str | Path = CKPT,
    embed_dim: int = 128,
    input_size: int = 160,
    batch_size: int = 256,
    backbone: str = "vit_b32_openai",
) -> None:
    """
    Embed all reference images and write:
      data/galleries/{dataset_name}.faiss   — IndexFlatIP
      data/galleries/{dataset_name}_labels.json  — [class_name per index]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder, transform = _load_embedder(backbone, embed_dim, input_size, ckpt_path, device)

    df = pd.read_parquet(ann_parquet)
    rows = df.to_dict("records")

    ds = CroppedLogoDataset(rows, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    all_embs, all_labels = [], []
    with torch.no_grad():
        for imgs, class_names in tqdm(loader, desc=f"Gallery {dataset_name}"):
            imgs = imgs.to(device)
            embs = embedder(imgs).cpu().numpy()
            all_embs.append(embs)
            all_labels.extend(class_names)

    embs = np.concatenate(all_embs).astype("float32")  # (N, D)

    # Build IndexFlatIP (inner product = cosine sim for L2-normalized vecs)
    index = faiss.IndexFlatIP(embed_dim)
    index.add(embs)

    GALLERY_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(GALLERY_DIR / f"{dataset_name}.faiss"))
    with open(GALLERY_DIR / f"{dataset_name}_labels.json", "w") as f:
        json.dump(all_labels, f)

    print(f"Gallery {dataset_name}: {index.ntotal} vectors → {GALLERY_DIR}")


def load_gallery(dataset_name: str) -> tuple[faiss.Index, list[str]]:
    index = faiss.read_index(str(GALLERY_DIR / f"{dataset_name}.faiss"))
    with open(GALLERY_DIR / f"{dataset_name}_labels.json") as f:
        labels = json.load(f)
    return index, labels


def check_duplicate(brand_name: str, dataset_name: str = "openlogodet3k") -> int:
    """
    Check if a brand already exists in the gallery.
    Returns the number of existing images (0 = not present).
    """
    labels_path = GALLERY_DIR / f"{dataset_name}_labels.json"
    if not labels_path.exists():
        return 0
    with open(labels_path) as f:
        labels = json.load(f)
    return labels.count(brand_name)


def remove_from_gallery(brand_name: str, dataset_name: str = "openlogodet3k") -> int:
    """
    Remove all vectors for a brand from the gallery.
    Returns the number of vectors removed.
    FAISS IndexFlatIP does not support direct deletion → rebuilds the index without that brand.
    """
    index, labels = load_gallery(dataset_name)

    keep_indices = [i for i, l in enumerate(labels) if l != brand_name]
    n_removed = len(labels) - len(keep_indices)

    if n_removed == 0:
        print(f"Brand '{brand_name}' not found in gallery.")
        return 0

    # Extract embeddings for entries to keep
    all_embs = np.zeros((index.ntotal, index.d), dtype="float32")
    index.reconstruct_n(0, index.ntotal, all_embs)
    kept_embs = all_embs[keep_indices]

    # Rebuild index
    new_index = faiss.IndexFlatIP(index.d)
    if len(kept_embs) > 0:
        new_index.add(kept_embs)
    new_labels = [labels[i] for i in keep_indices]

    faiss.write_index(new_index, str(GALLERY_DIR / f"{dataset_name}.faiss"))
    with open(GALLERY_DIR / f"{dataset_name}_labels.json", "w") as f:
        json.dump(new_labels, f)

    print(f"Removed brand '{brand_name}' ({n_removed} vectors) from gallery '{dataset_name}'")
    print(f"Gallery remaining: {new_index.ntotal} vectors")
    return n_removed


def add_to_gallery(
    image_paths: list[str | Path],
    brand_name: str,
    dataset_name: str = "openlogodet3k",
    ckpt_path: str | Path = CKPT,
    embed_dim: int = 128,
    input_size: int = 160,
    crop_box: tuple | None = None,
    on_duplicate: str = "append",  # "append" | "replace" | "skip"
    backbone: str = "vit_b32_openai",
) -> None:
    """
    Add a brand to an existing gallery without rebuilding from scratch.

    Args:
        image_paths:  list of images containing the brand logo
        brand_name:   brand name (label used during retrieval)
        dataset_name: gallery to update
        crop_box:     (x1, y1, x2, y2) to crop logo from image, None = use full image
        on_duplicate: how to handle an existing brand:
                        "append"  — add new images alongside existing ones (default)
                        "replace" — remove existing images, add new ones
                        "skip"    — do nothing
    """
    # ── Duplicate check ───────────────────────────────────────────────────
    existing_count = check_duplicate(brand_name, dataset_name)
    if existing_count > 0:
        if on_duplicate == "skip":
            print(f"  [SKIP] Brand '{brand_name}' already has {existing_count} images in gallery.")
            return
        elif on_duplicate == "replace":
            print(f"  [REPLACE] Brand '{brand_name}' has {existing_count} existing images → removing and re-adding.")
            remove_from_gallery(brand_name, dataset_name)
        else:  # append
            print(f"  [APPEND] Brand '{brand_name}' has {existing_count} existing images → appending new ones.")

    # ── Embed new images ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder, transform = _load_embedder(backbone, embed_dim, input_size, ckpt_path, device)
    new_embs = []

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            if crop_box is not None:
                x1, y1, x2, y2 = crop_box
                img = img.crop((max(0, x1), max(0, y1),
                                min(img.width, x2), min(img.height, y2)))
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = embedder(tensor).cpu().numpy()  # (1, D)
            new_embs.append(emb)
        except Exception as e:
            print(f"  [WARN] Skipping {img_path}: {e}")

    if not new_embs:
        print("No valid images found, skipping.")
        return

    new_embs = np.concatenate(new_embs).astype("float32")  # (N, D)

    # ── Add to gallery ────────────────────────────────────────────────────
    faiss_path = GALLERY_DIR / f"{dataset_name}.faiss"
    labels_path = GALLERY_DIR / f"{dataset_name}_labels.json"
    if faiss_path.exists() and labels_path.exists():
        index, labels = load_gallery(dataset_name)
    else:
        GALLERY_DIR.mkdir(exist_ok=True)
        index = faiss.IndexFlatIP(embed_dim)
        labels = []
        print(f"  Gallery '{dataset_name}' does not exist → creating new.")
    index.add(new_embs)
    labels.extend([brand_name] * len(new_embs))

    faiss.write_index(index, str(GALLERY_DIR / f"{dataset_name}.faiss"))
    with open(GALLERY_DIR / f"{dataset_name}_labels.json", "w") as f:
        json.dump(labels, f)

    print(f"Added brand '{brand_name}' ({len(new_embs)} images) → gallery '{dataset_name}'")
    print(f"Gallery updated: {index.ntotal} vectors total")
