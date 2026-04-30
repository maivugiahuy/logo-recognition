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

from src.data.transforms import val_transforms
from src.models.embedder_vit import build_vit_embedder

GALLERY_DIR = Path("data/galleries")
CKPT = Path("checkpoints/vit_hn.pt")


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
) -> None:
    """
    Embed all reference images and write:
      data/galleries/{dataset_name}.faiss   — IndexFlatIP
      data/galleries/{dataset_name}_labels.json  — [class_name per index]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = build_vit_embedder(embed_dim, input_size, freeze_blocks=0).to(device)
    state = torch.load(ckpt_path, map_location=device)
    embedder.load_state_dict(state["embedder"])
    embedder.eval()

    df = pd.read_parquet(ann_parquet)
    rows = df.to_dict("records")

    ds = CroppedLogoDataset(rows, transform=val_transforms(input_size))
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
