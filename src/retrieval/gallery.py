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


def check_duplicate(brand_name: str, dataset_name: str = "logodet3k") -> int:
    """
    Kiểm tra brand đã tồn tại trong gallery chưa.
    Trả về số lượng ảnh hiện có (0 = chưa có).
    """
    labels_path = GALLERY_DIR / f"{dataset_name}_labels.json"
    if not labels_path.exists():
        return 0
    with open(labels_path) as f:
        labels = json.load(f)
    return labels.count(brand_name)


def remove_from_gallery(brand_name: str, dataset_name: str = "logodet3k") -> int:
    """
    Xóa toàn bộ vectors của brand khỏi gallery.
    Trả về số vectors đã xóa.
    FAISS IndexFlatIP không hỗ trợ xóa trực tiếp → rebuild index không có brand đó.
    """
    index, labels = load_gallery(dataset_name)

    keep_indices = [i for i, l in enumerate(labels) if l != brand_name]
    n_removed = len(labels) - len(keep_indices)

    if n_removed == 0:
        print(f"Brand '{brand_name}' không có trong gallery.")
        return 0

    # Lấy embeddings của các entry cần giữ
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

    print(f"Đã xóa brand '{brand_name}' ({n_removed} vectors) khỏi gallery '{dataset_name}'")
    print(f"Gallery còn lại: {new_index.ntotal} vectors")
    return n_removed


def add_to_gallery(
    image_paths: list[str | Path],
    brand_name: str,
    dataset_name: str = "logodet3k",
    ckpt_path: str | Path = CKPT,
    embed_dim: int = 128,
    input_size: int = 160,
    crop_box: tuple | None = None,
    on_duplicate: str = "append",  # "append" | "replace" | "skip"
) -> None:
    """
    Thêm brand vào gallery hiện có mà không cần build lại từ đầu.

    Args:
        image_paths:  danh sách ảnh chứa logo của brand
        brand_name:   tên brand (label sẽ dùng khi retrieve)
        dataset_name: gallery cần update
        crop_box:     (x1, y1, x2, y2) nếu muốn crop logo từ ảnh, None = dùng toàn ảnh
        on_duplicate: xử lý khi brand đã tồn tại:
                        "append"  — thêm ảnh mới vào bên cạnh ảnh cũ (mặc định)
                        "replace" — xóa ảnh cũ, thêm ảnh mới
                        "skip"    — bỏ qua, không làm gì
    """
    # ── Kiểm tra trùng ────────────────────────────────────────────────────
    existing_count = check_duplicate(brand_name, dataset_name)
    if existing_count > 0:
        if on_duplicate == "skip":
            print(f"  [SKIP] Brand '{brand_name}' đã có {existing_count} ảnh trong gallery.")
            return
        elif on_duplicate == "replace":
            print(f"  [REPLACE] Brand '{brand_name}' đã có {existing_count} ảnh → xóa và thêm lại.")
            remove_from_gallery(brand_name, dataset_name)
        else:  # append
            print(f"  [APPEND] Brand '{brand_name}' đã có {existing_count} ảnh → thêm ảnh mới vào.")

    # ── Embed ảnh mới ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = build_vit_embedder(embed_dim, input_size, freeze_blocks=0).to(device)
    state = torch.load(ckpt_path, map_location=device)
    embedder.load_state_dict(state["embedder"])
    embedder.eval()

    transform = val_transforms(input_size)
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
            print(f"  [WARN] Bỏ qua {img_path}: {e}")

    if not new_embs:
        print("Không có ảnh hợp lệ, bỏ qua.")
        return

    new_embs = np.concatenate(new_embs).astype("float32")  # (N, D)

    # ── Thêm vào gallery ──────────────────────────────────────────────────
    index, labels = load_gallery(dataset_name)
    index.add(new_embs)
    labels.extend([brand_name] * len(new_embs))

    faiss.write_index(index, str(GALLERY_DIR / f"{dataset_name}.faiss"))
    with open(GALLERY_DIR / f"{dataset_name}_labels.json", "w") as f:
        json.dump(labels, f)

    print(f"Đã thêm brand '{brand_name}' ({len(new_embs)} ảnh) → gallery '{dataset_name}'")
    print(f"Gallery mới: {index.ntotal} vectors total")
