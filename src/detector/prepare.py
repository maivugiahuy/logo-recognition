"""
Convert LogoDet3K annotations to class-agnostic YOLO format.
All 3000 logo classes → single class 0 ("logo").
Output structure:
  data/processed/detector_yolo/
    images/{train,val,test}/
    labels/{train,val,test}/
    dataset.yaml
"""
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

RAW_LOGODET = Path("data/raw/logodet3k")
OUT = Path("data/processed/detector_yolo")
ANN = Path("data/processed/logodet3k/annotations.parquet")
SPLITS = Path("data/processed/logodet3k/splits")


def _to_yolo_bbox(x1, y1, x2, y2, img_w, img_h) -> tuple[float, float, float, float]:
    """Convert x1y1x2y2 → YOLO xc yc w h (normalized 0–1)."""
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h


def prepare_yolo_dataset() -> None:
    df = pd.read_parquet(ANN)
    # Use only LogoDet3K source for detector training
    df = df[df["source"] == "logodet3k"].copy()

    # Load image-level splits (any class in train_classes goes to train split)
    with open(SPLITS / "open_train.json") as f:
        train_cls = set(json.load(f))
    with open(SPLITS / "open_val.json") as f:
        val_cls = set(json.load(f))

    def get_split(cls):
        if cls in train_cls:
            return "train"
        elif cls in val_cls:
            return "val"
        return "test"

    df["yolo_split"] = df["class_name"].apply(get_split)

    for split in ["train", "val", "test"]:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    processed_imgs: set[str] = set()

    for img_path, group in tqdm(df.groupby("image_path"), desc="Preparing YOLO"):
        img_path = Path(img_path)
        split = group["yolo_split"].iloc[0]

        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception:
            continue

        # Tên file unique: hash 8 ký tự của full path + tên gốc
        # Tránh collision khi nhiều class dùng tên file giống nhau (1.jpg, 2.jpg...)
        uid = hashlib.md5(str(img_path).encode()).hexdigest()[:8]
        unique_stem = f"{uid}_{img_path.stem}"
        dest_img = OUT / "images" / split / f"{unique_stem}{img_path.suffix}"
        if not dest_img.exists():
            shutil.copy2(img_path, dest_img)

        # Write label file (class-agnostic: class 0)
        label_path = OUT / "labels" / split / f"{unique_stem}.txt"
        if str(img_path) not in processed_imgs:
            with open(label_path, "w") as f:
                for _, row in group.iterrows():
                    xc, yc, w, h = _to_yolo_bbox(
                        row["x1"], row["y1"], row["x2"], row["y2"], img_w, img_h
                    )
                    # clamp
                    xc = max(0.0, min(1.0, xc))
                    yc = max(0.0, min(1.0, yc))
                    w = max(0.001, min(1.0, w))
                    h = max(0.001, min(1.0, h))
                    f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            processed_imgs.add(str(img_path))

    # Write dataset.yaml
    dataset_cfg = {
        "path": str(OUT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["logo"],
    }
    with open(OUT / "dataset.yaml", "w") as f:
        yaml.dump(dataset_cfg, f)

    print(f"YOLO dataset → {OUT}")
    print(f"  train: {len(list((OUT / 'images/train').iterdir()))} images")
    print(f"  val:   {len(list((OUT / 'images/val').iterdir()))} images")
    print(f"  test:  {len(list((OUT / 'images/test').iterdir()))} images")


if __name__ == "__main__":
    prepare_yolo_dataset()
