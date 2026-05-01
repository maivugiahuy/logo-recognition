"""
Build LogoDet-3K + OpenLogo combined dataset.
Output: data/processed/openlogodet3k/annotations.parquet
Target (LogoDet-3K only):  ~2210 classes / ~101k images / ~101k objects
Target (combined):         ~2400+ classes after merge + dedup

Dataset paths:
  LogoDet-3K : DATASET_ROOT/LogoDet-3K/{category}/{ClassName}/{id}.jpg + {id}.xml
  OpenLogo   : DATASET_ROOT/openlogo/openlogo/Annotations/*.xml
               DATASET_ROOT/openlogo/openlogo/JPEGImages/*.jpg
"""
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import imagehash
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
# Default: absolute path to the dataset folder on this machine.
# Override with env var LOGO_DATASET_ROOT if datasets are elsewhere.
DATASET_ROOT = Path(os.environ.get(
    "LOGO_DATASET_ROOT",
    "data/raw"
))

OUT = Path("data/processed/openlogodet3k")
ALIASES_PATH = Path("src/data/aliases.yaml")
MIN_SIDE = 10
MIN_INSTANCES = 20


def load_aliases() -> dict[str, str]:
    """Return alias→canonical mapping (many-to-one)."""
    with open(ALIASES_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    mapping: dict[str, str] = {}
    for canonical, aliases in data.items():
        for alias in (aliases or []):
            mapping[_normalize(alias)] = canonical
    return mapping


def _normalize(name: str) -> str:
    """Appendix B step 1: lowercase, collapse separators, strip apostrophes."""
    name = name.lower().strip()
    name = re.sub(r"[-_ ]+", "_", name)
    name = name.replace("'", "")
    return name


def _apply_aliases(name: str, aliases: dict[str, str]) -> str:
    normed = _normalize(name)
    return aliases.get(normed, normed)


# ── Parsers for each source dataset ────────────────────────────────────────

def parse_logodet3k() -> list[dict]:
    """LogoDet-3K: Pascal VOC XML annotations.

    Layout: DATASET_ROOT/LogoDet-3K/{category}/{ClassName}/{id}.xml + {id}.jpg
    Image path = xml stem + .jpg (same folder). Do NOT use <filename> tag because
    some XMLs have mismatched filenames (e.g. "msn1 (38).jpg" vs actual "38.jpg").
    """
    root = DATASET_ROOT / "LogoDet-3K"
    if not root.exists():
        print(f"[WARN] LogoDet-3K not found at {root} — skipping")
        return []
    ann_files = list(root.rglob("*.xml"))
    records = []
    for xml_path in tqdm(ann_files, desc="LogoDet-3K"):
        # Derive image path from XML stem — more reliable than <filename> tag
        img_path = xml_path.with_suffix(".jpg")
        if not img_path.exists():
            img_path = xml_path.with_suffix(".png")
        if not img_path.exists():
            continue  # skip if no matching image
        try:
            tree = ET.parse(xml_path)
            xml_root = tree.getroot()
        except ET.ParseError:
            continue
        for obj in xml_root.findall("object"):
            name = obj.findtext("name", "unknown")
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            records.append({
                "image_path": str(img_path),
                "class_name": name,
                "x1": float(bnd.findtext("xmin", "0")),
                "y1": float(bnd.findtext("ymin", "0")),
                "x2": float(bnd.findtext("xmax", "0")),
                "y2": float(bnd.findtext("ymax", "0")),
                "source": "logodet3k",
            })
    return records


def parse_openlogo() -> list[dict]:
    """OpenLogo: Pascal VOC XML annotations.

    Layout:
      DATASET_ROOT/openlogo/openlogo/Annotations/{id}.xml
      DATASET_ROOT/openlogo/openlogo/JPEGImages/{id}.jpg

    Class name = <object><name> (already lowercase, e.g. "burgerking", "adidas").
    """
    root = DATASET_ROOT / "openlogo"
    ann_dir = root / "Annotations"
    img_dir = root / "JPEGImages"
    if not ann_dir.exists():
        print(f"[WARN] OpenLogo not found at {ann_dir} — skipping")
        return []
    ann_files = list(ann_dir.glob("*.xml"))
    records = []
    for xml_path in tqdm(ann_files, desc="OpenLogo"):
        img_path = img_dir / (xml_path.stem + ".jpg")
        if not img_path.exists():
            img_path = img_dir / (xml_path.stem + ".png")
        if not img_path.exists():
            continue
        try:
            tree = ET.parse(xml_path)
            xml_root = tree.getroot()
        except ET.ParseError:
            continue
        for obj in xml_root.findall("object"):
            name = obj.findtext("name", "unknown")
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            records.append({
                "image_path": str(img_path),
                "class_name": name,
                "x1": float(bnd.findtext("xmin", "0")),
                "y1": float(bnd.findtext("ymin", "0")),
                "x2": float(bnd.findtext("xmax", "0")),
                "y2": float(bnd.findtext("ymax", "0")),
                "source": "openlogo",
            })
    return records


# ── Main pipeline ───────────────────────────────────────────────────────────

def filter_min_side(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bboxes with min(width, height) < MIN_SIDE."""
    df = df.copy()
    df["w"] = df["x2"] - df["x1"]
    df["h"] = df["y2"] - df["y1"]
    df["min_side"] = df[["w", "h"]].min(axis=1)
    before = len(df)
    df = df[df["min_side"] >= MIN_SIDE]
    print(f"  min_side filter: {before} → {len(df)} objects")
    return df


def dedupe_images(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate images within same class via perceptual hash.

    BUG FIX (performance): phiên bản cũ gọi imagehash.phash() cho mỗi row trong
    200k+ rows, tức mở cùng 1 ảnh nhiều lần nếu có nhiều bbox. Phiên bản mới chỉ
    hash mỗi (class, image_path) unique một lần rồi broadcast kết quả về toàn bộ df.
    """
    before = len(df)

    # Bước 1: lấy danh sách (class_name, image_path) unique cần hash
    unique_pairs = df[["class_name", "image_path"]].drop_duplicates()

    seen: dict[str, set] = {}
    keep_image: set[tuple] = set()  # (class_name, image_path) được giữ lại

    for _, row in tqdm(unique_pairs.iterrows(), total=len(unique_pairs), desc="Deduping unique images"):
        cls = row["class_name"]
        img = row["image_path"]
        if cls not in seen:
            seen[cls] = set()
        try:
            phash = str(imagehash.phash(Image.open(img).convert("RGB")))
        except Exception:
            phash = img  # fallback: dùng path làm hash
        if phash not in seen[cls]:
            seen[cls].add(phash)
            keep_image.add((cls, img))

    # Bước 2: broadcast kết quả về toàn bộ df
    df["_key"] = list(zip(df["class_name"], df["image_path"]))
    df = df[df["_key"].isin(keep_image)].drop(columns=["_key"])
    print(f"  dedupe: {before} → {len(df)} objects")
    return df


def filter_min_instances(df: pd.DataFrame) -> pd.DataFrame:
    """Drop classes with fewer than MIN_INSTANCES objects."""
    counts = df["class_name"].value_counts()
    keep = counts[counts >= MIN_INSTANCES].index
    before = df["class_name"].nunique()
    df = df[df["class_name"].isin(keep)]
    print(f"  min_instances filter: {before} → {df['class_name'].nunique()} classes")
    return df


def build() -> pd.DataFrame:
    print("Parsing LogoDet-3K...")
    records_ld3k = parse_logodet3k()
    print(f"  LogoDet-3K: {len(records_ld3k)} objects")

    print("Parsing OpenLogo...")
    records_ol = parse_openlogo()
    print(f"  OpenLogo: {len(records_ol)} objects")

    records = records_ld3k + records_ol
    df = pd.DataFrame(records)
    print(f"\nCombined raw: {df['class_name'].nunique()} classes, {len(df)} objects")

    # Normalize class names + apply aliases
    aliases = load_aliases()
    df["class_name"] = df["class_name"].apply(lambda n: _apply_aliases(n, aliases))

    # Filters
    df = filter_min_side(df)
    df = dedupe_images(df)
    df = filter_min_instances(df)

    print(f"\nFinal: {df['class_name'].nunique()} classes, "
          f"{df['image_path'].nunique()} images, "
          f"{len(df)} objects")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "annotations.parquet", index=False)
    print(f"Saved → {OUT / 'annotations.parquet'}")
    return df


if __name__ == "__main__":
    build()