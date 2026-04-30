"""
Build OpenLogoDet3K47 composite dataset — paper Appendix B.
Merges LogoDet3K + QMUL-OpenLogo + FlickrLogos-47.
Output: data/processed/openlogodet3k47/annotations.parquet
Target: ~2714 classes / 181,552 images / 227,176 objects

Dataset paths (actual locations on disk):
  LogoDet-3K  : DATASET_ROOT/LogoDet-3K/{category}/{ClassName}/{id}.jpg + {id}.xml
  OpenLogo    : DATASET_ROOT/openlogo/Annotations/{id}.xml + JPEGImages/{id}.jpg
  FlickrLogos : DATASET_ROOT/FlickrLogos_47/train|test/{classID:06d}/{imageID}.png
                  + {imageID}.gt_data.txt  (format: x1 y1 x2 y2 class_id ...)
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

OUT = Path("data/processed/openlogodet3k47")
ALIASES_PATH = Path("src/data/aliases.yaml")
MIN_SIDE = 10
MIN_INSTANCES = 20


def load_aliases() -> dict[str, str]:
    """Return alias→canonical mapping (many-to-one)."""
    with open(ALIASES_PATH) as f:
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


def _parse_voc_xml(xml_path: Path, source: str, img_dir: Path | None = None) -> list[dict]:
    """Parse Pascal VOC format XML annotation.

    Args:
        xml_path: path to the .xml annotation file
        source:   dataset tag string (e.g. "logodet3k", "openlogo")
        img_dir:  explicit directory that contains the image file.
                  If None, the image is assumed to be in the same folder as
                  the XML (LogoDet-3K layout).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    fname = root.findtext("filename", "")
    if img_dir is not None:
        img_path = img_dir / fname
    else:
        img_path = xml_path.parent / fname
    records = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "unknown")
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        x1 = float(bnd.findtext("xmin", "0"))
        y1 = float(bnd.findtext("ymin", "0"))
        x2 = float(bnd.findtext("xmax", "0"))
        y2 = float(bnd.findtext("ymax", "0"))
        records.append({
            "image_path": str(img_path),
            "class_name": name,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "source": source,
        })
    return records


def parse_openlogo() -> list[dict]:
    """QMUL-OpenLogo: VOC-style XML annotations.

    Layout:
      DATASET_ROOT/openlogo/Annotations/{id}.xml
      DATASET_ROOT/openlogo/JPEGImages/{id}.jpg
    """
    root = DATASET_ROOT / "openlogo"
    if not root.exists():
        print(f"[WARN] OpenLogo not found at {root} — skipping")
        return []
    ann_dir = root / "Annotations"
    img_dir = root / "JPEGImages"
    records = []
    for xml_path in tqdm(list(ann_dir.glob("*.xml")), desc="OpenLogo"):
        records.extend(_parse_voc_xml(xml_path, "openlogo", img_dir=img_dir))
    return records


def parse_flickr47() -> list[dict]:
    """FlickrLogos-47: numeric class folders + .gt_data.txt annotations.

    Layout:
      DATASET_ROOT/FlickrLogos_47/train/{classID:06d}/{imageID}.png
      DATASET_ROOT/FlickrLogos_47/train/{classID:06d}/{imageID}.gt_data.txt
      DATASET_ROOT/FlickrLogos_47/test/  (same structure)
      DATASET_ROOT/FlickrLogos_47/className2ClassID.txt

    gt_data.txt line format (per README):
      <x1> <y1> <x2> <y2> <class_id> <dummy> <mask> <difficult> <truncated>
      x1,y1 = upper-left corner; x2,y2 = lower-right corner (already absolute coords).
    """
    root = DATASET_ROOT / "FlickrLogos_47"
    if not root.exists():
        print(f"[WARN] FlickrLogos-47 not found at {root} — skipping")
        return []

    # Load class-ID → class-name mapping
    mapping_file = root / "className2ClassID.txt"
    id_to_name: dict[int, str] = {}
    with open(mapping_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                name, cid = parts[0].strip(), int(parts[1].strip())
                id_to_name[cid] = name

    records = []
    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        # Collect numeric class folders (e.g. "000000", "000001", ...)
        class_dirs = [d for d in split_dir.iterdir()
                      if d.is_dir() and d.name.isdigit()]
        for class_dir in tqdm(class_dirs, desc=f"FlickrLogos-47 {split}"):
            class_id = int(class_dir.name)
            class_name = id_to_name.get(class_id, class_dir.name)
            for ann_file in class_dir.glob("*.gt_data.txt"):
                # Image has same stem as the .gt_data.txt file
                img_path = ann_file.parent / (ann_file.name.replace(".gt_data.txt", ".png"))
                if not img_path.exists():
                    continue
                with open(ann_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        x1, y1, x2, y2 = map(float, parts[:4])
                        records.append({
                            "image_path": str(img_path),
                            "class_name": class_name,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "source": "flickr47",
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

    # Bước 2: broadcast về toàn bộ df (mỗi bbox thuộc ảnh được giữ → giữ hết bbox của ảnh đó)
    keep_mask = df.apply(
        lambda r: (r["class_name"], r["image_path"]) in keep_image, axis=1
    )
    df = df[keep_mask]
    print(f"  dedupe: {before} → {len(df)} objects")
    return df


def build() -> pd.DataFrame:
    aliases = load_aliases()

    print("Parsing sources...")
    records = parse_logodet3k() + parse_openlogo() + parse_flickr47()
    df = pd.DataFrame(records)
    print(f"  raw objects: {len(df)}")

    # Normalize + merge class names (Appendix B steps 1–2)
    df["class_name"] = df["class_name"].apply(lambda n: _apply_aliases(n, aliases))

    # Filter bbox size
    df = filter_min_side(df)

    # Dedupe within class
    df = dedupe_images(df)

    # Drop classes with fewer than MIN_INSTANCES objects
    # BUG FIX: paper Appendix B nói "fewer than 20 instances" = bounding box objects,
    # không phải unique images. Dùng .size() thay vì .nunique() để đếm đúng số objects.
    class_counts = df.groupby("class_name").size()
    valid_classes = class_counts[class_counts >= MIN_INSTANCES].index
    before = df["class_name"].nunique()
    df = df[df["class_name"].isin(valid_classes)]
    print(f"  class filter (≥{MIN_INSTANCES} objects): {before} → {df['class_name'].nunique()} classes")

    print(f"\nFinal: {df['class_name'].nunique()} classes | "
          f"{df['image_path'].nunique()} images | {len(df)} objects")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "annotations.parquet", index=False)
    print(f"Saved → {OUT / 'annotations.parquet'}")
    return df


if __name__ == "__main__":
    build()
