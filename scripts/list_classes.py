"""
Xuất danh sách classes ra file.

Sources:
  --source dataset  : toàn bộ classes trong annotations.parquet (sau Step 3)
  --source gallery  : classes đang có trong FAISS gallery (sau Step 9)
  --source splits   : classes theo từng split (open/closed train/val/test)

Usage:
    python scripts/list_classes.py                          # dataset → classes.txt
    python scripts/list_classes.py --source gallery         # gallery → classes.txt
    python scripts/list_classes.py --source splits          # tất cả splits
    python scripts/list_classes.py --out my_classes.csv     # output tùy chỉnh
    python scripts/list_classes.py --format csv             # CSV với stats
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")

ANN = Path("data/processed/logodet3k/annotations.parquet")
SPLITS_DIR = Path("data/processed/logodet3k/splits")
GALLERY_DIR = Path("data/galleries")


def from_dataset(out_path: Path, fmt: str) -> None:
    import pandas as pd
    if not ANN.exists():
        print(f"[ERROR] {ANN} không tồn tại. Chạy Step 3 trước.")
        sys.exit(1)

    df = pd.read_parquet(ANN)
    stats = df.groupby("class_name").agg(
        n_images=("image_path", "nunique"),
        n_objects=("image_path", "count"),
    ).reset_index().sort_values("class_name")

    print(f"Dataset: {len(stats)} classes | "
          f"{stats['n_images'].sum()} images | "
          f"{stats['n_objects'].sum()} objects")

    _write(stats, out_path, fmt, source="dataset")


def from_gallery(gallery_name: str, out_path: Path, fmt: str) -> None:
    labels_path = GALLERY_DIR / f"{gallery_name}_labels.json"
    if not labels_path.exists():
        print(f"[ERROR] Gallery '{gallery_name}' chưa build. Chạy Step 9 trước.")
        sys.exit(1)

    with open(labels_path) as f:
        labels = json.load(f)

    counts = Counter(labels)
    import pandas as pd
    stats = pd.DataFrame(
        sorted(counts.items()),
        columns=["class_name", "n_images"]
    )
    stats["n_objects"] = stats["n_images"]  # gallery = 1 vector/image

    print(f"Gallery '{gallery_name}': {len(stats)} classes | {len(labels)} vectors")
    _write(stats, out_path, fmt, source=f"gallery:{gallery_name}")


def from_splits(out_dir: Path) -> None:
    split_files = {
        "open_train": SPLITS_DIR / "open_train.json",
        "open_val": SPLITS_DIR / "open_val.json",
        "open_test": SPLITS_DIR / "open_test.json",
        "closed_train": SPLITS_DIR / "closed_train.json",
        "closed_val": SPLITS_DIR / "closed_val.json",
        "closed_test": SPLITS_DIR / "closed_test.json",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for split_name, split_path in split_files.items():
        if not split_path.exists():
            print(f"  [SKIP] {split_name}: file không tồn tại")
            continue

        with open(split_path) as f:
            data = json.load(f)

        if isinstance(data, list):
            # open_set: list of class names
            classes = sorted(data)
            out_file = out_dir / f"{split_name}.txt"
            out_file.write_text("\n".join(classes), encoding="utf-8")
            print(f"  {split_name}: {len(classes)} classes → {out_file}")
            summary.append((split_name, len(classes), "-"))
        else:
            # closed_set: {class_name: [image_paths]}
            classes = sorted(data.keys())
            n_images = sum(len(v) for v in data.values())
            out_file = out_dir / f"{split_name}.txt"
            lines = [f"{cls}\t{len(data[cls])} images" for cls in classes]
            out_file.write_text("\n".join(lines), encoding="utf-8")
            print(f"  {split_name}: {len(classes)} classes, {n_images} images → {out_file}")
            summary.append((split_name, len(classes), n_images))

    # Summary file
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"{'Split':<20} {'Classes':>8} {'Images':>10}\n")
        f.write("-" * 42 + "\n")
        for name, n_cls, n_img in summary:
            f.write(f"{name:<20} {n_cls:>8} {str(n_img):>10}\n")
    print(f"\n  Summary → {summary_path}")


def _write(stats, out_path: Path, fmt: str, source: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        stats.to_csv(out_path, index=False, encoding="utf-8")
    else:
        # txt: một class per line
        out_path.write_text(
            "\n".join(stats["class_name"].tolist()),
            encoding="utf-8"
        )

    print(f"Saved → {out_path}  ({fmt.upper()}, source: {source})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="dataset",
                        choices=["dataset", "gallery", "splits"],
                        help="Nguồn dữ liệu (default: dataset)")
    parser.add_argument("--gallery", default="logodet3k",
                        help="Tên gallery (chỉ dùng khi --source gallery)")
    parser.add_argument("--out", default=None,
                        help="Output file path (default: classes.txt hoặc classes.csv)")
    parser.add_argument("--format", default="txt", choices=["txt", "csv"],
                        help="Output format: txt (1 class/line) hoặc csv (với stats)")
    args = parser.parse_args()

    if args.source == "splits":
        out_dir = Path(args.out) if args.out else Path("data/processed/logodet3k/splits_export")
        from_splits(out_dir)

    elif args.source == "gallery":
        default_name = f"gallery_{args.gallery}_classes.{args.format}"
        out_path = Path(args.out) if args.out else Path(default_name)
        from_gallery(args.gallery, out_path, args.format)

    else:  # dataset
        default_name = f"classes.{args.format}"
        out_path = Path(args.out) if args.out else Path(default_name)
        from_dataset(out_path, args.format)
