"""
Xuất danh sách classes ra file TXT.

Mỗi dòng: class_name | n_objects | n_images
Đọc từ data/processed/openlogodet3k/annotations.parquet (sau Step 3).

Usage:
    python scripts/list_classes.py                  # → classes.txt
    python scripts/list_classes.py --out my.txt     # output tùy chỉnh
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

ANN = Path("data/processed/openlogodet3k/annotations.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="classes.txt",
                        help="Output file path (default: classes.txt)")
    args = parser.parse_args()

    import pandas as pd
    if not ANN.exists():
        print(f"[ERROR] {ANN} không tồn tại. Chạy Step 3 trước.")
        sys.exit(1)

    df = pd.read_parquet(ANN)
    stats = df.groupby("class_name").agg(
        n_objects=("image_path", "count"),
        n_images=("image_path", "nunique"),
    ).reset_index().sort_values("class_name")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"{row.class_name}\t{row.n_objects}\t{row.n_images}"
             for row in stats.itertuples()]
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"{len(stats)} classes | {stats['n_objects'].sum()} objects | {stats['n_images'].sum()} images")
    print(f"Saved → {out_path}")
