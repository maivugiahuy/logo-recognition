"""
Export class list to a TXT file.

Each line: class_name | n_objects | n_images
Reads from data/processed/openlogodet3k/annotations.parquet (after Step 3).

Usage:
    python scripts/list_classes.py                  # → results/classes.txt
    python scripts/list_classes.py --out my.txt     # custom output path
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.utils.logging_utils import setup_logging

ANN = Path("data/processed/openlogodet3k/annotations.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/classes.txt",
                        help="Output file path (default: results/classes.txt)")
    args = parser.parse_args()

    setup_logging(__file__)
    import pandas as pd
    if not ANN.exists():
        print(f"[ERROR] {ANN} not found. Run Step 3 first.")
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
