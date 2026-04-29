"""Step 9: Build FAISS galleries for all 5 evaluation datasets."""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import pandas as pd
from src.retrieval.gallery import build_gallery

ANN = Path("data/processed/openlogodet3k47/annotations.parquet")
CKPT = "checkpoints/vit_hn.pt"

# Per-dataset parquet files (test splits)
DATASETS = {
    "logodet3k": "data/processed/logodet3k_test.parquet",
    "openlogo": "data/processed/openlogo_test.parquet",
    "flickr47": "data/processed/flickr47_test.parquet",
    "belga": "data/processed/belga_test.parquet",
    "litw": "data/processed/litw_test.parquet",
}


def _ensure_per_ds_parquet(name: str, parquet_path: str) -> None:
    p = Path(parquet_path)
    if p.exists():
        return
    df = pd.read_parquet(ANN)
    df = df[df["source"] == name]
    p.parent.mkdir(exist_ok=True)
    df.to_parquet(p, index=False)
    print(f"  Created {p} ({len(df)} objects)")


if __name__ == "__main__":
    for name, parquet in DATASETS.items():
        print(f"\n=== Building gallery: {name} ===")
        _ensure_per_ds_parquet(name, parquet)
        if not Path(parquet).exists() or pd.read_parquet(parquet).empty:
            print(f"  [SKIP] {parquet} empty")
            continue
        build_gallery(parquet, name, ckpt_path=CKPT)
    print("\nAll galleries built.")
