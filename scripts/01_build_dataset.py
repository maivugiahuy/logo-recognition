"""Step 3: Build LogoDet-3K + OpenLogo combined dataset + splits."""
import sys
sys.path.insert(0, ".")
from src.data.build_olg3k import build
from src.data.splits import build as build_splits

if __name__ == "__main__":
    print("=== Step 1: Build LogoDet-3K + OpenLogo ===")
    df = build()
    print("\n=== Step 2: Build splits ===")
    build_splits(df)
    print("\nDone.")
