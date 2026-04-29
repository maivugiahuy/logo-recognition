"""Step 3: Build OpenLogoDet3K47 + splits."""
import sys
sys.path.insert(0, ".")
from src.data.build_olg3k47 import build
from src.data.splits import build as build_splits

if __name__ == "__main__":
    print("=== Step 1: Build OpenLogoDet3K47 ===")
    df = build()
    print("\n=== Step 2: Build splits ===")
    build_splits(df)
    print("\nDone. Verify counts ≈ 2714 classes / 181,552 images / 227,176 objects.")
