"""Step 8: Build FAISS gallery for LogoDet-3K evaluation dataset."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, ".")
import pandas as pd
from src.retrieval.gallery import build_gallery
from src.utils.logging_utils import setup_logging

ANN = Path("data/processed/openlogodet3k/annotations.parquet")

DATASETS = {
    "openlogodet3k": "data/processed/openlogodet3k_test.parquet",
}

_DINOV2_DEFAULT_CKPT = "checkpoints/dinov2_hn.pt"
_VIT_DEFAULT_CKPT = "checkpoints/vit_hn.pt"
_DINOV2_INPUT_SIZE = 168
_VIT_INPUT_SIZE = 160


def _ensure_per_ds_parquet(name: str, parquet_path: str) -> None:
    p = Path(parquet_path)
    if p.exists():
        return
    df = pd.read_parquet(ANN)
    p.parent.mkdir(exist_ok=True)
    df.to_parquet(p, index=False)
    print(f"  Created {p} ({len(df)} objects)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="vit_b32_openai",
                        choices=["vit_b32_openai", "dinov2_vitb14"],
                        help="Backbone used to embed the gallery (default: vit_b32_openai)")
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint path (default: vit_hn.pt or dinov2_hn.pt based on backbone)")
    parser.add_argument("--suffix", default=None,
                        help="Gallery name suffix, e.g. '_dinov2' → 'openlogodet3k_dinov2' (auto-set for dinov2)")
    args = parser.parse_args()

    setup_logging(__file__)

    is_dinov2 = args.backbone == "dinov2_vitb14"
    ckpt = args.ckpt or (_DINOV2_DEFAULT_CKPT if is_dinov2 else _VIT_DEFAULT_CKPT)
    input_size = _DINOV2_INPUT_SIZE if is_dinov2 else _VIT_INPUT_SIZE
    suffix = args.suffix if args.suffix is not None else ("_dinov2" if is_dinov2 else "")

    for base_name, parquet in DATASETS.items():
        gallery_name = base_name + suffix
        print(f"\n=== Building gallery: {gallery_name} (backbone={args.backbone}) ===")
        _ensure_per_ds_parquet(base_name, parquet)
        if not Path(parquet).exists() or pd.read_parquet(parquet).empty:
            print(f"  [SKIP] {parquet} empty")
            continue
        build_gallery(parquet, gallery_name, ckpt_path=ckpt,
                      input_size=input_size, backbone=args.backbone)

    print("\nGallery built.")
