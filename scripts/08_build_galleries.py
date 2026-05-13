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

# (checkpoint, backbone, gallery_suffix)
_MODEL_PRESETS = {
    "b16_hn":   ("checkpoints/vit_b16_arcface_hn.pt",  "vit_b16_openai", ""),
    "b16_base": ("checkpoints/vit_b16_arcface_base.pt", "vit_b16_openai", "_b16_base"),
    "dinov3":   ("checkpoints/dinov3_arcface_base.pt",  "dinov3_vitb16",  "_dinov3"),
    "b32_hn":   ("checkpoints/vit_hn.pt",               "vit_b32_openai", "_b32"),
    "b32_base": ("checkpoints/vit_base.pt",             "vit_b32_openai", "_b32_base"),
}


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
    parser.add_argument("--model", default=None, choices=list(_MODEL_PRESETS),
                        help="Model preset — overrides --ckpt/--backbone/--suffix. "
                             "Choices: b16_hn (default best), b16_base, dinov3, b32_hn, b32_base")
    parser.add_argument("--backbone", default="vit_b16_openai",
                        choices=["vit_b16_openai", "vit_b32_openai", "dinov2_vitb14", "dinov3_vitb16"],
                        help="Backbone used to embed the gallery (default: vit_b16_openai)")
    parser.add_argument("--ckpt", default="checkpoints/vit_b16_arcface_hn.pt",
                        help="Checkpoint path")
    parser.add_argument("--suffix", default=None,
                        help="Gallery name suffix, e.g. '_dinov3' → 'openlogodet3k_dinov3' (auto-set by --model)")
    args = parser.parse_args()

    if args.model:
        args.ckpt, args.backbone, auto_suffix = _MODEL_PRESETS[args.model]
        if args.suffix is None:
            args.suffix = auto_suffix

    setup_logging(__file__)

    input_size = 168 if args.backbone == "dinov2_vitb14" else 160
    suffix = args.suffix if args.suffix is not None else ""

    for base_name, parquet in DATASETS.items():
        gallery_name = base_name + suffix
        print(f"\n=== Building gallery: {gallery_name} (backbone={args.backbone}) ===")
        _ensure_per_ds_parquet(base_name, parquet)
        if not Path(parquet).exists() or pd.read_parquet(parquet).empty:
            print(f"  [SKIP] {parquet} empty")
            continue
        build_gallery(parquet, gallery_name, ckpt_path=args.ckpt,
                      input_size=input_size, backbone=args.backbone)

    print("\nGallery built.")
