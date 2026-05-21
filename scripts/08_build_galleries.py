"""Step 8: Build FAISS galleries (eval + new_classes) for a given model."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, ".")
import pandas as pd
from src.retrieval.gallery import build_gallery, add_to_gallery
from src.utils.logging_utils import setup_logging

ANN = Path("data/processed/openlogodet3k/annotations.parquet")
NEW_CLASSES_DIR = Path("data/new_classes")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

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


def _build_new_classes_gallery(gallery_name: str, ckpt_path: str,
                               backbone: str, input_size: int) -> None:
    """Build new_classes gallery from data/new_classes/ subfolders."""
    if not NEW_CLASSES_DIR.exists():
        print(f"\n=== Skipping {gallery_name}: {NEW_CLASSES_DIR} not found ===")
        return

    subfolders = sorted(p for p in NEW_CLASSES_DIR.iterdir() if p.is_dir())
    if not subfolders:
        print(f"\n=== Skipping {gallery_name}: no subfolders in {NEW_CLASSES_DIR} ===")
        return

    print(f"\n=== Building gallery: {gallery_name} from {NEW_CLASSES_DIR} ({len(subfolders)} classes) ===")
    for subfolder in subfolders:
        imgs = sorted(p for p in subfolder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not imgs:
            print(f"  [SKIP] {subfolder.name}: no images")
            continue
        print(f"  {subfolder.name}: {len(imgs)} images")
        add_to_gallery(
            image_paths=imgs,
            brand_name=subfolder.name,
            dataset_name=gallery_name,
            ckpt_path=ckpt_path,
            input_size=input_size,
            backbone=backbone,
            on_duplicate="replace",
        )


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
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip building eval (openlogodet3k) gallery")
    parser.add_argument("--skip_new_classes", action="store_true",
                        help="Skip building new_classes gallery")
    args = parser.parse_args()

    if args.model:
        args.ckpt, args.backbone, auto_suffix = _MODEL_PRESETS[args.model]
        if args.suffix is None:
            args.suffix = auto_suffix

    setup_logging(__file__)

    input_size = 168 if args.backbone == "dinov2_vitb14" else 160
    suffix = args.suffix if args.suffix is not None else ""

    if not args.skip_eval:
        for base_name, parquet in DATASETS.items():
            gallery_name = base_name + suffix
            print(f"\n=== Building gallery: {gallery_name} (backbone={args.backbone}) ===")
            _ensure_per_ds_parquet(base_name, parquet)
            if not Path(parquet).exists() or pd.read_parquet(parquet).empty:
                print(f"  [SKIP] {parquet} empty")
                continue
            build_gallery(parquet, gallery_name, ckpt_path=args.ckpt,
                          input_size=input_size, backbone=args.backbone)

    if not args.skip_new_classes:
        nc_gallery = "new_classes" + suffix
        _build_new_classes_gallery(nc_gallery, args.ckpt, args.backbone, input_size)

    print("\nDone.")
