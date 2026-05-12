"""Step 7: Evaluate recall@1 on LogoDet-3K test set."""
import argparse
import sys
sys.path.insert(0, ".")
from src.eval.run_all import run_all
from src.utils.logging_utils import setup_logging

DEFAULT_CKPTS = ["checkpoints/vit_hn.pt"]
_ENSEMBLE_SPLITS = {
    "openlogodet3k_closedset": ("data/processed/openlogodet3k_test.parquet", "closed_set"),
    "openlogodet3k_openset":   ("data/processed/openlogodet3k_openset_test.parquet", "open_set"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint path. Omit to evaluate both vit_base.pt and vit_hn.pt.")
    parser.add_argument("--split", default="all",
                        choices=["all", "closedset", "openset"],
                        help="Which split to evaluate (default: all)")
    parser.add_argument("--backbone", default="vit_b32_openai",
                        choices=["vit_b32_openai", "dinov2_vitb14", "vit_s16"],
                        help="Embedder backbone matching the checkpoint (default: vit_b32_openai)")
    parser.add_argument("--ocr", action="store_true",
                        help="Enable OCR fusion: EasyOCR text fused with visual score at retrieval")
    parser.add_argument("--ocr_weight", type=float, default=0.3,
                        help="Weight for OCR text score in fusion (default: 0.3)")
    parser.add_argument("--ocr_rerank_k", type=int, default=10,
                        help="Top-k candidates to rerank with OCR (default: 10)")
    # Ensemble
    parser.add_argument("--ensemble", action="store_true",
                        help="Evaluate ViT+DINOv2 ensemble (ignores --ckpt/--backbone)")
    parser.add_argument("--vit_ckpt", default="checkpoints/vit_hn.pt",
                        help="ViT checkpoint for ensemble (default: checkpoints/vit_hn.pt)")
    parser.add_argument("--dinov2_ckpt", default="checkpoints/dinov2_hn.pt",
                        help="DINOv2 checkpoint for ensemble (default: checkpoints/dinov2_hn.pt)")
    parser.add_argument("--vit_weight", type=float, default=0.5,
                        help="ViT score weight in ensemble fusion (default: 0.5)")
    parser.add_argument("--ensemble_top_k", type=int, default=20,
                        help="Top-k per backbone for ensemble fusion (default: 20)")
    args = parser.parse_args()
    setup_logging(__file__)

    ocr_args = dict(ocr_enabled=args.ocr, ocr_weight=args.ocr_weight, ocr_rerank_k=args.ocr_rerank_k)

    if args.ensemble:
        from src.eval.recall_at_1 import evaluate_ensemble
        from pathlib import Path
        splits = _ENSEMBLE_SPLITS
        if args.split == "closedset":
            splits = {k: v for k, v in splits.items() if "closedset" in k}
        elif args.split == "openset":
            splits = {k: v for k, v in splits.items() if "openset" in k}
        print(f"\n{'#'*60}")
        print(f"# Ensemble: {args.vit_ckpt} + {args.dinov2_ckpt}  vit_weight={args.vit_weight}")
        for name, (parquet, mode) in splits.items():
            if not Path(parquet).exists():
                print(f"[SKIP] {name}: {parquet} not found")
                continue
            print(f"\n{'='*50}")
            print(f"Ensemble  |  Dataset: {name.upper()}")
            evaluate_ensemble(
                vit_ckpt=args.vit_ckpt,
                dinov2_ckpt=args.dinov2_ckpt,
                ann_parquet=parquet,
                mode=mode,
                vit_weight=args.vit_weight,
                ensemble_top_k=args.ensemble_top_k,
            )
    else:
        ckpts = [args.ckpt] if args.ckpt else DEFAULT_CKPTS
        all_results = {}
        for ckpt in ckpts:
            print(f"\n{'#'*60}")
            print(f"# Checkpoint: {ckpt}  Backbone: {args.backbone}")
            res = run_all(ckpt, split=args.split, ckpt_label=ckpt, backbone=args.backbone,
                          **ocr_args)
            all_results[ckpt] = res
