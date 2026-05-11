"""Step 7: Evaluate recall@1 on LogoDet-3K test set."""
import argparse
import sys
sys.path.insert(0, ".")
from src.eval.run_all import run_all
from src.utils.logging_utils import setup_logging

DEFAULT_CKPTS = ["checkpoints/vit_base.pt", "checkpoints/vit_hn.pt"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint path. Omit to evaluate both vit_base.pt and vit_hn.pt.")
    parser.add_argument("--split", default="all",
                        choices=["all", "closedset", "openset"],
                        help="Which split to evaluate (default: all)")
    parser.add_argument("--backbone", default="vit_b32_openai",
                        choices=["vit_b32_openai", "dinov2_vitb14"],
                        help="Embedder backbone matching the checkpoint (default: vit_b32_openai)")
    parser.add_argument("--qe", action="store_true",
                        help="Enable α-weighted Query Expansion")
    parser.add_argument("--qe_k", type=int, default=5,
                        help="Number of neighbors for QE (default: 5)")
    parser.add_argument("--qe_alpha", type=float, default=3.0,
                        help="α exponent for QE weighting (default: 3.0)")
    args = parser.parse_args()
    setup_logging(__file__)

    qe_args = dict(qe_enabled=args.qe, qe_k=args.qe_k, qe_alpha=args.qe_alpha)

    ckpts = [args.ckpt] if args.ckpt else DEFAULT_CKPTS
    all_results = {}
    for ckpt in ckpts:
        print(f"\n{'#'*60}")
        print(f"# Checkpoint: {ckpt}  Backbone: {args.backbone}")
        res = run_all(ckpt, split=args.split, ckpt_label=ckpt, backbone=args.backbone, **qe_args)
        all_results[ckpt] = res
    print("\nFinal results saved to eval_results.csv")
