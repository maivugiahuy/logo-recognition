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
    args = parser.parse_args()
    setup_logging(__file__)

    ckpts = [args.ckpt] if args.ckpt else DEFAULT_CKPTS
    all_results = {}
    for ckpt in ckpts:
        print(f"\n{'#'*60}")
        print(f"# Checkpoint: {ckpt}")
        res = run_all(ckpt, split=args.split, ckpt_label=ckpt)
        all_results[ckpt] = res
    print("\nFinal results saved to eval_results.csv")
