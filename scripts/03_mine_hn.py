"""Step 6: Mine hard negatives from confusion matrix on val set."""
import argparse
import sys
sys.path.insert(0, ".")
from src.training.mine_hn import mine

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/vit_base.pt")
    parser.add_argument("--alpha1", type=float, default=0.05)
    parser.add_argument("--alpha2", type=float, default=0.35)
    parser.add_argument("--lev_min", type=int, default=2)
    args = parser.parse_args()
    mine(args.ckpt, args.alpha1, args.alpha2, args.lev_min)
