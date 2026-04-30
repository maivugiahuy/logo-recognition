"""Step 10: Evaluate recall@1 on LogoDet-3K test set."""
import argparse
import sys
sys.path.insert(0, ".")
from src.eval.run_all import run_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/vit_hn.pt")
    args = parser.parse_args()
    results = run_all(args.ckpt)
    print("\nFinal results saved to eval_results.csv")
