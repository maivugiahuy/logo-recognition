"""Step 6: Mine hard negatives from confusion matrix on val set."""
import argparse
import sys
sys.path.insert(0, ".")

import yaml
from src.training.mine_hn import mine

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/vit_base.pt")
    parser.add_argument("--config", default="configs/base_vit.yaml",
                        help="Config yaml used for Phase A (reads freeze_blocks, embed_dim, input_size)")
    parser.add_argument("--alpha1", type=float, default=None)
    parser.add_argument("--alpha2", type=float, default=None)
    parser.add_argument("--lev_min", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    freeze_blocks = cfg.get("freeze_blocks", 0)
    alpha1 = args.alpha1 if args.alpha1 is not None else cfg.get("hn_mining", {}).get("alpha1", 0.05)
    alpha2 = args.alpha2 if args.alpha2 is not None else cfg.get("hn_mining", {}).get("alpha2", 0.35)
    lev_min = args.lev_min if args.lev_min is not None else cfg.get("hn_mining", {}).get("levenshtein_min", 2)
    ckpt = args.ckpt or cfg.get("hn_mining", {}).get("map_path", "checkpoints/vit_base.pt")

    print(f"Config: freeze_blocks={freeze_blocks}, alpha1={alpha1}, alpha2={alpha2}, lev_min={lev_min}")
    mine(ckpt, alpha1, alpha2, lev_min, freeze_blocks=freeze_blocks)
