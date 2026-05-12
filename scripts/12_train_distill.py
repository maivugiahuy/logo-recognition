"""Step 12: Train ViT-S/16 student via knowledge distillation from ViT+DINOv2 ensemble.

Run 11_precompute_teacher.py first to generate teacher embeddings.

Usage:
  python scripts/12_train_distill.py
  python scripts/12_train_distill.py --config configs/distill_vit_s.yaml --ckpt vit_s_distill.pt
"""
import argparse
import sys
sys.path.insert(0, ".")

from src.training.train_distill import train_distill
from src.utils.logging_utils import setup_logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distill_vit_s.yaml")
    parser.add_argument("--ckpt", default="vit_s_distill.pt",
                        help="Output checkpoint filename under checkpoints/ (default: vit_s_distill.pt)")
    args = parser.parse_args()

    setup_logging(__file__)
    train_distill(args.config, ckpt_name=args.ckpt)
