"""Step 7: Phase C — train ViT with ProxyNCAHN++ (closed-set)."""
import argparse
import sys
sys.path.insert(0, ".")
from src.training.train import train
from src.utils.logging_utils import setup_logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hn_vit.yaml")
    parser.add_argument("--ckpt", default="vit_hn.pt")
    args = parser.parse_args()
    setup_logging(__file__)
    train(args.config, args.ckpt)
