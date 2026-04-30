"""
Hard-negative mining — paper Sec 3.2.3.
For each class yi, h(yi) = {yj : α1 ≤ C[i,j] ≤ α2  AND  levenshtein(name_i, name_j) > 2}
Output: data/processed/hn_map.json  {class_name: [hn_class_name, ...]}
"""
import json
from pathlib import Path

import numpy as np
from Levenshtein import distance as levenshtein

from src.training.confusion import build_confusion_matrix

ALPHA1 = 0.05
ALPHA2 = 0.35
LEV_MIN = 2
OUT = Path("data/processed/hn_map.json")


def mine(
    ckpt_path: str | Path,
    alpha1: float = ALPHA1,
    alpha2: float = ALPHA2,
    lev_min: int = LEV_MIN,
    freeze_blocks: int = 0,
) -> dict[str, list[str]]:
    C, class_names = build_confusion_matrix(ckpt_path, freeze_blocks=freeze_blocks)
    num_classes = len(class_names)
    hn_map: dict[str, list[str]] = {}

    for i, name_i in enumerate(class_names):
        hard_negs = []
        for j, name_j in enumerate(class_names):
            if i == j:
                continue
            if alpha1 <= C[i, j] <= alpha2:
                if levenshtein(name_i, name_j) > lev_min:
                    hard_negs.append(name_j)
        hn_map[name_i] = hard_negs

    n_total = sum(len(v) for v in hn_map.values())
    n_with_hn = sum(1 for v in hn_map.values() if v)
    print(f"HN map: {n_with_hn}/{num_classes} classes have hard negatives ({n_total} total pairs)")

    # Spot-check expected examples from paper
    for probe in ["h_j_heinz", "heinz", "heineken"]:
        if probe in hn_map:
            print(f"  h({probe}) = {hn_map[probe]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(hn_map, f, indent=2)
    print(f"Saved → {OUT}")
    return hn_map


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/vit_base.pt"
    mine(ckpt)
