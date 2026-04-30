"""
Build open-set (64/16/20 class) and closed-set (within seen classes) splits.
Output: data/processed/openlogodet3k47/splits/open_{train,val,test}.json
        data/processed/openlogodet3k47/splits/closed_{train,val,test}.json
"""
import json
import random
from pathlib import Path

import pandas as pd

ANN = Path("data/processed/openlogodet3k47/annotations.parquet")
SPLITS_DIR = Path("data/processed/openlogodet3k47/splits")
SEED = 42
TRAIN_FRAC = 0.64
VAL_FRAC = 0.16
# test = remaining 0.20


def build_open_set_splits(df: pd.DataFrame) -> tuple[list, list, list]:
    """Split classes 64/16/20 (open-set: test classes unseen during training)."""
    classes = sorted(df["class_name"].unique().tolist())
    rng = random.Random(SEED)
    rng.shuffle(classes)
    n = len(classes)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    return classes[:n_train], classes[n_train:n_train + n_val], classes[n_train + n_val:]


def build_closed_set_splits(df: pd.DataFrame, train_classes: list, val_classes: list) -> dict:
    """Within seen classes (train+val), split objects 64/16/20 by image."""
    seen_classes = set(train_classes) | set(val_classes)
    result: dict[str, dict[str, list]] = {"train": {}, "val": {}, "test": {}}
    rng = random.Random(SEED + 1)

    for cls in sorted(seen_classes):
        images = sorted(df[df["class_name"] == cls]["image_path"].unique().tolist())
        rng.shuffle(images)
        n = len(images)
        n_tr = max(1, int(n * TRAIN_FRAC))
        n_va = max(1, int(n * VAL_FRAC))
        result["train"][cls] = images[:n_tr]
        result["val"][cls] = images[n_tr:n_tr + n_va]
        result["test"][cls] = images[n_tr + n_va:]
    return result


def build(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = pd.read_parquet(ANN)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_cls, val_cls, test_cls = build_open_set_splits(df)
    for split_name, cls_list in [("train", train_cls), ("val", val_cls), ("test", test_cls)]:
        path = SPLITS_DIR / f"open_{split_name}.json"
        with open(path, "w") as f:
            json.dump(cls_list, f, indent=2)
        print(f"  open_{split_name}: {len(cls_list)} classes → {path}")

    closed = build_closed_set_splits(df, train_cls, val_cls)
    for split_name, cls_dict in closed.items():
        path = SPLITS_DIR / f"closed_{split_name}.json"
        with open(path, "w") as f:
            json.dump(cls_dict, f, indent=2)
        n_imgs = sum(len(v) for v in cls_dict.values())
        print(f"  closed_{split_name}: {len(cls_dict)} classes, {n_imgs} images → {path}")


if __name__ == "__main__":
    build()
