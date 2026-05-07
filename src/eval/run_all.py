"""Evaluate recall@1 on LogoDet-3K test set."""
from pathlib import Path

import pandas as pd

from src.eval.recall_at_1 import evaluate

ANN_BASE = Path("data/processed")
CKPT = "checkpoints/vit_hn.pt"


EVAL_CONFIGS = {
    "openlogodet3k_closedset": {
        "parquet": ANN_BASE / "openlogodet3k_test.parquet",
        "split": None,
        "mode": "closed_set",
    },
    "openlogodet3k_openset": {
        "parquet": ANN_BASE / "openlogodet3k_openset_test.parquet",
        "split": None,
        "mode": "open_set",
    },
}


def _ensure_openlogodet3k_parquet() -> Path:
    """Create openlogodet3k_test.parquet from the main annotations if missing.
    Filters to only images in the closed_test split."""
    import json
    per_ds = ANN_BASE / "openlogodet3k_test.parquet"
    if per_ds.exists():
        return per_ds
    main = ANN_BASE / "openlogodet3k/annotations.parquet"
    closed_test_json = ANN_BASE / "openlogodet3k/splits/closed_test.json"
    df = pd.read_parquet(main)
    if closed_test_json.exists():
        with open(closed_test_json) as f:
            closed_test = json.load(f)  # {class_name: [image_paths]}
        keep_rows = []
        for cls, imgs in closed_test.items():
            img_set = set(imgs)
            rows = df[(df["class_name"] == cls) & (df["image_path"].isin(img_set))]
            keep_rows.append(rows)
        df = pd.concat(keep_rows, ignore_index=True) if keep_rows else df.iloc[:0]
        print(f"  Filtered to closed_test: {df['class_name'].nunique()} classes, {len(df)} objects")
    else:
        # Fallback: use all data if split file not found
        print("  [WARN] closed_test.json not found, using all openlogodet3k data")
        df = df[df["source"] == "logodet3k"]
    per_ds.parent.mkdir(exist_ok=True)
    df.to_parquet(per_ds, index=False)
    return per_ds


def _ensure_openset_test_parquet() -> Path:
    """Create parquet containing only classes in open_test.json (unseen classes)."""
    import json
    out = ANN_BASE / "openlogodet3k_openset_test.parquet"
    if out.exists():
        return out
    main = ANN_BASE / "openlogodet3k/annotations.parquet"
    open_test_json = ANN_BASE / "openlogodet3k/splits/open_test.json"
    if not open_test_json.exists():
        print("  [WARN] open_test.json not found — skipping open-set eval")
        return out
    with open(open_test_json) as f:
        test_classes = set(json.load(f))
    df = pd.read_parquet(main)
    df = df[df["class_name"].isin(test_classes)]
    print(f"  Open-set test: {df['class_name'].nunique()} classes, {len(df)} objects")
    out.parent.mkdir(exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def run_all(ckpt_path: str = CKPT, split: str = "all", ckpt_label: str = None) -> dict:
    """Run evaluation.
    split: 'all' | 'closedset' | 'openset'
    ckpt_label: label written to CSV 'checkpoint' column (defaults to ckpt_path)
    """
    configs = EVAL_CONFIGS
    if split == "closedset":
        configs = {k: v for k, v in EVAL_CONFIGS.items() if "closedset" in k}
    elif split == "openset":
        configs = {k: v for k, v in EVAL_CONFIGS.items() if "openset" in k}

    label = ckpt_label or ckpt_path

    all_results = {}
    for name, cfg in configs.items():
        parquet = cfg["parquet"]
        if not parquet.exists():
            if name == "openlogodet3k_openset":
                parquet = _ensure_openset_test_parquet()
            else:
                parquet = _ensure_openlogodet3k_parquet()
        if not parquet.exists():
            print(f"[SKIP] {name}: parquet not found")
            continue

        print(f"\n{'='*50}")
        print(f"Checkpoint: {label}  |  Dataset: {name.upper()}")

        results = evaluate(
            ckpt_path=ckpt_path,
            ann_parquet=parquet,
            split_json=None,
            mode=cfg["mode"],
        )
        all_results[name] = results

    # Append to CSV (accumulate across multiple run_all calls in one session)
    rows = []
    for ds, res in all_results.items():
        row = {"checkpoint": label, "dataset": ds}
        row.update(res)
        rows.append(row)
    if rows:
        out_csv = Path("results/eval_results.csv")
        Path("results").mkdir(exist_ok=True)
        new_df = pd.DataFrame(rows)
        if out_csv.exists():
            existing = pd.read_csv(out_csv)
            # Drop rows with same checkpoint+dataset (overwrite stale results)
            mask = ~(
                existing["checkpoint"].isin(new_df["checkpoint"]) &
                existing["dataset"].isin(new_df["dataset"])
            )
            combined = pd.concat([existing[mask], new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(out_csv, index=False)
        print(f"\nSaved → {out_csv}")
    return all_results


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else CKPT
    split = sys.argv[2] if len(sys.argv) > 2 else "all"
    run_all(ckpt, split=split)
