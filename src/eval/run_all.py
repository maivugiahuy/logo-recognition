"""
Reproduce Table 2: recall@1 across 5 public logo datasets.
Targets (paper Table 2, ViT IT Pre-trained row):
  LogoDet3K Q-vs-G: 0.9836
  OpenLogo  Q-vs-G: 0.9371  Text: 0.9568
  FlickrLogos-47 All: 0.9784
  BelgaLogos All: 0.9797
  LiTW Q-vs-G: 0.9778  Text: 0.9391
"""
import json
from pathlib import Path

import pandas as pd

from src.eval.recall_at_1 import evaluate

ANN_BASE = Path("data/processed")
CKPT = "checkpoints/vit_hn.pt"


EVAL_CONFIGS = {
    "logodet3k": {
        "parquet": ANN_BASE / "logodet3k_test.parquet",
        "split": None,  # test split parquet covers just test images
        "mode": "closed_set",
        "targets": {"qvg": 0.9836, "all_vs_all": 0.9886},
    },
    "openlogo": {
        "parquet": ANN_BASE / "openlogo_test.parquet",
        "split": None,
        "mode": "closed_set",
        "targets": {"qvg": 0.9371, "text_qvg": 0.9568, "all_vs_all": 0.9629},
    },
    "flickr47": {
        "parquet": ANN_BASE / "flickr47_test.parquet",
        "split": None,
        "mode": "closed_set",
        "targets": {"all_vs_all": 0.9784},
    },
    "belga": {
        "parquet": ANN_BASE / "belga_test.parquet",
        "split": None,
        "mode": "closed_set",
        "targets": {"all_vs_all": 0.9797},
    },
    "litw": {
        "parquet": ANN_BASE / "litw_test.parquet",
        "split": None,
        "mode": "closed_set",
        "targets": {"qvg": 0.9778, "text_qvg": 0.9391},
    },
}


def _per_dataset_parquet(dataset_name: str) -> Path:
    """Load and filter annotations to a specific dataset source."""
    # Delegate to each dataset's own annotation file if it exists;
    # otherwise filter main OLG3K47 parquet by source.
    per_ds = ANN_BASE / f"{dataset_name}_test.parquet"
    if per_ds.exists():
        return per_ds
    main = ANN_BASE / "openlogodet3k47/annotations.parquet"
    df = pd.read_parquet(main)
    df = df[df["source"] == dataset_name]
    per_ds.parent.mkdir(exist_ok=True)
    df.to_parquet(per_ds, index=False)
    return per_ds


def run_all(ckpt_path: str = CKPT) -> dict:
    all_results = {}
    for name, cfg in EVAL_CONFIGS.items():
        parquet = cfg["parquet"]
        if not parquet.exists():
            parquet = _per_dataset_parquet(name)
        if not parquet.exists():
            print(f"[SKIP] {name}: parquet not found")
            continue

        print(f"\n{'='*50}")
        print(f"Dataset: {name.upper()}")
        print(f"Target: {cfg['targets']}")

        results = evaluate(
            ckpt_path=ckpt_path,
            ann_parquet=parquet,
            split_json=None,  # evaluate on whole parquet
            mode=cfg["mode"],
        )
        all_results[name] = results

        # Check gates
        for metric, target in cfg["targets"].items():
            actual = results.get(metric, float("nan"))
            gap = actual - target
            status = "OK" if gap >= -0.02 else "FAIL (>2 pt below)"
            print(f"  [{status}] {metric}: {actual:.4f} (target {target:.4f}, gap {gap:+.4f})")

    # Save CSV
    rows = []
    for ds, res in all_results.items():
        row = {"dataset": ds}
        row.update(res)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("eval_results.csv", index=False)
    print(f"\nSaved → eval_results.csv")
    return all_results


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else CKPT
    run_all(ckpt)
