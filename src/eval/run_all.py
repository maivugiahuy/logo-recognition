"""
Evaluate recall@1 on LogoDet-3K test set.
Targets (paper Table 2, ViT IT Pre-trained row):
  LogoDet3K Q-vs-G: 0.9836
"""
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
}


def _ensure_logodet3k_parquet() -> Path:
    """Tạo logodet3k_test.parquet từ annotations chính nếu chưa có.
    Chỉ lấy ảnh thuộc closed_test split để eval đúng."""
    import json
    per_ds = ANN_BASE / "logodet3k_test.parquet"
    if per_ds.exists():
        return per_ds
    main = ANN_BASE / "logodet3k/annotations.parquet"
    closed_test_json = ANN_BASE / "logodet3k/splits/closed_test.json"
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
        # Fallback: dùng toàn bộ nếu không tìm thấy split file
        print("  [WARN] closed_test.json not found, using all logodet3k data")
        df = df[df["source"] == "logodet3k"]
    per_ds.parent.mkdir(exist_ok=True)
    df.to_parquet(per_ds, index=False)
    return per_ds


def run_all(ckpt_path: str = CKPT) -> dict:
    all_results = {}
    for name, cfg in EVAL_CONFIGS.items():
        parquet = cfg["parquet"]
        if not parquet.exists():
            parquet = _ensure_logodet3k_parquet()
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
