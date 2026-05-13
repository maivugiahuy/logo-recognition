"""Generate all matplotlib figures for the presentation slides."""
import sys
sys.path.insert(0, ".")

import json
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

OUT = Path("docs/slides/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

ANN = Path("data/processed/openlogodet3k/annotations.parquet")
SPLITS = Path("data/processed/openlogodet3k/splits")


def fig_sample_distribution():
    """Slide 7: Histogram of samples per class."""
    df = pd.read_parquet(ANN)
    spc = df.groupby("class_name").size()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(spc.values, bins=80, color="#4363d8", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of samples per class")
    ax.set_ylabel("Number of classes")
    ax.set_title("Distribution of samples per class (2,472 classes)")
    ax.axvline(spc.median(), color="#e6194b", linestyle="--", linewidth=2, label=f"Median = {spc.median():.0f}")
    ax.axvline(spc.mean(), color="#f58231", linestyle="--", linewidth=2, label=f"Mean = {spc.mean():.1f}")
    ax.legend()
    ax.set_xlim(0, 500)
    fig.savefig(OUT / "slide07_sample_distribution.png")
    plt.close(fig)
    print(f"  -> slide07_sample_distribution.png")


def fig_logo_sizes():
    """Slide 7: Scatter plot of logo width x height."""
    df = pd.read_parquet(ANN)
    df["logo_w"] = df.x2 - df.x1
    df["logo_h"] = df.y2 - df.y1

    sample = df.sample(n=min(5000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sample.logo_w, sample.logo_h, alpha=0.15, s=8, c="#4363d8")
    ax.set_xlabel("Logo width (px)")
    ax.set_ylabel("Logo height (px)")
    ax.set_title("Logo bounding box sizes (5K sample)")
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 1000)
    ax.axvline(df.logo_w.median(), color="#e6194b", linestyle="--", alpha=0.7, label=f"Median W={df.logo_w.median():.0f}")
    ax.axhline(df.logo_h.median(), color="#f58231", linestyle="--", alpha=0.7, label=f"Median H={df.logo_h.median():.0f}")
    ax.legend()
    fig.savefig(OUT / "slide07_logo_sizes.png")
    plt.close(fig)
    print(f"  -> slide07_logo_sizes.png")


def fig_source_distribution():
    """Slide 6: Pie/bar chart of data source."""
    df = pd.read_parquet(ANN)
    src_counts = df.source.value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar: annotations
    colors = ["#4363d8", "#e6194b"]
    axes[0].bar(src_counts.index, src_counts.values, color=colors, edgecolor="white")
    for i, v in enumerate(src_counts.values):
        axes[0].text(i, v + 1000, f"{v:,}", ha="center", fontweight="bold")
    axes[0].set_ylabel("Annotations")
    axes[0].set_title("Annotations per source")

    # Bar: classes
    cls_per_src = df.groupby("source")["class_name"].nunique()
    axes[1].bar(cls_per_src.index, cls_per_src.values, color=colors, edgecolor="white")
    for i, v in enumerate(cls_per_src.values):
        axes[1].text(i, v + 30, f"{v:,}", ha="center", fontweight="bold")
    axes[1].set_ylabel("Classes")
    axes[1].set_title("Classes per source")

    fig.suptitle("OpenLogoDet-3K: LogoDet-3K + OpenLogo", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "slide06_source_distribution.png")
    plt.close(fig)
    print(f"  -> slide06_source_distribution.png")


def fig_split_diagram():
    """Slide 8: Split sizes visualization."""
    # Open-set
    open_sizes = {"Train\n(1,582 cls)": 1582, "Val\n(395 cls)": 395, "Test\n(495 cls)": 495}
    # Closed-set
    with open(SPLITS / "closed_train.json") as f:
        ct = json.load(f)
    with open(SPLITS / "closed_val.json") as f:
        cv = json.load(f)
    with open(SPLITS / "closed_test.json") as f:
        cte = json.load(f)
    closed_sizes = {
        f"Train\n({sum(len(v) for v in ct.values()):,} imgs)": sum(len(v) for v in ct.values()),
        f"Val\n({sum(len(v) for v in cv.values()):,} imgs)": sum(len(v) for v in cv.values()),
        f"Test\n({sum(len(v) for v in cte.values()):,} imgs)": sum(len(v) for v in cte.values()),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#4363d8", "#f58231", "#e6194b"]

    # Open-set
    axes[0].barh(list(open_sizes.keys()), list(open_sizes.values()), color=colors, edgecolor="white")
    for i, v in enumerate(open_sizes.values()):
        axes[0].text(v + 20, i, str(v), va="center", fontweight="bold")
    axes[0].set_title("Open-set split (by class)")
    axes[0].set_xlabel("Number of classes")
    axes[0].invert_yaxis()

    # Closed-set
    axes[1].barh(list(closed_sizes.keys()), list(closed_sizes.values()), color=colors, edgecolor="white")
    for i, v in enumerate(closed_sizes.values()):
        axes[1].text(v + 300, i, f"{v:,}", va="center", fontweight="bold")
    axes[1].set_title("Closed-set split (by image, 1,977 classes)")
    axes[1].set_xlabel("Number of images")
    axes[1].invert_yaxis()

    fig.tight_layout()
    fig.savefig(OUT / "slide08_splits.png")
    plt.close(fig)
    print(f"  -> slide08_splits.png")


def fig_ablation_phase():
    """Slide 23: Phase A vs Phase C bar chart."""
    models = ["ViT-B/32\nProxyNCA", "ViT-B/16\nArcFace"]
    phase_a_closed = [0.9529, 0.9699]
    phase_c_closed = [0.9623, 0.9760]
    phase_a_open = [0.9663, 0.9788]
    phase_c_open = [0.9722, 0.9780]

    x = np.arange(len(models))
    w = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Closed-set
    bars1 = axes[0].bar(x - w/2, phase_a_closed, w, label="Phase A", color="#4363d8")
    bars2 = axes[0].bar(x + w/2, phase_c_closed, w, label="+ Phase C (HN)", color="#e6194b")
    axes[0].set_ylabel("Recall@1")
    axes[0].set_title("Closed-set Q-vs-G")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylim(0.94, 0.985)
    axes[0].legend()
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f"{bar.get_height():.4f}", ha="center", fontsize=9)
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f"{bar.get_height():.4f}", ha="center", fontsize=9)

    # Open-set
    bars1 = axes[1].bar(x - w/2, phase_a_open, w, label="Phase A", color="#4363d8")
    bars2 = axes[1].bar(x + w/2, phase_c_open, w, label="+ Phase C (HN)", color="#e6194b")
    axes[1].set_ylabel("Recall@1")
    axes[1].set_title("Open-set Q-vs-G")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_ylim(0.94, 0.985)
    axes[1].legend()
    for bar in bars1:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f"{bar.get_height():.4f}", ha="center", fontsize=9)
    for bar in bars2:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f"{bar.get_height():.4f}", ha="center", fontsize=9)

    fig.suptitle("Ablation: Phase A vs Phase C (Hard Negative Training)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "slide23_ablation_phase.png")
    plt.close(fig)
    print(f"  -> slide23_ablation_phase.png")


def fig_ablation_backbone_loss():
    """Slide 24: Backbone and loss comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Backbone comparison (Phase A, ArcFace)
    backbones = ["ViT-B/32", "DINOv3-B/16", "ViT-B/16"]
    closed_qvg = [0.9548, 0.9563, 0.9699]
    open_qvg = [0.9717, 0.9670, 0.9788]

    x = np.arange(len(backbones))
    w = 0.3
    bars1 = axes[0].bar(x - w/2, closed_qvg, w, label="Closed", color="#4363d8")
    bars2 = axes[0].bar(x + w/2, open_qvg, w, label="Open", color="#e6194b")
    axes[0].set_ylabel("Recall@1 Q-vs-G")
    axes[0].set_title("Backbone (ArcFace Phase A)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(backbones)
    axes[0].set_ylim(0.94, 0.99)
    axes[0].legend()
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f"{bar.get_height():.4f}", ha="center", fontsize=8)
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f"{bar.get_height():.4f}", ha="center", fontsize=8)

    # Loss comparison (B/32, after HN)
    losses = ["ProxyNCA", "ArcFace"]
    closed_loss = [0.9623, 0.9694]
    open_loss = [0.9722, 0.9734]

    x2 = np.arange(len(losses))
    bars1 = axes[1].bar(x2 - w/2, closed_loss, w, label="Closed", color="#4363d8")
    bars2 = axes[1].bar(x2 + w/2, open_loss, w, label="Open", color="#e6194b")
    axes[1].set_ylabel("Recall@1 Q-vs-G")
    axes[1].set_title("Loss (ViT-B/32, after HN)")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(losses)
    axes[1].set_ylim(0.94, 0.99)
    axes[1].legend()
    for bar in bars1:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f"{bar.get_height():.4f}", ha="center", fontsize=8)
    for bar in bars2:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f"{bar.get_height():.4f}", ha="center", fontsize=8)

    # Subset breakdown (ViT-B/16 ArcFace HN)
    subsets = ["Overall", "Text", "Small", "Large"]
    closed_sub = [0.9760, 0.9231, 0.9671, 0.9824]
    open_sub = [0.9780, 0.9255, 0.9647, 0.9875]

    x3 = np.arange(len(subsets))
    w3 = 0.3
    bars1 = axes[2].bar(x3 - w3/2, closed_sub, w3, label="Closed", color="#4363d8")
    bars2 = axes[2].bar(x3 + w3/2, open_sub, w3, label="Open", color="#e6194b")
    axes[2].set_ylabel("Recall@1 Q-vs-G")
    axes[2].set_title("Subset (ViT-B/16 ArcFace HN)")
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels(subsets)
    axes[2].set_ylim(0.88, 1.0)
    axes[2].legend()
    for bar in bars1:
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{bar.get_height():.3f}", ha="center", fontsize=8, rotation=45)
    for bar in bars2:
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{bar.get_height():.3f}", ha="center", fontsize=8, rotation=45)

    fig.suptitle("Ablation: Backbone, Loss, and Subset Breakdown", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "slide24_ablation_backbone_loss.png")
    plt.close(fig)
    print(f"  -> slide24_ablation_backbone_loss.png")


def fig_ensemble_improvement():
    """Slide 25: Ensemble vs single model."""
    subsets = ["Overall", "Text", "Small", "Large"]

    # Closed Q-vs-G
    vit_closed = [0.9760, 0.9231, 0.9671, 0.9824]
    ens_closed = [0.9797, 0.9538, 0.9741, 0.9836]
    delta_closed = [e - v for v, e in zip(vit_closed, ens_closed)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(subsets))
    w = 0.3
    bars1 = axes[0].bar(x - w/2, vit_closed, w, label="ViT-B/16 alone", color="#4363d8")
    bars2 = axes[0].bar(x + w/2, ens_closed, w, label="Ensemble", color="#3cb44b")
    axes[0].set_ylabel("Recall@1 Q-vs-G")
    axes[0].set_title("Closed-set: ViT-B/16 vs Ensemble")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(subsets)
    axes[0].set_ylim(0.90, 1.0)
    axes[0].legend()
    for bar, d in zip(bars2, delta_closed):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"+{d:.3f}", ha="center", fontsize=9, color="#e6194b", fontweight="bold")

    # Open Q-vs-G
    vit_open = [0.9780, 0.9255, 0.9647, 0.9875]
    ens_open = [0.9848, 0.9255, 0.9771, 0.9904]
    delta_open = [e - v for v, e in zip(vit_open, ens_open)]

    bars1 = axes[1].bar(x - w/2, vit_open, w, label="ViT-B/16 alone", color="#4363d8")
    bars2 = axes[1].bar(x + w/2, ens_open, w, label="Ensemble", color="#3cb44b")
    axes[1].set_ylabel("Recall@1 Q-vs-G")
    axes[1].set_title("Open-set: ViT-B/16 vs Ensemble")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(subsets)
    axes[1].set_ylim(0.90, 1.0)
    axes[1].legend()
    for bar, d in zip(bars2, delta_open):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"+{d:.3f}", ha="center", fontsize=9, color="#e6194b", fontweight="bold")

    fig.suptitle("Ensemble Improvement (ViT-B/16 + DINOv3, w=0.5)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "slide25_ensemble.png")
    plt.close(fig)
    print(f"  -> slide25_ensemble.png")


def fig_main_results_table():
    """Slide 22: Full comparison as a styled table image."""
    data = {
        "Model": [
            "ViT-B/32 ProxyNCA Phase A",
            "ViT-B/32 ProxyNCA HN",
            "ViT-B/32 ArcFace Phase A",
            "ViT-B/32 ArcFace HN",
            "ViT-B/16 ArcFace Phase A",
            "ViT-B/16 ArcFace HN",
            "DINOv3-B/16 ArcFace",
            "Ensemble (ViT+DINOv3)",
        ],
        "Closed\nQ-vs-G": [0.9529, 0.9623, 0.9548, 0.9694, 0.9699, 0.9760, 0.9563, 0.9797],
        "Closed\nAll": [0.9561, 0.9646, 0.9565, 0.9699, 0.9701, 0.9759, 0.9640, None],
        "Open\nQ-vs-G": [0.9663, 0.9722, 0.9717, 0.9734, 0.9788, 0.9780, 0.9670, 0.9848],
        "Open\nAll": [0.9723, 0.9753, 0.9735, 0.9759, 0.9806, 0.9817, 0.9768, None],
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    cell_text = []
    for i in range(len(data["Model"])):
        row = [data["Model"][i]]
        for col in list(data.keys())[1:]:
            v = data[col][i]
            row.append(f"{v:.4f}" if v is not None else "—")
        cell_text.append(row)

    col_labels = list(data.keys())
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4363d8")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best rows
    for j in range(len(col_labels)):
        table[6, j].set_facecolor("#e8f4e8")  # ViT-B/16 HN
        table[8, j].set_facecolor("#d4edda")   # Ensemble

    ax.set_title("Recall@1 — All Model Variants", fontsize=14, fontweight="bold", pad=20)
    fig.savefig(OUT / "slide22_main_results.png")
    plt.close(fig)
    print(f"  -> slide22_main_results.png")


def fig_patch_grid():
    """Slide 16: B/32 vs B/16 patch grid overlay on a real logo crop."""
    from PIL import Image as PILImage

    # Try to load a real logo crop from dataset
    ann_path = Path("data/processed/openlogodet3k/annotations.parquet")
    logo_img = None
    if ann_path.exists():
        _df = pd.read_parquet(ann_path)
        for target in ["starbucks", "apple", "bmw", "pepsi", "adidas", "nike"]:
            rows = _df[_df.class_name == target]
            if len(rows) == 0:
                continue
            rows = rows.copy()
            rows["area"] = (rows.x2 - rows.x1) * (rows.y2 - rows.y1)
            row = rows.nlargest(1, "area").iloc[0]
            img_path = Path(row.image_path)
            if img_path.exists():
                full = PILImage.open(img_path).convert("RGB")
                x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
                logo_img = full.crop((x1, y1, x2, y2)).resize((160, 160), PILImage.LANCZOS)
                break

    if logo_img is None:
        logo_img = PILImage.fromarray(
            np.random.RandomState(42).randint(50, 200, (160, 160, 3), dtype=np.uint8)
        )

    logo_arr = np.array(logo_img)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (patch_size, title) in zip(axes, [(32, "ViT-B/32: 5×5 = 25 tokens"), (16, "ViT-B/16: 10×10 = 100 tokens")]):
        ax.imshow(logo_arr)
        grid = 160 // patch_size
        for i in range(grid + 1):
            ax.axhline(i * patch_size - 0.5, color="red", linewidth=1.5, alpha=0.7)
            ax.axvline(i * patch_size - 0.5, color="red", linewidth=1.5, alpha=0.7)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Patch Grid: B/32 vs B/16 at 160×160", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "slide16_patch_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> slide16_patch_grid.png")


def fig_timing():
    """Slide 30: Inference timing breakdown."""
    stages = ["YOLO\nDetection", "Crop +\nEmbed", "FAISS\nSearch", "Total\n(1 logo)"]
    times = [40, 0.5, 0.1, 40.6]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e6194b", "#4363d8", "#3cb44b", "#f58231"]
    bars = ax.bar(stages, times, color=colors, edgecolor="white", width=0.5)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Inference Time Breakdown (per image, 1 logo)", fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 200)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                f"{t:.1f} ms", ha="center", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "slide30_timing.png")
    plt.close(fig)
    print(f"  -> slide30_timing.png")


def fig_dataset_samples():
    """Slide 6: Grid of real logo crops from dataset, picking largest per class."""
    from PIL import Image as PILImage

    df = pd.read_parquet(ANN)
    df["area"] = (df.x2 - df.x1) * (df.y2 - df.y1)

    target_classes = [
        "nike", "adidas", "apple", "pepsi", "coca_cola", "starbucks",
        "bmw", "shell", "fedex", "mcdonalds", "volkswagen", "samsung",
    ]

    available = df.class_name.unique().tolist()
    selected = [c for c in target_classes if c in available]
    if len(selected) < 12:
        top = df.groupby("class_name").size().sort_values(ascending=False)
        for cls in top.index:
            if cls not in selected:
                selected.append(cls)
            if len(selected) >= 12:
                break

    cols, rows = 6, 2
    cell = 160
    pad = 12

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    fig.suptitle("Dataset Samples: LogoDet-3K + OpenLogo", fontsize=15, fontweight="bold", y=0.98)

    for idx, cls in enumerate(selected[:cols * rows]):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]

        cls_df = df[df.class_name == cls]
        row = cls_df.nlargest(3, "area").iloc[min(1, len(cls_df) - 1)]

        crop = None
        img_path = Path(row.image_path)
        if img_path.exists():
            full = PILImage.open(img_path).convert("RGB")
            w, h = full.size
            x1 = max(0, int(row.x1))
            y1 = max(0, int(row.y1))
            x2 = min(w, int(row.x2))
            y2 = min(h, int(row.y2))
            if x2 > x1 and y2 > y1:
                crop = full.crop((x1, y1, x2, y2)).resize((cell, cell), PILImage.LANCZOS)

        if crop is not None:
            ax.imshow(np.array(crop))
        else:
            ax.imshow(np.ones((cell, cell, 3), dtype=np.uint8) * 220)

        ax.set_title(cls[:15], fontsize=10, fontweight="bold", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#ccc")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "slide06_dataset_samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> slide06_dataset_samples.png")


if __name__ == "__main__":
    print("Generating slide figures...\n")

    print("[Slide 6] Source distribution")
    fig_source_distribution()

    print("[Slide 7] Sample distribution")
    fig_sample_distribution()

    print("[Slide 7] Logo sizes")
    fig_logo_sizes()

    print("[Slide 8] Splits")
    fig_split_diagram()

    print("[Slide 16] Patch grid")
    fig_patch_grid()

    print("[Slide 22] Main results table")
    fig_main_results_table()

    print("[Slide 23] Ablation: Phase A vs C")
    fig_ablation_phase()

    print("[Slide 24] Ablation: Backbone & Loss")
    fig_ablation_backbone_loss()

    print("[Slide 25] Ensemble improvement")
    fig_ensemble_improvement()

    print("[Slide 30] Timing")
    fig_timing()

    print(f"\nDone! {len(list(OUT.glob('*.png')))} figures saved to {OUT}")
