"""Generate t-SNE visualization of embedding space for presentation."""
import sys
sys.path.insert(0, ".")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from src.data.dataset import LogoDataset
from src.data.transforms import val_transforms
from src.models.embedder_vit import build_vit_embedder

OUT = Path("docs/slides/figures")
ANN = Path("data/processed/openlogodet3k/annotations.parquet")
SPLITS = Path("data/processed/openlogodet3k/splits")
CKPT = Path("checkpoints/vit_b16_arcface_hn.pt")

NUM_CLASSES = 20
MAX_PER_CLASS = 50
SEED = 42


def extract_embeddings(ckpt_path, num_classes=NUM_CLASSES, max_per_class=MAX_PER_CLASS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = val_transforms(160)
    ds = LogoDataset.from_split(
        ANN, SPLITS / "closed_test.json",
        transform=transform, mode="closed_set",
    )

    # Pick top-N classes by sample count
    from collections import Counter
    label_counts = Counter(ds.labels)
    top_classes = [cls for cls, _ in label_counts.most_common(num_classes)]
    top_set = set(top_classes)

    # Filter indices
    indices = []
    class_count = {c: 0 for c in top_classes}
    for i, lbl in enumerate(ds.labels):
        if lbl in top_set and class_count[lbl] < max_per_class:
            indices.append(i)
            class_count[lbl] += 1

    subset = torch.utils.data.Subset(ds, indices)
    loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4)

    embedder = build_vit_embedder(128, 160, freeze_blocks=0, backbone="vit_b16_openai").to(device)
    state = torch.load(ckpt_path, map_location=device)
    embedder.load_state_dict(state["embedder"])
    embedder.eval()

    all_embs = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            embs = embedder(imgs.to(device)).cpu().numpy()
            all_embs.append(embs)
            all_labels.append(labels.numpy())

    embs = np.concatenate(all_embs)
    labels = np.concatenate(all_labels)

    # Map label indices to class names
    idx_to_name = {v: k for k, v in ds.class_to_idx.items()}
    names = [idx_to_name[l] for l in labels]

    return embs, labels, names


def plot_tsne(embs, labels, names):
    print(f"Running t-SNE on {len(embs)} embeddings...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=1000)
    coords = tsne.fit_transform(embs)

    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    label_to_color = {l: cmap(i) for i, l in enumerate(unique_labels)}

    idx_to_name = {}
    for l, n in zip(labels, names):
        idx_to_name[l] = n

    fig, ax = plt.subplots(figsize=(12, 10))
    for l in unique_labels:
        mask = labels == l
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[label_to_color[l]], s=15, alpha=0.7, label=idx_to_name[l])

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=2)
    ax.set_title(f"t-SNE Embedding Space (ViT-B/16 ArcFace HN, {NUM_CLASSES} classes)", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "slide29_tsne.png", dpi=200)
    plt.close(fig)
    print(f"  -> slide29_tsne.png")


if __name__ == "__main__":
    if not CKPT.exists():
        print(f"Checkpoint not found: {CKPT}")
        print("Skipping t-SNE generation.")
        sys.exit(0)

    embs, labels, names = extract_embeddings(CKPT)
    plot_tsne(embs, labels, names)
    print("Done!")
