"""Knowledge distillation training: ViT-S/16 student ← ViT+DINOv2 ensemble teacher."""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.distill_dataset import DistillDataset
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.models.embedder_vit_s import build_vit_s_embedder
import torchvision.transforms as T

ANN = Path("data/processed/openlogodet3k/annotations.parquet")
TEACHER_EMBS = Path("data/processed/teacher_embs.npy")
TEACHER_PARQUET = Path("data/processed/teacher_train.parquet")
CKPT_DIR = Path("checkpoints")


def _train_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        T.RandomPerspective(distortion_scale=0.2, p=0.5),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _val_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _recall_at_1(student: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    student.eval()
    all_embs, all_labels = [], []
    with torch.no_grad():
        for imgs, _teacher, labels in loader:
            embs = student(imgs.to(device))
            all_embs.append(embs.cpu())
            all_labels.extend(labels.tolist())
    embs = torch.cat(all_embs)
    labels = torch.tensor(all_labels)
    sim = embs @ embs.T
    sim.fill_diagonal_(-1e9)
    nn_labels = labels[sim.argmax(dim=1)]
    return (nn_labels == labels).float().mean().item()


def train_distill(cfg_path: str, ckpt_name: str = "vit_s_distill.pt") -> None:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    input_size = cfg.get("input_size", 160)
    embed_dim = cfg.get("embed_dim", 128)
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["optimizer"]["lr"]
    weight_decay = cfg["optimizer"].get("weight_decay", 0.05)
    patience = cfg.get("scheduler", {}).get("patience", 5)
    factor = cfg.get("scheduler", {}).get("factor", 0.25)
    es_patience = cfg.get("early_stopping_patience", 7)
    amp_enabled = cfg.get("amp", True)
    num_workers = cfg.get("num_workers", 8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (
        device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    ) else torch.float16

    teacher_parquet = cfg.get("teacher_parquet", str(TEACHER_PARQUET))
    teacher_embs_path = cfg.get("teacher_embs", str(TEACHER_EMBS))

    train_ds = DistillDataset.from_files(
        teacher_parquet, teacher_embs_path,
        transform=_train_transform(input_size),
    )
    val_ds = DistillDataset.from_files(
        teacher_parquet, teacher_embs_path,
        transform=_val_transform(input_size),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=True,
    )

    student = build_vit_s_embedder(embed_dim, input_size).to(device)
    print(f"Student: ViT-S/16  input={input_size}  embed_dim={embed_dim}")
    print(f"Train: {len(train_ds)} crops, {train_ds.df['class_name'].nunique()} classes")

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=patience, factor=factor)
    scaler = GradScaler(enabled=amp_enabled)

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_path = CKPT_DIR / ckpt_name
    best_recall = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        for imgs, teacher_embs, _ in tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=False):
            imgs = imgs.to(device)
            teacher_embs = teacher_embs.to(device)
            optimizer.zero_grad()
            with autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                student_embs = student(imgs)
                loss = (1.0 - F.cosine_similarity(student_embs, teacher_embs)).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        recall = _recall_at_1(student, val_loader, device)
        scheduler.step(recall)

        improved = recall > best_recall
        if improved:
            best_recall = recall
            torch.save({"embedder": student.state_dict()}, ckpt_path)
            no_improve = 0
        else:
            no_improve += 1

        print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | val recall@1 {recall:.4f}"
              + (f"\n  → saved {ckpt_path} (recall@1={recall:.4f})" if improved else ""))

        if no_improve >= es_patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {es_patience} epochs)")
            break

    print(f"\nBest val recall@1: {best_recall:.4f}")
