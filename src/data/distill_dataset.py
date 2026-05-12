"""Dataset for knowledge distillation: returns (student_img, teacher_emb, class_label)."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class DistillDataset(Dataset):
    def __init__(self, df: pd.DataFrame, teacher_embs: np.ndarray, transform=None):
        assert len(df) == len(teacher_embs), (
            f"df rows ({len(df)}) != teacher_embs rows ({len(teacher_embs)})"
        )
        self.df = df.reset_index(drop=True)
        self.teacher_embs = torch.from_numpy(teacher_embs.astype("float32"))
        self.transform = transform
        classes = sorted(df["class_name"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        w, h = img.size
        x1, y1 = max(0, int(row["x1"])), max(0, int(row["y1"]))
        x2, y2 = min(w, int(row["x2"])), min(h, int(row["y2"]))
        if x2 > x1 and y2 > y1:
            img = img.crop((x1, y1, x2, y2))
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[row["class_name"]]
        return img, self.teacher_embs[idx], label

    @classmethod
    def from_files(
        cls,
        parquet_path: str | Path,
        teacher_embs_path: str | Path,
        transform=None,
    ) -> "DistillDataset":
        df = pd.read_parquet(parquet_path)
        embs = np.load(teacher_embs_path)
        return cls(df, embs, transform=transform)
