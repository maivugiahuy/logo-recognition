"""OLG3K47Dataset: crops bboxes from images, returns (tensor, class_idx)."""
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class OLG3K47Dataset(Dataset):
    def __init__(
        self,
        annotations: pd.DataFrame,
        class_to_idx: dict[str, int],
        transform=None,
    ):
        self.df = annotations.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    @classmethod
    def from_split(
        cls,
        ann_path: str | Path,
        split_json: str | Path,
        transform=None,
        mode: str = "open_set",
    ) -> "OLG3K47Dataset":
        """
        mode='open_set': split_json is a list of class names (train/val/test classes).
        mode='closed_set': split_json is {class: [image_paths]} dict.
        """
        df = pd.read_parquet(ann_path)
        with open(split_json) as f:
            split_data = json.load(f)

        if mode == "open_set":
            classes = split_data
            mask = df["class_name"].isin(classes)
            df = df[mask]
        else:
            # closed_set: {class_name: [image_paths]}
            keep_rows = []
            for cls, imgs in split_data.items():
                img_set = set(imgs)
                rows = df[(df["class_name"] == cls) & (df["image_path"].isin(img_set))]
                keep_rows.append(rows)
            df = pd.concat(keep_rows, ignore_index=True) if keep_rows else df.iloc[:0]

        all_classes = sorted(df["class_name"].unique().tolist())
        class_to_idx = {c: i for i, c in enumerate(all_classes)}
        return cls(df, class_to_idx, transform)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        # clamp bbox to image bounds
        w, h = img.size
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            img = img.crop((x1, y1, x2, y2))
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[row["class_name"]]
        return img, label

    @property
    def labels(self) -> list[int]:
        return [self.class_to_idx[n] for n in self.df["class_name"]]

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)
