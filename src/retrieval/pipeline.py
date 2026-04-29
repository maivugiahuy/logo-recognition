"""
End-to-end inference pipeline (Step 11):
  predict(image) → [(box, brand_label, score)]

1. YOLOv8 detects logo boxes
2. Crop + resize 160×160
3. ViT embedder → 128-d L2 vector
4. FAISS IndexFlatIP → top-1 brand
"""
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.transforms import val_transforms
from src.detector.detect import LogoDetector
from src.models.embedder_vit import build_vit_embedder
from src.retrieval.gallery import load_gallery

DEFAULT_DETECTOR = "checkpoints/yolov8_logo/weights/best.pt"
DEFAULT_EMBEDDER = "checkpoints/vit_hn.pt"
DEFAULT_GALLERY = "logodet3k"


class LogoRecognitionPipeline:
    def __init__(
        self,
        detector_weights: str | Path = DEFAULT_DETECTOR,
        embedder_ckpt: str | Path = DEFAULT_EMBEDDER,
        gallery_name: str = DEFAULT_GALLERY,
        conf: float = 0.1,
        embed_dim: int = 128,
        input_size: int = 160,
        top_k: int = 1,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = val_transforms(input_size)
        self.top_k = top_k

        self.detector = LogoDetector(weights=detector_weights, conf=conf)

        embedder = build_vit_embedder(embed_dim, input_size).to(self.device)
        state = torch.load(embedder_ckpt, map_location=self.device)
        embedder.load_state_dict(state["embedder"])
        embedder.eval()
        self.embedder = embedder

        self.index, self.gallery_labels = load_gallery(gallery_name)

    def _embed_crop(self, image: Image.Image, box: dict) -> np.ndarray:
        """Crop PIL image by box dict, return 128-d L2-normalized embedding."""
        w, h = image.size
        x1 = max(0, int(box["x1"]))
        y1 = max(0, int(box["y1"]))
        x2 = min(w, int(box["x2"]))
        y2 = min(h, int(box["y2"]))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image.crop((x1, y1, x2, y2))
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.embedder(tensor).cpu().numpy()  # (1, D)
        return emb

    def predict(self, image_path: str | Path) -> list[dict]:
        """
        Returns:
          [{"box": {x1,y1,x2,y2,conf}, "brand": str, "score": float}, ...]
        """
        image = Image.open(image_path).convert("RGB")
        boxes = self.detector.detect(image_path)
        results = []

        for box in boxes:
            emb = self._embed_crop(image, box)
            if emb is None:
                continue
            scores, indices = self.index.search(emb.astype("float32"), self.top_k)
            brand = self.gallery_labels[indices[0][0]]
            score = float(scores[0][0])
            results.append({"box": box, "brand": brand, "score": score})

        return results
