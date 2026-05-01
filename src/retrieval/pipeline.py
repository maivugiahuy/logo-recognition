"""
End-to-end inference pipeline (Step 11):
  predict(image) → [(box, brand_label, score, is_unknown)]

1. YOLO26 detects logo boxes
2. Crop + resize 160×160
3. ViT embedder → 128-d L2 vector
4. FAISS IndexFlatIP → top-1 brand
5. Unknown threshold: nếu cosine similarity < threshold → "unknown"
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.transforms import val_transforms
from src.detector.detect import LogoDetector
from src.models.embedder_vit import build_vit_embedder
from src.retrieval.gallery import load_gallery

DEFAULT_DETECTOR = "runs/detect/checkpoints/yolo26m_logo/weights/best.pt"
DEFAULT_EMBEDDER = "checkpoints/vit_hn.pt"
DEFAULT_GALLERY = "openlogodet3k"

# Cosine similarity threshold để quyết định "unknown".
# Embeddings L2-normalized → inner product = cosine sim ∈ [-1, 1].
# Score < threshold → brand không đủ tự tin → trả về "unknown".
# Tune bằng cách chạy pipeline trên val set và tìm F1-optimal threshold.
DEFAULT_UNKNOWN_THRESHOLD = 0.50


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
        unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = val_transforms(input_size)
        self.top_k = top_k
        self.unknown_threshold = unknown_threshold

        self.detector = LogoDetector(weights=detector_weights, conf=conf)

        embedder = build_vit_embedder(embed_dim, input_size, freeze_blocks=0).to(self.device)
        state = torch.load(embedder_ckpt, map_location=self.device)
        embedder.load_state_dict(state["embedder"])
        embedder.eval()
        self.embedder = embedder

        self.index, self.gallery_labels = load_gallery(gallery_name)
        print(f"Pipeline ready. Gallery: {len(self.gallery_labels)} entries. "
              f"Unknown threshold: {self.unknown_threshold:.2f}")

    def _embed_crop(self, image: Image.Image, box: dict) -> np.ndarray | None:
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
        Returns list of dicts:
          {
            "box":        {x1, y1, x2, y2, conf},
            "brand":      str  (class name hoặc "unknown"),
            "score":      float  (cosine similarity, 0–1),
            "is_unknown": bool,
          }
        """
        image = Image.open(image_path).convert("RGB")
        boxes = self.detector.detect(image_path)
        results = []

        for box in boxes:
            emb = self._embed_crop(image, box)
            if emb is None:
                continue

            scores, indices = self.index.search(emb.astype("float32"), self.top_k)
            score = float(scores[0][0])
            is_unknown = score < self.unknown_threshold
            brand = "unknown" if is_unknown else self.gallery_labels[indices[0][0]]

            results.append({
                "box": box,
                "brand": brand,
                "score": score,
                "is_unknown": is_unknown,
            })

        return results
