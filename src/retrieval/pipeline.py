"""
End-to-end inference pipeline (Step 11):
  predict(image) → [(box, brand_label, score, is_unknown)]

1. YOLO detects logo boxes
2. Crop + resize 160×160
3. ViT embedder → 128-d L2 vector
4. FAISS IndexFlatIP → top-k retrieval
5. α-weighted Query Expansion (αQE): re-query with weighted avg of query + top-k
6. Unknown threshold: if cosine similarity < threshold → "unknown"
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.transforms import val_transforms, val_transforms_dinov2
from src.detector.detect import LogoDetector
from src.models.embedder_dinov2 import build_dinov2_embedder
from src.models.embedder_vit import build_vit_embedder
from src.retrieval.gallery import load_gallery

DEFAULT_DETECTOR = "runs/detect/checkpoints/yolov8_logo/weights/best.pt"
DEFAULT_EMBEDDER = "checkpoints/vit_hn.pt"
DEFAULT_GALLERY = "openlogodet3k"

_DINOV2_BACKBONES = {"dinov2_vitb14", "dinov2"}

# Cosine similarity threshold for "unknown" decision.
# Embeddings are L2-normalized → inner product = cosine sim ∈ [-1, 1].
# Score < threshold → brand confidence too low → return "unknown".
# Tune by running the pipeline on the val set and finding the F1-optimal threshold.
DEFAULT_UNKNOWN_THRESHOLD = 0.50
DEFAULT_QE_K = 5
DEFAULT_QE_ALPHA = 3.0


class LogoRecognitionPipeline:
    def __init__(
        self,
        detector_weights: str | Path = DEFAULT_DETECTOR,
        embedder_ckpt: str | Path = DEFAULT_EMBEDDER,
        gallery_name: str = DEFAULT_GALLERY,
        backbone: str = "vit_b32_openai",
        conf: float = 0.1,
        embed_dim: int = 128,
        input_size: int = 160,
        top_k: int = 1,
        unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
        qe_enabled: bool = True,
        qe_k: int = DEFAULT_QE_K,
        qe_alpha: float = DEFAULT_QE_ALPHA,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_dinov2 = backbone in _DINOV2_BACKBONES
        self.transform = val_transforms_dinov2(input_size) if is_dinov2 else val_transforms(input_size)
        self.top_k = top_k
        self.unknown_threshold = unknown_threshold
        self.qe_enabled = qe_enabled
        self.qe_k = qe_k
        self.qe_alpha = qe_alpha

        self.detector = LogoDetector(weights=detector_weights, conf=conf)

        if is_dinov2:
            embedder = build_dinov2_embedder(embed_dim, input_size, freeze_blocks=0).to(self.device)
        else:
            embedder = build_vit_embedder(embed_dim, input_size, freeze_blocks=0).to(self.device)
        state = torch.load(embedder_ckpt, map_location=self.device)
        embedder.load_state_dict(state["embedder"])
        embedder.eval()
        self.embedder = embedder

        self.index, self.gallery_labels = load_gallery(gallery_name)

        if self.qe_enabled:
            self._gallery_vecs = np.zeros(
                (self.index.ntotal, self.index.d), dtype="float32"
            )
            self.index.reconstruct_n(0, self.index.ntotal, self._gallery_vecs)

        print(f"Pipeline ready. Gallery: {len(self.gallery_labels)} entries. "
              f"Unknown threshold: {self.unknown_threshold:.2f}  "
              f"QE: {'on' if self.qe_enabled else 'off'}"
              f"{f' (k={self.qe_k}, α={self.qe_alpha})' if self.qe_enabled else ''}")

    def _query_expand(self, query: np.ndarray) -> np.ndarray:
        """α-weighted Query Expansion: average query with top-k gallery neighbors,
        weighting each neighbor by score^α. Higher α → closer neighbors dominate."""
        k = min(self.qe_k, self.index.ntotal)
        scores, indices = self.index.search(query.astype("float32"), k)
        scores, indices = scores[0], indices[0]

        weights = np.power(scores.clip(0), self.qe_alpha).astype("float32")
        neighbor_vecs = self._gallery_vecs[indices]  # (k, D)
        expanded = query[0] + (weights[:, None] * neighbor_vecs).sum(axis=0)
        norm = np.linalg.norm(expanded)
        if norm > 0:
            expanded /= norm
        return expanded.reshape(1, -1).astype("float32")

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
            "brand":      str  (class name or "unknown"),
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

            query = emb.astype("float32")
            if self.qe_enabled:
                query = self._query_expand(query)

            scores, indices = self.index.search(query, self.top_k)
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
