"""
End-to-end inference pipeline:
  predict(image) → [(box, brand_label, score, is_unknown)]

1. YOLO detects logo boxes
2. Crop + resize 160×160
3. ViT/DINOv2 embedder → 128-d L2 vector
4. FAISS IndexFlatIP → top-k retrieval
5. Unknown threshold: if score < threshold → "unknown"
"""
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.transforms import val_transforms, val_transforms_dinov2
from src.detector.detect import LogoDetector
from src.models.embedder_dinov2 import build_dinov2_embedder
from src.models.embedder_dinov3 import build_dinov3_embedder
from src.models.embedder_vit import build_vit_embedder
from src.retrieval.gallery import load_gallery

DEFAULT_DETECTOR = "runs/detect/checkpoints/yolov8_logo/weights/best.pt"
DEFAULT_EMBEDDER = "checkpoints/vit_hn.pt"
DEFAULT_GALLERY = "openlogodet3k"

_DINOV2_BACKBONES = {"dinov2_vitb14", "dinov2"}
_DINOV3_BACKBONES = {"dinov3_vitb16"}
_IMAGENET_BACKBONES = _DINOV2_BACKBONES | _DINOV3_BACKBONES

# Cosine similarity threshold for "unknown" decision.
# Embeddings are L2-normalized → inner product = cosine sim ∈ [-1, 1].
# Score < threshold → brand confidence too low → return "unknown".
# Tune by running the pipeline on the val set and finding the F1-optimal threshold.
DEFAULT_UNKNOWN_THRESHOLD = 0.50


class LogoRecognitionPipeline:
    def __init__(
        self,
        detector_weights: str | Path = DEFAULT_DETECTOR,
        embedder_ckpt: str | Path = DEFAULT_EMBEDDER,
        gallery_name: str = DEFAULT_GALLERY,
        backbone: str = "vit_b16_openai",
        conf: float = 0.1,
        embed_dim: int = 128,
        input_size: int = 160,
        top_k: int = 1,
        unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_dinov2 = backbone in _DINOV2_BACKBONES
        is_dinov3 = backbone in _DINOV3_BACKBONES
        self.transform = val_transforms_dinov2(input_size) if backbone in _IMAGENET_BACKBONES else val_transforms(input_size)
        self.top_k = top_k
        self.unknown_threshold = unknown_threshold

        self.detector = LogoDetector(weights=detector_weights, conf=conf)

        if is_dinov2:
            embedder = build_dinov2_embedder(embed_dim, input_size, freeze_blocks=0).to(self.device)
        elif is_dinov3:
            embedder = build_dinov3_embedder(embed_dim, input_size, freeze_blocks=0).to(self.device)
        else:
            embedder = build_vit_embedder(embed_dim, input_size, freeze_blocks=0, backbone=backbone).to(self.device)
        state = torch.load(embedder_ckpt, map_location=self.device)
        embedder.load_state_dict(state["embedder"])
        embedder.eval()
        # Warmup: trigger CUDA kernel compile so first real inference is not skewed
        with torch.no_grad():
            embedder(torch.zeros(1, 3, input_size, input_size, device=self.device))
        self.embedder = embedder

        self.index, self.gallery_labels = load_gallery(gallery_name)

        print(f"Pipeline ready. Gallery: {len(self.gallery_labels)} entries. "
              f"Unknown threshold: {self.unknown_threshold:.2f}")

    def _extract_crop(self, image: Image.Image, box: dict) -> Image.Image | None:
        """Return PIL crop for box, or None if degenerate."""
        w, h = image.size
        x1 = max(0, int(box["x1"]))
        y1 = max(0, int(box["y1"]))
        x2 = min(w, int(box["x2"]))
        y2 = min(h, int(box["y2"]))
        if x2 <= x1 or y2 <= y1:
            return None
        return image.crop((x1, y1, x2, y2))

    def _embed(self, crop: Image.Image) -> np.ndarray:
        """Embed PIL crop → (1, D) L2-normalized float32 array."""
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.embedder(tensor).cpu().numpy()
        return emb.astype("float32")

    def _search_box(
        self, image: Image.Image, box: dict, top_k: int = 20
    ) -> list[tuple[float, str]]:
        """Return top-k (score, label) from gallery for a single box. Used by EnsemblePipeline."""
        crop = self._extract_crop(image, box)
        if crop is None:
            return []
        query = self._embed(crop)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)
        return [(float(scores[0][i]), self.gallery_labels[int(indices[0][i])]) for i in range(k)]

    def predict(self, image_path: str | Path, return_timing: bool = False):
        """
        Returns list of dicts:
          {
            "box":        {x1, y1, x2, y2, conf},
            "brand":      str  (class name or "unknown"),
            "score":      float  (cosine similarity, 0–1),
            "is_unknown": bool,
          }
        If return_timing=True, returns (results, {"detection_ms": float, "recognition_ms": float}).
        """
        image = Image.open(image_path).convert("RGB")

        t0 = time.time()
        boxes = self.detector.detect(image_path)
        detection_ms = (time.time() - t0) * 1000

        t0 = time.time()
        results = []
        for box in boxes:
            crop = self._extract_crop(image, box)
            if crop is None:
                continue

            query = self._embed(crop)
            raw_scores, indices = self.index.search(query, self.top_k)
            score = float(raw_scores[0][0])
            best_idx = int(indices[0][0])

            is_unknown = score < self.unknown_threshold
            brand = "unknown" if is_unknown else self.gallery_labels[best_idx]

            results.append({
                "box": box,
                "brand": brand,
                "score": score,
                "is_unknown": is_unknown,
            })
        recognition_ms = (time.time() - t0) * 1000

        if return_timing:
            return results, {"detection_ms": detection_ms, "recognition_ms": recognition_ms}
        return results
