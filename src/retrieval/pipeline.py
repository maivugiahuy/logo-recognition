"""
End-to-end inference pipeline:
  predict(image) → [(box, brand_label, score, is_unknown)]

1. YOLO detects logo boxes
2. Crop + resize 160×160
3. ViT/DINOv2 embedder → 128-d L2 vector
4. FAISS IndexFlatIP → top-k retrieval
5. OCR fusion (optional): EasyOCR on crop → fuse visual + text similarity scores
6. Unknown threshold: if score < threshold → "unknown"
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.transforms import val_transforms, val_transforms_dinov2
from src.detector.detect import LogoDetector
from src.models.embedder_dinov2 import build_dinov2_embedder
from src.models.embedder_vit import build_vit_embedder
from src.models.embedder_vit_s import build_vit_s_embedder
from src.retrieval.gallery import load_gallery
from src.retrieval.ocr import run_ocr, text_similarity

DEFAULT_DETECTOR = "runs/detect/checkpoints/yolov8_logo/weights/best.pt"
DEFAULT_EMBEDDER = "checkpoints/vit_hn.pt"
DEFAULT_GALLERY = "openlogodet3k"

_DINOV2_BACKBONES = {"dinov2_vitb14", "dinov2"}
_VIT_S_BACKBONES = {"vit_s16"}

# Cosine similarity threshold for "unknown" decision.
# Embeddings are L2-normalized → inner product = cosine sim ∈ [-1, 1].
# Score < threshold → brand confidence too low → return "unknown".
# Tune by running the pipeline on the val set and finding the F1-optimal threshold.
DEFAULT_UNKNOWN_THRESHOLD = 0.50
DEFAULT_OCR_WEIGHT = 0.3
DEFAULT_OCR_RERANK_K = 10


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
        ocr_enabled: bool = False,
        ocr_weight: float = DEFAULT_OCR_WEIGHT,
        ocr_rerank_k: int = DEFAULT_OCR_RERANK_K,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_dinov2 = backbone in _DINOV2_BACKBONES
        is_vit_s = backbone in _VIT_S_BACKBONES
        # ViT-S uses ImageNet normalization (same as DINOv2)
        self.transform = val_transforms(input_size) if not (is_dinov2 or is_vit_s) else val_transforms_dinov2(input_size)
        self.top_k = top_k
        self.unknown_threshold = unknown_threshold
        self.ocr_enabled = ocr_enabled
        self.ocr_weight = ocr_weight
        self.ocr_rerank_k = ocr_rerank_k

        self.detector = LogoDetector(weights=detector_weights, conf=conf)

        if is_dinov2:
            embedder = build_dinov2_embedder(embed_dim, input_size, freeze_blocks=0).to(self.device)
        elif is_vit_s:
            embedder = build_vit_s_embedder(embed_dim, input_size).to(self.device)
        else:
            embedder = build_vit_embedder(embed_dim, input_size, freeze_blocks=0).to(self.device)
        state = torch.load(embedder_ckpt, map_location=self.device)
        embedder.load_state_dict(state["embedder"])
        embedder.eval()
        self.embedder = embedder

        self.index, self.gallery_labels = load_gallery(gallery_name)

        ocr_status = f"  OCR: on (w={self.ocr_weight}, k={self.ocr_rerank_k})" if self.ocr_enabled else ""
        print(f"Pipeline ready. Gallery: {len(self.gallery_labels)} entries. "
              f"Unknown threshold: {self.unknown_threshold:.2f}{ocr_status}")

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

    def _ocr_rerank(self, query: np.ndarray, crop: Image.Image) -> tuple[float, int]:
        """Search top-k, run OCR on crop, fuse visual+text scores, return (best_score, best_idx)."""
        k = min(self.ocr_rerank_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)
        scores, indices = scores[0], indices[0]

        use_gpu = self.device.type == "cuda"
        ocr_text = run_ocr(crop, gpu=use_gpu)

        best_score, best_idx = -1.0, int(indices[0])
        for vis_score, idx in zip(scores, indices):
            label = self.gallery_labels[idx]
            tsim = text_similarity(ocr_text, label)
            fused = (1 - self.ocr_weight) * float(vis_score) + self.ocr_weight * tsim if ocr_text else float(vis_score)
            if fused > best_score:
                best_score, best_idx = fused, int(idx)
        return best_score, best_idx

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
            crop = self._extract_crop(image, box)
            if crop is None:
                continue

            query = self._embed(crop)

            if self.ocr_enabled:
                score, best_idx = self._ocr_rerank(query, crop)
            else:
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

        return results
