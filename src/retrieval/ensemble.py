"""
Ensemble pipeline: score-level fusion of ViT + DINOv2 galleries.

For each detected crop:
  1. Search top-k from ViT gallery   → {label: max_vit_score}
  2. Search top-k from DINOv2 gallery → {label: max_dinov2_score}
  3. fused[label] = vit_weight * vit_score + (1-vit_weight) * dinov2_score
  4. Pick label with highest fused score

Requires two pre-built galleries:
  data/galleries/openlogodet3k.faiss          (ViT)
  data/galleries/openlogodet3k_dinov2.faiss   (DINOv2)

Build DINOv2 gallery first:
  python scripts/08_build_galleries.py --backbone dinov2_vitb14
"""
from pathlib import Path

from PIL import Image

from src.retrieval.pipeline import DEFAULT_UNKNOWN_THRESHOLD, LogoRecognitionPipeline

DEFAULT_VIT_CKPT = "checkpoints/vit_hn.pt"
DEFAULT_DINOV2_CKPT = "checkpoints/dinov2_hn.pt"
DEFAULT_VIT_GALLERY = "openlogodet3k"
DEFAULT_DINOV2_GALLERY = "openlogodet3k_dinov2"
DEFAULT_VIT_WEIGHT = 0.5
DEFAULT_ENSEMBLE_TOP_K = 20


class EnsemblePipeline:
    """Score-level fusion of ViT and DINOv2 retrieval pipelines."""

    def __init__(
        self,
        vit_pipeline: LogoRecognitionPipeline,
        dinov2_pipeline: LogoRecognitionPipeline,
        vit_weight: float = DEFAULT_VIT_WEIGHT,
        unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
        top_k: int = DEFAULT_ENSEMBLE_TOP_K,
    ):
        self.vit = vit_pipeline
        self.dinov2 = dinov2_pipeline
        self.vit_weight = vit_weight
        self.dinov2_weight = 1.0 - vit_weight
        self.unknown_threshold = unknown_threshold
        self.top_k = top_k
        print(f"Ensemble ready. ViT weight={vit_weight:.2f}  DINOv2 weight={self.dinov2_weight:.2f}  "
              f"Unknown threshold={unknown_threshold:.2f}")

    def predict(self, image_path: str | Path) -> list[dict]:
        """
        Returns list of dicts:
          {
            "box":        {x1, y1, x2, y2, conf},
            "brand":      str  (class name or "unknown"),
            "score":      float  (fused score),
            "is_unknown": bool,
          }
        """
        image = Image.open(image_path).convert("RGB")
        boxes = self.vit.detector.detect(image_path)
        results = []

        for box in boxes:
            vit_hits = self.vit._search_box(image, box, top_k=self.top_k)
            dinov2_hits = self.dinov2._search_box(image, box, top_k=self.top_k)

            if not vit_hits and not dinov2_hits:
                continue

            # Per-label max score from each backbone
            vit_scores: dict[str, float] = {}
            for score, label in vit_hits:
                vit_scores[label] = max(vit_scores.get(label, 0.0), score)

            dinov2_scores: dict[str, float] = {}
            for score, label in dinov2_hits:
                dinov2_scores[label] = max(dinov2_scores.get(label, 0.0), score)

            # Fuse
            all_labels = set(vit_scores) | set(dinov2_scores)
            fused = {
                label: self.vit_weight * vit_scores.get(label, 0.0)
                       + self.dinov2_weight * dinov2_scores.get(label, 0.0)
                for label in all_labels
            }

            best_label = max(fused, key=fused.__getitem__)
            best_score = fused[best_label]
            is_unknown = best_score < self.unknown_threshold

            results.append({
                "box": box,
                "brand": "unknown" if is_unknown else best_label,
                "score": best_score,
                "is_unknown": is_unknown,
            })

        return results


def build_ensemble_pipeline(
    detector_weights: str = "runs/detect/checkpoints/yolov8_logo/weights/best.pt",
    vit_ckpt: str = DEFAULT_VIT_CKPT,
    dinov2_ckpt: str = DEFAULT_DINOV2_CKPT,
    vit_gallery: str = DEFAULT_VIT_GALLERY,
    dinov2_gallery: str = DEFAULT_DINOV2_GALLERY,
    conf: float = 0.1,
    unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
    vit_weight: float = DEFAULT_VIT_WEIGHT,
    top_k: int = DEFAULT_ENSEMBLE_TOP_K,
) -> EnsemblePipeline:
    """Convenience factory that builds both pipelines and wraps them."""
    vit = LogoRecognitionPipeline(
        detector_weights=detector_weights,
        embedder_ckpt=vit_ckpt,
        gallery_name=vit_gallery,
        backbone="vit_b32_openai",
        conf=conf,
        input_size=160,
        unknown_threshold=unknown_threshold,
    )
    dinov2 = LogoRecognitionPipeline(
        detector_weights=detector_weights,
        embedder_ckpt=dinov2_ckpt,
        gallery_name=dinov2_gallery,
        backbone="dinov2_vitb14",
        conf=conf,
        input_size=168,
        unknown_threshold=unknown_threshold,
    )
    return EnsemblePipeline(vit, dinov2, vit_weight=vit_weight,
                            unknown_threshold=unknown_threshold, top_k=top_k)
