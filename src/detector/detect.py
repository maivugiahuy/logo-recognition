"""
Class-agnostic logo detection with objectness threshold sweep.
Paper (Appendix A, Fig 7) shows {0.4, 0.1, 0.01} thresholds.
"""
from pathlib import Path
from typing import Union

import torch
from ultralytics import YOLO


class LogoDetector:
    def __init__(
        self,
        weights: str | Path = "runs/detect/checkpoints/yolov8_logo/weights/best.pt",
        conf: float = 0.1,
        iou: float = 0.45,
        imgsz: int = 512,
        device: str | None = None,
    ):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def detect(self, image_path: str | Path) -> list[dict]:
        """
        Returns list of {"x1", "y1", "x2", "y2", "conf"} dicts (pixel coords).
        """
        results = self.model.predict(
            str(image_path),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": float(box.conf[0]),
                })
        return boxes

    def sweep_thresholds(
        self,
        image_path: str | Path,
        thresholds: list[float] = [0.4, 0.1, 0.01],
    ) -> dict[float, list[dict]]:
        """Return detections for each threshold (Fig 7)."""
        results = {}
        for conf in thresholds:
            self.conf = conf
            results[conf] = self.detect(image_path)
        return results


def evaluate_ap(
    weights: str | Path,
    data_yaml: str = "data/processed/detector_yolo/dataset.yaml",
    imgsz: int = 512,
    split: str = "test",
) -> dict:
    """Compute AP@0.5 on the specified split. Gate: ≥0.70."""
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, imgsz=imgsz, split=split, verbose=True)
    ap50 = metrics.box.ap50.mean() if hasattr(metrics.box, "ap50") else metrics.box.map50
    print(f"AP@0.5 ({split}): {ap50:.4f}  [gate: ≥0.70]")
    return {"ap50": float(ap50)}
