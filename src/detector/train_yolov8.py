"""
Fine-tune YOLOv8m as class-agnostic logo detector (Appendix A substitute).
Paper used YoloV4 on proprietary PL2K; we use YOLOv8m on LogoDet3K.
"""
from pathlib import Path

import yaml
from ultralytics import YOLO


def train_detector(cfg_path: str = "configs/detector_yolov8.yaml") -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model = YOLO(cfg["model"])
    results = model.train(
        data=cfg["data"],
        imgsz=cfg["imgsz"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        device=cfg["device"],
        project=cfg["project"],
        name=cfg["name"],
        exist_ok=cfg.get("exist_ok", True),
        patience=cfg.get("patience", 10),
        save_period=cfg.get("save_period", 10),
    )
    print("Detector training done.")
    print(f"Best weights: {Path(cfg['project']) / cfg['name'] / 'weights/best.pt'}")
    return results


if __name__ == "__main__":
    train_detector()
