"""
Fine-tune YOLO26m as class-agnostic logo detector (Appendix A substitute).
Paper used YoloV4 on proprietary PL2K; we use YOLO26m on LogoDet-3K + OpenLogo.
"""
from pathlib import Path

import yaml
from ultralytics import YOLO


def train_detector(cfg_path: str = "configs/detector_yolo26.yaml") -> None:
    with open(cfg_path, encoding="utf-8") as f:
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
        workers=cfg.get("workers", 4),
        cache=cfg.get("cache", False),
    )
    # Lấy path thực từ results.save_dir (ultralytics mới có thể prepend runs/detect/)
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print("Detector training done.")
    print(f"Best weights: {best_weights}")
    return best_weights


if __name__ == "__main__":
    train_detector()
