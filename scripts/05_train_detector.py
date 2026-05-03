"""Step 8: Prepare YOLO data + fine-tune YOLOv8m logo detector."""
import argparse
import sys
sys.path.insert(0, ".")
from src.detector.prepare import prepare_yolo_dataset
from src.detector.train_yolov8 import train_detector
from src.detector.detect import evaluate_ap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--config", default="configs/detector_yolov8.yaml")
    args = parser.parse_args()

    best_weights = "runs/detect/checkpoints/yolov8_logo/weights/best.pt"

    if not args.eval_only:
        if not args.skip_prepare:
            print("=== Preparing YOLO dataset ===")
            prepare_yolo_dataset()
        print("=== Training YOLOv8m ===")
        best_weights = train_detector(args.config)

    print("=== Evaluating AP@0.5 ===")
    evaluate_ap(best_weights)
