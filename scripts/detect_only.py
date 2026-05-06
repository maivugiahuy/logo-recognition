"""
Detect logos in image(s) — bounding boxes only, no recognition.
Usage:
    python scripts/detect_only.py image.jpg
    python scripts/detect_only.py *.jpg --conf 0.2 --save_dir results/
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, ".")

from PIL import Image, ImageDraw

from src.detector.detect import LogoDetector
from src.utils.logging_utils import setup_logging


def draw_boxes(image_path: str, boxes: list[dict], out_path: str | None = None) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for b in boxes:
        draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline="#00ff00", width=3)
        draw.text((b["x1"], max(0, b["y1"] - 14)), f"{b['conf']:.2f}", fill="#00ff00")
    if out_path:
        img.save(out_path)
        print(f"  Saved → {out_path}")
    else:
        img.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Image paths")
    parser.add_argument("--detector", default="runs/detect/checkpoints/yolov8_logo/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()

    setup_logging(__file__)
    detector = LogoDetector(weights=args.detector, conf=args.conf, iou=args.iou)

    for img_path in args.images:
        print(f"\nImage: {img_path}")
        boxes = detector.detect(img_path)
        if not boxes:
            print("  No logos detected.")
            continue
        for i, b in enumerate(boxes):
            print(f"  [{i:02d}] box [{b['x1']:.0f},{b['y1']:.0f},{b['x2']:.0f},{b['y2']:.0f}]  conf:{b['conf']:.4f}")

        if args.save_dir:
            out = Path(args.save_dir) / (Path(img_path).stem + "_detect.jpg")
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            draw_boxes(img_path, boxes, str(out))
        else:
            draw_boxes(img_path, boxes)
