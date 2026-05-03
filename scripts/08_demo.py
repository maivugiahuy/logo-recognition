"""
Step 11: End-to-end demo — input image(s) → brand labels with bounding boxes.
Usage: python scripts/08_demo.py path/to/image.jpg [path/to/image2.jpg ...]
       python scripts/08_demo.py --gallery openlogodet3k --conf 0.1   # eval gallery
       python scripts/08_demo.py --gallery new_classes --conf 0.1     # user-added classes
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, ".")

from PIL import Image, ImageDraw, ImageFont

from src.retrieval.pipeline import LogoRecognitionPipeline

_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]
_UNKNOWN_COLOR = "#ff6600"
_brand_color_cache: dict[str, str] = {}


def _brand_color(brand: str) -> str:
    if brand not in _brand_color_cache:
        _brand_color_cache[brand] = _PALETTE[len(_brand_color_cache) % len(_PALETTE)]
    return _brand_color_cache[brand]


def draw_results(image_path: str, results: list[dict], out_path: str | None = None) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for r in results:
        b = r["box"]
        color = _UNKNOWN_COLOR if r.get("is_unknown") else _brand_color(r["brand"])
        draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=color, width=3)
        label = f"{r['brand']} (det:{b['conf']:.2f} sim:{r['score']:.2f})"
        draw.text((b["x1"], max(0, b["y1"] - 15)), label, fill=color)
    if out_path:
        img.save(out_path)
        print(f"  Saved → {out_path}")
    else:
        img.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Image paths")
    parser.add_argument("--detector", default="runs/detect/checkpoints/yolov8_logo/weights/best.pt")
    parser.add_argument("--embedder", default="checkpoints/vit_hn.pt")
    parser.add_argument("--gallery", default="openlogodet3k",
                        help="Gallery name: 'openlogodet3k' (eval, default) or 'new_classes' (user-added via add_classes.py)")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--unknown_threshold", type=float, default=0.50,
                        help="Cosine similarity threshold below which logo is 'unknown' (default: 0.50)")
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()

    print("Loading pipeline...")
    pipeline = LogoRecognitionPipeline(
        detector_weights=args.detector,
        embedder_ckpt=args.embedder,
        gallery_name=args.gallery,
        conf=args.conf,
        unknown_threshold=args.unknown_threshold,
    )

    for img_path in args.images:
        print(f"\nImage: {img_path}")
        results = pipeline.predict(img_path)
        if not results:
            print("  No logos detected.")
            continue
        for r in results:
            b = r["box"]
            status = "UNKNOWN" if r["is_unknown"] else r["brand"]
            print(f"  box [{b['x1']:.0f},{b['y1']:.0f},{b['x2']:.0f},{b['y2']:.0f}] "
                  f"det:{b['conf']:.4f}  → brand: {status}  sim:{r['score']:.4f}")

        if args.save_dir:
            out = Path(args.save_dir) / (Path(img_path).stem + "_result.jpg")
            Path(args.save_dir).mkdir(exist_ok=True)
            draw_results(img_path, results, str(out))
        else:
            draw_results(img_path, results)
