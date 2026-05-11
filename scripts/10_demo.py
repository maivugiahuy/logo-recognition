"""
Step 10: End-to-end demo — input image(s) → brand labels with bounding boxes.
Usage: python scripts/10_demo.py path/to/image.jpg [path/to/image2.jpg ...]
       python scripts/10_demo.py --gallery openlogodet3k --conf 0.1   # eval gallery
       python scripts/10_demo.py --gallery new_classes --conf 0.1     # user-added classes
       python scripts/10_demo.py img.jpg --save_crops                 # save cropped logos
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, ".")

from PIL import Image, ImageDraw, ImageFont

from src.retrieval.pipeline import LogoRecognitionPipeline
from src.utils.logging_utils import setup_logging

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
    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except Exception:
        font = ImageFont.load_default(size=24)
    for r in results:
        b = r["box"]
        color = _UNKNOWN_COLOR if r.get("is_unknown") else _brand_color(r["brand"])
        draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=color, width=3)
        label = r["brand"]
        bbox = font.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = int(b["x1"]), max(0, int(b["y1"]) - th - 6)
        draw.rectangle([tx, ty, tx + tw + 6, ty + th + 6], fill=color)
        draw.text((tx + 3, ty + 3), label, fill="white", font=font)
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
    parser.add_argument("--backbone", default="vit_b32_openai",
                        choices=["vit_b32_openai", "dinov2_vitb14"],
                        help="Embedder backbone (default: vit_b32_openai)")
    parser.add_argument("--input_size", type=int, default=None,
                        help="Override input resolution (default: 160 for ViT, 224 for DINOv2)")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--unknown_threshold", type=float, default=0.50,
                        help="Cosine similarity threshold below which logo is 'unknown' (default: 0.50)")
    parser.add_argument("--no_qe", action="store_true",
                        help="Disable α-weighted Query Expansion")
    parser.add_argument("--qe_k", type=int, default=5,
                        help="Number of neighbors for Query Expansion (default: 5)")
    parser.add_argument("--qe_alpha", type=float, default=3.0,
                        help="α exponent for QE weighting — higher = closer neighbors dominate (default: 3.0)")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--save_crops", action="store_true",
                        help="Save each detected logo crop as a separate image")
    args = parser.parse_args()

    setup_logging(__file__)
    print("Loading pipeline...")
    default_input_size = 168 if args.backbone == "dinov2_vitb14" else 160
    input_size = args.input_size if args.input_size is not None else default_input_size

    pipeline = LogoRecognitionPipeline(
        detector_weights=args.detector,
        embedder_ckpt=args.embedder,
        gallery_name=args.gallery,
        backbone=args.backbone,
        conf=args.conf,
        input_size=input_size,
        unknown_threshold=args.unknown_threshold,
        qe_enabled=not args.no_qe,
        qe_k=args.qe_k,
        qe_alpha=args.qe_alpha,
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

        if args.save_crops:
            crop_dir = Path(args.save_dir) if args.save_dir else Path("results/crops")
            crop_dir.mkdir(parents=True, exist_ok=True)
            img_orig = Image.open(img_path).convert("RGB")
            stem = Path(img_path).stem
            for i, r in enumerate(results):
                b = r["box"]
                x1, y1 = max(0, int(b["x1"])), max(0, int(b["y1"]))
                x2, y2 = min(img_orig.width, int(b["x2"])), min(img_orig.height, int(b["y2"]))
                crop = img_orig.crop((x1, y1, x2, y2))
                brand = r["brand"].replace("/", "_")
                crop_path = crop_dir / f"{stem}_{i:02d}_{brand}.jpg"
                crop.save(crop_path)
                print(f"  Crop saved → {crop_path}")

        if args.save_dir:
            out = Path(args.save_dir) / (Path(img_path).stem + "_result.jpg")
            Path(args.save_dir).mkdir(exist_ok=True)
            draw_results(img_path, results, str(out))
        else:
            draw_results(img_path, results)
