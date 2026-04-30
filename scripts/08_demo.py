"""
Step 11: End-to-end demo — input image(s) → brand labels with bounding boxes.
Usage: python scripts/08_demo.py path/to/image.jpg [path/to/image2.jpg ...]
       python scripts/08_demo.py --gallery logodet3k --conf 0.1
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, ".")

from PIL import Image, ImageDraw, ImageFont

from src.retrieval.pipeline import LogoRecognitionPipeline


def draw_results(image_path: str, results: list[dict], out_path: str | None = None) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for r in results:
        b = r["box"]
        draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline="red", width=3)
        label = f"{r['brand']} ({r['score']:.2f})"
        draw.text((b["x1"], max(0, b["y1"] - 15)), label, fill="red")
    if out_path:
        img.save(out_path)
        print(f"  Saved → {out_path}")
    else:
        img.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Image paths")
    parser.add_argument("--detector", default="checkpoints/yolov8_logo/weights/best.pt")
    parser.add_argument("--embedder", default="checkpoints/vit_hn.pt")
    parser.add_argument("--gallery", default="logodet3k")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()

    print("Loading pipeline...")
    pipeline = LogoRecognitionPipeline(
        detector_weights=args.detector,
        embedder_ckpt=args.embedder,
        gallery_name=args.gallery,
        conf=args.conf,
    )

    for img_path in args.images:
        print(f"\nImage: {img_path}")
        results = pipeline.predict(img_path)
        if not results:
            print("  No logos detected.")
            continue
        for r in results:
            b = r["box"]
            print(f"  box [{b['x1']:.0f},{b['y1']:.0f},{b['x2']:.0f},{b['y2']:.0f}] "
                  f"→ brand: {r['brand']}  score: {r['score']:.4f}")

        if args.save_dir:
            out = Path(args.save_dir) / (Path(img_path).stem + "_result.jpg")
            Path(args.save_dir).mkdir(exist_ok=True)
            draw_results(img_path, results, str(out))
        else:
            draw_results(img_path, results)
