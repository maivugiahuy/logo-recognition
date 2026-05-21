"""
Add new classes to the gallery without retraining.

Each subfolder in folder_root = 1 class (subfolder name = class name).

Usage:
    # Run from data/new_classes/ (default)
    python scripts/09_add_classes.py

    # Specify a different folder_root
    python scripts/09_add_classes.py --folder_root path/to/brands/

    # Use YOLO to auto-detect logos
    python scripts/09_add_classes.py --use_detector

    # List classes in gallery
    python scripts/09_add_classes.py --list

    # Remove a class from gallery
    python scripts/09_add_classes.py --remove pepsi
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from src.retrieval.gallery import GALLERY_DIR, add_to_gallery, check_duplicate
from src.utils.logging_utils import setup_logging

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def list_classes(dataset_name: str) -> None:
    labels_path = GALLERY_DIR / f"{dataset_name}_labels.json"
    if not labels_path.exists():
        print(f"Gallery '{dataset_name}' not found. Run 08_build_galleries.py first.")
        return
    with open(labels_path) as f:
        labels = json.load(f)
    counts = Counter(labels)
    print(f"Gallery '{dataset_name}': {len(counts)} classes, {len(labels)} vectors\n")
    for cls, count in sorted(counts.items()):
        print(f"  {cls}: {count} images")


def add_with_detector(
    image_paths: list[Path],
    class_name: str,
    detector_weights: str,
    conf: float,
    **kwargs,
) -> None:
    """Run YOLO detection on each image, crop detected logos, then embed into the gallery."""
    from src.detector.detect import LogoDetector
    from PIL import Image
    import tempfile, shutil

    detector = LogoDetector(weights=detector_weights, conf=conf)
    tmp_dir = Path(tempfile.mkdtemp())
    cropped_paths = []

    print(f"  Detecting logos in {len(image_paths)} images...")
    for img_path in image_paths:
        boxes = detector.detect(img_path)
        if not boxes:
            print(f"    [SKIP] {img_path.name}: no logo detected")
            continue
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        for i, box in enumerate(boxes):
            x1 = max(0, int(box["x1"]))
            y1 = max(0, int(box["y1"]))
            x2 = min(w, int(box["x2"]))
            y2 = min(h, int(box["y2"]))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img.crop((x1, y1, x2, y2))
            out = tmp_dir / f"{img_path.stem}_crop{i}.jpg"
            crop.save(out)
            cropped_paths.append(out)
        print(f"    {img_path.name}: {len(boxes)} logo(s) detected")

    if not cropped_paths:
        print(f"  [WARN] No logos detected in any image for '{class_name}'")
        shutil.rmtree(tmp_dir)
        return

    print(f"  → Embedding {len(cropped_paths)} crop(s) cho class '{class_name}'...")
    add_to_gallery(image_paths=cropped_paths, brand_name=class_name, **kwargs)
    shutil.rmtree(tmp_dir)


# (checkpoint, backbone, gallery_suffix)
_MODEL_PRESETS = {
    "b16_hn":   ("checkpoints/vit_b16_arcface_hn.pt",  "vit_b16_openai", ""),
    "b16_base": ("checkpoints/vit_b16_arcface_base.pt", "vit_b16_openai", "_b16_base"),
    "dinov3":   ("checkpoints/dinov3_arcface_base.pt",  "dinov3_vitb16",  "_dinov3"),
    "b32_hn":   ("checkpoints/vit_hn.pt",               "vit_b32_openai", "_b32"),
    "b32_base": ("checkpoints/vit_base.pt",             "vit_b32_openai", "_b32_base"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add new classes to the gallery from data/new_classes/ (each subfolder = 1 class)"
    )
    parser.add_argument("--folder_root", default="data/new_classes",
                        help="Parent folder containing class subfolders (default: data/new_classes)")
    parser.add_argument("--use_detector", action="store_true",
                        help="Use YOLO to auto-detect logos instead of using the full image")
    parser.add_argument("--detector", default="runs/detect/checkpoints/yolov8_logo/weights/best.pt",
                        help="Path to YOLO weights")
    parser.add_argument("--det_conf", type=float, default=0.1,
                        help="YOLO confidence threshold (default: 0.1)")
    parser.add_argument("--gallery", default="new_classes",
                        help="Gallery to update (default: new_classes; use 'openlogodet3k' for eval gallery)")
    parser.add_argument("--model", default=None, choices=list(_MODEL_PRESETS),
                        help="Model preset — overrides --embedder/--backbone. "
                             "Choices: b16_hn (default best), b16_base, dinov3, b32_hn, b32_base")
    parser.add_argument("--embedder", default="checkpoints/vit_b16_arcface_hn.pt")
    parser.add_argument("--backbone", default="vit_b16_openai",
                        choices=["vit_b16_openai", "vit_b32_openai", "dinov2_vitb14", "dinov3_vitb16"],
                        help="Embedder backbone matching the checkpoint (default: vit_b16_openai)")
    parser.add_argument("--suffix", default=None,
                        help="Gallery name suffix, e.g. '_dinov3' → 'new_classes_dinov3' (auto-set by --model)")
    parser.add_argument("--on_duplicate", default="ask",
                        choices=["ask", "append", "replace", "skip"],
                        help="How to handle existing class: ask/append/replace/skip (default: ask)")
    parser.add_argument("--list", action="store_true",
                        help="List classes in gallery")
    parser.add_argument("--remove", default=None, metavar="CLASS",
                        help="Remove a class from gallery")
    args = parser.parse_args()

    if args.model:
        args.embedder, args.backbone, auto_suffix = _MODEL_PRESETS[args.model]
        if args.suffix is None:
            args.suffix = auto_suffix

    suffix = args.suffix if args.suffix is not None else ""
    args.gallery = args.gallery + suffix

    setup_logging(__file__)

    if args.list:
        list_classes(args.gallery)
        sys.exit(0)

    if args.remove:
        from src.retrieval.gallery import remove_from_gallery
        remove_from_gallery(args.remove, args.gallery)
        sys.exit(0)

    common_kwargs = dict(dataset_name=args.gallery, ckpt_path=args.embedder, backbone=args.backbone)
    detector_kwargs = dict(detector_weights=args.detector, conf=args.det_conf)

    def resolve_duplicate_action(class_name: str) -> str:
        if args.on_duplicate != "ask":
            return args.on_duplicate
        existing = check_duplicate(class_name, args.gallery)
        if existing == 0:
            return "append"
        print(f"\n⚠️  Class '{class_name}' already has {existing} images in gallery.")
        print("   [1] append  — add new images alongside existing (increases coverage)")
        print("   [2] replace — remove existing images, use new ones only")
        print("   [3] skip    — do nothing")
        while True:
            choice = input("   Choose (1/2/3): ").strip()
            if choice == "1":
                return "append"
            elif choice == "2":
                return "replace"
            elif choice == "3":
                return "skip"
            print("   Enter 1, 2, or 3.")

    def run_add(imgs: list[Path], class_name: str) -> None:
        action = resolve_duplicate_action(class_name)
        kwargs = {**common_kwargs, "on_duplicate": action}
        if args.use_detector:
            add_with_detector(imgs, class_name, **detector_kwargs, **kwargs)
        else:
            add_to_gallery(image_paths=imgs, brand_name=class_name, **kwargs)

    root = Path(args.folder_root)
    if not root.exists():
        print(f"Folder '{root}' does not exist.")
        print(f"Create subfolders for each class inside it (e.g. {root}/nike/, {root}/adidas/)")
        sys.exit(1)

    subfolders = sorted(p for p in root.iterdir() if p.is_dir())
    if not subfolders:
        print(f"No subfolders found in {root}")
        print("Each subfolder = 1 class (subfolder name = class name)")
        sys.exit(1)

    print(f"Found {len(subfolders)} class(es) in {root}\n")
    for subfolder in subfolders:
        imgs = collect_images(subfolder)
        if not imgs:
            print(f"  [SKIP] {subfolder.name}: no images found")
            continue
        print(f"\n── Class: {subfolder.name} ({len(imgs)} images) ──")
        run_add(imgs, subfolder.name)
