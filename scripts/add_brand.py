"""
Thêm brand mới vào gallery mà không cần train lại.

Usage:
    # Ảnh đã crop sẵn logo → dùng folder
    python scripts/add_brand.py --brand pepsi --folder path/to/pepsi_logos/

    # Ảnh thực tế (có cảnh xung quanh) → YOLO tự detect logo
    python scripts/add_brand.py --brand pepsi --folder photos/ --use_detector

    # Nhiều brand cùng lúc (mỗi subfolder = 1 brand)
    python scripts/add_brand.py --folder_root path/to/brands/ [--use_detector]

    # Danh sách ảnh lẻ
    python scripts/add_brand.py --brand pepsi --images logo1.jpg logo2.jpg

    # Xem brands trong gallery
    python scripts/add_brand.py --list
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from src.retrieval.gallery import GALLERY_DIR, add_to_gallery, check_duplicate

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def list_brands(dataset_name: str) -> None:
    labels_path = GALLERY_DIR / f"{dataset_name}_labels.json"
    if not labels_path.exists():
        print(f"Gallery '{dataset_name}' chưa được build. Chạy 06_build_galleries.py trước.")
        return
    with open(labels_path) as f:
        labels = json.load(f)
    counts = Counter(labels)
    print(f"Gallery '{dataset_name}': {len(counts)} brands, {len(labels)} vectors\n")
    for brand, count in sorted(counts.items()):
        print(f"  {brand}: {count} ảnh")


def add_with_detector(
    image_paths: list[Path],
    brand_name: str,
    detector_weights: str,
    conf: float,
    **kwargs,
) -> None:
    """Chạy YOLO detect trên từng ảnh, crop logo tìm được, rồi embed vào gallery."""
    from src.detector.detect import LogoDetector
    from src.retrieval.gallery import add_to_gallery
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
        print(f"  [WARN] Không tìm thấy logo nào trong toàn bộ ảnh của '{brand_name}'")
        shutil.rmtree(tmp_dir)
        return

    print(f"  → Embedding {len(cropped_paths)} crop(s) cho brand '{brand_name}'...")
    add_to_gallery(image_paths=cropped_paths, brand_name=brand_name, **kwargs)
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", default=None, help="Tên brand (dùng với --folder hoặc --images)")
    parser.add_argument("--folder", default=None, help="Folder chứa ảnh logo của 1 brand")
    parser.add_argument("--folder_root", default=None,
                        help="Folder cha chứa nhiều subfolder, mỗi subfolder = 1 brand")
    parser.add_argument("--images", nargs="+", default=[], help="Danh sách ảnh lẻ")
    parser.add_argument("--crop", nargs=4, type=int, default=None,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="Crop box thủ công áp dụng cho tất cả ảnh")
    parser.add_argument("--use_detector", action="store_true",
                        help="Dùng YOLO detect logo tự động thay vì dùng toàn ảnh")
    parser.add_argument("--detector", default="runs/detect/checkpoints/yolov8_logo/weights/best.pt",
                        help="Path tới YOLO weights")
    parser.add_argument("--det_conf", type=float, default=0.1,
                        help="YOLO confidence threshold (default: 0.1)")
    parser.add_argument("--gallery", default="logodet3k", help="Tên gallery cần update")
    parser.add_argument("--embedder", default="checkpoints/vit_hn.pt")
    parser.add_argument("--on_duplicate", default="ask",
                        choices=["ask", "append", "replace", "skip"],
                        help="Xử lý khi brand đã tồn tại: ask/append/replace/skip (default: ask)")
    parser.add_argument("--list", action="store_true", help="Liệt kê brands trong gallery")
    parser.add_argument("--remove", default=None, metavar="BRAND",
                        help="Xóa brand khỏi gallery")
    args = parser.parse_args()

    if args.list:
        list_brands(args.gallery)
        sys.exit(0)

    if args.remove:
        from src.retrieval.gallery import remove_from_gallery
        remove_from_gallery(args.remove, args.gallery)
        sys.exit(0)

    crop_box = tuple(args.crop) if args.crop else None
    common_kwargs = dict(dataset_name=args.gallery, ckpt_path=args.embedder, crop_box=crop_box)
    detector_kwargs = dict(detector_weights=args.detector, conf=args.det_conf)

    def resolve_duplicate_action(brand: str) -> str:
        """Nếu --on_duplicate=ask thì hỏi user, ngược lại dùng giá trị CLI."""
        if args.on_duplicate != "ask":
            return args.on_duplicate
        existing = check_duplicate(brand, args.gallery)
        if existing == 0:
            return "append"  # brand mới, không có gì để hỏi
        print(f"\n⚠️  Brand '{brand}' đã có {existing} ảnh trong gallery.")
        print("   [1] append  — thêm ảnh mới vào bên cạnh (tăng coverage)")
        print("   [2] replace — xóa ảnh cũ, dùng ảnh mới hoàn toàn")
        print("   [3] skip    — bỏ qua, không thay đổi gì")
        while True:
            choice = input("   Chọn (1/2/3): ").strip()
            if choice == "1":
                return "append"
            elif choice == "2":
                return "replace"
            elif choice == "3":
                return "skip"
            print("   Nhập 1, 2, hoặc 3.")

    def run_add(imgs: list[Path], brand: str) -> None:
        action = resolve_duplicate_action(brand)
        kwargs = {**common_kwargs, "on_duplicate": action}
        if args.use_detector:
            add_with_detector(imgs, brand, **detector_kwargs, **kwargs)
        else:
            add_to_gallery(image_paths=imgs, brand_name=brand, **kwargs)

    # ── Nhiều brand từ folder_root ─────────────────────────────────────────
    if args.folder_root:
        root = Path(args.folder_root)
        subfolders = sorted(p for p in root.iterdir() if p.is_dir())
        if not subfolders:
            print(f"Không tìm thấy subfolder nào trong {root}")
            sys.exit(1)
        print(f"Tìm thấy {len(subfolders)} brand(s) trong {root}\n")
        for subfolder in subfolders:
            imgs = collect_images(subfolder)
            if not imgs:
                print(f"  [SKIP] {subfolder.name}: không có ảnh")
                continue
            print(f"\n── Brand: {subfolder.name} ({len(imgs)} ảnh) ──")
            run_add(imgs, subfolder.name)
        sys.exit(0)

    # ── 1 brand từ folder ─────────────────────────────────────────────────
    if args.folder:
        folder = Path(args.folder)
        imgs = collect_images(folder)
        brand = args.brand or folder.name
        if not imgs:
            print(f"Không tìm thấy ảnh trong {folder}")
            sys.exit(1)
        print(f"Brand '{brand}': {len(imgs)} ảnh từ {folder}")
        run_add(imgs, brand)
        sys.exit(0)

    # ── Danh sách ảnh lẻ ──────────────────────────────────────────────────
    if args.images:
        if not args.brand:
            print("Cần truyền --brand khi dùng --images")
            sys.exit(1)
        run_add([Path(p) for p in args.images], args.brand)
        sys.exit(0)

    parser.print_help()

