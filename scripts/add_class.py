"""
Thêm class mới vào gallery mà không cần train lại.

Mỗi subfolder trong folder_root = 1 class (tên subfolder = tên class).

Usage:
    # Chạy từ data/new_classes/ (mặc định)
    python scripts/add_class.py

    # Chỉ định folder_root khác
    python scripts/add_class.py --folder_root path/to/brands/

    # Dùng YOLO detect logo tự động
    python scripts/add_class.py [--use_detector]

    # Xem classes trong gallery
    python scripts/add_class.py --list

    # Xóa class khỏi gallery
    python scripts/add_class.py --remove pepsi
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


def list_classes(dataset_name: str) -> None:
    labels_path = GALLERY_DIR / f"{dataset_name}_labels.json"
    if not labels_path.exists():
        print(f"Gallery '{dataset_name}' chưa được build. Chạy 06_build_galleries.py trước.")
        return
    with open(labels_path) as f:
        labels = json.load(f)
    counts = Counter(labels)
    print(f"Gallery '{dataset_name}': {len(counts)} classes, {len(labels)} vectors\n")
    for cls, count in sorted(counts.items()):
        print(f"  {cls}: {count} ảnh")


def add_with_detector(
    image_paths: list[Path],
    class_name: str,
    detector_weights: str,
    conf: float,
    **kwargs,
) -> None:
    """Chạy YOLO detect trên từng ảnh, crop logo tìm được, rồi embed vào gallery."""
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
        print(f"  [WARN] Không tìm thấy logo nào trong toàn bộ ảnh của '{class_name}'")
        shutil.rmtree(tmp_dir)
        return

    print(f"  → Embedding {len(cropped_paths)} crop(s) cho class '{class_name}'...")
    add_to_gallery(image_paths=cropped_paths, brand_name=class_name, **kwargs)
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thêm class mới vào gallery từ data/new_classes/ (mỗi subfolder = 1 class)"
    )
    parser.add_argument("--folder_root", default="data/new_classes",
                        help="Folder cha chứa các subfolder class (default: data/new_classes)")
    parser.add_argument("--use_detector", action="store_true",
                        help="Dùng YOLO detect logo tự động thay vì dùng toàn ảnh")
    parser.add_argument("--detector", default="runs/detect/checkpoints/yolo26m_logo/weights/best.pt",
                        help="Path tới YOLO weights")
    parser.add_argument("--det_conf", type=float, default=0.1,
                        help="YOLO confidence threshold (default: 0.1)")
    parser.add_argument("--gallery", default="openlogodet3k",
                        help="Tên gallery cần update")
    parser.add_argument("--embedder", default="checkpoints/vit_hn.pt")
    parser.add_argument("--on_duplicate", default="ask",
                        choices=["ask", "append", "replace", "skip"],
                        help="Xử lý khi class đã tồn tại: ask/append/replace/skip (default: ask)")
    parser.add_argument("--list", action="store_true",
                        help="Liệt kê classes trong gallery")
    parser.add_argument("--remove", default=None, metavar="CLASS",
                        help="Xóa class khỏi gallery")
    args = parser.parse_args()

    if args.list:
        list_classes(args.gallery)
        sys.exit(0)

    if args.remove:
        from src.retrieval.gallery import remove_from_gallery
        remove_from_gallery(args.remove, args.gallery)
        sys.exit(0)

    common_kwargs = dict(dataset_name=args.gallery, ckpt_path=args.embedder)
    detector_kwargs = dict(detector_weights=args.detector, conf=args.det_conf)

    def resolve_duplicate_action(class_name: str) -> str:
        if args.on_duplicate != "ask":
            return args.on_duplicate
        existing = check_duplicate(class_name, args.gallery)
        if existing == 0:
            return "append"
        print(f"\n⚠️  Class '{class_name}' đã có {existing} ảnh trong gallery.")
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

    def run_add(imgs: list[Path], class_name: str) -> None:
        action = resolve_duplicate_action(class_name)
        kwargs = {**common_kwargs, "on_duplicate": action}
        if args.use_detector:
            add_with_detector(imgs, class_name, **detector_kwargs, **kwargs)
        else:
            add_to_gallery(image_paths=imgs, brand_name=class_name, **kwargs)

    root = Path(args.folder_root)
    if not root.exists():
        print(f"Folder '{root}' không tồn tại.")
        print(f"Tạo các subfolder cho mỗi class bên trong (VD: {root}/nike/, {root}/adidas/)")
        sys.exit(1)

    subfolders = sorted(p for p in root.iterdir() if p.is_dir())
    if not subfolders:
        print(f"Không tìm thấy subfolder nào trong {root}")
        print("Mỗi subfolder = 1 class (tên subfolder = tên class)")
        sys.exit(1)

    print(f"Tìm thấy {len(subfolders)} class(es) trong {root}\n")
    for subfolder in subfolders:
        imgs = collect_images(subfolder)
        if not imgs:
            print(f"  [SKIP] {subfolder.name}: không có ảnh")
            continue
        print(f"\n── Class: {subfolder.name} ({len(imgs)} ảnh) ──")
        run_add(imgs, subfolder.name)
