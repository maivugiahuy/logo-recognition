"""
Web demo for logo recognition pipeline.
Run from repo root:  python scripts/app.py
"""
import sys
sys.path.insert(0, ".")

import json
import shutil
import tempfile
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.retrieval.gallery import GALLERY_DIR, add_to_gallery
from src.retrieval.pipeline import LogoRecognitionPipeline

DETECTOR = "runs/detect/checkpoints/yolov8_logo/weights/best.pt"

# (ckpt, backbone, input_size, gallery)
MODEL_PRESETS = {
    "ViT-B/16 ArcFace HN": ("checkpoints/vit_b16_arcface_hn.pt", "vit_b16_openai", 160, "openlogodet3k"),
}

_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9",
]
_UNKNOWN_COLOR = "#ff6600"
_brand_colors: dict = {}


def _brand_color(brand: str) -> str:
    if brand not in _brand_colors:
        _brand_colors[brand] = _PALETTE[len(_brand_colors) % len(_PALETTE)]
    return _brand_colors[brand]


_state: dict = {
    "pipeline": None,
    "pipeline_key": None,
    "gallery_embs": None,
    "gallery_labels_full": None,
    "gallery_unique": None,
}

# new_classes gallery (per-model suffix: "" for b16, "_b32" for b32)
_nc_state: dict = {"embs": None, "labels": None}

# new_classes gallery suffix per model
_NC_SUFFIX = {
    "ViT-B/16 ArcFace HN": "",
}


def _nc_gallery_name(model_name: str) -> str:
    return "new_classes" + _NC_SUFFIX.get(model_name, "")


def _reload_nc(model_name: str) -> None:
    """Load new_classes gallery for current model into _nc_state."""
    gname = _nc_gallery_name(model_name)
    idx_path = GALLERY_DIR / f"{gname}.faiss"
    lbl_path = GALLERY_DIR / f"{gname}_labels.json"
    if not idx_path.exists():
        _nc_state["embs"] = None
        _nc_state["labels"] = None
        return
    idx = faiss.read_index(str(idx_path))
    embs = np.zeros((idx.ntotal, idx.d), dtype="float32")
    if idx.ntotal > 0:
        idx.reconstruct_n(0, idx.ntotal, embs)
    with open(lbl_path) as f:
        _nc_state["labels"] = json.load(f)
    _nc_state["embs"] = embs


def _load_gallery_meta(gallery_name: str) -> list[str]:
    """Reconstruct embedding matrix + label list; return sorted unique class names."""
    index_path = GALLERY_DIR / f"{gallery_name}.faiss"
    labels_path = GALLERY_DIR / f"{gallery_name}_labels.json"
    if not index_path.exists() or not labels_path.exists():
        _state.update(gallery_embs=None, gallery_labels_full=[], gallery_unique=[])
        return []
    idx = faiss.read_index(str(index_path))
    embs = np.zeros((idx.ntotal, idx.d), dtype="float32")
    if idx.ntotal > 0:
        idx.reconstruct_n(0, idx.ntotal, embs)
    with open(labels_path) as f:
        labels = json.load(f)
    _state["gallery_embs"] = embs
    _state["gallery_labels_full"] = labels
    _state["gallery_unique"] = sorted(set(labels))
    return _state["gallery_unique"]


def _ensure_pipeline(model_name: str, conf: float, threshold: float):
    ckpt, backbone, input_size, gallery_name = MODEL_PRESETS[model_name]
    key = model_name
    if _state["pipeline"] is None or _state["pipeline_key"] != key:
        _state["pipeline"] = LogoRecognitionPipeline(
            detector_weights=DETECTOR,
            embedder_ckpt=ckpt,
            gallery_name=gallery_name,
            backbone=backbone,
            conf=conf,
            input_size=input_size,
            unknown_threshold=threshold,
        )
        _state["pipeline_key"] = key
    else:
        _state["pipeline"].detector.conf = conf
        _state["pipeline"].unknown_threshold = threshold
    return _state["pipeline"]


def _build_sub_index(selected: list[str]):
    """Build FAISS sub-index from selected classes (main gallery + new_classes filtered by selection)."""
    if not selected:
        return None, None
    sel_set = set(selected)
    sub_embs = []
    sub_labels = []

    # Main gallery subset
    embs = _state["gallery_embs"]
    labels = _state["gallery_labels_full"]
    if embs is not None and labels:
        mask = [i for i, l in enumerate(labels) if l in sel_set]
        if mask:
            sub_embs.append(embs[mask].astype("float32"))
            sub_labels += [labels[i] for i in mask]

    # new_classes subset (only selected brands, not all)
    if _nc_state["embs"] is not None and _nc_state["labels"]:
        nc_mask = [i for i, l in enumerate(_nc_state["labels"]) if l in sel_set]
        if nc_mask:
            sub_embs.append(_nc_state["embs"][nc_mask].astype("float32"))
            sub_labels += [_nc_state["labels"][i] for i in nc_mask]

    if not sub_embs:
        return None, None
    all_embs = np.concatenate(sub_embs)
    sub_idx = faiss.IndexFlatIP(all_embs.shape[1])
    sub_idx.add(all_embs)
    return sub_idx, sub_labels


def _build_merged_index(pipeline):
    """Merge pipeline gallery with new_classes. Returns (index, labels) or None."""
    if _nc_state["embs"] is None or not _nc_state["labels"]:
        return None, None
    orig_embs = np.zeros((pipeline.index.ntotal, pipeline.index.d), dtype="float32")
    pipeline.index.reconstruct_n(0, pipeline.index.ntotal, orig_embs)
    merged_embs = np.concatenate([orig_embs, _nc_state["embs"].astype("float32")])
    merged_labels = list(pipeline.gallery_labels) + list(_nc_state["labels"])
    merged_idx = faiss.IndexFlatIP(merged_embs.shape[1])
    merged_idx.add(merged_embs)
    return merged_idx, merged_labels


def draw_results(image: Image.Image, results: list) -> Image.Image:
    img = image.copy()
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
        try:
            bb = font.getbbox(label)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = len(label) * 12, 16
        tx = int(b["x1"])
        ty = max(0, int(b["y1"]) - th - 6)
        draw.rectangle([tx, ty, tx + tw + 6, ty + th + 6], fill=color)
        draw.text((tx + 3, ty + 3), label, fill="white", font=font)
    return img


# --- Gradio callbacks ---

def on_model_change(model_name: str):
    _, _, _, gallery_name = MODEL_PRESETS[model_name]
    classes = _load_gallery_meta(gallery_name)
    _reload_nc(model_name)
    nc_brands = sorted(set(_nc_state["labels"])) if _nc_state["labels"] else []
    all_choices = sorted(set(classes + nc_brands))
    return gr.update(choices=all_choices, value=[])


def cb_clear():
    return gr.update(value=[])


def run_inference(image_path, model_name, selected_classes, conf, threshold):
    if image_path is None:
        return None, "Upload an image first.", []
    pipeline = _ensure_pipeline(model_name, conf, threshold)

    all_unique = set(_state["gallery_unique"] or [])
    use_full = not selected_classes or set(selected_classes) == all_unique
    has_nc = _nc_state["embs"] is not None and bool(_nc_state["labels"])

    # Normalize: empty = full gallery
    if not selected_classes:
        selected_classes = list(all_unique)

    orig_index = orig_labels = None
    if not use_full:
        # Subset selected: build sub-index (includes new_classes)
        sub_idx, sub_labels = _build_sub_index(selected_classes)
        if sub_idx is not None:
            orig_index, orig_labels = pipeline.index, pipeline.gallery_labels
            pipeline.index = sub_idx
            pipeline.gallery_labels = sub_labels
    elif has_nc:
        # Full gallery + merge new_classes
        merged_idx, merged_labels = _build_merged_index(pipeline)
        if merged_idx is not None:
            orig_index, orig_labels = pipeline.index, pipeline.gallery_labels
            pipeline.index = merged_idx
            pipeline.gallery_labels = merged_labels

    try:
        results, timing = pipeline.predict(image_path, return_timing=True)
    finally:
        if orig_index is not None:
            pipeline.index = orig_index
            pipeline.gallery_labels = orig_labels

    image = Image.open(image_path).convert("RGB")
    annotated = draw_results(image, results)

    det = timing["detection_ms"]
    rec = timing["recognition_ms"]
    nc_count = len(set(_nc_state["labels"])) if _nc_state["labels"] else 0
    nc_str = f" + {nc_count} new" if nc_count else ""
    info = (
        f"{len(results)} logo(s)  |  "
        f"det {det:.0f} ms  ·  rec {rec:.0f} ms  ·  total {det + rec:.0f} ms  |  "
        f"{len(selected_classes)}/{len(all_unique)} classes{nc_str}"
    )

    table = [
        [
            r["brand"],
            f"{r['score']:.4f}",
            "unknown" if r["is_unknown"] else "recognized",
            f"[{r['box']['x1']:.0f}, {r['box']['y1']:.0f}, {r['box']['x2']:.0f}, {r['box']['y2']:.0f}]",
        ]
        for r in results
    ]

    return annotated, info, table


def add_new_brand(brand_name: str, files, on_duplicate: str, model_name: str):
    if not brand_name or not brand_name.strip():
        return "Enter a brand name.", gr.update(), gr.update()
    if not files:
        return "Upload at least one image.", gr.update(), gr.update()
    brand_name = brand_name.strip()
    ckpt, backbone, input_size, _ = MODEL_PRESETS[model_name]
    nc_name = _nc_gallery_name(model_name)
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    # files may be list of str paths (Gradio File component)
    image_paths = [Path(f) for f in (files if isinstance(files, list) else [files])]

    try:
        add_to_gallery(
            image_paths=image_paths,
            brand_name=brand_name,
            dataset_name=nc_name,
            ckpt_path=ckpt,
            backbone=backbone,
            input_size=input_size,
            on_duplicate=on_duplicate,
        )
    except Exception as e:
        return f"Error: {e}", gr.update(), gr.update()

    _reload_nc(model_name)
    nc_brands = sorted(set(_nc_state["labels"])) if _nc_state["labels"] else []
    status = f"Added '{brand_name}' to {nc_name}. Gallery now has {len(nc_brands)} brand(s)."

    # Merge new_classes into class_select choices
    main_classes = list(_state["gallery_unique"] or [])
    all_choices = sorted(set(main_classes + nc_brands))
    return status, gr.update(value="\n".join(nc_brands)), gr.update(choices=all_choices)


# --- Layout ---

_DEFAULT_MODEL = "ViT-B/16 ArcFace HN"

# Delete all new_classes gallery files on startup
for _suffix in _NC_SUFFIX.values():
    _nc_name = "new_classes" + _suffix
    for _ext in (".faiss", "_labels.json"):
        _p = GALLERY_DIR / f"{_nc_name}{_ext}"
        if _p.exists():
            _p.unlink()

with gr.Blocks(title="Logo Recognition Demo") as demo:
    gr.Markdown(
        "# Logo Recognition Demo\n"
        "LogoDet-3K + OpenLogo · 2 400+ brands"
    )

    with gr.Row():
        # ── Left panel ──────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=320):
            image_input = gr.Image(type="filepath", label="Input Image", sources=["upload"])
            model_dd = gr.Dropdown(
                choices=list(MODEL_PRESETS),
                value=_DEFAULT_MODEL,
                label="Model",
            )
            conf_sl = gr.Slider(0.05, 0.90, value=0.10, step=0.05, label="Detection Confidence")
            thr_sl  = gr.Slider(0.00, 1.00, value=0.50, step=0.05, label="Unknown Threshold")
            run_btn = gr.Button("Run Recognition", variant="primary", size="lg")
            with gr.Group():
                btn_clr = gr.Button("Clear", size="sm", variant="secondary")
                class_select = gr.Dropdown(
                    choices=[],
                    multiselect=True,
                    label="Active Classes  (type to filter)",
                    info="Leave empty = use full gallery",
                )

        # ── Right panel ─────────────────────────────────────────────────────
        with gr.Column(scale=2):
            image_output = gr.Image(label="Annotated Result")
            info_box = gr.Textbox(label="Info", interactive=False)
            result_tbl = gr.Dataframe(
                headers=["Brand", "Score", "Status", "Box [x1,y1,x2,y2]"],
                label="Detections",
                wrap=True,
            )
            with gr.Accordion("Add New Brand", open=False):
                gr.Markdown(
                    "Upload logo images for a brand not in the main gallery. "
                    "Embedded with active model → saved to `new_classes` gallery → "
                    "automatically included in all future recognition runs."
                )
                with gr.Row():
                    brand_name_in = gr.Textbox(label="Brand Name", placeholder="e.g. my_brand")
                    dup_dd = gr.Dropdown(
                        ["append", "replace", "skip"],
                        value="append",
                        label="If brand exists",
                    )
                brand_files = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="Brand Images (jpg/png/webp)",
                )
                add_btn    = gr.Button("Add to Gallery", variant="secondary")
                add_status = gr.Textbox(label="Status", interactive=False)
                nc_brands  = gr.Textbox(
                    label="Brands in new_classes gallery",
                    interactive=False,
                    lines=3,
                )

    # ── Events ──────────────────────────────────────────────────────────────
    demo.load(fn=on_model_change,  inputs=model_dd,  outputs=class_select)
    model_dd.change(fn=on_model_change, inputs=model_dd, outputs=class_select)
    btn_clr.click(fn=cb_clear, outputs=class_select)
    run_btn.click(
        fn=run_inference,
        inputs=[image_input, model_dd, class_select, conf_sl, thr_sl],
        outputs=[image_output, info_box, result_tbl],
    )
    add_btn.click(
        fn=add_new_brand,
        inputs=[brand_name_in, brand_files, dup_dd, model_dd],
        outputs=[add_status, nc_brands, class_select],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
