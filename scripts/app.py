"""
Web demo for logo recognition pipeline.
Run from repo root:  python scripts/app.py
"""
import sys
sys.path.insert(0, ".")

import json

import faiss
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.retrieval.gallery import GALLERY_DIR
from src.retrieval.pipeline import LogoRecognitionPipeline

DETECTOR = "runs/detect/checkpoints/yolov8_logo/weights/best.pt"

MODEL_PRESETS = {
    "ViT-B/16 ArcFace HN (best)": ("checkpoints/vit_b16_arcface_hn.pt",  "vit_b16_openai",  160),
    "ViT-B/16 ArcFace Base":      ("checkpoints/vit_b16_arcface_base.pt", "vit_b16_openai",  160),
    "DINOv3-B/16 ArcFace Base":   ("checkpoints/dinov3_arcface_base.pt",  "dinov3_vitb16",   160),
    "ViT-B/32 ArcFace HN":        ("checkpoints/vit_hn.pt",               "vit_b32_openai",  160),
    "ViT-B/32 ArcFace Base":      ("checkpoints/vit_base.pt",             "vit_b32_openai",  160),
}

GALLERIES = ["openlogodet3k", "new_classes"]

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


def _ensure_pipeline(model_name: str, gallery_name: str, conf: float, threshold: float):
    key = (model_name, gallery_name)
    if _state["pipeline"] is None or _state["pipeline_key"] != key:
        ckpt, backbone, input_size = MODEL_PRESETS[model_name]
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
    embs = _state["gallery_embs"]
    labels = _state["gallery_labels_full"]
    if embs is None or not labels or not selected:
        return None, None
    sel_set = set(selected)
    mask = [i for i, l in enumerate(labels) if l in sel_set]
    if not mask:
        return None, None
    sub_embs = embs[mask].astype("float32")
    sub_labels = [labels[i] for i in mask]
    sub_idx = faiss.IndexFlatIP(embs.shape[1])
    sub_idx.add(sub_embs)
    return sub_idx, sub_labels


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

def on_gallery_change(gallery_name: str):
    classes = _load_gallery_meta(gallery_name)
    return gr.update(choices=classes, value=classes)


def cb_select_all(gallery_name: str):
    return gr.update(value=list(_state["gallery_unique"] or []))


def cb_clear():
    return gr.update(value=[])


def run_inference(image_path, model_name, gallery_name, selected_classes, conf, threshold):
    if image_path is None:
        return None, "Upload an image first.", []
    # Empty selection = use full gallery (no sub-index)
    if not selected_classes:
        selected_classes = list(_state["gallery_unique"] or [])

    pipeline = _ensure_pipeline(model_name, gallery_name, conf, threshold)

    all_unique = set(_state["gallery_unique"] or [])
    use_full = set(selected_classes) == all_unique

    orig_index = orig_labels = None
    if not use_full:
        sub_idx, sub_labels = _build_sub_index(selected_classes)
        if sub_idx is not None:
            orig_index, orig_labels = pipeline.index, pipeline.gallery_labels
            pipeline.index = sub_idx
            pipeline.gallery_labels = sub_labels

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
    info = (
        f"{len(results)} logo(s)  |  "
        f"det {det:.0f} ms  ·  rec {rec:.0f} ms  ·  total {det + rec:.0f} ms  |  "
        f"{len(selected_classes)}/{len(all_unique)} classes active"
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


# --- Layout ---

with gr.Blocks(title="Logo Recognition Demo") as demo:
    gr.Markdown(
        "# Logo Recognition Demo\n"
        "ViT-B/16 ArcFace · LogoDet-3K + OpenLogo · 2 400+ brands"
    )

    with gr.Row():
        # ── Left panel ──────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=320):
            image_input = gr.Image(type="filepath", label="Input Image")
            with gr.Group():
                model_dd = gr.Dropdown(
                    choices=list(MODEL_PRESETS),
                    value="ViT-B/16 ArcFace HN (best)",
                    label="Model",
                )
                gallery_dd = gr.Dropdown(
                    choices=GALLERIES,
                    value="openlogodet3k",
                    label="Gallery",
                )
            conf_sl = gr.Slider(0.05, 0.90, value=0.10, step=0.05, label="Detection Confidence")
            thr_sl  = gr.Slider(0.00, 1.00, value=0.50, step=0.05, label="Unknown Threshold")
            run_btn = gr.Button("Run Recognition", variant="primary", size="lg")
            with gr.Group():
                with gr.Row():
                    btn_all = gr.Button("Select All", size="sm", variant="secondary")
                    btn_clr = gr.Button("Clear",      size="sm", variant="secondary")
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

    # ── Events ──────────────────────────────────────────────────────────────
    demo.load(fn=on_gallery_change,  inputs=gallery_dd,  outputs=class_select)
    gallery_dd.change(fn=on_gallery_change, inputs=gallery_dd, outputs=class_select)
    btn_all.click(fn=cb_select_all,  inputs=gallery_dd,  outputs=class_select)
    btn_clr.click(fn=cb_clear,                           outputs=class_select)
    run_btn.click(
        fn=run_inference,
        inputs=[image_input, model_dd, gallery_dd, class_select, conf_sl, thr_sl],
        outputs=[image_output, info_box, result_tbl],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
