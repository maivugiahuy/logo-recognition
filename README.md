# Logo Recognition

## What gets trained

### ViT-B/16 pipeline (main)

| Component | Loss | Data split | Output |
|---|---|---|---|
| ViT-B/16 embedder Phase A | ArcFace (K=1) | Open-set train classes (~1600) | `checkpoints/vit_b16_arcface_base.pt` |
| ViT-B/16 embedder Phase C | Sub-center ArcFace (K=3) | Closed-set train classes (~1900) | `checkpoints/vit_b16_arcface_hn.pt` |
| YOLOv8m detector | YOLO obj det loss | LogoDet-3K + OpenLogo boxes (class-agnostic) | `runs/detect/checkpoints/yolov8_logo/weights/best.pt` |

### DINOv3-B/16 pipeline (modification)

| Component | Loss | Data split | Output |
|---|---|---|---|
| DINOv3-B/16 embedder Phase A | ArcFace (K=1) | Open-set train classes (~1600) | `checkpoints/dinov3_arcface_base.pt` |
| YOLOv8m detector | shared with ViT pipeline | — | `runs/detect/checkpoints/yolov8_logo/weights/best.pt` |

## Backbone comparison

| | ViT-B/16 (main) | DINOv3-B/16 | ViT-B/32 (baseline) |
|---|---|---|---|
| Pretrain | OpenAI CLIP (image-text) | Self-supervised DINOv3 (LVD-1.689B) | OpenAI CLIP (image-text) |
| Input | 160×160 | 160×160 | 160×160 |
| Patch size | 16 | 16 | 32 |
| Trunk output | 512-d | 768-d | 512-d |
| FC head | 512→128 | 768→128 | 512→128 |
| Normalization | CLIP mean/std | ImageNet mean/std | CLIP mean/std |
| Patches at 160px | 10×10 = 100 | 10×10 = 100 | 5×5 = 25 |

## Modifications over base paper

1. **ViT-B/16 backbone** — finer 16×16 patches (100 tokens vs 25 at 160px) vs ViT-B/32 baseline. Same OpenAI CLIP pretraining.

2. **ArcFace loss** — replaces ProxyNCA++ with ArcFace (K=1, Phase A) and Sub-center ArcFace (K=3, Phase C). Angular margin enforces tighter intra-class clusters.

3. **DINOv3-B/16 backbone** — swaps CLIP ViT for DINOv3 pretrained on LVD-1.689B images. Same patch size (16), same input (160×160), uses ImageNet normalization.

4. **ViT-B/16 + DINOv3 ensemble** — score-level fusion: `fused = vit_weight × vit_score + (1 − vit_weight) × dino_score`. Run with `--ensemble` flag.

## Results

### ViT-B/16 ArcFace HN (main model)

| Split | Q-vs-G | All-vs-All | Text logos | Small logos | Large logos |
|---|---|---|---|---|---|
| Closed-set | **0.9760** | **0.9759** | **0.9231** | **0.9671** | **0.9824** |
| Open-set | **0.9780** | **0.9817** | **0.9255** | **0.9647** | **0.9875** |

### ViT-B/16 + DINOv3 Ensemble

| Split | Q-vs-G | Text logos | Small logos | Large logos |
|---|---|---|---|---|
| Closed-set | **0.9797** | **0.9538** | **0.9741** | **0.9836** |
| Open-set | **0.9848** | **0.9255** | **0.9771** | **0.9904** |

### Backbone comparison (closed-set)

| Model | Q-vs-G | All-vs-All |
|---|---|---|
| ViT-B/32 ProxyNCA HN | 0.9623 | 0.9646 |
| ViT-B/32 ArcFace HN | 0.9694 | 0.9699 |
| DINOv3-B/16 ArcFace Phase A | 0.9563 | 0.9640 |
| **ViT-B/16 ArcFace HN** | **0.9760** | **0.9759** |
| ViT-B/16 + DINOv3 Ensemble | **0.9797** | — |

### Detector

| Metric | Value |
|---|---|
| YOLO AP@0.5 (val) | **0.653** |

## Speed optimizations applied

- AMP mixed precision (bfloat16) — ~2× faster
- `torch.compile` — **disabled on Windows** (Triton not supported on Windows)
- `freeze_blocks=0` — unfreeze all 12 ViT blocks (required for accuracy)
- `num_workers=10` + `persistent_workers=True`
- TF32 enabled on Ampere/Ada (`allow_tf32=True`)
- Early stopping patience=8 (Phase A), 6 (Phase C)

---

## Execution steps

### Step 0 — One-time setup

Skip if environment already configured.

#### 0.1 — Install prerequisites
| Tool | Link | Notes |
|---|---|---|
| NVIDIA driver + CUDA 12.8 | https://developer.nvidia.com/cuda-12-8-0-download-archive | Windows → x86_64 → 11 → exe (local); reboot after |
| Git | https://git-scm.com/download/win | Install with defaults |
| Python 3.11 (64-bit) | https://www.python.org/downloads/release/python-3119/ | Check **"Add python.exe to PATH"** |
| VS Build Tools | https://visualstudio.microsoft.com/visual-cpp-build-tools/ | Select **"Desktop development with C++"** |

Verify GPU: `nvidia-smi`

#### 0.2 — Clone repo and install dependencies
```cmd
git clone https://github.com/maivugiahuy/logo-recognition "C:\Logo Recognition"
cd "C:\Logo Recognition"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```

Verify:
```cmd
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
python -c "import torch, open_clip, faiss, ultralytics, pandas; print('OK')"
```

#### 0.3 — Activate environment (every session)
```cmd
cd "C:\Logo Recognition"
.venv\Scripts\activate
```

#### 0.4 — Download datasets (manual, one-time)

| Dataset | Size | Download | Destination |
|---|---|---|---|
| LogoDet-3K | ~3 GB | [Kaggle](https://www.kaggle.com/datasets/lyly99/logodet3k) | `data/raw/LogoDet-3K/` |
| OpenLogo | ~2 GB | [GitHub](https://hangsu0730.github.io/qmul-openlogo/) | `data/raw/openlogo/` |

Expected layouts:
```
data/raw/LogoDet-3K/{category}/{ClassName}/{id}.jpg + {id}.xml
data/raw/openlogo/Annotations/*.xml  JPEGImages/*.jpg  ImageSets/
```

> Keep `data/`, `checkpoints/`, `data/galleries/` on a disk with ≥100 GB free.

---

### Step 1 — Build dataset
```bash
python scripts/01_build_dataset.py
```
Parses both datasets, normalizes bboxes, deduplicates, builds splits.

Output: `data/processed/openlogodet3k/annotations.parquet` + `data/processed/openlogodet3k/splits/`

| Source | Classes | Images |
|---|---|---|
| LogoDet-3K | ~2210 | ~101k |
| OpenLogo | ~355 | ~27k |
| **Combined** | **~2400+** | **~125k+** |

Splits: `open_train.json` (~1600 classes, Phase A), `open_val/test.json`, `closed_train/val/test.json`.

### Step 2 — Smoke test
```bash
python scripts/02_smoke_test.py
```
Verifies loss decreases over a few batches. Recall@1 = 0.0 at start is expected.

### Step 3 — Phase A: base embedder

**ViT-B/16 (main):**
```bash
python scripts/03_train_base.py --config configs/base_arcface_vit.yaml --ckpt vit_b16_arcface_base.pt
```
Output: `checkpoints/vit_b16_arcface_base.pt`

**DINOv3-B/16:**
```bash
python scripts/03_train_base.py --config configs/base_arcface_dinov3.yaml --ckpt dinov3_arcface_base.pt
```
Output: `checkpoints/dinov3_arcface_base.pt`

**ViT-B/32 (baseline):**
```bash
python scripts/03_train_base.py --config configs/base_vit.yaml
```
Output: `checkpoints/vit_base.pt`

ArcFace on ~1600 open-set classes. Batch 512 (k=64 × m=8), max 60 epochs, AMP bfloat16, ~16 GB VRAM.

### Step 4 — Hard-negative mining

**ViT-B/16:**
```bash
python scripts/04_mine_hn.py --ckpt checkpoints/vit_b16_arcface_base.pt --config configs/base_arcface_vit.yaml
```

**ViT-B/32 (baseline):**
```bash
python scripts/04_mine_hn.py --ckpt checkpoints/vit_base.pt --config configs/base_vit.yaml
```

Finds hard-negative pairs: confusion similarity 0.01–0.35, Levenshtein distance ≥ 2.

Output: `data/processed/hn_map.json`

### Step 5 — Phase C: fine-tune with hard negatives

**ViT-B/16:**
```bash
python scripts/05_train_hn.py --config configs/hn_arcface_vit.yaml --ckpt vit_b16_arcface_hn.pt
```
Output: `checkpoints/vit_b16_arcface_hn.pt`

Sub-center ArcFace (K=3) on ~1900 closed-set classes. Init from Phase A checkpoint, max 60 epochs, ~20 GB VRAM.

### Step 6 — Train logo detector
```bash
python scripts/06_train_detector.py
```
YOLOv8m, class-agnostic (1 class: "logo"), imgsz=512, batch=48, 50 epochs. Shared by all pipelines.

Output: `runs/detect/checkpoints/yolov8_logo/weights/best.pt`

> To improve AP@0.5: set `imgsz: 640` or switch to `yolov8l.pt` in `configs/detector_yolov8.yaml`.

### Step 7 — Evaluate embedder

```bash
# ViT-B/16 ArcFace HN (main)
python scripts/07_eval.py --ckpt checkpoints/vit_b16_arcface_hn.pt --backbone vit_b16_openai

# ViT-B/16 ArcFace base
python scripts/07_eval.py --ckpt checkpoints/vit_b16_arcface_base.pt --backbone vit_b16_openai

# DINOv3
python scripts/07_eval.py --ckpt checkpoints/dinov3_arcface_base.pt --backbone dinov3_vitb16

# ViT-B/32 baseline
python scripts/07_eval.py --ckpt checkpoints/vit_hn.pt --backbone vit_b32_openai

# Ensemble (ViT-B/16 + DINOv3)
python scripts/07_eval.py --ensemble
python scripts/07_eval.py --ensemble --vit_weight 0.6

# Split-specific
python scripts/07_eval.py --ckpt checkpoints/vit_b16_arcface_hn.pt --backbone vit_b16_openai --split closedset
python scripts/07_eval.py --ckpt checkpoints/vit_b16_arcface_hn.pt --backbone vit_b16_openai --split openset
```
Output: `results/eval_results.csv`

### Step 8 — Build gallery
```bash
python scripts/08_build_galleries.py
```
Embeds all reference crops into a FAISS index.

Output: `data/galleries/openlogodet3k.faiss` + `data/galleries/openlogodet3k_labels.json`

### Step 9 — Add new classes (no retraining)

Place logo images in `{folder_root}/{class_name}/` (one subfolder per class), then:
```bash
# Add all subfolders → new_classes gallery (default folder_root = data/new_classes)
python scripts/09_add_classes.py

# Use the test/new_classes staging folder instead
python scripts/09_add_classes.py --folder_root test/new_classes

# Use YOLO to crop logos before embedding
python scripts/09_add_classes.py --use_detector

# Add into eval gallery instead
python scripts/09_add_classes.py --gallery openlogodet3k

# List / remove
python scripts/09_add_classes.py --list
python scripts/09_add_classes.py --remove nike
```

`--on_duplicate`: `ask` (default) / `append` / `replace` / `skip`

### Step 10 — Demo

```bash
# ViT-B/16 pipeline (main) — input from test/input/, output to test/output/
python scripts/10_demo.py test/input/your_image.jpg --embedder checkpoints/vit_b16_arcface_hn.pt --backbone vit_b16_openai --save_dir test/output

# DINOv3 pipeline
python scripts/10_demo.py test/input/your_image.jpg --embedder checkpoints/dinov3_arcface_base.pt --backbone dinov3_vitb16 --save_dir test/output

# Ensemble
python scripts/10_demo.py test/input/your_image.jpg --ensemble --save_dir test/output

# User-added classes gallery
python scripts/10_demo.py test/input/your_image.jpg --gallery new_classes --save_dir test/output

# Tune unknown threshold (default 0.50)
python scripts/10_demo.py test/input/your_image.jpg --unknown_threshold 0.65 --save_dir test/output

# Save cropped logo detections
python scripts/10_demo.py test/input/your_image.jpg --save_crops --save_dir test/output
```

**Inference pipeline:** YOLO detect → crop → embed (160×160) → FAISS top-1 → label

- Box colored = recognized (score ≥ threshold), color per brand
- Box **orange** = unknown (score < threshold)

### Step 11 — Web demo (Gradio)

```bash
python scripts/app.py
```
Browser UI for the ViT-B/16 pipeline. Upload an image → detect + recognize, with sliders for detector confidence and unknown threshold. Optionally filter retrieval to a subset of classes via the class dropdown (empty = full gallery).

**Add new brands at runtime** (no retraining): enter a brand name + upload reference images → embedded into a per-session `new_classes` gallery and merged into the class dropdown. `on_duplicate` controls append/replace/skip.

> The `new_classes*` galleries are **wiped on every startup** — runtime-added brands do not persist across restarts. For persistent additions, use `scripts/09_add_classes.py` against the `openlogodet3k` gallery.

### Utility — List training classes
```bash
python scripts/list_classes.py              # → results/classes.txt
python scripts/list_classes.py --out my.txt
```
Reads `annotations.parquet`; outputs `class_name | n_objects | n_images` per line.

---

## Key hyperparameters

### Phase A — ViT-B/16 ArcFace (`configs/base_arcface_vit.yaml`)

| Param | Value |
|---|---|
| Backbone | ViT-B/16, OpenAI CLIP pretrained |
| Input size | 160×160 |
| Normalization | CLIP mean/std |
| Embedding dim | 128 |
| ArcFace scale | 30.0 |
| ArcFace margin | 0.5 (≈28.6°) |
| ArcFace K | 1 (standard) |
| Trunk LR | 2.3e-6 |
| FC LR | 1.5e-3 |
| ArcFace weight LR | 0.05 |
| Trunk weight decay | 0.2 |
| Batch | 512 (k=64 × m=8) |
| Max epochs | 60 |
| Early stopping patience | 8 |
| freeze_blocks | 0 (full fine-tune) |

### Phase A — DINOv3-B/16 ArcFace (`configs/base_arcface_dinov3.yaml`)

| Param | Value |
|---|---|
| Backbone | DINOv3 ViT-B/16, LVD-1.689B pretrained |
| Input size | 160×160 (16×16 patch → 10×10 = 100 tokens) |
| Normalization | ImageNet mean/std |
| Embedding dim | 128 |
| ArcFace scale | 30.0 |
| ArcFace margin | 0.5 |
| ArcFace K | 1 |
| Trunk LR | 5.0e-6 |
| FC LR | 1.5e-3 |
| ArcFace weight LR | 0.05 |
| Trunk weight decay | 0.05 |
| Batch | 512 (k=64 × m=8) |
| Max epochs | 60 |
| Early stopping patience | 8 |
| freeze_blocks | 0 (full fine-tune) |

### Phase C — ViT-B/16 Sub-center ArcFace (`configs/hn_arcface_vit.yaml`)

| Param | Value |
|---|---|
| Init from | `checkpoints/vit_b16_arcface_base.pt` |
| ArcFace K | 3 (sub-center — handles multi-modal brands) |
| ArcFace scale | 30.0 |
| ArcFace margin | 0.5 |
| HN α1 / α2 | 0.01 / 0.35 |
| Levenshtein min | ≥ 2 |
| Trunk LR | 3.0e-6 |
| FC LR | 3.0e-4 |
| Batch | 512 (k=64 × m=8) |
| Max epochs | 60 |
| Early stopping patience | 6 |
| freeze_blocks | 0 |

### Detector (`configs/detector_yolov8.yaml`)

| Param | Value |
|---|---|
| Model | yolov8m.pt |
| imgsz | 512 |
| batch | 48 |
| epochs | 50 |
| patience | 5 |
| cache | disk |

---

## Known limitations

- YOLO AP@0.5 = 0.65 — needs `imgsz: 640` or `yolov8l.pt` to improve
- `torch.compile` disabled on Windows — loses ~15–20% speed
- Unknown threshold (0.50) needs calibration on val set to optimize F1
- DINOv3 Phase C (HN fine-tune) not yet trained — only Phase A results available
