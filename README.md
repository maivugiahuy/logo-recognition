# Logo Recognition

## What gets trained

### ViT-B/32 pipeline (baseline)

| Component | Loss | Data split | Output |
|---|---|---|---|
| ViT-B/32 embedder Phase A | ProxyNCA++ | Open-set train classes (~1600) | `checkpoints/vit_base.pt` |
| ViT-B/32 embedder Phase C | ProxyNCAHN++ | Closed-set train classes (~1900) | `checkpoints/vit_hn.pt` |
| YOLOv8m detector | YOLO obj det loss | LogoDet-3K + OpenLogo boxes (class-agnostic) | `runs/detect/checkpoints/yolov8_logo/weights/best.pt` |

### DINOv2-B/14 pipeline (modification)

| Component | Loss | Data split | Output |
|---|---|---|---|
| DINOv2-B/14 embedder Phase A | ProxyNCA++ | Open-set train classes (~1600) | `checkpoints/dinov2_base.pt` |
| DINOv2-B/14 embedder Phase C | ProxyNCAHN++ | Closed-set train classes (~1900) | `checkpoints/dinov2_hn.pt` |
| YOLOv8m detector | shared with ViT pipeline | — | `runs/detect/checkpoints/yolov8_logo/weights/best.pt` |

## Backbone comparison

| | ViT-B/32 | DINOv2-B/14 |
|---|---|---|
| Pretrain | OpenAI CLIP (image-text) | Self-supervised DINO v2 (LVD-142M) |
| Input | 160×160 | 224×224 |
| Patch size | 32 | 14 |
| Trunk output | 512-d | 768-d |
| FC head | 512→128 | 768→128 |
| Normalization | CLIP mean/std | ImageNet mean/std |
| Trunk params | ~86M | ~86M |

## Modifications over base paper

1. **DINOv2-B/14 backbone** — swaps CLIP ViT-B/32 for DINOv2, trained with same ProxyNCA++/ProxyNCAHN++ losses. Self-supervised pretraining optimized for visual similarity rather than image-text alignment.

2. **α-weighted Query Expansion (αQE)** — post-retrieval re-ranking. After initial FAISS search, averages query vector with top-k gallery neighbors weighted by `score^α`, then re-queries. Enabled by default in inference (`qe_k=5`, `qe_alpha=3.0`). Adds <1ms per query.

## Results

### v2 — LogoDet-3K + OpenLogo (ViT-B/32)

| Metric | Value |
|---|---|
| Phase A val recall@1 | **0.9651** |
| Phase C val recall@1 | **0.9369** |
| YOLO AP@0.5 (val) | **0.653** |
| Eval Q-vs-G recall@1 | **0.9331** |
| Eval All-vs-All recall@1 | **0.9445** |
| Eval small logo Q-vs-G | **0.9134** |
| Eval large logo Q-vs-G | **0.9485** |

### v1 — LogoDet-3K only

| Metric | Value |
|---|---|
| Phase A val recall@1 | **0.9675** |
| Phase C val recall@1 | **0.9356** |
| YOLO AP@0.5 (test) | **0.6498** |
| Eval Q-vs-G recall@1 | **0.9340** |
| Eval All-vs-All recall@1 | **0.9414** |
| Eval small logo Q-vs-G | **0.9099** |
| Eval large logo Q-vs-G | **0.9498** |

## Speed optimizations applied

- AMP mixed precision (bfloat16) — ~2× faster
- `torch.compile` — **disabled on Windows** (Triton not supported on Windows)
- `freeze_blocks=0` — unfreeze all 12 ViT blocks (required for accuracy)
- `num_workers=8` + `persistent_workers=True`
- TF32 enabled on Ampere/Ada (`allow_tf32=True`)
- Early stopping patience=5 instead of fixed epochs

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

**ViT-B/32:**
```bash
python scripts/03_train_base.py --config configs/base_vit.yaml
```
Output: `checkpoints/vit_base.pt`

**DINOv2-B/14:**
```bash
python scripts/03_train_base.py --config configs/base_dinov2.yaml --ckpt dinov2_base.pt
```
Output: `checkpoints/dinov2_base.pt`

ProxyNCA++ on ~1600 open-set classes. Batch 512 (k=64 × m=8), max 50 epochs, AMP bfloat16, ~16 GB VRAM.

### Step 4 — Hard-negative mining

**ViT-B/32:**
```bash
python scripts/04_mine_hn.py --ckpt checkpoints/vit_base.pt --config configs/base_vit.yaml
```

**DINOv2-B/14:**
```bash
python scripts/04_mine_hn.py --ckpt checkpoints/dinov2_base.pt --config configs/base_dinov2.yaml
```

Finds hard-negative pairs: confusion similarity 0.05–0.35, Levenshtein distance > 2.

Output: `data/processed/hn_map.json`

### Step 5 — Phase C: fine-tune with hard negatives

**ViT-B/32:**
```bash
python scripts/05_train_hn.py --config configs/hn_vit.yaml
```
Output: `checkpoints/vit_hn.pt`

**DINOv2-B/14:**
```bash
python scripts/05_train_hn.py --config configs/hn_dinov2.yaml --ckpt dinov2_hn.pt
```
Output: `checkpoints/dinov2_hn.pt`

ProxyNCAHN++ on ~1900 closed-set classes. Init from Phase A checkpoint, max 50 epochs, ~20 GB VRAM.

### Step 6 — Train logo detector
```bash
python scripts/06_train_detector.py
```
YOLOv8m, class-agnostic (1 class: "logo"), imgsz=512, batch=48, 50 epochs. Shared by both pipelines.

Output: `runs/detect/checkpoints/yolov8_logo/weights/best.pt`

> To improve AP@0.5: set `imgsz: 640` or switch to `yolov8l.pt` in `configs/detector_yolov8.yaml`.

### Step 7 — Evaluate embedder

```bash
# ViT (default)
python scripts/07_eval.py                                            # both checkpoints, both splits
python scripts/07_eval.py --split closedset
python scripts/07_eval.py --split openset
python scripts/07_eval.py --ckpt checkpoints/vit_base.pt

# DINOv2
python scripts/07_eval.py --ckpt checkpoints/dinov2_hn.pt --backbone dinov2_vitb14

# With α-weighted Query Expansion
python scripts/07_eval.py --qe
python scripts/07_eval.py --qe --qe_k 10 --qe_alpha 2.0
python scripts/07_eval.py --ckpt checkpoints/dinov2_hn.pt --backbone dinov2_vitb14 --qe
```
Output: `results/eval_results.csv`

### Step 8 — Build gallery
```bash
python scripts/08_build_galleries.py
```
Embeds all reference crops into a FAISS index.

Output: `data/galleries/openlogodet3k.faiss` + `data/galleries/openlogodet3k_labels.json`

### Step 9 — Add new classes (no retraining)

Place logo images in `data/new_classes/{class_name}/`, then:
```bash
# Add all subfolders → new_classes gallery
python scripts/09_add_classes.py

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
# ViT pipeline (default)
python scripts/10_demo.py your_image.jpg --save_dir results/

# DINOv2 pipeline
python scripts/10_demo.py your_image.jpg --backbone dinov2_vitb14 --embedder checkpoints/dinov2_hn.pt --save_dir results/

# User-added classes gallery
python scripts/10_demo.py your_image.jpg --gallery new_classes --save_dir results/

# Tune unknown threshold (default 0.50)
python scripts/10_demo.py your_image.jpg --unknown_threshold 0.65 --save_dir results/

# Query Expansion options (QE on by default)
python scripts/10_demo.py your_image.jpg --no_qe                   # disable QE
python scripts/10_demo.py your_image.jpg --qe_k 10 --qe_alpha 2.0  # tune QE
```

**Inference pipeline:** YOLO detect → crop → embed (160×160 ViT or 224×224 DINOv2) → α-QE → FAISS top-1 → label

- Box **red** = recognized (score ≥ threshold)
- Box **orange** = unknown (score < threshold)

### Utility — List training classes
```bash
python scripts/list_classes.py              # → results/classes.txt
python scripts/list_classes.py --out my.txt
```
Reads `annotations.parquet`; outputs `class_name | n_objects | n_images` per line.

---

## Key hyperparameters

### Phase A — ViT-B/32 (`configs/base_vit.yaml`)

| Param | Value |
|---|---|
| Backbone | ViT-B/32, OpenAI CLIP pretrained |
| Input size | 160×160 |
| Normalization | CLIP mean/std |
| Embedding dim | 128 |
| Temperature σ | 0.06 |
| Trunk LR | 2.3e-6 |
| FC LR | 1.5e-3 |
| Proxy LR | 71 |
| Trunk weight decay | 0.2 |
| β2 trunk/fc | 0.98 |
| β2 proxy | 0.999 |
| Batch | 512 (k=64 × m=8) |
| Max epochs | 50 |
| Early stopping patience | 5 |
| freeze_blocks | 0 (full fine-tune) |

### Phase A — DINOv2-B/14 (`configs/base_dinov2.yaml`)

| Param | Value |
|---|---|
| Backbone | DINOv2-B/14, facebookresearch pretrained |
| Input size | 224×224 (patch 14, 16×16 grid) |
| Normalization | ImageNet mean/std |
| Embedding dim | 128 |
| Temperature σ | 0.06 |
| Trunk LR | 5.0e-6 |
| FC LR | 1.5e-3 |
| Proxy LR | 71 |
| Trunk weight decay | 0.05 |
| Batch | 512 (k=64 × m=8) |
| Max epochs | 50 |
| Early stopping patience | 5 |
| freeze_blocks | 0 (full fine-tune) |

### Phase C — ViT-B/32 (`configs/hn_vit.yaml`)

| Param | Value |
|---|---|
| Init from | `checkpoints/vit_base.pt` |
| HN α1 / α2 | 0.05 / 0.35 |
| Levenshtein min | > 2 |
| Batch | 512 (k=64 × m=8) |
| Max epochs | 50 |
| freeze_blocks | 0 |

### Phase C — DINOv2-B/14 (`configs/hn_dinov2.yaml`)

| Param | Value |
|---|---|
| Init from | `checkpoints/dinov2_base.pt` |
| HN α1 / α2 | 0.05 / 0.35 |
| Levenshtein min | > 2 |
| Batch | 512 (k=64 × m=8) |
| Max epochs | 50 |
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

### Query Expansion (αQE)

| Param | Default | Notes |
|---|---|---|
| `qe_k` | 5 | Neighbors to average with query |
| `qe_alpha` | 3.0 | Weight exponent — higher = closer neighbors dominate |
| Enabled in demo | yes | Disable with `--no_qe` |
| Enabled in eval | no | Enable with `--qe` |

---

## Known limitations

- YOLO AP@0.5 = 0.65 — needs `imgsz: 640` or `yolov8l.pt` to improve
- `torch.compile` disabled on Windows — loses ~15–20% speed
- Unknown threshold (0.50) needs calibration on val set to optimize F1
- DINOv2 results pending — ViT benchmarks above are baseline
