# Logo Recognition

## What gets trained

| Component | Loss | Data split | Output |
|---|---|---|---|
| ViT-B/32 embedder Phase A | ProxyNCA++ | Open-set train classes (~1600) | `checkpoints/vit_base.pt` |
| ViT-B/32 embedder Phase C | ProxyNCAHN++ | Closed-set train classes (~1900) | `checkpoints/vit_hn.pt` |
| YOLOv8m detector | YOLO obj det loss | LogoDet-3K + OpenLogo boxes (class-agnostic) | `runs/detect/checkpoints/yolov8_logo/weights/best.pt` |

## Results

### v2 — LogoDet-3K + OpenLogo

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
- Early stopping patience=6 instead of fixed epochs

---

## Execution steps

### Step 0 — Fresh Windows setup

#### 0.1 — Check GPU
```cmd
nvidia-smi
```
If not found → install NVIDIA driver first (step 0.2).

#### 0.2 — Install NVIDIA driver + CUDA Toolkit
RTX 5060 Ti (Blackwell) requires CUDA 12.8+.
1. Download driver: https://www.nvidia.com/Download/index.aspx
2. Download CUDA 12.8 Toolkit: https://developer.nvidia.com/cuda-12-8-0-download-archive
   - Select: Windows → x86_64 → 11 → exe (local)
3. Reboot → verify: `nvidia-smi` shows GPU + CUDA 12.8

#### 0.3 — Install Git
https://git-scm.com/download/win → install with defaults.

#### 0.4 — Install Python 3.11
https://www.python.org/downloads/release/python-3119/
- Pick **Windows installer (64-bit)**
- **Check "Add python.exe to PATH"** before installing

#### 0.5 — Install VS Build Tools
https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Select **"Desktop development with C++"** workload

#### 0.6 — Clone repo + create venv
```cmd
git clone https://github.com/maivugiahuy/logo-recognition "C:\Logo Recognition"
cd "C:\Logo Recognition"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

#### 0.7 — Install PyTorch (CUDA 12.8)
```cmd
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```
Must print `True 12.8`.

#### 0.8 — Install project dependencies
```cmd
pip install -r requirements.txt
```

#### 0.9 — Verify
```cmd
python -c "import torch, open_clip, faiss, ultralytics, pandas; print('OK')"
```

> **Note:** Keep `data/`, `checkpoints/`, `data/galleries/` on a large disk (≥100 GB free).

---

### Step 1 — Activate environment
```cmd
cd "C:\Logo Recognition"
.venv\Scripts\activate
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 2 — Download datasets (manual)

| Dataset | Size | Download | Destination |
|---|---|---|---|
| LogoDet-3K | ~3 GB | [Kaggle](https://www.kaggle.com/datasets/lyly99/logodet3k) | `data/raw/LogoDet-3K/` |
| OpenLogo | ~2 GB | [GitHub](https://hangsu0730.github.io/qmul-openlogo/) | `data/raw/openlogo/` |

LogoDet-3K layout: `LogoDet-3K/{category}/{ClassName}/{id}.jpg` + `{id}.xml`

OpenLogo layout:
```
data/raw/openlogo/
  Annotations/   ← *.xml (Pascal VOC)
  JPEGImages/    ← *.jpg
  ImageSets/
```

### Step 3 — Build dataset (~45–60 min)
```bash
python scripts/01_build_dataset.py
```
Parse LogoDet-3K + OpenLogo → normalize → dedup → filter → build splits.

Output: `data/processed/openlogodet3k/annotations.parquet` + `data/processed/openlogodet3k/splits/`

| Source | Classes | Images |
|---|---|---|
| LogoDet-3K | ~2210 | ~101k |
| OpenLogo | ~355 | ~27k |
| **Combined** | **~2400+** | **~125k+** |

Splits:
- `open_train.json` — ~1600 classes (Phase A)
- `open_val.json` — ~400 classes
- `open_test.json` — ~500 classes
- `closed_train/val/test.json` — image-level splits on seen classes

### Step 4 — Smoke test (~5–10 min)
```bash
python scripts/02_smoke_test.py
```
Gate: loss decreasing. Recall@1 = 0.0 at start is normal.

### Step 5 — Phase A: base ViT (~2–3 h)
```bash
python scripts/03_train_base.py --config configs/base_vit.yaml
```
- LR: trunk 2.3e-6, FC 1.5e-3, proxy 71
- Batch 512 (k=64 × m=8), max 15 epochs, early stopping patience 6, σ=0.06
- freeze_blocks=0, AMP bfloat16, num_workers=8
- VRAM: ~9GB

Output: `checkpoints/vit_base.pt`

### Step 6 — Hard-negative mining (~1 min)
```bash
python scripts/04_mine_hn.py --ckpt checkpoints/vit_base.pt --config configs/base_vit.yaml
```
- h(yi) = {yj : 0.05 ≤ C[i,j] ≤ 0.35 AND levenshtein(name_i, name_j) > 2}
- Output: `data/processed/hn_map.json`

### Step 7 — Phase C: ProxyNCAHN++ (~5 h)
```bash
python scripts/05_train_hn.py --config configs/hn_vit.yaml
```
- Init from `checkpoints/vit_base.pt`
- Closed-set, max 25 epochs, early stopping patience 6
- Batch 512 (k=64 × m=8), freeze_blocks=0
- VRAM: ~15GB

Output: `checkpoints/vit_hn.pt`

### Step 8 — Logo detector (~3.5 h)
```bash
python scripts/06_train_detector.py
```
Config: `configs/detector_yolov8.yaml`
- YOLOv8m, class-agnostic (1 class: "logo"), imgsz=512, batch=48
- 25 epochs, patience=5, cache=disk

Output: `runs/detect/checkpoints/yolov8_logo/weights/best.pt`

> To improve AP@0.5: increase `imgsz: 640` or use `yolov8l.pt`.

### Step 9 — Build gallery (~5–10 min)
```bash
python scripts/08_build_galleries.py
```
Output: `data/galleries/openlogodet3k.faiss` + `data/galleries/openlogodet3k_labels.json`

> Required before running demo. Not needed for eval.

### Step 10 — Evaluation (~10–15 min)
```bash
python scripts/07_eval.py
```

### Step 11 — Demo
```bash
# Eval gallery (openlogodet3k — 2400+ classes, default)
python scripts/10_demo.py your_image.jpg --save_dir results/

# New-classes gallery (brands added via 09_add_classes.py)
python scripts/10_demo.py your_image.jpg --gallery new_classes --save_dir results/

# Custom unknown threshold (default 0.50)
python scripts/10_demo.py your_image.jpg --unknown_threshold 0.65 --save_dir results/
```
Pipeline: YOLO26 detect → crop 160×160 → ViT embed → FAISS top-1 → class label
- Box **red** = recognized class (score ≥ threshold)
- Box **orange** = unknown (score < threshold)

### Step 11b — List classes
```bash
# Export classes.txt (class_name | n_objects | n_images)
python scripts/list_classes.py

# Custom output path
python scripts/list_classes.py --out my_classes.txt
```

### Step 12 — Add new classes to gallery (no retraining needed)

Classes are added to the `new_classes` gallery (separate from the eval gallery `openlogodet3k`).
Use `--gallery openlogodet3k` to add directly into the eval gallery.

```bash
# Add from data/new_classes/ (each subfolder = 1 class) → new_classes gallery
python scripts/09_add_classes.py [--use_detector]

# Specify a different folder_root
python scripts/09_add_classes.py --folder_root path/to/brands/

# Add into eval gallery instead of new_classes
python scripts/09_add_classes.py --gallery openlogodet3k

# List classes in gallery
python scripts/09_add_classes.py --list
python scripts/09_add_classes.py --list --gallery openlogodet3k

# Remove a class from gallery
python scripts/09_add_classes.py --remove nike
```

`data/new_classes/` layout:
```
data/new_classes/
  nike/       ← Nike logo images
  adidas/     ← Adidas logo images
  ...
```

`--on_duplicate`: `ask` (default) / `append` / `replace` / `skip`

---

## Compute budget (1× RTX 5060 Ti 16 GB, Windows)

| Phase | Actual |
|---|---|
| Build dataset | ~45–60 min |
| Smoke test | ~5 min |
| Phase A (15 epochs, batch 512) | ~2–3 h |
| HN mining | ~1 min |
| Phase C (25 epochs, batch 512) | ~5 h |
| Detector (YOLOv8m) | ~3.5 h |
| Gallery build | ~5–10 min |
| Eval | ~10–15 min |
| **Total** | **~12–14 h** |

---

## Key hyperparameters

### Phase A (`configs/base_vit.yaml`)

| Param | Value |
|---|---|
| Backbone | ViT-B/32, OpenAI CLIP pretrained |
| Input size | 160×160 |
| Embedding dim | 128 |
| Temperature σ | 0.06 |
| Trunk LR | 2.3e-6 |
| FC LR | 1.5e-3 |
| Proxy LR | 71 |
| Trunk weight decay | 0.2 |
| β2 trunk/fc | 0.98 |
| β2 proxy | 0.999 |
| Batch | 512 (k=64 × m=8) |
| Max epochs | 15 |
| Early stopping patience | 6 |
| freeze_blocks | 0 (full fine-tune) |

### Phase C (`configs/hn_vit.yaml`)

| Param | Value |
|---|---|
| Init from | `checkpoints/vit_base.pt` |
| HN α1 / α2 | 0.05 / 0.35 |
| Levenshtein min | > 2 |
| Batch | 512 (k=64 × m=8) |
| Max epochs | 25 |
| freeze_blocks | 0 |

### Detector (`configs/detector_yolov8.yaml`)

| Param | Value |
|---|---|
| Model | yolov8m.pt |
| imgsz | 512 |
| batch | 48 |
| epochs | 25 |
| patience | 5 |
| cache | disk |

---

## Known limitations

- YOLO AP@0.5 = 0.65 — needs `imgsz: 640` or `yolov8l.pt` to improve
- `torch.compile` disabled on Windows — loses ~15–20% speed
- Unknown threshold (0.50) needs calibration on val set to optimize F1
