# Logo Recognition — Reproduction Plan

Paper: *Image-Text Pre-Training for Logo Recognition* (Hubenthal & Kumar, Amazon, arXiv:2309.10206)

## What gets trained

| Component | Loss | Data split | Output |
|---|---|---|---|
| ViT-B/32 embedder Phase A | ProxyNCA++ (eq 3) | Open-set train classes | `checkpoints/vit_base.pt` |
| ViT-B/32 embedder Phase C | ProxyNCAHN++ (eq 5) | Closed-set all classes | `checkpoints/vit_hn.pt` |
| YOLOv8m detector | YOLO obj det loss | LogoDet3K boxes (class-agnostic) | `checkpoints/yolov8_logo/weights/best.pt` |

## Substitutions vs paper

| Paper original | Substitution | Reason |
|---|---|---|
| CLIP pretraining on 20M e-comm pairs | OpenAI pretrained ViT-B/32 weights | Table 3: OpenAI IT ≥ E-comm IT; 20M pairs unavailable |
| YoloV4 on Amazon PL2K | YOLOv8m on LogoDet3K | PL2K is proprietary |
| 4× V100 | 1× RTX 5060 Ti 16GB + AMP bfloat16 | Hardware |

## Speed optimizations applied

- AMP mixed precision (bfloat16) — ~2× faster
- `torch.compile` — **disabled on Windows** (Triton không hỗ trợ Windows); mất ~15–20%
- Freeze first 8 of 12 ViT blocks — ~60% less backward compute
- `num_workers=4` + `persistent_workers=True` — tối ưu cho Windows multiprocessing
- Phase A epochs 25→10, Phase C 25→18 (early stopping dừng sớm hơn)
- Scheduler patience 4→2

## Execution steps

### Step 0 — Fresh Windows remote desktop setup

#### 0.1 — Check GPU
Open **Device Manager** → Display Adapters → confirm NVIDIA GPU present.
Then open Command Prompt:
```cmd
nvidia-smi
```
If `nvidia-smi` not found → install NVIDIA driver first (step 0.2).

#### 0.2 — Install NVIDIA driver + CUDA Toolkit
RTX 5060 Ti (Blackwell) yêu cầu CUDA 12.8+.
1. Download driver: https://www.nvidia.com/Download/index.aspx (chọn RTX 5060 Ti, Windows 11)
2. Download CUDA 12.8 Toolkit: https://developer.nvidia.com/cuda-12-8-0-download-archive
   - Select: Windows → x86_64 → 11 → exe (local)
   - Install with default options
3. Reboot
4. Verify: `nvidia-smi` shows GPU + CUDA 12.8

#### 0.3 — Install Git
Download: https://git-scm.com/download/win → install with defaults.
```cmd
git --version
```

#### 0.4 — Install Python 3.11
Download: https://www.python.org/downloads/release/python-3119/
- Pick **Windows installer (64-bit)**
- **Check "Add python.exe to PATH"** before installing
```cmd
python --version
```

#### 0.5 — Install VS Build Tools (required by some pip packages)
Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Select **"Desktop development with C++"** workload → Install

#### 0.6 — Clone repo + create venv
```cmd
git clone <your-repo-url> "C:\Logo Recognition"
cd "C:\Logo Recognition"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

#### 0.7 — Install PyTorch (CUDA 12.8)
RTX 5060 cần PyTorch nightly vì stable chưa có wheel cu128 chính thức:
```cmd
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```
Must print `True 12.8`. If `False` → CUDA driver mismatch, reinstall CUDA Toolkit.

#### 0.8 — Install project dependencies
```cmd
pip install -r requirements.txt
```

#### 0.9 — Verify all imports
```cmd
python -c "import torch, open_clip, faiss, ultralytics, pandas; print('OK')"
```

> **Note:** Keep `data/`, `checkpoints/`, `data/galleries/` on a large disk partition (need ≥100 GB free). Check with `dir C:\` or use D:\ if C:\ is small.

### Step 1 — Environment (already done in Step 0 for fresh machine)
```cmd
cd "C:\Logo Recognition"
.venv\Scripts\activate
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 2 — Download datasets (manual)

**Dùng để train + build:**

| Dataset | Size | Download | Destination |
|---|---|---|---|
| LogoDet-3K | ~3 GB | [Kaggle](https://www.kaggle.com/datasets/lyly99/logodet3k) | `data/raw/LogoDet-3K/` |
| QMUL-OpenLogo | ~2 GB | [qmul-openlogo.github.io](https://qmul-openlogo.github.io/) | `data/raw/openlogo/` |
| FlickrLogos-47 | ~150 MB | [Kaggle](https://www.kaggle.com/datasets/samikshakolhe/flicker-47-logo-images-dataset) | `data/raw/FlickrLogos_47/` |

**Dùng để eval only (Step 10):**

| Dataset | Size | Download | Destination |
|---|---|---|---|
| BelgaLogos | ~70 MB | [INRIA](http://www-sop.inria.fr/members/Alexis.Joly/BelgaLogos/BelgaLogos.html) | `data/raw/belga/` |
| LogosInTheWild | ~1 GB | [Zenodo](https://zenodo.org/records/5101018) | `data/raw/litw/` |

> **Lưu ý:**
> - **LogoDet-3K**: giải nén, đặt vào `data/raw/LogoDet-3K/`. Cấu trúc: `LogoDet-3K/{category}/{ClassName}/{id}.jpg` + `{id}.xml`
> - **QMUL-OpenLogo**: giải nén, đặt vào `data/raw/openlogo/`. Cấu trúc: `openlogo/Annotations/*.xml` + `openlogo/JPEGImages/*.jpg`
> - **FlickrLogos-47**: giải nén, đặt vào `data/raw/FlickrLogos_47/`. Cấu trúc: `FlickrLogos_47/train/{classID:06d}/*.png` + `*.gt_data.txt`
> - **BelgaLogos** và **LogosInTheWild**: chỉ cần khi chạy `07_eval.py` để so sánh với paper Table 2. Có thể bỏ qua nếu chỉ muốn train và test trên 3 dataset chính.

### Step 3 — Build OpenLogoDet3K47
```bash
python scripts/01_build_dataset.py
```
Target: **2714 classes / 181,552 images / 227,176 objects**

### Step 4 — Smoke test
```bash
python scripts/smoke_test.py
```
Gate: loss decreasing + recall@1 > 0.5

### Step 5 — Phase A: base ViT (~3–4 h)
```bash
python scripts/02_train_base.py --config configs/base_vit.yaml
```
- LR: trunk 2.3e-6, FC 1.5e-3, proxy 71
- Batch 192 (k=24, m=4), max 10 epochs, early stopping patience 6, σ=0.06
- Gate: val recall@1 ≈ 0.97–0.98

### Step 6 — Hard-negative mining (~30 min)
```bash
python scripts/03_mine_hn.py --ckpt checkpoints/vit_base.pt
```
- h(yi) = {yj : 0.05 ≤ C[i,j] ≤ 0.35 AND levenshtein(name_i, name_j) > 2}
- Output: `data/processed/hn_map.json`

### Step 7 — Phase C: ProxyNCAHN++ (~5–7 h)
```bash
python scripts/04_train_hn.py --config configs/hn_vit.yaml
```
- Init từ `vit_base.pt`, closed-set, max 18 epochs, early stopping patience 6
- Gate: recall@1 ≈ 0.96

### Step 8 — Logo detector (6–10 h)
```bash
python scripts/05_train_detector.py
```
- YOLOv8m, class-agnostic, 512px, 50 epochs
- Gate: AP@0.5 ≥ 0.70

### Step 9 — Build galleries (1 h)
```bash
python scripts/06_build_galleries.py
```
Output: `data/galleries/{dataset}.faiss` for all 5 datasets

### Step 10 — Evaluation
```bash
python scripts/07_eval.py
```
| Dataset | Metric | Target |
|---|---|---|
| LogoDet3K | Q-vs-G recall@1 | ≥ 0.97 |
| OpenLogo | Text recall@1 | ≥ 0.95 |
| FlickrLogos-47 | All recall@1 | ≥ 0.97 |

### Step 11 — Demo
```bash
python scripts/08_demo.py your_image.jpg --save_dir results/
```
Pipeline: YOLOv8 detect → crop 160×160 → ViT embed → FAISS top-1 → brand label

## Compute budget (1× RTX 5060 Ti 16 GB, Windows, no torch.compile)

| Phase | Time |
|---|---|
| Build dataset | 1–2 h |
| Smoke test | ~15 min |
| Phase A (base train) | ~3–4 h |
| HN mining | ~30 min |
| Phase C (HN train) | ~5–7 h |
| Detector | ~6–8 h |
| Galleries + eval | ~1–2 h |
| **Total** | **~17–24 h** |

## Key hyperparameters (Tables 3 + 5)

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
| FC weight decay | 0.001 |
| β2 trunk/fc | 0.98 |
| β2 proxy | 0.999 |
| ε trunk/fc | 1e-6 |
| ε proxy | 1.0 |
| Batch | 192 (k=24, m=4) |
| HN α1 / α2 | 0.05 / 0.35 |
| Levenshtein min | > 2 |
