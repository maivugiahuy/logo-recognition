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
| 4× V100 | Single GPU + AMP + torch.compile | Hardware |

## Speed optimizations applied

- AMP mixed precision (bfloat16) — ~2× faster
- `torch.compile` — ~10–20%
- Freeze first 8 of 12 ViT blocks — ~60% less backward compute
- `num_workers=8`
- Phase A epochs 25→15
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
1. Download driver: https://www.nvidia.com/Download/index.aspx (pick your GPU, Windows 11)
2. Download CUDA 12.1 Toolkit: https://developer.nvidia.com/cuda-12-1-0-download-archive
   - Select: Windows → x86_64 → 11 → exe (local)
   - Install with default options
3. Reboot
4. Verify: `nvidia-smi` shows GPU + CUDA version

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

#### 0.7 — Install PyTorch (CUDA 12.1)
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```
Must print `True`. If `False` → CUDA driver mismatch, reinstall CUDA Toolkit.

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
| Dataset | Size | Destination |
|---|---|---|
| LogoDet3K | ~3 GB | `data/raw/logodet3k/` |
| QMUL-OpenLogo | ~2 GB | `data/raw/openlogo/` |
| FlickrLogos-47 | ~150 MB | `data/raw/flickr47/` |
| BelgaLogos | ~70 MB | `data/raw/belga/` |
| LogosInTheWild | ~1 GB | `data/raw/litw/` |

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

### Step 5 — Phase A: base ViT (8–12 h → ~4–6 h with opts)
```bash
python scripts/02_train_base.py --config configs/base_vit.yaml
```
- LR: trunk 2.3e-6, FC 1.5e-3, proxy 71
- Batch 192 (k=24, m=4), 15 epochs, σ=0.06
- Gate: val recall@1 ≈ 0.97–0.98

### Step 6 — Hard-negative mining (30–60 min)
```bash
python scripts/03_mine_hn.py --ckpt checkpoints/vit_base.pt
```
- h(yi) = {yj : 0.05 ≤ C[i,j] ≤ 0.35 AND levenshtein(name_i, name_j) > 2}
- Output: `data/processed/hn_map.json`

### Step 7 — Phase C: ProxyNCAHN++ (10–14 h)
```bash
python scripts/04_train_hn.py --config configs/hn_vit.yaml
```
- Init from `vit_base.pt`, closed-set, 25 epochs
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

## Compute budget (1× RTX 4090)

| Phase | Time |
|---|---|
| Download + build dataset | 4–8 h |
| Smoke test | 30 min |
| Phase A (base train) | 4–6 h |
| HN mining | 30–60 min |
| Phase C (HN train) | 10–14 h |
| Detector | 6–10 h |
| Galleries + eval | 2–3 h |
| **Total** | **~27–43 h** |

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
