# Logo Recognition — Reproduction Plan

Paper: *Image-Text Pre-Training for Logo Recognition* (Hubenthal & Kumar, Amazon, arXiv:2309.10206)

## What gets trained

| Component | Loss | Data split | Output |
|---|---|---|---|
| ViT-B/32 embedder Phase A | ProxyNCA++ (eq 3) | Open-set train classes (1414) | `checkpoints/vit_base.pt` |
| ViT-B/32 embedder Phase C | ProxyNCAHN++ (eq 5) | Closed-set train classes (~1728) | `checkpoints/vit_hn.pt` |
| YOLOv8m detector | YOLO obj det loss | LogoDet-3K boxes (class-agnostic) | `runs/detect/checkpoints/yolov8_logo/weights/best.pt` |

## Substitutions vs paper

| Paper original | Substitution | Reason |
|---|---|---|
| CLIP pretraining on 20M e-comm pairs | OpenAI pretrained ViT-B/32 weights | Table 3: OpenAI IT ≥ E-comm IT; 20M pairs unavailable |
| YoloV4 on Amazon PL2K | YOLOv8m on LogoDet-3K | PL2K is proprietary |
| 4× V100 | 1× RTX 5060 Ti 16GB + AMP bfloat16 | Hardware |
| OpenLogoDet3K47 composite (4 datasets) | LogoDet-3K only | Đơn giản hóa pipeline; 1 dataset công khai duy nhất |

## Actual results achieved

| Metric | Ours | Paper target | Gap |
|---|---|---|---|
| Phase A val recall@1 | **0.9675** | ~0.97 | -0.25pp |
| Phase C val recall@1 | **0.9356** | ~0.96 | -2.44pp |
| YOLO AP@0.5 (test) | **0.6498** | ≥0.70 | -5.0pp |
| Eval Q-vs-G recall@1 | **0.9340** | 0.9836 | -4.96pp |
| Eval All-vs-All recall@1 | **0.9414** | 0.9886 | -4.72pp |
| Eval small logo Q-vs-G | **0.9099** | — | — |
| Eval large logo Q-vs-G | **0.9498** | — | — |

> Gap ~5pp so với paper là hợp lý: paper dùng 4 dataset, mình chỉ dùng LogoDet-3K (~44% ít data hơn).

## Speed optimizations applied

- AMP mixed precision (bfloat16) — ~2× faster
- `torch.compile` — **disabled on Windows** (Triton không hỗ trợ Windows)
- `freeze_blocks=0` — unfreeze toàn bộ 12 ViT blocks (đúng paper, tốt hơn freeze)
- `num_workers=8` + `persistent_workers=True`
- TF32 enabled trên Ampere/Ada (`allow_tf32=True`)
- Early stopping patience=6 thay vì fixed epochs

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

| Dataset | Size | Download | Destination |
|---|---|---|---|
| LogoDet-3K | ~3 GB | [Kaggle](https://www.kaggle.com/datasets/lyly99/logodet3k) | `data/raw/LogoDet-3K/` |

> **Lưu ý:** Giải nén, đặt vào `data/raw/LogoDet-3K/`. Cấu trúc: `LogoDet-3K/{category}/{ClassName}/{id}.jpg` + `{id}.xml`

### Step 3 — Build LogoDet-3K dataset (~30–45 min)
```bash
python scripts/01_build_dataset.py
```
**Kết quả thực tế:** 2210 classes / 101,222 images / objects
Output: `data/processed/logodet3k/annotations.parquet` + `data/processed/logodet3k/splits/`

Splits tạo ra:
- `open_train.json` — 1414 classes (Phase A training)
- `open_val.json` — 354 classes (Phase A validation)
- `open_test.json` — 442 classes (open-set test)
- `closed_train.json` / `closed_val.json` / `closed_test.json` — image-level splits trên seen classes

### Step 4 — Smoke test (~5–10 min)
```bash
python scripts/smoke_test.py
```
Gate: loss decreasing. Recall@1 có thể = 0.0 ban đầu — bình thường.

### Step 5 — Phase A: base ViT (~2–3 h)
```bash
python scripts/02_train_base.py --config configs/base_vit.yaml
```
Config hiện tại (`configs/base_vit.yaml`):
- LR: trunk 2.3e-6, FC 1.5e-3, proxy 71
- Batch 768 (k=96 × m=8), max 15 epochs, early stopping patience 6, σ=0.06
- freeze_blocks=0 (unfreeze toàn bộ 12 blocks)
- AMP bfloat16, num_workers=8

**Kết quả thực tế:** best val recall@1 = **0.9675** (epoch ~15)
VRAM usage: ~9GB

### Step 6 — Hard-negative mining (~1 min)
```bash
python scripts/03_mine_hn.py --ckpt checkpoints/vit_base.pt --config configs/base_vit.yaml
```
- Đọc `freeze_blocks` và `alpha1/alpha2/lev_min` tự động từ config
- h(yi) = {yj : 0.05 ≤ C[i,j] ≤ 0.35 AND levenshtein(name_i, name_j) > 2}
- Output: `data/processed/hn_map.json`

**Kết quả thực tế:** 68/1414 classes có hard negatives (76 total pairs)
> Model tốt → ít nhầm lẫn → ít pairs — đây là dấu hiệu tốt.

### Step 7 — Phase C: ProxyNCAHN++ (~5 h)
```bash
python scripts/04_train_hn.py --config configs/hn_vit.yaml
```
Config hiện tại (`configs/hn_vit.yaml`):
- Init từ `checkpoints/vit_base.pt`
- Closed-set, max 25 epochs, early stopping patience 6
- Batch 768 (k=96 × m=8), freeze_blocks=0

**Kết quả thực tế:** best val recall@1 = **0.9356** (epoch 25, không early stop)
VRAM usage: ~15GB

### Step 8 — Logo detector (~3.5 h)
```bash
python scripts/05_train_detector.py
```
Config hiện tại (`configs/detector_yolov8.yaml`):
- YOLOv8m, class-agnostic (1 class: "logo"), imgsz=512, batch=64
- 30 epochs, patience=7, cache=disk

**Kết quả thực tế:** early stop epoch 16, best AP@0.5 = **0.6498** (epoch 9)
Best weights: `runs/detect/checkpoints/yolov8_logo/weights/best.pt`

> Gate ≥0.70 chưa đạt. Để cải thiện: tăng `imgsz: 640` hoặc dùng `yolov8l.pt`.

### Step 9 — Build gallery (~5–10 min)
```bash
python scripts/06_build_galleries.py
```
Output: `data/galleries/logodet3k.faiss` + `data/galleries/logodet3k_labels.json`

> Bắt buộc phải chạy trước `08_demo.py`. Không cần cho `07_eval.py`.

### Step 10 — Evaluation (~10–15 min)
```bash
python scripts/07_eval.py
```

**Kết quả thực tế** (closed_test: 1767 classes, 21497 objects):

| Metric | Ours | Paper target |
|---|---|---|
| Q-vs-G recall@1 | **0.9340** | 0.9836 |
| All-vs-All recall@1 | **0.9414** | 0.9886 |
| Small logo Q-vs-G | **0.9099** | — |
| Large logo Q-vs-G | **0.9498** | — |

### Step 11 — Demo
```bash
# Inference trên ảnh bất kỳ
python scripts/08_demo.py your_image.jpg --save_dir results/

# Với unknown threshold tùy chỉnh (mặc định 0.50)
python scripts/08_demo.py your_image.jpg --unknown_threshold 0.65 --save_dir results/
```
Pipeline: YOLOv8 detect → crop 160×160 → ViT embed → FAISS top-1 → brand label
- Box **đỏ** = brand nhận dạng được (score ≥ threshold)
- Box **cam** = unknown (score < threshold)

### Step 12 — Thêm brand mới vào gallery (không cần train lại)
```bash
# Ảnh đã crop sẵn logo (thêm 1 brand)
python scripts/add_brand.py --brand nike --folder path/to/nike_logos/

# Ảnh thực tế — YOLO tự detect và crop logo
python scripts/add_brand.py --brand nike --folder photos/ --use_detector

# Nhiều brand cùng lúc (mỗi subfolder = 1 brand)
python scripts/add_brand.py --folder_root my_brands/ --use_detector

# Xem danh sách brands trong gallery
python scripts/add_brand.py --list

# Xóa brand khỏi gallery
python scripts/add_brand.py --remove nike
```

**Xử lý brand trùng** (`--on_duplicate`):
- `ask` *(mặc định)* — hỏi interactively khi brand đã tồn tại
- `append` — thêm ảnh mới vào bên cạnh ảnh cũ (tăng coverage)
- `replace` — xóa ảnh cũ, dùng ảnh mới hoàn toàn
- `skip` — bỏ qua nếu brand đã có

```bash
# Ví dụ: tự động replace không hỏi
python scripts/add_brand.py --brand nike --folder logos/nike/ --on_duplicate replace
```

## Compute budget (1× RTX 5060 Ti 16 GB, Windows, no torch.compile)

| Phase | Thực tế |
|---|---|
| Build dataset | ~30–45 min |
| Smoke test | ~5 min |
| Phase A (15 epochs, batch 768) | ~2–3 h |
| HN mining | ~1 min |
| Phase C (25 epochs, batch 768) | ~5 h |
| Detector (YOLOv8m, 16 epochs early stop) | ~3.5 h |
| Gallery build | ~5–10 min |
| Eval | ~10–15 min |
| **Total** | **~11–13 h** |

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
| Batch | 768 (k=96 × m=8) |
| Max epochs | 15 |
| Early stopping patience | 6 |
| freeze_blocks | 0 (full fine-tune) |

### Phase C (`configs/hn_vit.yaml`)

| Param | Value |
|---|---|
| Init from | `checkpoints/vit_base.pt` |
| HN α1 / α2 | 0.05 / 0.35 |
| Levenshtein min | > 2 |
| Batch | 768 (k=96 × m=8) |
| Max epochs | 25 |
| freeze_blocks | 0 |

### Detector (`configs/detector_yolov8.yaml`)

| Param | Value |
|---|---|
| Model | yolov8m.pt |
| imgsz | 512 |
| batch | 64 |
| epochs | 30 (early stop at 16) |
| patience | 7 |
| cache | disk |

## Known limitations

- YOLO AP@0.5 = 0.65 (gate 0.70 chưa đạt) — cần `imgsz: 640` hoặc model lớn hơn
- `text_qvg = nan` — không có text-variant classes (chỉ LogoDet-3K, không có OpenLogo)
- `torch.compile` disabled trên Windows — mất ~15–20% speed
- Unknown threshold (0.50) cần calibrate trên val set để tối ưu F1
