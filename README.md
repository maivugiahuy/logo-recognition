# Logo Recognition

Reproduction of *Image-Text Pre-Training for Logo Recognition* (Hubenthal & Kumar, Amazon, arXiv:2309.10206).

## What gets trained

| Component | Loss | Data split | Output |
|---|---|---|---|
| ViT-B/32 embedder Phase A | ProxyNCA++ (eq 3) | Open-set train classes (~1600) | `checkpoints/vit_base.pt` |
| ViT-B/32 embedder Phase C | ProxyNCAHN++ (eq 5) | Closed-set train classes (~1900) | `checkpoints/vit_hn.pt` |
| YOLO26m detector | YOLO obj det loss | LogoDet-3K + OpenLogo boxes (class-agnostic) | `runs/detect/checkpoints/yolo26m_logo/weights/best.pt` |

## Substitutions vs paper

| Paper original | Substitution | Reason |
|---|---|---|
| CLIP pretraining on 20M e-comm pairs | OpenAI pretrained ViT-B/32 weights | Table 3: OpenAI IT ≥ E-comm IT; 20M pairs unavailable |
| YoloV4 on Amazon PL2K | YOLO26m on LogoDet-3K + OpenLogo | PL2K is proprietary |
| 4× V100 | 1× RTX 5060 Ti 16GB + AMP bfloat16 | Hardware |
| OpenLogoDet3K47 composite (4 datasets) | LogoDet-3K + OpenLogo | PL2K, Flickr32, BelgaLogos không có; 2 dataset công khai |

## Results (v1 — LogoDet-3K only)

> Kết quả dưới đây là run đầu tiên chỉ dùng LogoDet-3K. Sau khi retrain với LogoDet-3K + OpenLogo, cập nhật lại.

| Metric | Ours | Paper target | Gap |
|---|---|---|---|
| Phase A val recall@1 | **0.9675** | ~0.97 | -0.25pp |
| Phase C val recall@1 | **0.9356** | ~0.96 | -2.44pp |
| YOLO AP@0.5 (test) | **0.6498** | ≥0.70 | -5.0pp |
| Eval Q-vs-G recall@1 | **0.9340** | 0.9836 | -4.96pp |
| Eval All-vs-All recall@1 | **0.9414** | 0.9886 | -4.72pp |
| Eval small logo Q-vs-G | **0.9099** | — | — |
| Eval large logo Q-vs-G | **0.9498** | — | — |

Gap ~5pp so với paper là hợp lý: paper dùng 4 dataset, v1 chỉ dùng LogoDet-3K.

## Speed optimizations applied

- AMP mixed precision (bfloat16) — ~2× faster
- `torch.compile` — **disabled on Windows** (Triton không hỗ trợ Windows)
- `freeze_blocks=0` — unfreeze toàn bộ 12 ViT blocks (đúng paper, tốt hơn freeze)
- `num_workers=8` + `persistent_workers=True`
- TF32 enabled trên Ampere/Ada (`allow_tf32=True`)
- Early stopping patience=6 thay vì fixed epochs

---

## Execution steps

### Step 0 — Fresh Windows setup

#### 0.1 — Check GPU
```cmd
nvidia-smi
```
If not found → install NVIDIA driver first (step 0.2).

#### 0.2 — Install NVIDIA driver + CUDA Toolkit
RTX 5060 Ti (Blackwell) yêu cầu CUDA 12.8+.
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
| OpenLogo | ~2 GB | [GitHub](https://github.com/hang21/OpenLogo) | `data/raw/openlogo/openlogo/` |

LogoDet-3K cấu trúc: `LogoDet-3K/{category}/{ClassName}/{id}.jpg` + `{id}.xml`

OpenLogo cấu trúc:
```
data/raw/openlogo/openlogo/
  Annotations/   ← *.xml (Pascal VOC)
  JPEGImages/    ← *.jpg
  ImageSets/
```

### Step 3 — Build dataset (~45–60 min)
```bash
python scripts/01_build_dataset.py
```
Parse LogoDet-3K + OpenLogo → normalize → dedup → filter → tạo splits.

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
- `closed_train/val/test.json` — image-level splits trên seen classes

### Step 4 — Smoke test (~5–10 min)
```bash
python scripts/smoke_test.py
```
Gate: loss decreasing. Recall@1 = 0.0 ban đầu là bình thường.

### Step 5 — Phase A: base ViT (~2–3 h)
```bash
python scripts/02_train_base.py --config configs/base_vit.yaml
```
- LR: trunk 2.3e-6, FC 1.5e-3, proxy 71
- Batch 512 (k=64 × m=8), max 15 epochs, early stopping patience 6, σ=0.06
- freeze_blocks=0, AMP bfloat16, num_workers=8
- VRAM: ~9GB

Output: `checkpoints/vit_base.pt`

### Step 6 — Hard-negative mining (~1 min)
```bash
python scripts/03_mine_hn.py --ckpt checkpoints/vit_base.pt --config configs/base_vit.yaml
```
- h(yi) = {yj : 0.05 ≤ C[i,j] ≤ 0.35 AND levenshtein(name_i, name_j) > 2}
- Output: `data/processed/hn_map.json`

### Step 7 — Phase C: ProxyNCAHN++ (~5 h)
```bash
python scripts/04_train_hn.py --config configs/hn_vit.yaml
```
- Init từ `checkpoints/vit_base.pt`
- Closed-set, max 25 epochs, early stopping patience 6
- Batch 512 (k=64 × m=8), freeze_blocks=0
- VRAM: ~15GB

Output: `checkpoints/vit_hn.pt`

### Step 8 — Logo detector (~3.5 h)
```bash
python scripts/05_train_detector.py
```
Config: `configs/detector_yolo26.yaml`
- YOLO26m, class-agnostic (1 class: "logo"), imgsz=512, batch=64
- 30 epochs, patience=7, cache=disk

Output: `runs/detect/checkpoints/yolo26m_logo/weights/best.pt`

> Để cải thiện AP@0.5: tăng `imgsz: 640` hoặc dùng `yolo26l.pt`.

### Step 9 — Build gallery (~5–10 min)
```bash
python scripts/06_build_galleries.py
```
Output: `data/galleries/openlogodet3k.faiss` + `data/galleries/openlogodet3k_labels.json`

> Bắt buộc chạy trước demo. Không cần cho eval.

### Step 10 — Evaluation (~10–15 min)
```bash
python scripts/07_eval.py
```

### Step 11 — Demo
```bash
# Inference trên ảnh bất kỳ
python scripts/08_demo.py your_image.jpg --save_dir results/

# Với unknown threshold tùy chỉnh (mặc định 0.50)
python scripts/08_demo.py your_image.jpg --unknown_threshold 0.65 --save_dir results/
```
Pipeline: YOLO26 detect → crop 160×160 → ViT embed → FAISS top-1 → class label
- Box **đỏ** = class nhận dạng được (score ≥ threshold)
- Box **cam** = unknown (score < threshold)

### Step 11b — Xem danh sách classes
```bash
# Toàn bộ classes trong dataset → classes.txt
python scripts/list_classes.py

# Kèm stats (số ảnh, số objects) → classes.csv
python scripts/list_classes.py --format csv --out classes.csv

# Classes đang có trong gallery
python scripts/list_classes.py --source gallery

# Xuất từng split riêng
python scripts/list_classes.py --source splits
```

### Step 12 — Thêm class mới vào gallery (không cần train lại)
```bash
# Nhiều class cùng lúc từ data/new_classes/ (mặc định)
python scripts/add_class.py [--use_detector]

# 1 class từ folder
python scripts/add_class.py --class_name nike --folder path/to/nike_logos/

# Ảnh thực tế — YOLO tự detect và crop logo
python scripts/add_class.py --class_name nike --folder photos/ --use_detector

# Xem danh sách classes trong gallery
python scripts/add_class.py --list

# Xóa class khỏi gallery
python scripts/add_class.py --remove nike
```

Cấu trúc `data/new_classes/`:
```
data/new_classes/
  nike/       ← ảnh logo Nike
  adidas/     ← ảnh logo Adidas
  ...
```

`--on_duplicate`: `ask` (mặc định) / `append` / `replace` / `skip`

---

## Compute budget (1× RTX 5060 Ti 16 GB, Windows)

| Phase | Thực tế |
|---|---|
| Build dataset | ~45–60 min |
| Smoke test | ~5 min |
| Phase A (15 epochs, batch 512) | ~2–3 h |
| HN mining | ~1 min |
| Phase C (25 epochs, batch 512) | ~5 h |
| Detector (YOLO26m) | ~3.5 h |
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

### Detector (`configs/detector_yolo26.yaml`)

| Param | Value |
|---|---|
| Model | yolo26m.pt |
| imgsz | 512 |
| batch | 64 |
| epochs | 30 |
| patience | 7 |
| cache | disk |

---

## Known limitations

- YOLO AP@0.5 = 0.65 (gate 0.70 chưa đạt) — cần `imgsz: 640` hoặc `yolo26l.pt`
- `torch.compile` disabled trên Windows — mất ~15–20% speed
- Unknown threshold (0.50) cần calibrate trên val set để tối ưu F1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    