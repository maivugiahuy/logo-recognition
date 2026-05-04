# Kế hoạch slide trình bày — Hệ thống nhận diện logo

**Tổng số slide dự kiến:** 20–22 slide
**Thời lượng:** 30 phút (~1.5 phút/slide)
**Ảnh minh họa:** dùng image placeholder, người làm thay thế sau 

---

## Slide 1 — Trang bìa

- Tên đồ án: "Hệ thống nhận diện logo dựa trên học biểu diễn sâu và truy vấn ảnh"
- Tên tiếng Anh: "Logo Recognition System Based on Deep Representation Learning and Image Retrieval"
- Môn: Nhập môn Thị giác Máy tính
- Tên sinh viên, MSSV
- Nền tối, logo trường, font sạch

---

## Slide 2 — Mục lục trình bày

5 phần:
1. Tầm quan trọng của bài toán
2. Phát biểu bài toán
3. Phương pháp giải quyết
4. Kết quả thực nghiệm
5. Đánh giá và hạn chế

---

## PHẦN 1 — TẦM QUAN TRỌNG (2 slide)

### Slide 3 — Ứng dụng thực tế

4 icon/ảnh minh họa, mỗi cái kèm 1 dòng ngắn:
- Giám sát thương hiệu — theo dõi logo trên mạng xã hội
- Phát hiện hàng giả — phát hiện sử dụng logo trái phép
- Đo lường tài trợ — đếm logo trên truyền hình/sự kiện
- Tìm kiếm ảnh — chụp sản phẩm → tra thương hiệu

### Slide 4 — Thách thức

4 ảnh ví dụ minh họa từng thách thức:
- Đa dạng kích thước và góc độ (logo rất nhỏ vs rất lớn)
- Che khuất, mờ, ngược sáng
- Số lượng thương hiệu lớn, liên tục thay đổi
- Logo tương tự nhau giữa các thương hiệu khác nhau

---

## PHẦN 2 — PHÁT BIỂU BÀI TOÁN (3 slide)

### Slide 5 — Input / Output

Chia slide thành 2 cột:
- Bên trái (Input): ảnh tự nhiên bất kỳ chứa logo
- Bên phải (Output): ảnh có bounding box + nhãn thương hiệu + điểm similarity
- Ghi chú: output có thể là "unknown" nếu logo không có trong gallery

### Slide 6 — Tại sao không dùng phân loại?

So sánh 2 cột:
- Classification: số lớp cố định, thêm lớp mới phải train lại, không xử lý unknown
- Retrieval (đồ án): thêm lớp mới chỉ cần thêm ảnh, xử lý unknown bằng ngưỡng, linh hoạt

### Slide 7 — Pipeline tổng quan

Sơ đồ ngang 3 khối nối mũi tên:
```
Ảnh → [YOLOv8m Detection] → Vùng logo → [ViT-B/32 Embedding] → Vector 128-d → [FAISS Retrieval] → Nhãn
```
Mỗi khối ghi tên model và chức năng ngắn gọn. Đây là slide quan trọng, sẽ quay lại nhiều lần.

---

## PHẦN 3 — PHƯƠNG PHÁP (8 slide)

### Slide 8 — Detection: YOLOv8m

- Ảnh minh họa: ảnh gốc → ảnh có bounding boxes
- Gạch đầu dòng: class-agnostic (1 class "logo"), ảnh 640×640, 50 epochs
- Nhấn mạnh: detector không cần biết thương hiệu, chỉ phát hiện vị trí

### Slide 9 — Embedder: ViT-B/32 kiến trúc

Sơ đồ kiến trúc embedder:
```
Logo crop (160×160) → Patch Embedding (5×5 patches) → 12 Transformer Blocks → [CLS] token (512-d) → FC (128-d) → L2 Normalize
```
Ghi chú: backbone từ CLIP (pretrained trên 400M cặp ảnh-text)

### Slide 10 — Embedding space

Hình minh họa không gian embedding:
- Các chấm màu đại diện các logo, cùng thương hiệu gần nhau, khác thương hiệu xa nhau
- Mũi tên cosine similarity giữa query và gallery
- Ghi chú: inner product trên vector chuẩn hóa L2 = cosine similarity

### Slide 11 — ProxyNCA++ loss

- Hình minh họa: ảnh so sánh với proxy thay vì so sánh với nhau
- Công thức loss (đơn giản hóa): -log(exp(-d(x,p+)) / Σ exp(-d(x,p-)))
- Gạch đầu dòng: mỗi lớp 1 proxy (learnable), temperature σ = 0.06, khoảng cách squared L2

### Slide 12 — Huấn luyện Phase A

- Phase A: ProxyNCA++ trên open-set (~1536 lớp)
- Batch: 64 lớp × 8 ảnh = 512 mẫu (MPerClassSampler)
- Mục tiêu: học không gian embedding tổng quát
- Output: checkpoint vit_base.pt

### Slide 13 — Hard-Negative Mining

Sơ đồ quy trình:
```
Checkpoint Phase A → Chạy toàn bộ dữ liệu → Confusion matrix → Lọc cặp nhầm lẫn 5%–35% → hn_map.json
```
- Ảnh ví dụ: 2 logo giống nhau mà model hay nhầm
- Điều kiện thêm: Levenshtein distance > 2 (loại biến thể tên)

### Slide 14 — Huấn luyện Phase C

- Phase C: ProxyNCAHN++ trên closed-set (~1920 lớp)
- Khác biệt với Phase A: thêm embedding thật của hard-negative vào mẫu số loss
- HardNegativeBatchSampler: thêm mẫu hard-negative vào mỗi batch
- Init từ checkpoint Phase A
- Output: checkpoint vit_hn.pt (cuối cùng)

### Slide 15 — Retrieval: FAISS

- Sơ đồ: query vector → FAISS IndexFlatIP → top-1 nearest neighbor → nhãn
- Ngưỡng τ = 0.50 (cosine similarity): ≥ 0.50 → gán nhãn, < 0.50 → "unknown"
- Hình minh họa: góc 60° trong không gian embedding
- Mở rộng gallery: thêm ảnh mẫu → embed → thêm vào FAISS, không cần train lại

---

## PHẦN 4 — KẾT QUẢ THỰC NGHIỆM (4 slide)

### Slide 16 — Dataset và tiền xử lý

- OpenLogoDet-3K = LogoDet-3K (2210 lớp) + OpenLogo (355 lớp) = 2400+ lớp
- Tiền xử lý: chuẩn hóa tên, loại trùng (perceptual hash), lọc crop < 15px
- Chia tập: open-set (64/16/20 theo lớp), closed-set (64/16/20 theo ảnh)
- Phần cứng: RTX 5060 Ti 16GB, PyTorch 2.4+

### Slide 17 — Bảng kết quả chính

Bảng lớn chiếm giữa slide:

| Giao thức | All-vs-All | QvG | Text QvG | Small QvG | Large QvG |
|---|---|---|---|---|---|
| Closed-set | 95.78% | 95.50% | 89.23% | 93.55% | 96.87% |
| Open-set | 97.32% | 96.76% | 93.62% | 95.23% | 97.86% |

Highlight ô QvG bằng màu nổi bật

### Slide 18 — Phân tích kết quả

4 điểm phân tích, mỗi điểm 1 dòng kèm con số:
- Open-set > Closed-set: 96.76% vs 95.50% → embedding tổng quát hóa tốt
- Text logos thấp nhất: 89.23% → patch size 32px khó bắt chi tiết ký tự
- Small < Large: 93.55% vs 96.87% → resize từ crop nhỏ mất chi tiết
- Phase C cải thiện rõ rệt so với Phase A nhờ hard-negative mining

### Slide 19 — Demo pipeline

- Ảnh kết quả inference end-to-end: ảnh gốc bên trái, ảnh có boxes + nhãn bên phải
- Hoặc chạy demo trực tiếp nếu có thời gian

---

## PHẦN 5 — ĐÁNH GIÁ VÀ HẠN CHẾ (2 slide)

### Slide 20 — Ưu điểm

5 điểm, mỗi điểm 1 dòng ngắn:
- Retrieval-based: thêm logo mới không cần train lại
- Open-set mạnh: 96.76% trên logo chưa thấy
- 2 pha hiệu quả: Phase A tổng quát + Phase C tinh chỉnh
- Pipeline end-to-end hoàn chỉnh
- 95.50% trên 2400+ lớp với GPU consumer-grade

### Slide 21 — Hạn chế và hướng phát triển

Chia 2 cột:

Hạn chế:
- Text logos yếu (89.23%) — patch size lớn khó bắt chi tiết ký tự
- Logo nhỏ mất chi tiết khi resize lên 160×160
- FAISS exact search chưa tối ưu cho gallery hàng triệu vector
- Chưa xử lý video real-time

Hướng phát triển:
- ViT-B/16 hoặc tăng resolution cho text logos
- Super-resolution cho logo nhỏ
- FAISS approximate search (IVF, HNSW) cho scale lớn
- Tích hợp tracking cho video real-time

---

## Slide 22 — Kết thúc

- Tóm tắt 1 câu: "Hệ thống 3 giai đoạn Detection → Embedding → Retrieval đạt Recall@1 trên 95%, mở rộng linh hoạt không cần huấn luyện lại"
- "Cảm ơn — Q&A"

---

## Ghi chú thiết kế

- **Font:** sans-serif sạch (Montserrat, Inter, hoặc Roboto)
- **Màu chủ đạo:** xanh dương đậm + trắng, accent cam/vàng cho highlight
- **Nền:** tối (dark theme) hoặc trắng tùy sở thích
- **Hình:** ưu tiên sơ đồ tự vẽ, ảnh minh họa thực tế, tránh wall of text
- **Mỗi slide tối đa 5–6 dòng text**, còn lại là hình/sơ đồ
- **Slide pipeline (slide 7):** in nhỏ lặp lại ở góc các slide phương pháp, highlight phần đang nói
