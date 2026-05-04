# Kế hoạch trình bày đồ án: Hệ thống nhận diện logo

**Môn:** Nhập môn Thị giác Máy tính
**Thời lượng:** 30 phút
**Phân bổ thời gian gợi ý:** 3 + 4 + 12 + 7 + 4 phút


---

## Phần 1 — Tầm quan trọng của bài toán (3 phút)

Mở đầu bằng câu hỏi: "Mỗi ngày hàng tỷ ảnh được đăng lên mạng xã hội, trong đó logo thương hiệu xuất hiện ở khắp nơi — nhưng làm sao để tự động biết ảnh nào chứa logo gì?"

Nêu 4 ứng dụng thực tế:

- **Giám sát thương hiệu:** Doanh nghiệp cần biết logo mình xuất hiện ở đâu trên Internet, đo lường mức độ nhận diện, phân tích sentiment. Không thể quét thủ công hàng triệu ảnh mỗi ngày.
- **Phát hiện hàng giả / vi phạm bản quyền:** Logo được bảo hộ pháp lý. Hệ thống tự động giúp phát hiện sử dụng trái phép trên quy mô lớn.
- **Đo lường hiệu quả tài trợ thể thao:** Đo thời lượng logo xuất hiện trên truyền hình, livestream.
- **Tìm kiếm ảnh:** Chụp ảnh sản phẩm → nhận diện logo → tra thông tin thương hiệu.

Chuyển sang lý do bài toán khó — khác biệt so với phân loại ảnh thông thường:

- Logo xuất hiện ở mọi kích thước (vài pixel đến biển quảng cáo), bị xoay, nghiêng, biến dạng phối cảnh.
- Thường bị che khuất một phần, mờ, ngược sáng.
- Có hàng triệu thương hiệu và liên tục thay đổi — không thể đóng cứng số lớp.
- Nhiều logo trông rất giống nhau (font serif tương tự, logo hình tròn đơn giản chỉ khác chi tiết nhỏ).

**Kết luận phần này:** Cần một hệ thống vừa chính xác, vừa linh hoạt mở rộng mà không cần huấn luyện lại.


---

## Phần 2 — Phát biểu bài toán (4 phút)

### Input
Một ảnh tự nhiên bất kỳ (ảnh chụp sản phẩm, quảng cáo, ảnh đường phố...) có thể chứa 0, 1, hoặc nhiều logo.

### Output
Với mỗi logo trong ảnh:
- Vị trí (bounding box)
- Tên thương hiệu (hoặc "unknown" nếu logo không có trong cơ sở dữ liệu)
- Điểm tin cậy (cosine similarity)

### Điểm then chốt cần nhấn mạnh

Bài toán này **không phải** phân loại ảnh thông thường (closed-set classification). Lý do:

- Số lượng thương hiệu quá lớn (hàng triệu) và liên tục thay đổi.
- Phân loại truyền thống cần huấn luyện lại khi thêm lớp mới — tốn thời gian, tài nguyên.
- Cần xử lý logo chưa từng thấy (open-set).

→ Giải pháp: chuyển sang bài toán **image retrieval** (truy vấn ảnh). Thay vì phân loại, ta biến mỗi logo thành một vector rồi so sánh với cơ sở dữ liệu. Thêm thương hiệu mới chỉ cần thêm ảnh mẫu vào gallery — không cần huấn luyện lại.

### Pipeline tổng quan (3 giai đoạn)

```
Ảnh đầu vào → [Detection] → Vùng chứa logo → [Embedding] → Vector 128-d → [Retrieval] → Nhãn thương hiệu
```

- **Detection:** YOLOv8m phát hiện vị trí logo (class-agnostic — chỉ phân biệt "có logo" vs "không có logo").
- **Embedding:** ViT-B/32 chuyển mỗi vùng logo thành vector 128 chiều, chuẩn hóa L2.
- **Retrieval:** FAISS tìm vector gần nhất trong gallery bằng cosine similarity.


---

## Phần 3 — Phương pháp giải quyết (12 phút)

Phần chiếm nhiều thời gian nhất. Chia thành 4 mục nhỏ:

### 3.1 — Detector: YOLOv8m (2 phút)

- Nhiệm vụ: phát hiện tất cả vùng chứa logo trong ảnh.
- **Class-agnostic:** chỉ có 1 class duy nhất là "logo" — không cần biết thương hiệu. Lý do: nếu detector phân loại luôn thì phải huấn luyện lại khi thêm thương hiệu → không linh hoạt.
- Huấn luyện trên toàn bộ bounding box annotations (~128.000 crops) của OpenLogoDet-3K.
- Cấu hình: ảnh 640×640, batch 16, 50 epochs, optimizer SGD.
- Kết quả đạt mAP50 cao, đủ tin cậy để cung cấp crop cho giai đoạn embedding.

### 3.2 — Embedder: ViT-B/32 + CLIP (4 phút)

Đây là thành phần cốt lõi — cần giải thích kỹ.

**Backbone:** Dùng ViT-B/32 từ CLIP (OpenAI pretrained). Tại sao CLIP?
- Đã được pretrain trên 400 triệu cặp ảnh-text → hiểu tốt các đặc trưng thị giác.
- ViT xử lý ảnh bằng cách chia thành patches 32×32 pixel, rồi dùng Transformer encode.

**FC Head:** Thêm một lớp Linear (512 → 128) + L2 normalize. Vector 128-d này chính là "danh tính" của logo.

**Kích thước ảnh:** 160×160 thay vì 224×224 mặc định.
- Tiết kiệm ~50% bộ nhớ GPU → batch size lớn hơn.
- Phù hợp vì logo crop thường nhỏ (50-100px gốc).
- Positional embedding được interpolate từ 7×7 xuống 5×5 bằng bicubic.

**Ý tưởng quan trọng:** Sau khi huấn luyện, các logo cùng thương hiệu sẽ có vector gần nhau, khác thương hiệu sẽ xa nhau trong không gian 128 chiều. Inner product giữa 2 vector chuẩn hóa L2 chính là cosine similarity.

### 3.3 — Hàm mất mát: ProxyNCA++ (4 phút)

**Vấn đề:** Metric learning truyền thống (triplet loss, contrastive loss) cần tạo cặp/bộ ba mẫu → rất chậm, số lượng cặp tăng theo bình phương.

**Giải pháp — Proxy:** Thay vì so sánh trực tiếp giữa các ảnh, mỗi lớp được đại diện bằng một vector proxy (learnable parameter). Ảnh chỉ cần so sánh với proxy → giảm từ O(n²) xuống O(n×C).

**ProxyNCA++ loss:**
- Softmax-based: xác suất gán ảnh cho proxy đúng phải lớn nhất.
- Temperature σ = 0.06 (giá trị nhỏ → phân bố sắc nét, buộc mô hình tự tin hơn).
- Khoảng cách: squared L2 trên vector đã chuẩn hóa.

**Huấn luyện 2 pha:**

| | Phase A | Phase C |
|---|---|---|
| Hàm mất mát | ProxyNCA++ | ProxyNCAHN++ |
| Dữ liệu | Open-set (~1536 lớp) | Closed-set (~1920 lớp) |
| Sampler | MPerClassSampler (64 lớp × 8 ảnh) | HardNegativeBatchSampler |
| Mục tiêu | Học không gian embedding tổng quát | Tinh chỉnh cho các cặp khó |

**Phase A** — học tổng quát: huấn luyện trên 64% lớp (open-set), để mô hình học không gian embedding tốt cho cả lớp chưa thấy.

**Hard-negative mining:** Sau Phase A, chạy tất cả dữ liệu qua model, tính confusion matrix. Các cặp lớp có tỷ lệ nhầm lẫn từ 5% đến 35% (và tên lớp không quá giống nhau, Levenshtein > 2) được xác định là hard-negative pairs.

**Phase C** — tinh chỉnh: ProxyNCAHN++ mở rộng mẫu số của loss bằng cách thêm embedding thật của hard-negative (không chỉ proxy). Buộc mô hình phải phân biệt tốt hơn các cặp logo hay bị nhầm.

### 3.4 — Retrieval: FAISS (2 phút)

- Gallery: mỗi thương hiệu có nhiều ảnh mẫu → nhiều vector trong gallery.
- Dùng FAISS IndexFlatIP (exact search, inner product = cosine similarity trên vector chuẩn hóa).
- Tìm top-1 nearest neighbor. Nếu cosine similarity ≥ 0.50 → gán nhãn. Nếu < 0.50 → "unknown".
- Ngưỡng 0.50 tương đương góc 60° trong không gian embedding — query phải nằm trong vùng 60° quanh gallery vector.
- **Mở rộng:** Thêm logo mới chỉ cần embed vài ảnh mẫu rồi thêm vector vào gallery. Không cần train lại bất cứ gì.


---

## Phần 4 — Kết quả thực nghiệm (7 phút)

### 4.1 — Dataset (1.5 phút)

- **OpenLogoDet-3K:** hợp nhất LogoDet-3K (~2210 lớp, ~101.000 ảnh) + OpenLogo (~355 lớp, ~27.000 ảnh).
- Tổng: hơn 2400 lớp thương hiệu, ~128.000 ảnh.
- Tiền xử lý: chuẩn hóa tên lớp, loại trùng bằng perceptual hash (imagehash), lọc crop quá nhỏ (min_side < 15px).
- Chia tập:
  - Open-set: 64/16/20 theo lớp (lớp test chưa thấy khi train).
  - Closed-set: 64/16/20 theo ảnh (lớp test đã thấy khi train).

### 4.2 — Độ đo (1 phút)

- **Recall@1:** tỷ lệ query mà nearest neighbor thuộc đúng lớp. Đây là metric chuẩn cho image retrieval.
- Hai giao thức:
  - **Query-vs-Gallery (QvG):** query là test set, gallery là train set — mô phỏng thực tế.
  - **All-vs-All (AvA):** mỗi ảnh test lần lượt là query, phần còn lại là gallery.

### 4.3 — Phần cứng và lập trình (0.5 phút)

- GPU: NVIDIA RTX 5060 Ti 16GB
- Framework: PyTorch 2.4+, open_clip_torch, ultralytics (YOLOv8), faiss-cpu
- Toàn bộ pipeline viết bằng Python, chạy từng bước qua scripts

### 4.4 — Kết quả chính (2.5 phút)

Trình bày bảng kết quả:

| Giao thức | All-vs-All | QvG | Text QvG | Small QvG | Large QvG |
|---|---|---|---|---|---|
| Closed-set | 95.78% | 95.50% | 89.23% | 93.55% | 96.87% |
| Open-set | 97.32% | 96.76% | 93.62% | 95.23% | 97.86% |

Phân tích từng điểm:

- **Open-set cao hơn closed-set:** Có vẻ phản trực giác, nhưng giải thích: closed-set có ~1920 lớp (mật độ cao hơn trong embedding space) vs open-set chỉ ~480 lớp → ít nhầm lẫn hơn.
- **Text logos thấp nhất (89.23%):** Logo dạng chữ chỉ khác nội dung text, font và màu tương tự. ViT-B/32 với patch 32×32 trên ảnh 160×160 có thể không nắm bắt đủ chi tiết ở mức ký tự.
- **Small vs Large (93.55% vs 96.87%):** Logo nhỏ bị resize lên 160×160, quá trình nội suy tạo artifact và mất chi tiết.
- **Phase C cải thiện đáng kể so với Phase A** nhờ hard-negative mining buộc mô hình phân biệt tốt hơn các cặp logo dễ nhầm.

### 4.5 — Demo pipeline (1.5 phút)

Chiếu ảnh kết quả demo: ảnh đầu vào → bounding boxes + nhãn + điểm similarity.
Nhấn mạnh: pipeline chạy end-to-end, từ ảnh thô đến kết quả cuối cùng.


---

## Phần 5 — Đánh giá ưu điểm và hạn chế (4 phút)

### Ưu điểm

- **Kiến trúc retrieval-based linh hoạt:** Thêm logo mới chỉ cần vài ảnh mẫu, không cần huấn luyện lại. Đây là lợi thế lớn nhất so với classification-based.
- **Xử lý open-set tốt:** Recall@1 open-set 96.76% — mô hình tổng quát hóa tốt cho cả logo chưa từng thấy khi huấn luyện.
- **Huấn luyện 2 pha hiệu quả:** Phase A học tổng quát, Phase C tinh chỉnh cho cặp khó. Hard-negative mining giúp tập trung tài nguyên vào những lỗi thực sự.
- **Pipeline end-to-end hoàn chỉnh:** Từ ảnh thô → detection → embedding → retrieval → kết quả. Có thể triển khai thực tế.
- **Kết quả cạnh tranh:** 95.50% QvG trên bộ dữ liệu 2400+ lớp với phần cứng consumer-grade.

### Hạn chế

- **Text logos còn yếu (89.23%):** ViT-B/32 với patch size lớn và ảnh 160×160 khó nắm bắt chi tiết ký tự. Cải thiện: dùng patch size nhỏ hơn (ViT-B/16) hoặc tăng resolution.
- **Logo nhỏ mất chi tiết khi resize:** Crop < 30px gốc bị nội suy quá mức. Cải thiện: dùng super-resolution trước khi embed, hoặc multi-scale feature extraction.
- **FAISS exact search chưa tối ưu cho scale lớn:** IndexFlatIP tìm chính xác nhưng O(n) — khi gallery lên hàng triệu vector cần chuyển sang approximate search (IVF, HNSW).
- **Chưa xử lý real-time video:** Pipeline hiện tại xử lý từng ảnh. Mở rộng cho video cần tối ưu tốc độ inference.

### Hướng phát triển

- Thử ViT-B/16 hoặc tăng resolution cho text logos.
- Thêm giai đoạn super-resolution cho logo nhỏ.
- Chuyển FAISS sang IndexIVFFlat hoặc HNSW cho gallery lớn.
- Tích hợp tracking cho video real-time.


---

## Ghi chú trình bày

- Slide nên có nhiều hình minh họa: ảnh ví dụ logo, sơ đồ pipeline, biểu đồ kết quả.
- Phần 3 (Phương pháp) dài nhất — nên dùng animation để từng bước hiện lên, tránh quá tải thông tin.
- Chuẩn bị sẵn 1-2 ảnh demo chạy pipeline để chiếu trực tiếp nếu có thời gian.
- Dự phòng: nếu thiếu thời gian, rút ngắn mục 3.3 (ProxyNCA++) — chỉ giữ ý chính về proxy và 2 pha.
- Nếu thừa thời gian, mở rộng phần phân tích lỗi hoặc demo trực tiếp.
