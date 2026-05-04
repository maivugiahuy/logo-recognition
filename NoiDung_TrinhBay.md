# Nội dung nói trình bày — Hệ thống nhận diện logo

---

## SLIDE MỞ ĐẦU

Xin chào mọi người, hôm nay mình sẽ trình bày đồ án "Hệ thống nhận diện logo dựa trên học biểu diễn sâu và truy vấn ảnh".

---

## PHẦN 1 — TẦM QUAN TRỌNG CỦA BÀI TOÁN

Trước tiên mình đặt câu hỏi: mỗi ngày có hàng tỷ ảnh được đăng lên mạng xã hội, trong đó logo thương hiệu xuất hiện ở khắp nơi. Vậy làm sao để một máy tính tự động biết ảnh nào chứa logo gì?

Bài toán nhận diện logo có nhiều ứng dụng thực tế rất thiết thực.

Thứ nhất là giám sát thương hiệu. Các doanh nghiệp cần theo dõi logo mình xuất hiện ở đâu trên Internet để đo lường mức độ nhận diện thương hiệu. Mỗi ngày có hàng triệu ảnh mới, không thể làm thủ công được.

Thứ hai là phát hiện hàng giả và vi phạm bản quyền. Logo được bảo hộ pháp lý, nên cần hệ thống tự động để phát hiện việc sử dụng trái phép trên quy mô lớn.

Thứ ba là đo lường hiệu quả tài trợ. Trong thể thao hay sự kiện, nhà tài trợ muốn biết logo mình xuất hiện bao nhiêu lần, bao lâu trên truyền hình hay livestream.

Và cuối cùng là hỗ trợ tìm kiếm. Người dùng có thể chụp ảnh sản phẩm, hệ thống nhận ra logo rồi tra thông tin thương hiệu.

Tuy nhiên bài toán này khó hơn nhiều so với phân loại ảnh thông thường. Logo có thể xuất hiện ở mọi kích thước, từ vài pixel trên nhãn sản phẩm đến rất lớn trên biển quảng cáo. Chúng thường bị che khuất một phần, bị mờ, hoặc chụp ở góc nghiêng. Trên thế giới có hàng triệu thương hiệu và liên tục có thương hiệu mới. Ngoài ra, nhiều logo trông rất giống nhau, chỉ khác nhau ở chi tiết nhỏ.

Vì vậy, mình cần một hệ thống vừa chính xác, vừa có thể mở rộng linh hoạt mà không cần huấn luyện lại.

---

## PHẦN 2 — PHÁT BIỂU BÀI TOÁN

Bài toán được phát biểu như sau.

Input là một ảnh tự nhiên bất kỳ, có thể là ảnh chụp sản phẩm, ảnh quảng cáo, hay ảnh đường phố. Ảnh có thể chứa không, một, hoặc nhiều logo.

Output là với mỗi logo trong ảnh, hệ thống trả về vị trí dưới dạng bounding box, tên thương hiệu hoặc nhãn "unknown" nếu logo không có trong cơ sở dữ liệu, và điểm tin cậy.

Điểm quan trọng ở đây là mình không giải bài toán này bằng phân loại truyền thống. Lý do là số lượng thương hiệu quá lớn, liên tục thay đổi, và nếu dùng phân loại thì mỗi lần thêm thương hiệu mới phải huấn luyện lại toàn bộ mô hình, rất tốn kém.

Thay vào đó, mình tiếp cận bằng truy vấn ảnh, tức là image retrieval. Ý tưởng là biến mỗi logo thành một vector số, rồi so sánh với cơ sở dữ liệu để tìm logo giống nhất. Muốn thêm thương hiệu mới thì chỉ cần thêm ảnh mẫu vào cơ sở dữ liệu, không cần huấn luyện lại gì cả.

Pipeline tổng quan gồm ba giai đoạn. Đầu tiên là Detection, dùng YOLOv8m để phát hiện vị trí logo. Tiếp theo là Embedding, dùng ViT-B/32 để chuyển mỗi vùng logo thành vector 128 chiều. Cuối cùng là Retrieval, dùng FAISS để tìm vector gần nhất trong gallery và trả về nhãn.

---

## PHẦN 3 — PHƯƠNG PHÁP GIẢI QUYẾT

Giờ mình sẽ đi chi tiết từng thành phần.

### 3.1 — Detector: YOLOv8m

Thành phần đầu tiên là detector. Mình dùng YOLOv8m để phát hiện tất cả vùng chứa logo trong ảnh.

Điểm đặc biệt là detector này hoạt động theo kiểu class-agnostic, nghĩa là chỉ có một class duy nhất là "logo". Nó không quan tâm đó là logo gì, chỉ cần xác định "ở đây có logo". Lý do mình thiết kế như vậy là để detector không phụ thuộc vào số lượng thương hiệu, khi thêm thương hiệu mới thì detector vẫn hoạt động bình thường.

Mô hình được huấn luyện trên toàn bộ bounding box annotations của bộ dữ liệu, khoảng 128 nghìn crops, với ảnh đầu vào 640×640, batch size 16, trong 50 epochs.

### 3.2 — Embedder: ViT-B/32

Thành phần thứ hai và cũng là cốt lõi nhất là embedder. Nhiệm vụ của nó là chuyển mỗi vùng logo đã crop thành một vector 128 chiều đại diện cho "danh tính" của logo đó.

Mình dùng backbone là ViT-B/32 từ CLIP của OpenAI. Tại sao chọn CLIP? Vì nó đã được pretrain trên 400 triệu cặp ảnh và text, nên đã học được rất nhiều đặc trưng thị giác hữu ích. ViT xử lý ảnh bằng cách chia thành các patches 32×32 pixel, rồi dùng Transformer để encode.

Trên backbone này mình thêm một lớp Fully Connected từ 512 chiều xuống 128 chiều, rồi chuẩn hóa L2. Vector 128 chiều cuối cùng này chính là embedding của logo.

Về kích thước ảnh, mình dùng 160×160 thay vì 224×224 mặc định. Lý do là tiết kiệm khoảng 50% bộ nhớ GPU, cho phép batch size lớn hơn. Hơn nữa, logo crop thường chỉ có kích thước gốc 50 đến 100 pixel, nên resize lên 160 đã là nội suy đáng kể rồi. Positional embedding của ViT được interpolate từ grid 7×7 xuống 5×5 bằng bicubic.

Mục tiêu huấn luyện là làm sao cho các logo cùng thương hiệu có vector gần nhau, và khác thương hiệu thì xa nhau trong không gian 128 chiều.

### 3.3 — Hàm mất mát và huấn luyện 2 pha

Để huấn luyện embedder, mình dùng metric learning. Cụ thể là ProxyNCA++.

Metric learning truyền thống như triplet loss hay contrastive loss cần tạo cặp hoặc bộ ba mẫu để so sánh, số lượng cặp tăng rất nhanh nên rất chậm. ProxyNCA++ giải quyết vấn đề này bằng khái niệm proxy. Mỗi lớp được đại diện bởi một vector proxy, là tham số học được. Thay vì so sánh ảnh với ảnh, ta so sánh ảnh với proxy, giảm đáng kể chi phí tính toán.

Loss function hoạt động theo kiểu softmax: xác suất gán ảnh cho proxy đúng phải lớn nhất. Temperature sigma bằng 0.06, giá trị nhỏ này buộc mô hình phải tự tin trong quyết định.

Quá trình huấn luyện chia thành hai pha.

Phase A huấn luyện với ProxyNCA++ trên tập open-set, khoảng 1536 lớp. Mỗi batch gồm 64 lớp nhân 8 ảnh bằng 512 mẫu. Mục tiêu của phase này là học một không gian embedding tổng quát, hoạt động tốt cho cả những logo chưa thấy.

Sau Phase A, mình chạy hard-negative mining. Đưa toàn bộ dữ liệu qua model, tính confusion matrix, rồi tìm các cặp lớp có tỷ lệ nhầm lẫn từ 5% đến 35%. Đây là những cặp logo mà mô hình hay nhầm, ví dụ hai logo hình tròn giống nhau hoặc hai logo chữ dùng font tương tự.

Phase C tiếp tục huấn luyện từ checkpoint Phase A, nhưng dùng ProxyNCAHN++. Phiên bản mở rộng này thêm embedding thật của hard-negative vào mẫu số của loss, buộc mô hình phải học phân biệt tốt hơn chính những cặp logo hay bị nhầm nhất.

### 3.4 — Retrieval bằng FAISS

Giai đoạn cuối là truy vấn. Gallery chứa nhiều vector embedding cho mỗi thương hiệu. Khi có một logo cần nhận diện, mình embed nó thành vector 128 chiều rồi dùng FAISS tìm vector gần nhất trong gallery bằng cosine similarity.

Nếu cosine similarity lớn hơn hoặc bằng 0.50 thì gán nhãn thương hiệu đó. Nếu nhỏ hơn 0.50 thì trả về "unknown", nghĩa là logo này không có trong cơ sở dữ liệu. Ngưỡng 0.50 tương đương góc 60 độ trong không gian embedding.

Ưu điểm lớn nhất: muốn thêm logo mới chỉ cần cung cấp vài ảnh mẫu, hệ thống tự embed và thêm vào gallery, không cần huấn luyện lại bất cứ mô hình nào.

---

## PHẦN 4 — KẾT QUẢ THỰC NGHIỆM

### Dataset

Mình sử dụng bộ dữ liệu OpenLogoDet-3K, được tạo bằng cách hợp nhất hai bộ dữ liệu: LogoDet-3K với khoảng 2210 lớp và 101 nghìn ảnh, cùng với OpenLogo khoảng 355 lớp và 27 nghìn ảnh. Tổng cộng hơn 2400 lớp thương hiệu.

Dữ liệu được tiền xử lý: chuẩn hóa tên lớp, loại bỏ ảnh trùng bằng perceptual hash, và lọc những crop quá nhỏ dưới 15 pixel.

Chia tập theo hai giao thức. Open-set chia 64/16/20 theo lớp, nghĩa là các lớp trong test set chưa từng thấy khi huấn luyện. Closed-set chia 64/16/20 theo ảnh, các lớp đã thấy nhưng ảnh test thì chưa thấy.

### Độ đo

Mình dùng Recall@1, là tỷ lệ query mà nearest neighbor thuộc đúng lớp. Đây là metric chuẩn cho image retrieval.

Có hai cách đánh giá: Query-vs-Gallery là dùng test set làm query và train set làm gallery, mô phỏng đúng tình huống thực tế. All-vs-All là mỗi ảnh test lần lượt làm query, phần còn lại làm gallery.

### Phần cứng

Toàn bộ huấn luyện và đánh giá chạy trên một GPU NVIDIA RTX 5060 Ti 16GB, hệ điều hành Windows 11. Framework chính là PyTorch, cùng với open_clip, ultralytics cho YOLOv8, và faiss.

### Kết quả

Đây là bảng kết quả chính.

Closed-set đạt Recall@1 là 95.78% All-vs-All và 95.50% Query-vs-Gallery. Open-set đạt 97.32% All-vs-All và 96.76% Query-vs-Gallery.

Một điểm đáng chú ý: open-set lại cao hơn closed-set. Nghe có vẻ phản trực giác vì open-set là các lớp chưa thấy. Nhưng giải thích là closed-set có khoảng 1920 lớp, mật độ trong embedding space cao hơn nên dễ nhầm hơn. Trong khi open-set chỉ có khoảng 480 lớp, ít hơn nhiều nên ít nhầm lẫn. Điều này cũng cho thấy không gian embedding tổng quát hóa rất tốt.

Text logos là tập con khó nhất, chỉ đạt 89.23% ở closed-set. Logo dạng chữ thường chỉ khác nhau ở nội dung text, trong khi font và màu sắc tương tự. ViT-B/32 với patch size 32 pixel trên ảnh 160×160 có thể không nắm bắt đủ chi tiết ở mức ký tự.

Logo nhỏ cho kết quả thấp hơn logo lớn khoảng 2 đến 3 phần trăm, do quá trình resize từ kích thước rất nhỏ lên 160×160 tạo ra artifact và mất chi tiết.

---

## PHẦN 5 — ĐÁNH GIÁ ƯU ĐIỂM VÀ HẠN CHẾ

### Ưu điểm

Ưu điểm lớn nhất là kiến trúc retrieval-based rất linh hoạt. Thêm logo mới chỉ cần vài ảnh mẫu, không cần huấn luyện lại. Đây là khác biệt lớn so với phương pháp phân loại truyền thống.

Hệ thống xử lý open-set rất tốt với Recall@1 đạt 96.76%, chứng tỏ mô hình tổng quát hóa tốt cho cả logo chưa từng thấy.

Chiến lược huấn luyện 2 pha hiệu quả: Phase A học không gian tổng quát, Phase C tinh chỉnh cho các cặp khó nhờ hard-negative mining.

Pipeline hoàn chỉnh end-to-end, từ ảnh thô đến kết quả cuối cùng, có thể triển khai thực tế.

Và kết quả 95.50% trên bộ dữ liệu hơn 2400 lớp với phần cứng consumer-grade là khá tốt.

### Hạn chế

Text logos còn yếu, chỉ đạt 89.23%. Nguyên nhân chính là patch size 32 pixel quá lớn và resolution 160×160 quá nhỏ để nắm bắt chi tiết ký tự. Có thể cải thiện bằng cách dùng ViT-B/16 với patch nhỏ hơn hoặc tăng resolution.

Logo nhỏ mất chi tiết khi resize. Các crop dưới 30 pixel gốc bị nội suy quá mức khi resize lên 160×160. Có thể cải thiện bằng super-resolution trước khi embed.

FAISS exact search hiện tại có độ phức tạp O(n), khi gallery lên hàng triệu vector sẽ chậm, cần chuyển sang approximate search như IVF hoặc HNSW.

Và hiện tại pipeline chưa xử lý được video real-time, chỉ xử lý từng ảnh riêng lẻ.

### Hướng phát triển

Thử ViT-B/16 hoặc tăng resolution cho text logos. Thêm super-resolution cho logo nhỏ. Chuyển FAISS sang approximate search cho gallery lớn. Và tích hợp object tracking để xử lý video real-time.

---

## SLIDE KẾT THÚC

Tóm lại, đồ án xây dựng một hệ thống nhận diện logo end-to-end gồm ba giai đoạn: detection bằng YOLOv8m, embedding bằng ViT-B/32 với ProxyNCA++, và retrieval bằng FAISS. Kết quả đạt Recall@1 trên 95% cho cả closed-set và open-set. Ưu điểm nổi bật nhất là khả năng mở rộng linh hoạt mà không cần huấn luyện lại.

Cảm ơn mọi người đã lắng nghe. Mình sẵn sàng nhận câu hỏi.
