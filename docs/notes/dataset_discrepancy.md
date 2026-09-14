# VietSuperSpeech: số liệu thật khác đáng kể so với đề cương

Phát hiện khi chạy `src/data/survey.py` lần đầu (2026-09-14, chạy trực tiếp
từ máy local, không qua Kaggle — script CPU-only, không cần GPU).

## Ba nguồn số liệu, ba kết quả khác nhau

| Nguồn | Tổng mẫu | Tổng giờ | Train | Dev/Validation | Độ dài audio |
|---|---|---|---|---|---|
| **Đề cương đã đăng ký** | 52.023 | 267,39h | 46.822 (240,67h) | "dev-test" 5.201 (26,72h) | 3–30 giây |
| **README của dataset trên HF Hub** | 32.267 | 103,18h | 29.041 | "dev" 3.226 | ~12s trung bình |
| **Đo thật (`load_dataset`, 2026-09-14)** | **67.405** | **245,42h** | **60.656 (220,84h)** | **"validation" 6.749 (24,57h)** | **10–15 giây (gần như đồng nhất)** |

Ba con số này **không khớp nhau ở bất kỳ chỗ nào** — không phải do dataset
vừa được cập nhật: đã kiểm tra lịch sử commit của repo HF
(`thanhnew2001/VietSuperSpeech`, 1.323 commit), commit mới nhất là
**2026-02-22**, tức là dữ liệu đã đứng yên từ 7 tháng trước, trước cả khi
đề cương được viết (2026-09-07). README của chính dataset cũng đã lỗi thời
so với dữ liệu thật (khả năng README viết trước khi tác giả dataset upload
xong toàn bộ, rồi không cập nhật lại).

**Kết luận: số liệu trong đề cương (52.023 mẫu, 267,39h, độ dài 3-30s,
tên split "dev-test") không khớp với dữ liệu thật hiện có trên HF Hub, và
không phải do dataset thay đổi sau khi đề cương được viết.**

## Chi tiết đo thật (2026-09-14)

- **Tên split đúng**: `train` và `validation` (KHÔNG phải `dev-test`).
- **train**: 60.656 mẫu, 220,84 giờ, 645 nguồn video khác nhau.
- **validation**: 6.749 mẫu, 24,57 giờ, 562 nguồn video khác nhau.
- **Độ dài audio gần như đồng nhất 10–15 giây** (p50=13,4s, p90=15,0s,
  p99=15,0s, min=10,0s, max=15,0s) — **không có mẫu nào ngắn hơn 10s hoặc
  dài hơn 15s** trong toàn bộ dữ liệu đo được. Điều này khác hẳn tuyên bố
  "câu dài 3-30 giây" trong đề cương.
- Transcript toàn bộ là **CHỮ HOA** (UPPERCASE), 96 ký tự khác nhau (bảng
  chữ cái tiếng Việt có dấu + khoảng trắng, không có dấu câu).
  `empty_text_count: 0` — không có transcript rỗng.
- File kết quả đầy đủ:
  `reports/results/dataset_survey_train.json`,
  `reports/results/dataset_survey_validation.json`.

## Vì sao việc này quan trọng — ảnh hưởng đến RQ2

Đề cương đặt RQ2 (hiệu năng suy luận) dựa trên giả định có **dải độ dài
audio đa dạng (3-30s)** để thấy rõ lợi thế O(n) của Mamba so với O(n²) của
Conformer *"đặc biệt khi độ dài audio đầu vào tăng lên"*. Với dữ liệu thật
gần như đồng nhất 10-15 giây, **không có đủ dải độ dài để quan sát xu
hướng này một cách thuyết phục** như thiết kế ban đầu. Đây không phải lỗi
kỹ thuật có thể tự sửa bằng code — cần quyết định về phương pháp.

## Các hướng xử lý (cần quyết định, chưa tự ý chọn)

1. **Giữ nguyên, chấp nhận dải hẹp hơn**: vẫn đo RTF theo bucket nhỏ hơn
   (vd. 10-11s, 11-12s, ..., 15s) — chênh lệch tuyệt đối giữa 10s và 15s
   vẫn có thể cho thấy xu hướng, dù kém thuyết phục hơn 3-30s.
2. **Ghép nhiều đoạn liên tiếp cùng nguồn** (cùng giá trị `source`) thành
   audio dài hơn để tạo dải độ dài nhân tạo cho riêng thí nghiệm RTF (không
   dùng để train, chỉ dùng để đo latency) — cần nêu rõ phương pháp ghép
   trong báo cáo.
3. **Tìm thêm nguồn dữ liệu bổ sung** có audio dài hơn chỉ để phục vụ RQ2
   — tăng phạm vi, thêm rủi ro/khối lượng công việc.
4. **Điều chỉnh lại đề cương** cho khớp số liệu thật (báo GVHD) — xử lý
   cùng lúc với việc số liệu tổng thể cũng sai khác (mẫu 3, dòng "Dữ liệu
   sử dụng").

Xem thêm: [`docs/notes/mamba_ssm_install_log.md`](mamba_ssm_install_log.md)
cho tiền lệ tương tự (đề cương vs Kaggle) — lần đó người dùng chọn "chỉ sửa
tài liệu nội bộ, giữ nguyên đề cương đã nộp". Trường hợp này nghiêm trọng
hơn vì ảnh hưởng trực tiếp đến thiết kế RQ2, không chỉ là chi tiết hạ tầng.
