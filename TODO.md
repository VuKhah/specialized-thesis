# TODO tổng — Mamba vs Conformer ASR

Cập nhật lần cuối: **2026-09-16**. File này là danh sách việc tổng, cập nhật
theo tiến độ thật (không phải kế hoạch tĩnh) — xem thêm README.md mục
"Trạng thái hiện tại" cho chi tiết từng tuần.

## 🔴 Đang chặn — chờ quyết định từ GVHD

- [ ] **Hướng xử lý sai lệch số liệu dataset ảnh hưởng RQ2**: số liệu thật
      (67.405 mẫu/245,42h, độ dài audio gần như đồng nhất 10-15s) khác đề
      cương đã đăng ký (52.023/267,39h, dải 3-30s) — làm RQ2 (RTF theo độ
      dài audio) không còn đủ dải để chứng minh ưu thế O(n) của Mamba. 4
      phương án trong `docs/notes/dataset_discrepancy.md`, đang chờ ý kiến
      thầy Hoàng Văn Dũng. **Không tự chọn phương án khi chưa có ý kiến.**

## 🟡 Sẵn sàng làm ngay — không phụ thuộc quyết định trên

- [ ] **Chạy prefetch audio đầy đủ** (67.405 file, ước tính ~27GB) — script
      đã viết xong và test logic (`src/data/prefetch_audio.py`), nhưng
      *chưa chạy full*. Cần quyết định chạy ở đâu: máy local (đã xác nhận
      có internet trực tiếp) hay để dành chạy trên Kaggle lúc train thật
      (Tuần 4-5) — cache tải ở local không tự chuyển sang Kaggle được.
- [ ] **Hoàn thiện training loop thật** trong `src/training/train.py` (hiện
      là skeleton): checkpoint/resume (quan trọng — Kaggle session cap
      ~9-12h), eval loop tính WER định kỳ, logging (tensorboard).
- [ ] **Tune tham số Mamba khớp Conformer**: chạy
      `python -m src.models.param_count` (cần mamba-ssm tag v2.3.1 đã cài)
      để chỉnh `configs/model_mamba.yaml`, mục tiêu chênh lệch < 5% so với
      `configs/model_conformer.yaml`.
- [ ] **Train baseline Conformer-CTC** trước (đúng thứ tự đề cương, Tuần
      6-7) — cần 2 việc trên xong trước.
- [ ] **Hoàn tất upload file doc/báo cáo lên Google Drive** — mới tạo xong
      cấu trúc thư mục (`specialized-thesis/docs/{de_cuong,bieu_mau,notes}`,
      `specialized-thesis/reports`), **chưa upload file nào** (bị ngắt giữa
      chừng khi đang lấy base64 các file .docx + json).

## 🟢 Việc tay — làm song song bất cứ lúc nào, không chặn code

- [ ] Nghe & hiệu đính thủ công 250 câu trong
      `data/processed/clean_test_manifest.json` (cột `corrected_text` còn
      trống) — dùng mục 5 của `notebooks/02_dataset_eda.ipynb`.

## ⚠️ Cần kiểm tra riêng (không phải việc cần làm, mà là điểm bất thường)

- [ ] `docs/de_cuong/De_Cuong_Chi_Tiet_Mamba_ASR.docx` đang hiện là đã sửa
      (`git status`) nhưng không phải do session này chỉnh — có thể do Word
      tự lưu khi file đang mở (thấy file khóa `~$...` cùng thư mục trước
      đó). Kiểm tra nội dung trước khi commit để tránh mất/ghi đè bản edit
      tay của bạn.

## ✅ Đã hoàn thành

- [x] Đề cương chi tiết + tóm lược, đăng ký với GVHD (Tuần 1-2).
- [x] Cấu trúc project + skeleton code cho toàn bộ pipeline.
- [x] **Rủi ro kỹ thuật lớn nhất**: cài `mamba-ssm` CUDA kernel trên Kaggle
      T4 (pin tag v2.3.1) — xác nhận forward pass thật chạy được (Tuần 1).
- [x] Khảo sát thống kê đầy đủ VietSuperSpeech (`src/data/survey.py`),
      phát hiện sai lệch số liệu so với đề cương (Tuần 3).
- [x] Train tokenizer BPE tiếng Việt (vocab_size=1000, 60.656 transcript).
- [x] Lấy mẫu clean-test manifest 250 câu (seed=42).
- [x] Notebook EDA (xem waveform/spectrogram, nghe audio, thống kê),
      `notebooks/02_dataset_eda.ipynb`, đã test chạy thật.
- [x] Sửa bug decode audio: cột `audio` là string path, không tự giải mã.
- [x] Viết `src/data/prefetch_audio.py` — tải hàng loạt song song đúng
      67.405 file cần dùng (không tải thừa như snapshot_download cả thư
      mục), resume qua session, lỗi từng file không sập cả batch. Đã test
      logic (5 file thật + 1 file lỗi cố ý) — *chỉ còn thiếu bước chạy full*
      (xem mục 🟡 ở trên).
- [x] Push toàn bộ code lên GitHub:
      https://github.com/VuKhah/specialized-thesis
