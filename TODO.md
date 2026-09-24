# TODO tổng — Mamba vs Conformer ASR

Cập nhật lần cuối: **2026-09-24**. **Nguồn sự thật duy nhất cho việc cần làm /
đang chặn / đã xong.** Kế hoạch tuần + quyết định + nhật ký phiên ở
[`Plan.md`](Plan.md); kiến trúc ở [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 🔴 Đang chặn — chờ quyết định từ GVHD

- [ ] **Hướng xử lý sai lệch số liệu dataset ảnh hưởng RQ2**: số liệu thật
      (67.405 mẫu/245,42h, audio gần như đồng nhất 10-15s) khác đề cương đã
      đăng ký (52.023/267,39h, dải 3-30s) — RQ2 (RTF theo độ dài audio) không
      còn đủ dải để chứng minh ưu thế O(n) của Mamba. 4 phương án trong
      `docs/notes/dataset_discrepancy.md`, đang chờ ý kiến thầy Hoàng Văn
      Dũng. **Không tự chọn phương án khi chưa có ý kiến.**

## ⏸ Treo — người dùng chủ động hoãn quyết định

- [ ] **Prefetch audio đầy đủ** (67.405 file, ~27GB): hướng nêu 2026-09-19 là
      chạy trên Kaggle, nhưng quyết định cuối đang treo. Cần xác nhận trước
      khi chốt: hạn mức đĩa Kaggle (`/kaggle/working` = **20GB, người dùng xác nhận
      2026-09-21** → 27GB **không vừa**; các vị trí đĩa khác chưa kiểm chứng),
      cách lưu cache (thư mục tạm / đóng gói Kaggle Dataset riêng /
      tải lại mỗi phiên), chi phí GPU-giờ nếu tải lại. Cache local không
      chuyển sang Kaggle được. Script đã xong (`src/data/prefetch_audio.py`,
      test 5 file thật + 1 lỗi cố ý), chỉ thiếu bước chạy full. Chặn: train
      Conformer thật.

## 🟡 Sẵn sàng làm ngay

- [ ] **Xác nhận trên Kaggle: số tham số Mamba + chạy thử `train.py`**. Yaml
      đã đặt `n_layers: 28`, `expand: 2` (2026-09-24) → 12.292.352 tham số,
      +0,72% so với Conformer — số này đếm bằng bản sao shape của
      `mamba_ssm.Mamba` v2.3.1 trên CPU. Trên Kaggle: chạy
      `python -m src.models.param_count` (phải in ĐẠT, đúng 12.292.352) rồi
      vài step `python -m src.training.train --config configs/model_mamba.yaml`
      (28 layer, T≈1000-1500 khung, batch 16 — để ý VRAM T4).
- [ ] **Train baseline Conformer-CTC** (Tuần 6-7) — sau khi prefetch xong (⏸).

## 🟢 Việc tay — song song bất cứ lúc nào, không chặn code

- [ ] Nghe & hiệu đính 250 câu trong `data/processed/clean_test_manifest.json`
      (cột `corrected_text` còn trống) — dùng mục 5 của
      `notebooks/02_dataset_eda.ipynb`.

## 📘 Khóa luận (bản viết) — kế hoạch, mốc và quy cách ở `Plan.md` mục 6

**Làm ngay / sớm**
- [ ] **Gặp GVHD (tối đa 2 tuần/lần, GVHD ký logbook)**: cho AI biết ngày gặp
      gần nhất + ngày hẹn kế tiếp để ghi vào bảng ở `Plan.md` mục 6. Nên hỏi:
      🔴 quyết định RQ2 · hình thức nộp hiện hành (biểu mẫu 2018) · dạng "thiết
      kế" ở Chương 2. *(việc tay của người dùng)*
- [ ] **Sao lưu file Word khóa luận lên Drive** `specialized-thesis/khoa_luan/`
      (`docs/khoa_luan/KhoaLuan_Mamba_vs_Conformer_ASR.docx`, gitignore). Đây là
      **bản làm việc duy nhất**; script sinh file đã bỏ — đừng dựng lại kẻo mất
      bản sửa tay.
- [ ] **Điền chỗ trống trên bìa** (tô vàng): Bộ môn, Khóa, logo khoa. Xác nhận
      tên đề tài trên bìa: bìa ghi `Conformer`, đề cương đã nộp ghi
      `Con-Former` — hỏi GVHD nên theo bản nào.
- [ ] **Rà bản nháp Chương 1** (~9 trang, đoạn tô vàng = ghi chú cần xử lý):
      đối chiếu 12 trích dẫn với nguồn gốc (điền từ trí nhớ AI, chưa kiểm),
      đọc lại 3 bài Mamba-ASR (số liệu AISHELL-1 lấy từ đề cương), chuyển công
      thức sang Word Equation, vẽ Hình 1.1-1.3. Mục tiêu xong trước Tuần 7.
- [ ] **Bắt đầu ghi Tài liệu tham khảo** ngay khi trích dẫn (chỉ tài liệu thực
      sự trích dẫn; quy ước ghi ở Mẫu 5).

**Theo tuần (chưa đến hạn)**
- [ ] Tuần 6-7: Ch2 Phân tích & thiết kế (từ `ARCHITECTURE.md`).
- [ ] Sau Tuần 8/12/13: Ch3 Thực nghiệm — chèn kết quả từ `reports/`; phần mô
      tả dữ liệu chờ 🔴.
- [ ] Tuần 14: Ch4 Demo & tổng kết. Tuần 15: Mở đầu, Kết luận, Tóm tắt, Phụ lục.
- [ ] **Trước hết đợt Kaggle:** sao lưu `best.pt` của cả hai thí nghiệm lên
      Drive (cần cho đĩa CD, mục `SETUP`).
- [ ] Khi nộp: danh sách kiểm tra ở `Plan.md` mục 6 (2 cuốn + 2 CD, đề cương có
      chữ ký, GVHD duyệt trước khi in).

## ⚠️ Cần kiểm tra / rủi ro đã biết (không phải việc làm ngay)

**Repo public**
- [ ] `notebooks/02_dataset_eda.ipynb` nhúng audio YouTube (20 output
      `<audio>`, ~5MB) — trong repo public và lịch sử git. Người dùng chọn bỏ
      qua (2026-09-19). Chưa kiểm license VietSuperSpeech. Khi xử lý: xoá
      output (`nbstripout`); muốn gỡ khỏi lịch sử phải viết lại lịch sử + force
      push (hỏi trước).
- [ ] Lịch sử git vẫn chứa các file đã gỡ index: `.docx` (đề cương, biểu mẫu
      khoa) và file TLCN có tên+MSSV người thứ ba trong tên file. Chưa quyết
      định có viết lại lịch sử không.
- [ ] `docs/de_cuong/De_Cuong_Chi_Tiet_Mamba_ASR.docx` trên đĩa **khác bản đã
      commit** (trước đó `git status` báo modified; nghi Word tự lưu). Giờ file
      không còn track nên git không hiện diff nữa — mở kiểm tra nội dung trước
      khi upload Drive, đừng để mất bản edit tay. Không sửa nội dung file.

**Code / thiết kế** (chi tiết ở `ARCHITECTURE.md` mục 8)
- [ ] `MambaEncoder` + `train.py` **chưa xác nhận trên Kaggle** (máy local
      không CUDA). Cần chạy vài step
      `python -m src.training.train --config configs/model_mamba.yaml` trên
      Kaggle trước khi coi training loop dùng được cho cả hai kiến trúc.
- [ ] Mamba đơn hướng, chưa mask padding (TODO trong `mamba_encoder.py`); hai
      encoder không subsampling (T'=T ≈ 1000-1500 khung). Cần nêu trong phần
      thảo luận; kiểm tra đề cương đã đề cập chưa.
- [ ] `train.py`: 1 GPU (Kaggle có 2x T4), không AMP — ảnh hưởng thời gian
      train so với hạn mức 30 GPU-giờ/tuần. Chưa quyết định có tối ưu không
      (sửa training loop là sửa code dùng chung, phải áp dụng cho cả hai).
- [ ] `best.pt` chọn theo WER `validation`, mà clean-test lấy từ `validation`
      → lưu ý khi diễn giải kết quả cuối.
- [ ] `rtf.py`/`survey.py` chia bucket độ dài theo đề cương (3-30s), không
      khớp dữ liệu thật — **không sửa** khi 🔴 chưa có quyết định.

## ✅ Đã hoàn thành

- [x] **Push lên GitHub (2026-09-24)**: 6 commit (`f101acb`..`f8d5af4`), gồm
      gỡ 12 file `.docx` khỏi index. File cũ vẫn nằm trong lịch sử git (xem ⚠️).

- [x] ~~Upload tài liệu `.docx` (đề cương, biểu mẫu, notes) lên Google Drive~~
      — **người dùng bỏ qua (2026-09-24)**. Các file chỉ còn trên đĩa local
      (gitignore) + bản cũ trong lịch sử git.

- [x] **Script đo WER clean-test (2026-09-21)**:
      `python -m src.evaluation.eval_clean_test --config configs/model_<x>.yaml`.
      Test thật với 250 câu thật (Conformer, CPU, trọng số ngẫu nhiên: chạy hết
      pipeline, WER vô nghĩa). Còn: chạy với checkpoint thật khi có; Mamba chưa
      test (Kaggle); reference hiện là `pseudo_label` cho cả 250 câu vì
      `corrected_text` chưa hiệu đính (🟢) — script in rõ số câu mỗi loại.

- [x] **`param_count.py` đọc yaml (2026-09-21)**: dùng `build_encoder` của
      `train.py`; test local: Conformer 12.204.288 khớp số đã đo, đổi
      `n_layers` trong yaml thì số đổi theo. Nhánh Mamba chưa chạy (không CUDA).

- [x] **Chuẩn hoá tài liệu gốc (2026-09-19)**: tách `Plan.md` (kế hoạch/trạng
      thái/quyết định/nhật ký), `ARCHITECTURE.md` (sơ đồ, luồng dữ liệu, bản đồ
      file), `docs/CONVENTIONS.md` (quy ước), rút gọn `README.md`, cập nhật
      `CLAUDE.md`. Rà repo public, bổ sung `.gitignore`, gỡ 12 file nhị phân
      khỏi index. *(Đã commit local, chưa push — xem 🟡.)*
- [x] **Training loop thật** (`139f018`): checkpoint/resume tự động
      (`checkpoints/<experiment_name>/{latest,best}.pt`), eval WER định kỳ,
      tensorboard. Test thật với dữ liệu thật trên ConformerEncoder (CPU) — resume
      đúng epoch/global_step. Sửa bug: `lr: 3e-4` bị PyYAML đọc thành string →
      `3.0e-4`. Nhánh Mamba chưa test (xem ⚠️).
- [x] Đề cương chi tiết + tóm lược, đăng ký với GVHD (Tuần 1-2).
- [x] Cấu trúc project + skeleton code toàn bộ pipeline.
- [x] **Rủi ro kỹ thuật lớn nhất**: cài `mamba-ssm` CUDA kernel trên Kaggle T4
      (pin tag v2.3.1), forward pass thật chạy được (Tuần 1).
- [x] Khảo sát thống kê VietSuperSpeech (`src/data/survey.py`), phát hiện sai
      lệch số liệu so với đề cương (Tuần 3).
- [x] Tokenizer BPE tiếng Việt (vocab_size=1000, 60.656 transcript).
- [x] Clean-test manifest 250 câu (seed=42).
- [x] Notebook EDA `notebooks/02_dataset_eda.ipynb` (đã test chạy thật).
- [x] Sửa bug decode audio: cột `audio` là string path, không tự giải mã.
- [x] `src/data/prefetch_audio.py`: tải song song đúng 67.405 file cần dùng,
      resume qua session, lỗi từng file không sập cả batch (test logic xong,
      chưa chạy full — xem ⏸).
- [x] Push code lên GitHub: https://github.com/VuKhah/specialized-thesis
