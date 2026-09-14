# Mamba vs Conformer Encoder cho ASR Tiếng Việt Hội Thoại

Khóa luận tốt nghiệp — Học kỳ 1, năm học 2026-2027
Trần Vũ Khanh — 22110349 — Khoa CNTT, Chuyên ngành Trí tuệ Nhân tạo
GVHD: PGS.TS. Hoàng Văn Dũng

## Đề tài

So sánh có kiểm soát (matched-parameter, cùng pipeline CTC) giữa kiến trúc
**Mamba (Selective State Space Model)** và **Conformer** làm encoder cho hệ
thống ASR tiếng Việt hội thoại tự nhiên, trên bộ dữ liệu **VietSuperSpeech**.

Đề cương đầy đủ: [`docs/de_cuong/De_Cuong_Chi_Tiet_Mamba_ASR.docx`](docs/de_cuong/De_Cuong_Chi_Tiet_Mamba_ASR.docx)

> **Lưu ý:** đề cương đã nộp ghi hạ tầng là Google Colab free-tier; trên
> thực tế đang chạy trên **Kaggle** (2x T4). Tài liệu làm việc (README này,
> notebook, config) phản ánh đúng Kaggle — bản đề cương chính thức giữ
> nguyên như đã đăng ký với GVHD.

## Câu hỏi nghiên cứu

1. **RQ1 (WER)** — Cùng số tham số, cùng pipeline: Mamba hay Conformer cho WER thấp hơn?
2. **RQ2 (RTF/latency)** — Kiến trúc nào nhanh hơn, đặc biệt khi audio dài?
3. **RQ3 (Error taxonomy)** — Hai kiến trúc mắc lỗi khác nhau ra sao?

## Thiết kế pipeline (matched-parameter, ablation study)

```
audio (16kHz wav)
   │
   ▼
[Front-end: log-mel spectrogram]   src/features/
   │
   ▼
[Encoder]  ◄── điểm hoán đổi duy nhất giữa 2 thí nghiệm
   │  ├── MambaEncoder     src/models/mamba_encoder.py
   │  └── ConformerEncoder src/models/conformer_encoder.py
   │      (khớp số tham số — xem src/models/param_count.py)
   ▼
[CTC head + BPE tokenizer tiếng Việt]  src/tokenizer/, src/models/ctc_model.py
   │
   ▼
[WER / RTF / error taxonomy]  src/evaluation/
```

Front-end, tokenizer, CTC head **dùng chung** cho cả hai thí nghiệm — chỉ
encoder thay đổi. Đây là điều kiện bắt buộc để so sánh công bằng.

## Cấu trúc thư mục

```
docs/          đề cương, biểu mẫu KLTN, ghi chú lịch sử lựa chọn đề tài
notebooks/     notebook chạy trên Kaggle (2x T4, 30 giờ GPU/tuần)
src/
  data/        load & tiền xử lý VietSuperSpeech
  features/    trích xuất log-mel spectrogram
  tokenizer/   BPE tokenizer tiếng Việt
  models/      encoder (Mamba/Conformer), CTC head
  training/    training loop dùng chung
  evaluation/  WER, RTF/latency, error taxonomy
  demo/        demo Gradio
configs/       config siêu tham số (yaml) cho từng thí nghiệm
reports/       bảng số liệu, biểu đồ dùng cho báo cáo
checkpoints/   model checkpoint đã train (không commit vào git)
data/          dữ liệu audio (không commit vào git — quá lớn cho GitHub)
```

## Trạng thái hiện tại (Tuần 1)

- [x] Đề cương chi tiết + tóm lược đã hoàn thiện, đã đăng ký với GVHD.
- [x] Cấu trúc project + skeleton code.
- [ ] **Kiểm tra rủi ro kỹ thuật cao nhất: cài `mamba-ssm` trên Kaggle T4.**
      Đã cài được (build từ source, pin tag `v2.3.1`) và xác nhận fallback
      thuần PyTorch hoạt động; **chưa verify CUDA kernel thật chạy đúng sau
      khi pin** — xem phần "Rủi ro" bên dưới và log chi tiết trong
      [`docs/notes/mamba_ssm_install_log.md`](docs/notes/mamba_ssm_install_log.md).
- [x] Xác nhận schema VietSuperSpeech (`audio`, `text`, `duration`, `source`).
- [ ] Tải & khảo sát thống kê đầy đủ VietSuperSpeech.
- [ ] Xây tokenizer BPE tiếng Việt.

Xem kế hoạch chi tiết theo tuần trong đề cương (Tuần 1–15).

## Rủi ro kỹ thuật lớn nhất: cài đặt `mamba-ssm`

`mamba-ssm` cần biên dịch CUDA kernel (`causal-conv1d` + `selective_scan`).
Lịch sử 3 lần thử trên Kaggle T4 (chi tiết đầy đủ, kèm log lỗi thật:
[`docs/notes/mamba_ssm_install_log.md`](docs/notes/mamba_ssm_install_log.md)):

1. **Cài qua pip thẳng — thất bại**: lỗi build metadata do PEP517 build
   isolation không thấy torch cài sẵn.
2. **Build từ source (`--no-build-isolation`), nhánh `main` — cài được**,
   nhưng import lỗi `AttributeError: __dict__ of 'type' objects is not
   writable`.
3. **Chẩn đoán bằng full traceback → xác định nguyên nhân thật**: nhánh
   `main` (từ v2.3.2) gộp thêm kiến trúc **Mamba-3** mới, và
   `mamba_ssm/__init__.py` luôn import nó dù không dùng tới — kéo theo
   `tilelang` → `tvm`, và `tvm` có bug không tương thích Python 3.12.
   **Fix**: pin cài đặt về tag `v2.3.1` (bản cuối trước Mamba-3) — đề tài
   chỉ cần Mamba/S6 cổ điển nên không mất gì. Xem
   [`docs/notes/mamba_versions.md`](docs/notes/mamba_versions.md) để hiểu
   rõ khác biệt Mamba/S6 vs Mamba-2 vs Mamba-3 và vì sao chọn bản cổ điển.

**Phương án dự phòng đã xác nhận hoạt động**: `mamba_ssm` có sẵn
implementation thuần PyTorch không cần CUDA kernel custom
(`selective_scan_ref`, chậm hơn nhưng đúng về mặt số học) — đề tài không bị
chặn ngay cả khi CUDA kernel thật còn vấn đề.

Việc này **phải xác nhận xong trong Tuần 1–2** theo đúng kế hoạch trong đề
cương — đây là rủi ro cao nhất của toàn bộ đề tài. Còn thiếu: verify lại
CUDA kernel thật chạy đúng sau khi pin `v2.3.1` (Lần 4, chưa chạy).

## Dataset

[VietSuperSpeech](https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech) —
52.023 cặp audio-text (267,39 giờ), tiếng Việt hội thoại tự nhiên từ YouTube.
Train 46.822 mẫu / dev-test 5.201 mẫu (seed cố định). **Lưu ý**: transcript
là pseudo-label (Zipformer-30M-RNNT-6000h qua Sherpa-ONNX), chưa qua kiểm
định người — cần xây thêm clean-test set 200-300 câu hiệu đính thủ công.
