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

## Trạng thái hiện tại

**Tuần 1-2:**
- [x] Đề cương chi tiết + tóm lược đã hoàn thiện, đã đăng ký với GVHD.
- [x] Cấu trúc project + skeleton code.
- [x] **Rủi ro kỹ thuật cao nhất: cài `mamba-ssm` trên Kaggle T4 — ĐÃ GIẢI
      QUYẾT.** Build từ source, pin tag `v2.3.1`, CUDA kernel thật đã chạy
      được (forward pass xác nhận, Lần 4). Xem
      [`docs/notes/mamba_ssm_install_log.md`](docs/notes/mamba_ssm_install_log.md).

**Tuần 3:**
- [x] Khảo sát thống kê đầy đủ VietSuperSpeech (`src/data/survey.py`, chạy
      local, không cần GPU) — kết quả: `reports/results/dataset_survey_*.json`.
- [x] Xây tokenizer BPE tiếng Việt (`configs/tokenizer.model`, vocab_size=1000,
      train trên 60.656 transcript thật).
- [x] Lấy mẫu clean-test 200-300 câu (seed cố định) —
      `data/processed/clean_test_manifest.json` (250 mẫu, seed=42, còn
      trống cột `corrected_text`, cần nghe & hiệu đính thủ công).
- [ ] ⚠️ **VẤN ĐỀ MỞ, CHƯA QUYẾT ĐỊNH HƯỚNG XỬ LÝ**: số liệu dataset thật
      khác đáng kể so với đề cương đã đăng ký (67.405 mẫu/245,42h thực đo
      so với 52.023/267,39h trong đề cương; tên split đúng là
      `train`/`validation` không phải `dev-test`; **độ dài audio thực tế
      gần như đồng nhất 10-15 giây, không phải dải 3-30 giây** như đề cương
      giả định) — ảnh hưởng trực tiếp đến thiết kế RQ2 (RTF theo độ dài
      audio). Toàn bộ phân tích + 4 hướng xử lý khả dĩ:
      [`docs/notes/dataset_discrepancy.md`](docs/notes/dataset_discrepancy.md).
      **Cần quyết định trước khi triển khai thực nghiệm RQ2** (Tuần 8, 12).

Xem kế hoạch chi tiết theo tuần trong đề cương (Tuần 1–15).

## Rủi ro kỹ thuật lớn nhất: cài đặt `mamba-ssm` — ĐÃ GIẢI QUYẾT (Tuần 1)

`mamba-ssm` cần biên dịch CUDA kernel (`causal-conv1d` + `selective_scan`).
Sau 4 lần thử trên Kaggle T4 (toàn bộ log lỗi + chẩn đoán chi tiết:
[`docs/notes/mamba_ssm_install_log.md`](docs/notes/mamba_ssm_install_log.md)),
lệnh cài đặt đúng đã xác nhận:

```bash
pip install packaging ninja
pip install --no-build-isolation git+https://github.com/Dao-AILab/causal-conv1d
pip install --no-build-isolation git+https://github.com/state-spaces/mamba@v2.3.1
```

Tóm tắt quá trình: pip cài thẳng thất bại do PEP517 build isolation không
thấy torch cài sẵn → build từ source (`--no-build-isolation`) cài được
nhưng nhánh `main` (từ v2.3.2) kéo theo kiến trúc **Mamba-3** mới (qua
`tilelang`/`tvm`, có bug không tương thích Python 3.12) → pin về tag
`v2.3.1` (bản cuối trước Mamba-3, đúng Mamba/S6 cổ điển mà đề cương cần) →
**CUDA kernel thật chạy đúng, đã xác nhận bằng forward pass thật.** Xem
[`docs/notes/mamba_versions.md`](docs/notes/mamba_versions.md) để hiểu rõ
khác biệt Mamba/S6 vs Mamba-2 vs Mamba-3.

Phương án dự phòng (`selective_scan_ref`, thuần PyTorch, không cần CUDA
kernel) cũng đã xác nhận khả dụng — vẫn giữ làm backup nếu môi trường
Kaggle thay đổi version torch/CUDA sau này.

## Dataset

[VietSuperSpeech](https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech) —
52.023 cặp audio-text (267,39 giờ), tiếng Việt hội thoại tự nhiên từ YouTube.
Train 46.822 mẫu / dev-test 5.201 mẫu (seed cố định). **Lưu ý**: transcript
là pseudo-label (Zipformer-30M-RNNT-6000h qua Sherpa-ONNX), chưa qua kiểm
định người — cần xây thêm clean-test set 200-300 câu hiệu đính thủ công.
