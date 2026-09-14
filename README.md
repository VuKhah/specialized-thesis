# Mamba vs Conformer Encoder cho ASR Tiếng Việt Hội Thoại

Khóa luận tốt nghiệp — Học kỳ 1, năm học 2026-2027
Trần Vũ Khanh — 22110349 — Khoa CNTT, Chuyên ngành Trí tuệ Nhân tạo
GVHD: PGS.TS. Hoàng Văn Dũng

## Đề tài

So sánh có kiểm soát (matched-parameter, cùng pipeline CTC) giữa kiến trúc
**Mamba (Selective State Space Model)** và **Conformer** làm encoder cho hệ
thống ASR tiếng Việt hội thoại tự nhiên, trên bộ dữ liệu **VietSuperSpeech**.

Đề cương đầy đủ: [`docs/de_cuong/De_Cuong_Chi_Tiet_Mamba_ASR.docx`](docs/de_cuong/De_Cuong_Chi_Tiet_Mamba_ASR.docx)

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
notebooks/     notebook chạy trên Google Colab (T4 free-tier)
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
- [ ] **Kiểm tra rủi ro kỹ thuật cao nhất: cài `mamba-ssm` trên Colab T4.**
      Chạy [`notebooks/00_setup_environment.ipynb`](notebooks/00_setup_environment.ipynb)
      trên Colab để xác nhận. Notebook có sẵn 3 phương án cài đặt theo thứ tự
      ưu tiên + fallback thuần PyTorch nếu build CUDA kernel thất bại
      (xem phần "Rủi ro" bên dưới).
- [ ] Tải & khảo sát thống kê VietSuperSpeech.
- [ ] Xây tokenizer BPE tiếng Việt.

Xem kế hoạch chi tiết theo tuần trong đề cương (Tuần 1–15).

## Rủi ro kỹ thuật lớn nhất: cài đặt `mamba-ssm`

`mamba-ssm` cần biên dịch CUDA kernel (`causal-conv1d` + `selective_scan`).
Đây là lỗi build phổ biến trên Colab do version PyTorch/CUDA thay đổi liên tục
(xác nhận qua khảo sát các issue trên GitHub state-spaces/mamba, 2026-09).
Notebook setup thử theo thứ tự:

1. Cài qua pip bình thường (`pip install mamba-ssm`).
2. Nếu build wheel lỗi: cài từ source với `--no-build-isolation`, pin đúng
   version torch có sẵn trên Colab.
3. **Phương án dự phòng bắt buộc phải verify sớm**: `mamba_ssm` có sẵn
   implementation thuần PyTorch không cần CUDA kernel custom
   (`selective_scan_ref`, chậm hơn nhưng đúng về mặt số học) — dùng để không
   bị chặn tiến độ nếu build kernel thất bại trên T4.

Việc này **phải xác nhận trong Tuần 1–2** theo đúng kế hoạch trong đề cương —
đây là rủi ro cao nhất của toàn bộ đề tài.

## Dataset

[VietSuperSpeech](https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech) —
52.023 cặp audio-text (267,39 giờ), tiếng Việt hội thoại tự nhiên từ YouTube.
Train 46.822 mẫu / dev-test 5.201 mẫu (seed cố định). **Lưu ý**: transcript
là pseudo-label (Zipformer-30M-RNNT-6000h qua Sherpa-ONNX), chưa qua kiểm
định người — cần xây thêm clean-test set 200-300 câu hiệu đính thủ công.
