# Mamba vs Conformer Encoder cho ASR Tiếng Việt Hội Thoại

Khóa luận tốt nghiệp — Học kỳ 1, năm học 2026-2027
Trần Vũ Khanh — 22110349 — Khoa CNTT, Chuyên ngành Trí tuệ Nhân tạo
GVHD: PGS.TS. Hoàng Văn Dũng

## Đề tài

So sánh có kiểm soát (matched-parameter, cùng pipeline CTC) giữa kiến trúc
**Mamba (Selective State Space Model)** và **Conformer** làm encoder cho hệ
thống ASR tiếng Việt hội thoại tự nhiên, trên bộ dữ liệu **VietSuperSpeech**.

Đề cương và các biểu mẫu (`.docx`) không đưa lên repo công khai này.

> **Lưu ý:** đề cương đã nộp ghi hạ tầng là Google Colab free-tier; thực tế
> chạy trên **Kaggle** (2x T4). Tài liệu làm việc (README, notebook, config)
> phản ánh đúng Kaggle; đề cương chính thức giữ nguyên như đã đăng ký.

## Câu hỏi nghiên cứu

1. **RQ1 (WER)** — Cùng số tham số, cùng pipeline: Mamba hay Conformer cho WER thấp hơn?
2. **RQ2 (RTF/latency)** — Kiến trúc nào nhanh hơn, đặc biệt khi audio dài?
3. **RQ3 (Error taxonomy)** — Hai kiến trúc mắc lỗi khác nhau ra sao?

## Thiết kế pipeline (matched-parameter, ablation study)

```
audio (16kHz wav) → [log-mel] → [Encoder: Mamba | Conformer] → [CTC head + BPE] → WER / RTF / lỗi
                                     ▲ điểm hoán đổi duy nhất
```

Front-end, tokenizer, CTC head, training loop, đánh giá **dùng chung** cho hai
thí nghiệm — chỉ encoder thay đổi. Sơ đồ chi tiết, shape tensor, luồng dữ liệu:
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Tiến độ

Trạng thái theo tuần và các quyết định: [`Plan.md`](Plan.md). Việc cần làm
chi tiết: [`TODO.md`](TODO.md). *(Không chép lại ở đây để khỏi lệch.)*

## Cấu trúc thư mục

```
ARCHITECTURE.md  sơ đồ & luồng dữ liệu       Plan.md   kế hoạch, trạng thái, quyết định, nhật ký
TODO.md          việc cần làm                CLAUDE.md luật cho AI (Claude Code)
docs/            CONVENTIONS.md (quy ước) · notes/ (điều tra, quyết định)
notebooks/       notebook chạy trên Kaggle (2x T4, 30 giờ GPU/tuần)
src/
  data/          load & tiền xử lý VietSuperSpeech, prefetch audio, khảo sát
  features/      trích xuất log-mel spectrogram
  tokenizer/     BPE tokenizer tiếng Việt
  models/        encoder (Mamba/Conformer), CTC head
  training/      training loop dùng chung
  evaluation/    WER, RTF/latency (error taxonomy: chưa có)
  demo/          demo Gradio (chưa có)
configs/         yaml cho từng thí nghiệm + tokenizer
reports/         bảng số liệu, biểu đồ dùng cho báo cáo
checkpoints/     model checkpoint (không commit)
data/            dữ liệu audio (không commit — quá lớn cho GitHub)
```

## Cách chạy (từ gốc repo)

```bash
pip install -r requirements.txt          # mamba-ssm: xem mục cài đặt bên dưới

python -m src.data.prefetch_audio        # tải ~27GB audio (67.405 file); --verify để kiểm tra đủ
python -m src.models.param_count         # so số tham số hai encoder
python -m src.training.train --config configs/model_conformer.yaml
python -m src.training.train --config configs/model_mamba.yaml   # cần GPU + mamba-ssm (Kaggle)
```

Train tự resume từ `checkpoints/<experiment_name>/latest.pt` nếu có (`--no_resume`
để bỏ qua). Trên Windows đặt `PYTHONIOENCODING=utf-8` khi chạy script in tiếng
Việt.

## Cài đặt `mamba-ssm` (rủi ro kỹ thuật lớn nhất — đã giải quyết)

`mamba-ssm` cần biên dịch CUDA kernel (`causal-conv1d` + `selective_scan`).
Lệnh cài đúng, đã xác nhận trên Kaggle T4 (forward pass thật chạy được):

```bash
pip install packaging ninja
pip install --no-build-isolation git+https://github.com/Dao-AILab/causal-conv1d
pip install --no-build-isolation git+https://github.com/state-spaces/mamba@v2.3.1
```

Bắt buộc pin tag `v2.3.1` (bản cuối trước Mamba-3, đúng Mamba/S6 cổ điển đề
cương cần); nhánh `main` kéo Mamba-3/tilelang/tvm, lỗi với Python 3.12. Log 4
lần thử: [`docs/notes/mamba_ssm_install_log.md`](docs/notes/mamba_ssm_install_log.md);
khác biệt Mamba/S6 vs Mamba-2 vs Mamba-3:
[`docs/notes/mamba_versions.md`](docs/notes/mamba_versions.md). Phương án dự
phòng `selective_scan_ref` (thuần PyTorch) cũng đã xác nhận khả dụng.

## Dataset

[VietSuperSpeech](https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech) —
tiếng Việt hội thoại tự nhiên từ YouTube. **Số liệu đo thật** (2026-09-14, xem
[`docs/notes/dataset_discrepancy.md`](docs/notes/dataset_discrepancy.md) cho lý
do khác đề cương): split `train` 60.656 mẫu (220,84h) + `validation` 6.749 mẫu
(24,57h) = 67.405 mẫu / 245,42h. Độ dài audio gần như đồng nhất 10-15 giây.
Transcript toàn bộ CHỮ HOA, là pseudo-label (Zipformer-30M-RNNT-6000h qua
Sherpa-ONNX), chưa qua kiểm định người — clean-test set 250 câu hiệu đính thủ
công (seed=42): `data/processed/clean_test_manifest.json` (chưa hiệu đính xong).
