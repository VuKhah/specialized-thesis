# Khảo sát Mamba làm encoder cho ASR tiếng Việt hội thoại

Khóa luận tốt nghiệp — Học kỳ 1, năm học 2026-2027
Trần Vũ Khanh — 22110349 — Khoa CNTT, Chuyên ngành Trí tuệ Nhân tạo
GVHD: PGS.TS. Hoàng Văn Dũng

> Tên trên là tên làm việc. Sau buổi GVHD lần 1 (2026-10-01), đề tài chuyển từ
> *so sánh có kiểm soát Mamba vs Conformer* sang **khảo sát nhiều mô hình với
> Mamba là trọng tâm**. Đề cương mới đang soạn; đề cương đã nộp và các biểu mẫu
> (`.docx`) không đưa lên repo công khai này.

## Đề tài

Khảo sát kiến trúc **Mamba (Selective State Space Model)** làm encoder cho ASR
tiếng Việt hội thoại tự nhiên trên **VietSuperSpeech**. Mamba được đặt cạnh
Conformer train từ đầu và một mô hình pre-train lớn, rồi nhận xét theo đánh
đổi **độ chính xác – tốc độ – tài nguyên**.

### Đội hình mô hình

| Mô hình | Vai trò | Cách dùng |
|---|---|---|
| **Mamba hai chiều** (ExtBiMamba chồng, ~12M) | Trọng tâm | Train từ đầu |
| **ConExtBiMamba** (khung Conformer, thay self-attention bằng Mamba hai chiều, ~12M) | Ứng viên trọng tâm — chọn một trong hai biến thể Mamba sau train thử ngắn | Train từ đầu |
| **Conformer** (~12M) | Baseline có kiểm soát | Train từ đầu, cùng pipeline |
| **Parakeet-CTC-0.6B-vi** (dự bị: PhoWhisper-small) | Tham chiếu có pre-train | Zero-shot, fine-tune nếu đủ tài nguyên |
| Mamba một chiều | Chỉ trích dẫn (arXiv:2405.12609) | Không train |

### Hai mức so sánh

- **Nhóm train từ đầu — có kiểm soát.** Cùng front-end, tokenizer, CTC head,
  vòng train và đánh giá, cùng cỡ ~12M tham số; **chỉ encoder khác nhau**.
- **Nhóm pre-train — tham chiếu.** Dùng pipeline và front-end riêng của mô
  hình; chỉ chung bước chuẩn hóa văn bản trước WER và tập đánh giá. Kết luận
  tách riêng hai nhóm.

### Câu hỏi nghiên cứu (bản làm việc)

1. **RQ1 — độ chính xác:** WER trên clean-test (192 câu hiệu đính tay) và tập
   val, theo epoch với các mô hình train từ đầu.
2. **RQ2 — hiệu quả:** RTF, độ trễ, VRAM, số tham số, GPU-giờ train; theo độ
   dài trên **audio ghép nhân tạo** từ các đoạn liên tiếp (dữ liệu gốc chỉ dài
   10-15 s).
3. **RQ3 — phân tích lỗi** định tính.

## Pipeline của nhóm train từ đầu

```
audio 16 kHz → log-mel 80 (25/10 ms) → CMVN toàn cục → SpecAugment (chỉ khi train)
            → [Encoder: Mamba hai chiều | ConExtBiMamba | Conformer] → CTC head (BPE 1000) → greedy
                        ▲ điểm hoán đổi duy nhất
```

Không subsampling (giữ 100 khung/giây để RQ2 phản ánh đúng chi phí theo độ
dài); giải mã CTC greedy, không LM. Lý do chọn CTC, CMVN, SpecAugment, dropout
(có trích nguồn): [`docs/notes/frontend_decoder_survey.md`](docs/notes/frontend_decoder_survey.md).
Sơ đồ chi tiết, shape tensor, luồng dữ liệu: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Tiến độ

Mục tiêu, lộ trình theo tuần, trạng thái và các quyết định: [`Plan.md`](Plan.md).
Việc cần làm chi tiết: [`TODO.md`](TODO.md). *(Không chép lại ở đây để khỏi lệch.)*

## Cấu trúc thư mục

```
ARCHITECTURE.md  sơ đồ & luồng dữ liệu       Plan.md   mục tiêu, lộ trình, quyết định, nhật ký
TODO.md          việc cần làm                CLAUDE.md luật cho AI (Claude Code)
docs/            CONVENTIONS.md (quy ước) · notes/ (điều tra, quyết định, đối chiếu tài liệu)
notebooks/       notebook chạy trên Kaggle (2x T4, 30 giờ GPU/tuần)
scripts/kaggle/  kernel Kaggle chạy từ local qua Kaggle CLI (tạo dataset, LID, benchmark, zero-shot)
src/
  data/          VietSuperSpeech, manifest shard, lọc ngôn ngữ, giải nén tar, khảo sát
  features/      log-mel + CMVN + SpecAugment, tính thống kê CMVN
  tokenizer/     BPE tokenizer tiếng Việt
  models/        encoder (Mamba, Conformer), CTC head
  training/      training loop dùng chung (DDP + AMP, resume giữa epoch)
  evaluation/    WER (chuẩn hóa văn bản chung), clean-test, RTF (error taxonomy: chưa có)
  demo/          demo Gradio (chưa có)
configs/         yaml cho từng thí nghiệm, tokenizer, thống kê CMVN
data/splits/     manifest 4 shard train + val, danh sách đoạn bị loại (excluded.tsv)
reports/         bảng số liệu, biểu đồ dùng cho báo cáo
checkpoints/     model checkpoint (không commit)
data/raw/        audio (không commit — quá lớn, xem mục Dataset)
```

## Cách chạy (từ gốc repo)

```bash
pip install -r requirements.txt                 # mamba-ssm: xem mục cài đặt bên dưới

python -m src.data.vietsuperspeech_dataset      # làm ấm cache metadata HF (một lần, có mạng)
export HF_HUB_OFFLINE=1                         # sau đó đọc cache ~0,2 s thay vì ~45 s

python -m src.models.param_count                # số tham số encoder theo yaml
python -m src.training.train --config configs/model_conformer.yaml
python -m src.training.train --config configs/model_mamba.yaml        # cần GPU + mamba-ssm (Kaggle)
python -m src.evaluation.eval_clean_test --config configs/model_conformer.yaml
```

Trên Kaggle (2 GPU), audio lấy từ các Kaggle Dataset dạng tar:

```bash
python -m src.data.extract_audio --input /kaggle/input --dest /tmp/audio_cache
AUDIO_CACHE_DIR=/tmp/audio_cache HF_HUB_OFFLINE=1 torchrun --nproc_per_node 2 -m src.training.train \
    --config configs/model_mamba.yaml --max_minutes 510 --ckpt_dir /kaggle/working/checkpoints
```

Train tự resume (kể cả giữa epoch) từ `<ckpt_dir>/<experiment_name>/latest.pt`
hoặc `--resume_from`; `--no_resume` để train lại. Train thử một shard:
`--train_manifests train_shard0 --epochs 5`. Trên Windows đặt
`PYTHONIOENCODING=utf-8` khi chạy script in tiếng Việt.

## Cài đặt `mamba-ssm`

`mamba-ssm` cần biên dịch CUDA kernel (`causal-conv1d` + `selective_scan`).
Lệnh cài đúng, đã xác nhận trên Kaggle T4:

```bash
pip install packaging ninja
pip install --no-build-isolation git+https://github.com/Dao-AILab/causal-conv1d
pip install --no-build-isolation git+https://github.com/state-spaces/mamba@v2.3.1
```

Bắt buộc pin tag `v2.3.1` (Mamba-1/S6 cổ điển); nhánh `main` kéo
Mamba-3/tilelang/tvm, lỗi với Python 3.12. Log các lần thử:
[`docs/notes/mamba_ssm_install_log.md`](docs/notes/mamba_ssm_install_log.md);
khác biệt Mamba/S6 vs Mamba-2 vs Mamba-3:
[`docs/notes/mamba_versions.md`](docs/notes/mamba_versions.md). Thực nghiệm
chạy trên **Kaggle** (2x T4), không phải Google Colab như đề cương cũ ghi.

## Dataset

[VietSuperSpeech](https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech)
(pin revision `cbf624ae9b`) — tiếng Việt hội thoại tự nhiên từ YouTube,
`train` 60.656 mẫu (220,84 h) + `validation` 6.749 mẫu (24,57 h); audio gần như
đồng nhất 10-15 s (khác số liệu trong đề cương cũ — xem
[`docs/notes/dataset_discrepancy.md`](docs/notes/dataset_discrepancy.md)).
Transcript là pseudo-label (Zipformer-30M-RNNT-6000h), chưa qua kiểm định người.

**Lọc dữ liệu (A1):** một phần audio không phải tiếng Việt (phỏng vấn khách
nước ngoài, một video tiếng Nhật) mà nhãn vẫn bị phiên thành chữ Việt/giả tiếng
Anh. Loại theo nhận dạng ngôn ngữ trên audio (Whisper-small) và tỉ lệ từ có
dấu của nhãn → [`data/splits/excluded.tsv`](data/splits/excluded.tsv). Sau lọc:

| Tập | Câu | Giờ |
|---|---|---|
| train (4 shard) | 48.340 | 175,77 |
| val = `validation` trừ clean-test | 5.140 | 18,65 |
| clean-test (seed 42, từ `validation`, hiệu đính tay) | 192 | — |

Tokenizer BPE (1000 piece) train lại trên nhãn train đã lọc. Clean-test:
`data/processed/clean_test_manifest.json` (đang hiệu đính).
