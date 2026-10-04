# Front-end, decoder, dropout — đối chiếu tài liệu và quyết định (2026-10-04)

Người dùng duyệt 2026-10-04 cả 5 đề xuất dưới đây. Mỗi mục ghi **căn cứ đã
kiểm từ nguồn gốc** (bài báo hoặc file cấu hình chính thức); chỗ không tìm được
bằng chứng công bố thì ghi rõ. Áp cho cả 3 mô hình train từ đầu (BiMamba B1,
ConExtBiMamba, Conformer-12M) — đều là code dùng chung. Mô hình pre-train
(Parakeet, PhoWhisper) bắt buộc giữ front-end riêng của chúng.

## Tóm tắt quyết định

| # | Quyết định | Căn cứ chính | Code |
|---|---|---|---|
| 1 | **Giữ CTC** (không thêm decoder attention/Mamba) | Speech Slytherin Bảng 3 | không đổi |
| 2 | **Thêm SpecAugment** (chỉ lúc train) | SpecAugment Bảng 3, 4, 6 | `log_mel.py`, mục `features.spec_augment` |
| 3 | **Thêm CMVN toàn cục** | Thực hành của mọi hệ đối chứng (không có ablation riêng) | `log_mel.py`, `configs/cmvn_stats.json` |
| 4 | **Dropout 0,1 cho Mamba** | 2405.12609 Bảng I-III; ConMamba yaml | `mamba_encoder.py`, `dropout` trong yaml |
| 5 | **Giữ B1, không làm biến thể B1 + FFN**; ConExtBiMamba vào train thử | 2405.12609 Bảng XII, XVI | — |
| — | Giữ log-mel 80 / 25 ms / 10 ms, n_fft 400; giữ không subsampling (chốt 2026-09-26) | Bảng front-end bên dưới | — |

## 1. CTC hay encoder-decoder

Speech Slytherin / ConMamba (arXiv:2407.09732), LibriSpeech, WER test-clean / test-other:

| Cấu hình | Không LM | Có LM |
|---|---|---|
| ConMamba + CTC (Bảng 3) | 3,9 / 10,3 | 2,4 / 5,7 |
| Conformer + CTC (Bảng 3) | 4,3 / 11,3 | 2,5 / 6,1 |
| ConMamba (S) + decoder Transformer | 4,0 / 9,5 | — |
| ConMamba (S) + decoder Mamba | 4,0 / 9,7 | — |

- CTC chỉ kém decoder ở test-other; thứ tự Mamba > Conformer giữ nguyên với CTC
  → so encoder bằng CTC cho kết luận cùng chiều, và đo encoder "sạch" hơn
  (không có LM ngầm của decoder che điểm yếu encoder).
- Decoder Mamba kém decoder Transformer ("Mamba decoders perform slightly worse
  than transformer decoders") → không có căn cứ làm decoder Mamba.
- RQ2: CTC giữ RTF ≈ chi phí encoder; decoder autoregressive làm mờ khác biệt
  O(T) vs O(T²).
- Transducer không khả thi khi không subsampling: tensor joint
  B×T×U×V ≈ 16×1500×60×1001 ≈ 1,4 tỉ phần tử.
- 2405.12609 dùng ESPnet hybrid CTC/attention (trọng số CTC 0,3, decoder
  Transformer) → **số WER của bài không so thẳng** với CTC thuần ở đây.
- Để sau: LM ngoài + beam search cho mọi mô hình CTC (Bảng 3: 10,3 → 5,7).

## 2. SpecAugment

Park và cs. 2019 (arXiv:1904.08779), số trích từ bảng:
- LibriSpeech 960h, LAS không LM: 4,1 / 12,5 → **2,8 / 6,8** (Bảng 3).
- Switchboard 300h (hội thoại, cỡ gần 176 h ở đây): 11,2 / 21,6 → **7,2 / 14,6** (Bảng 4).
- Bảng 6 (LAS-4-1024, test-other/test): đủ 10,0/3,7; bỏ time warp 10,1/3,8;
  bỏ freq mask 11,0/4,0; bỏ time mask 10,9/4,1. Tác giả: time warping "should
  be the first augmentation to be dropped given any budgetary limitations".

Cấu hình chọn = Conformer (Gulati 2020, arXiv:2005.08100: "SpecAugment with
mask parameter (F = 27), and ten time masks with maximum time-mask ratio
(pS = 0.05)"), cũng là mặc định NeMo FastConformer (`freq_masks: 2,
freq_width: 27, time_masks: 10, time_width: 0.05`). Không time warping. Đo
thật trên front-end: trung bình che ~22% khung, ~33% kênh mel. Ghi chú NeMo:
"you may use lower time_masks for smaller models to have a faster convergence"
— nếu train thử cho thấy hội tụ quá chậm thì đây là núm điều chỉnh đầu tiên
(người dùng quyết).

Recipe khác để tham khảo: torchaudio LibriSpeech Conformer dùng
`FrequencyMasking(27)` × 2 + `TimeMasking(100, p=0.2)` × 2.

## 3. Front-end và CMVN — các hệ đối chứng dùng gì

| Hệ thống | Đặc trưng | Cửa sổ / bước | Chuẩn hoá | Subsampling |
|---|---|---|---|---|
| Conformer (Gulati 2020) | "80-channel filterbanks" | 25 / 10 ms | (bài không nêu) | 4× |
| Whisper / PhoWhisper (arXiv:2212.04356) | "80-channel log-magnitude Mel spectrogram" | 25 / 10 ms | toàn cục: "globally scale the input to be between -1 and 1 with approximately zero mean" | 2× |
| FastConformer / Parakeet (NeMo `fast-conformer_ctc_bpe.yaml`) | 80 mel, log, n_fft 512, dither 1e-5 | 25 / 10 ms | `per_feature` (theo câu) | 8× |
| ConMamba (`hparams/CTC/conmamba_large.yaml`) | `Fbank`, 80 mel, n_fft 512 | win 25 ms | `InputNormalization(norm_type: global)` | 4× (CNN stride 2,2) |
| torchaudio recipe LibriSpeech Conformer RNN-T | 80 mel, n_fft 400, hop 160 | 25 / 10 ms | `GlobalStatsNormalization` | có |
| **Dự án** | 80 mel, log clamp 1e-5, n_fft 400 | 25 / 10 ms | **toàn cục (từ 2026-10-04)** | **không** |

- Tham số log-mel đã đúng chuẩn → không đổi.
- CMVN: mọi hệ có ghi rõ đều chuẩn hoá. **Không tìm được bài ablation đo riêng
  hiệu quả CMVN** cho mô hình end-to-end — căn cứ là thực hành của bộ công cụ.
- Chọn **toàn cục** (ConMamba, torchaudio, gần Whisper) thay vì `per_feature`:
  phép biến đổi cố định, không phụ thuộc độ dài câu → audio ghép dài của RQ2
  nhận đúng phép biến đổi như lúc train.
- Thống kê: `python -m src.features.compute_cmvn` — 2000 câu ngẫu nhiên (seed
  42) từ train đã bỏ `excluded.tsv`, chỉ khung thật, float64. Lưu
  `configs/cmvn_stats.json` (track git) và thành buffer trong checkpoint.

## 4. Dropout cho Mamba

- 2405.12609 Bảng I-III: dropout 0,1 cho các mô hình ASR có Mamba.
- ConMamba yaml: `transformer_dropout: 0.1` truyền vào encoder ConMamba.
- Conformer gốc và Conformer của dự án: 0,1.
- mamba-ssm gốc không dropout — công thức LM dữ liệu lớn, không phải ASR.
- **Vị trí** (đầu ra mỗi khối Mamba, trước khi cộng residual, cả hai chiều) là
  lựa chọn của dự án: bài không nêu chi tiết. Không đổi số tham số; eval tắt
  dropout nên bất biến padding (`ARCHITECTURE.md` mục 8-e) giữ nguyên (kiểm bằng
  khối giả nhân quả: lệch 9,5e-7).
- Bỏ được biến gây nhiễu "Conformer có dropout, Mamba không" (`TODO.md` ⚠️ cũ).

## 5. Mamba chồng thuần hay thêm FFN

2405.12609 Bảng XVI (LibriSpeech100, dev/test): ExtBiMamba chồng 38,5/37,7;
+ FFN 34,9/36,1 (~1,6 điểm). Bảng XII: ConExtBiMamba 5,9/6,0 vs Conformer
6,3/6,5. Cải thiện lớn chỉ đến từ khung Conformer → biến thể "B1 + FFN" không
đáng tốn quota; ConExtBiMamba đã nằm trong train thử. Chi tiết:
`mamba_bidirectional.md`.

## Nguồn

- SpecAugment — https://arxiv.org/abs/1904.08779
- Conformer — https://arxiv.org/abs/2005.08100
- Whisper — https://arxiv.org/abs/2212.04356
- Fast Conformer — https://arxiv.org/abs/2305.05084
- NeMo config — https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/fast-conformer_ctc_bpe.yaml
- Speech Slytherin / ConMamba — https://arxiv.org/abs/2407.09732 ; yaml: https://github.com/xi-j/Mamba-ASR/blob/main/hparams/CTC/conmamba_large.yaml
- Mamba in Speech (ExtBiMamba) — https://arxiv.org/abs/2405.12609
- Samba-ASR (Mamba encoder + decoder) — https://arxiv.org/abs/2501.02832
- torchaudio recipe — https://github.com/pytorch/audio/blob/main/examples/asr/librispeech_conformer_rnnt/transforms.py
