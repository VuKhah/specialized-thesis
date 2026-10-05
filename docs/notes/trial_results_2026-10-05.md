# Kết quả train thử 3 mô hình train từ đầu (2026-10-05)

Số liệu gốc: `reports/results/trial_2026-10-05/` (`trial_report.md/.json` do
`src/evaluation/trial_report.py` sinh; `metrics_*.jsonl` từng epoch;
`trial_report_lan1_b1_fp16.md` = báo cáo lần 1 khi B1 còn lỗi NaN). Nhật ký:
`Plan.md` 2026-10-05 (chiều, tối).

## 1. Điều kiện chạy

| | Conformer-12M | ConExtBiMamba | Mamba B1 |
|---|---|---|---|
| Kernel | `train-trial-asr` v3 | `train-trial-asr` v3 | `retrain-b1-asr` v1 (lần 1 trong `train-trial-asr` hỏng NaN) |
| Code | commit `39e0a44` | commit `39e0a44` | `39e0a44` + OVERLAY 2 file mô hình (`mamba_fp32`) |
| Dữ liệu | `train_shard0` (11.364 câu), eval val 4.824 + val_unseen 3.011 | như trái | như trái |
| Train | 5 epoch × 711 step, batch 16 (2×8), seed 42, AMP fp16, 2× T4 DDP | như trái | như trái |
| **Tiền xử lý** | log-mel 80 (25/10 ms, n_fft 400) → log clamp 1e-5 → CMVN toàn cục → SpecAugment (2 F27 + 10 T 5%) | **giống hệt** | **giống hệt** |
| Số học khối Mamba | — (không có Mamba) | **fp16** (autocast) | **fp32** (`mamba_fp32`) |
| Tham số encoder | 12.204.288 | 12.241.536 | 12.285.696 |

## 2. Chất lượng (epoch cuối, CI95 bootstrap theo khối video)

| Mô hình | WER val | WER val_unseen | CER val_unseen | Δ WER val_unseen so với Conformer |
|---|---|---|---|---|
| Conformer-12M | 0,570 [0,563; 0,577] | 0,564 [0,543; 0,587] | 0,412 | (mốc) |
| ConExtBiMamba | **0,441** [0,432; 0,449] | **0,439** [0,415; 0,467] | **0,290** | −0,125 [−0,131; −0,117] |
| Mamba B1 | 0,463 [0,455; 0,472] | 0,465 [0,438; 0,496] | 0,316 | −0,099 [−0,108; −0,089] |

B1 − ConExt (val_unseen, bootstrap cặp theo video, 29 video): **+0,026 [+0,019; +0,034]**.
Thứ tự ở mốc này: ConExtBiMamba > B1 > Conformer, cả ba khoảng chênh đều không chứa 0.

## 3. Học và ổn định

| Mô hình | WER val_unseen epoch 0 → 4 | Giảm WER epoch cuối | Train / val loss | Step bỏ (NaN + AMP) | Grad norm tb / max | Câu rỗng | S / D / I |
|---|---|---|---|---|---|---|---|
| Conformer-12M | 1,000 → 1,000 → 0,999 → 0,649 → 0,564 | 13,2% | 4,19 / 3,02 | 0 + 6 | 2,92 / 299,7 | 0% | 0,374 / 0,175 / 0,014 |
| ConExtBiMamba | 1,000 → 0,734 → 0,542 → 0,479 → 0,439 | 8,3% | 3,60 / 2,19 | 0 + 6 | 3,60 / 222,9 | 0% | 0,340 / 0,064 / 0,035 |
| Mamba B1 (fp32) | 1,000 → 0,799 → 0,610 → 0,507 → 0,465 | 8,2% | 3,70 / 2,40 | 0 + 5 | 3,92 / 406,0 | 0% | 0,356 / 0,076 / 0,032 |
| *B1 lần 1 (fp16, hỏng)* | *1,000 → 0,764 → 0,681 → 0,719 → 0,747* | — | *4,15 / NaN* | *1318 + 5* | — | *41,6%* | — |

Step AMP bị bỏ đều ở epoch 0 (GradScaler hạ scale ban đầu) — bình thường. Grad
norm max đều ở epoch 0. Conformer kẹt bình nguyên blank 3 epoch (deletion cao
nhất: 0,175); hai mô hình Mamba thoát sau 1 epoch. Cả ba còn đang giảm WER.

## 4. Tài nguyên

| Mô hình | s/step | VRAM đỉnh | Ước GPU-giờ / 30 epoch full | Ghi chú |
|---|---|---|---|---|
| Conformer-12M | 0,287 | 2,62 GiB | 7,1 | không có Mamba — số này giữ được |
| ConExtBiMamba | 0,326 | 2,43 GiB | 8,1 | **đo khi khối Mamba fp16 — phải đo lại với fp32** |
| Mamba B1 | 0,733 | 3,28 GiB | 18,1 | khối Mamba fp32 (fp16 cũ 0,47 s/step) |

Cờ "B1 bị ConExtBiMamba trội Pareto" trong `trial_report.md` dựa trên số chưa
cùng điều kiện số học → chưa dùng để kết luận.

## 5. Cổng loại cứng (G0 dùng được · G1 không sụp blank · G2 ổn định · G3 tài nguyên)

| Mô hình | G0 | G1 | G2 | G3 |
|---|---|---|---|---|
| Conformer-12M | ✓ | ✓ | ✓ | ✓ |
| ConExtBiMamba | ✓ | ✓ | ✓ | ✓ |
| Mamba B1 | ✓ | ✓ | ✓ | ✓ |

**Giới hạn:** 1 seed, 5 epoch, 1/4 dữ liệu; thứ hạng có thể đảo khi train full.
RTF ở report là proxy, không phải phép đo RQ2.

## 6. Tiền xử lý: 2 mô hình kia có khác B1 không?

**Không.** Cả ba dùng cùng commit `39e0a44` cho `src/features/log_mel.py`,
`src/data/vietsuperspeech_dataset.py`, `configs/cmvn_stats.json`, mục
`features` trong yaml; OVERLAY của lần train lại B1 chỉ ghi đè
`mamba_encoder.py` và `conextbimamba_encoder.py` (log kernel: "OVERLAY: ghi đè
src/models/…"). Người dùng chọn giữ nguyên front-end cho lần train lại.

Thứ khác nhau là **số học bên trong khối Mamba** (không phải tiền xử lý):

| | Conformer | ConExtBiMamba | B1 |
|---|---|---|---|
| Ở train thử | fp16 AMP (không có khối Mamba) | khối Mamba fp16 | khối Mamba fp32 |
| Từ code hiện tại (train full) | không đổi | **tự động fp32** (code dùng chung) | fp32 |

**Có nên đổi cho 2 mô hình đó không:**
- *Tiền xử lý:* không cần để "bằng" B1 — đã giống hệt. Nếu sau này áp đề xuất ở
  `audio_preprocessing_survey.md` (volume/speed perturbation) thì phải áp **cho
  cả ba** cùng lúc (luật "chỉ encoder khác nhau"), không đổi riêng 2 mô hình.
- *Số học:* Conformer không có gì để đổi. ConExtBiMamba đã được đổi trong code
  (`mamba_fp32`). Kết quả WER của nó ở train thử vẫn dùng được: kernel
  `diag-b1-asr` cho checkpoint ConExt chạy fp16 và fp32 trên 80 câu ra **cùng WER
  0,435**, 0 NaN. Chỉ **tốc độ** phải đo lại.

## 7. Phương án cho bước tiếp theo (người dùng/GVHD quyết)

**(a) Đội hình train full** — số liệu để cân nhắc, không phải khuyến nghị chọn:
- Giữ cả 3: đủ hai biến thể Mamba (thuần vs khung Conformer) cho câu hỏi "Mamba
  làm encoder"; tốn nhất (ước ~7 + ~8-12 + ~18 GPU-h, chưa tính eval; số ConExt chờ
  đo lại).
- Giữ Conformer + ConExtBiMamba: tốt nhất về WER ở mốc này, rẻ hơn; mất mô hình
  Mamba thuần (đề tài lấy Mamba làm trọng tâm).
- Giữ Conformer + B1: Mamba thuần đúng nghĩa; B1 kém ConExt 0,026 và tốn ~2× GPU.

**(b) Đo lại tốc độ ConExtBiMamba với khối Mamba fp32:**
1. Đo trong epoch đầu của train full — không tốn thêm quota, nhưng quyết định (a)
   phải đưa ra trước khi có số. *(Đề xuất nếu (a) không phụ thuộc tốc độ.)*
2. Kernel ngắn đo s/step ConExt fp32 (~15-20 phút GPU gồm setup) — có số công bằng
   trước khi quyết (a).
3. Train thử lại ConExt fp32 đủ 5 epoch (~1 GPU-h) — chỉ cần nếu muốn cả WER lẫn
   tốc độ cùng điều kiện tuyệt đối; WER gần như chắc không đổi (mục 6).

**(c) Tiền xử lý cho train full:** giữ nguyên, hoặc thêm volume/speed perturbation
cho cả 3 (xem `audio_preprocessing_survey.md` mục 4; speed perturbation đổi
s/step → ước lại GPU-giờ).
