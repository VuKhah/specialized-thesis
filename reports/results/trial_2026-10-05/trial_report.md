# Báo cáo train thử — mốc: `conformer_ctc_baseline`

## 1. Chất lượng (epoch cuối, CI95 bootstrap theo khối video)

| Mô hình | Epoch | WER val | WER val_unseen | CER val_unseen | Δ WER val so với mốc [CI95] | Δ WER val_unseen so với mốc [CI95] | P(tốt hơn mốc, val_unseen) |
|---|---|---|---|---|---|---|---|
| conformer_ctc_baseline | 5 | 0.5697 [0.5626, 0.5769] | 0.5638 [0.5427, 0.5873] | 0.4116 | (mốc) | (mốc) | – |
| conextbimamba_ctc | 5 | 0.4405 [0.4323, 0.4489] | 0.4392 [0.4145, 0.4668] | 0.2896 | -0.1292 [-0.1320, -0.1264] | -0.1246 [-0.1312, -0.1174] | 1.00 |
| mamba_ctc | 5 | 0.4630 [0.4545, 0.4717] | 0.4652 [0.4375, 0.4956] | 0.3156 | -0.1066 [-0.1097, -0.1035] | -0.0986 [-0.1077, -0.0886] | 1.00 |

## 2. Học và ổn định

| Mô hình | WER val_unseen theo epoch | Giảm WER epoch cuối | val_unseen − val | Train loss | Val loss | Step bị bỏ | Grad norm tb / max | Câu rỗng | S / D / I (val_unseen) |
|---|---|---|---|---|---|---|---|---|---|
| conformer_ctc_baseline | 1.000 → 1.000 → 0.999 → 0.649 → 0.564 | 13.2% | -0.0059 | 4.193 | 3.020 | 6/3555 | 2.92 / 299.71 | 0.0% | 0.374 / 0.175 / 0.014 |
| conextbimamba_ctc | 1.000 → 0.734 → 0.542 → 0.479 → 0.439 | 8.3% | -0.0013 | 3.602 | 2.193 | 6/3555 | 3.60 / 222.91 | 0.0% | 0.340 / 0.064 / 0.035 |
| mamba_ctc | 1.000 → 0.799 → 0.610 → 0.507 → 0.465 | 8.2% | +0.0022 | 3.703 | 2.398 | 5/3555 | 3.92 / 406.02 | 0.0% | 0.356 / 0.076 / 0.032 |

## 3. Tài nguyên

| Mô hình | Tham số encoder | s/step | VRAM đỉnh (GiB) | Ước GPU-giờ train full 30 epoch | RTF eval (proxy) | Bị trội Pareto bởi |
|---|---|---|---|---|---|---|
| conformer_ctc_baseline | 12,204,288 | 0.287 | 2.62 | 7.1 | 0.0004 | không |
| conextbimamba_ctc | 12,241,536 | 0.326 | 2.43 | 8.1 | 0.0004 | không |
| mamba_ctc | 12,285,696 | 0.733 | 3.28 | 18.1 | 0.0009 | conextbimamba_ctc |

## 4. Cổng loại cứng (tự đánh giá) — quyết định giữ/bỏ: người dùng/GVHD

G0 lần chạy dùng được (WER val_unseen mốc < 0.95) · G1 không sụp về blank (câu rỗng ≤ 5%) · G2 ổn định (loss hữu hạn, step bỏ ≤ 1%, WER val giảm) · G3 tài nguyên (≤ 30 GPU-giờ, VRAM ≤ 14 GiB)

| Mô hình | G0 | G1 | G2 | G3 |
|---|---|---|---|---|
| conformer_ctc_baseline | ✓ | ✓ | ✓ | ✓ |
| conextbimamba_ctc | ✓ | ✓ | ✓ | ✓ |
| mamba_ctc | ✓ | ✓ | ✓ | ✓ |

Ghi chú: 1 seed, 5 epoch trên 1/4 dữ liệu — thứ hạng có thể đảo khi train full. RTF eval là proxy (batch, 2 GPU, gồm đọc dữ liệu), không phải phép đo RQ2.
