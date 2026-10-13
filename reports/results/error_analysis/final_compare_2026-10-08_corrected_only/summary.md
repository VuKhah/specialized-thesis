# Phân tích lỗi clean-test — final_compare_2026-10-08_corrected_only

Tham chiếu: {'corrected_text': 11} (ref_mode=corrected_only). Ngưỡng chuỗi chèn/xoá dài: ≥5 từ. CI95 bootstrap theo video, 10000 mẫu.

## Phân rã WER (% số từ tham chiếu; số tuyệt đối trong ngoặc; CI95)

| Mô hình | WER | S-thanh | S-dấu phụ | S-khác | D | I |
|---|---|---|---|---|---|---|
| conformer | 22.30 (128) [19.68, 27.09] | 0.35 (2) [0.27, 0.49] | 0.35 (2) [0.27, 0.49] | 13.94 (80) [11.32, 18.72] | 6.10 (35) [5.39, 7.39] | 1.57 (9) [0.00, 2.43] |
| mamba | 12.20 (70) [11.05, 14.29] | 0.35 (2) [0.27, 0.49] | 0.35 (2) [0.00, 0.54] | 8.71 (50) [6.47, 12.81] | 1.57 (9) [0.99, 1.89] | 1.22 (7) [0.00, 1.89] |
| conextbimamba | 14.46 (83) [12.94, 17.24] | 0.00 (0) [0.00, 0.00] | 0.52 (3) [0.00, 0.81] | 9.41 (54) [7.28, 13.30] | 2.79 (16) [2.46, 2.96] | 1.74 (10) [1.48, 1.89] |
| phowhisper | 10.98 (63) [8.09, 16.26] | 0.00 (0) [0.00, 0.00] | 0.35 (2) [0.00, 0.54] | 5.40 (31) [3.50, 8.87] | 1.92 (11) [0.49, 2.70] | 3.31 (19) [1.35, 6.90] |
| parakeet_fp32 | 10.28 (59) [7.01, 16.26] | 0.00 (0) [0.00, 0.00] | 0.35 (2) [0.00, 0.54] | 6.97 (40) [4.04, 12.32] | 0.52 (3) [0.49, 0.54] | 2.44 (14) [1.89, 3.45] |

Số từ tham chiếu: 574; số câu: 11; số video: 2.

## Câu rỗng / chuỗi chèn-xoá dài

| Mô hình | hyp rỗng | câu có chèn dài | từ chèn dài | câu có xoá dài | từ xoá dài |
|---|---|---|---|---|---|
| conformer | 0 | 0 | 0 | 1 | 5 |
| mamba | 0 | 0 | 0 | 0 | 0 |
| conextbimamba | 0 | 0 | 0 | 0 | 0 |
| phowhisper | 0 | 0 | 0 | 0 | 0 |
| parakeet_fp32 | 0 | 0 | 0 | 0 | 0 |

## Nhóm ghi chú tên riêng / tiếng Anh

| Mô hình | WER nhóm có ghi chú (câu, từ) | WER nhóm còn lại (câu, từ) |
|---|---|---|
| conformer | — (0, 0) | 22.30 (11, 574) |
| mamba | — (0, 0) | 12.20 (11, 574) |
| conextbimamba | — (0, 0) | 14.46 (11, 574) |
| phowhisper | — (0, 0) | 10.98 (11, 574) |
| parakeet_fp32 | — (0, 0) | 10.28 (11, 574) |

## WER theo độ dài câu (số từ ref)

| Mô hình | <40 | 40-59 | >=60 |
|---|---|---|---|
| conformer | 20.29 (2 câu) | 24.41 (7 câu) | 16.94 (2 câu) |
| mamba | 11.59 (2 câu) | 12.86 (7 câu) | 10.48 (2 câu) |
| conextbimamba | 15.94 (2 câu) | 13.91 (7 câu) | 15.32 (2 câu) |
| phowhisper | 17.39 (2 câu) | 9.97 (7 câu) | 10.48 (2 câu) |
| parakeet_fp32 | 11.59 (2 câu) | 10.24 (7 câu) | 9.68 (2 câu) |

## Top 10 cặp nhầm lẫn (ref → hyp: số lần)

- **conformer**: trung→chung: 2, nhiều→nhiềuàiã: 1, mình→là: 1, nọ→nào: 1, nhưng→thì: 1, thấy→thì: 1, chả→cho: 1, ai→gái: 1, cả→là: 1, kém→kémập: 1
- **mamba**: trung→chung: 2, mà→và: 1, hoài→hoàn: 1, bão→bã: 1, ai→hai: 1, cả→là: 1, rồi→thu: 1, thua→b: 1, thua→thôi: 1, viết→với: 1
- **conextbimamba**: thua→với: 2, trung→chung: 2, nhiều→nhiềuài: 1, hoài→bã: 1, thế→thếọ: 1, chả→ch: 1, cả→là: 1, rồi→vớiuộc: 1, viết→với: 1, nó→nữa: 1
- **phowhisper**: đấy→ấy: 2, mươi→tư: 1, bảy→tảy: 1, nhiên→nhận: 1, thành→này: 1, thì→đi: 1, st→sen: 1, em→kênh: 1, sên→sen: 1, canh→kênh: 1
- **parakeet_fp32**: trung→chung: 2, hoài→hoàião: 1, thế→hay: 1, cả→là: 1, thua→thuế: 1, viết→biết: 1, nhạc→nhà: 1, mình→nhình: 1, viết→nhất: 1, đấy→để: 1
