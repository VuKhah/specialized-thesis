# Phân tích lỗi clean-test — final_compare_2026-10-08

Tham chiếu: {'corrected_text': 11, 'pseudo_label': 192} (ref_mode=auto). Ngưỡng chuỗi chèn/xoá dài: ≥5 từ. CI95 bootstrap theo video, 10000 mẫu.

## Phân rã WER (% số từ tham chiếu; số tuyệt đối trong ngoặc; CI95)

| Mô hình | WER | S-thanh | S-dấu phụ | S-khác | D | I |
|---|---|---|---|---|---|---|
| conformer | 26.46 (2735) [23.69, 29.35] | 0.42 (43) [0.30, 0.55] | 0.15 (15) [0.08, 0.21] | 15.91 (1644) [14.16, 17.67] | 7.91 (818) [6.81, 9.20] | 2.08 (215) [1.64, 2.56] |
| mamba | 17.11 (1768) [14.86, 19.61] | 0.36 (37) [0.24, 0.49] | 0.18 (19) [0.11, 0.27] | 11.51 (1190) [9.94, 13.25] | 2.63 (272) [2.12, 3.23] | 2.42 (250) [1.92, 2.95] |
| conextbimamba | 20.28 (2096) [17.82, 22.84] | 0.24 (25) [0.16, 0.33] | 0.20 (21) [0.10, 0.31] | 13.58 (1404) [11.81, 15.44] | 3.79 (392) [3.25, 4.44] | 2.46 (254) [2.02, 2.92] |
| phowhisper | 15.46 (1598) [12.87, 18.27] | 0.35 (36) [0.22, 0.50] | 0.14 (14) [0.07, 0.22] | 7.75 (801) [6.20, 9.40] | 1.85 (191) [1.53, 2.19] | 5.38 (556) [4.26, 6.59] |
| parakeet_fp32 | 14.03 (1450) [11.74, 16.49] | 0.16 (17) [0.08, 0.26] | 0.11 (11) [0.04, 0.19] | 7.30 (754) [5.88, 8.84] | 1.83 (189) [1.37, 2.32] | 4.63 (479) [3.83, 5.50] |

Số từ tham chiếu: 10335; số câu: 203; số video: 29.

## Câu rỗng / chuỗi chèn-xoá dài

| Mô hình | hyp rỗng | câu có chèn dài | từ chèn dài | câu có xoá dài | từ xoá dài |
|---|---|---|---|---|---|
| conformer | 0 | 0 | 0 | 6 | 35 |
| mamba | 0 | 1 | 5 | 1 | 8 |
| conextbimamba | 0 | 2 | 10 | 2 | 11 |
| phowhisper | 0 | 7 | 46 | 0 | 0 |
| parakeet_fp32 | 0 | 3 | 25 | 2 | 17 |

## Nhóm ghi chú tên riêng / tiếng Anh

| Mô hình | WER nhóm có ghi chú (câu, từ) | WER nhóm còn lại (câu, từ) |
|---|---|---|
| conformer | — (0, 0) | 26.46 (203, 10335) |
| mamba | — (0, 0) | 17.11 (203, 10335) |
| conextbimamba | — (0, 0) | 20.28 (203, 10335) |
| phowhisper | — (0, 0) | 15.46 (203, 10335) |
| parakeet_fp32 | — (0, 0) | 14.03 (203, 10335) |

## WER theo độ dài câu (số từ ref)

| Mô hình | <40 | 40-59 | >=60 |
|---|---|---|---|
| conformer | 33.33 (29 câu) | 26.55 (134 câu) | 23.70 (40 câu) |
| mamba | 24.09 (29 câu) | 17.12 (134 câu) | 14.47 (40 câu) |
| conextbimamba | 28.15 (29 câu) | 19.81 (134 câu) | 18.56 (40 câu) |
| phowhisper | 24.19 (29 câu) | 15.05 (134 câu) | 13.27 (40 câu) |
| parakeet_fp32 | 22.97 (29 câu) | 13.97 (134 câu) | 10.85 (40 câu) |

## Top 10 cặp nhầm lẫn (ref → hyp: số lần)

- **conformer**: nó→đó: 7, làm→là: 6, nó→nói: 5, và→mà: 5, là→và: 4, và→vào: 4, đó→nó: 4, về→với: 4, là→lại: 3, trong→cho: 3
- **mamba**: làm→là: 7, mà→là: 4, đó→nó: 4, về→với: 3, vào→và: 3, nói→nó: 3, mà→và: 2, đi→thì: 2, ngay→nghe: 2, tắt→tác: 2
- **conextbimamba**: làm→là: 8, mà→và: 6, vào→và: 5, đó→nó: 4, nó→đó: 4, lại→là: 4, và→mà: 4, mà→là: 3, tắt→t: 3, với→cái: 3
- **phowhisper**: cảm→cám: 5, ok→kê: 4, ấy→ý: 4, thật→thực: 4, và→mà: 4, đấy→ấy: 3, hóa→hoá: 3, dna→ây: 3, đó→nó: 3, web→goép: 3
- **parakeet_fp32**: ok→kê: 6, nó→đó: 4, thật→thực: 4, bar→ba: 4, marketing→tinh: 3, và→mà: 3, trung→chung: 2, res→rít: 2, ruby→rui: 2, và→vào: 2
