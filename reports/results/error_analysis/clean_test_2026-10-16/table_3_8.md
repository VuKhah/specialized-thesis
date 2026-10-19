# Bảng 3.8 — Phân rã lỗi clean-test (203 câu, 29 video)

Số là % trên số từ ref (cùng mẫu số với WER, nên S + D + I = WER và thanh + dấu phụ + khác = S), trừ cột ghi khác. 40 ms: trung bình ± sd 3 seed. Quy tắc: `docs/notes/rq3_phan_ra_loi_2026-10-16.md` mục 1.

### 3.8a — Train từ đầu, cấu hình 10 ms, ep37, 1 seed, greedy

| Hệ thống | WER | S | D | I | S: thanh | S: dấu phụ | S: khác | WER từ Việt | WER từ ngoại |
|---|---|---|---|---|---|---|---|---|---|
| Conformer relPE | 16,94 | 10,20 | 4,90 | 1,84 | 0,22 | 0,14 | 9,84 | 14,44 | 95,60 |
| B1 Mamba | 14,32 | 9,81 | 2,70 | 1,82 | 0,25 | 0,18 | 9,37 | 11,74 | 95,60 |
| ConExtBiMamba | 17,05 | 11,24 | 3,94 | 1,87 | 0,28 | 0,15 | 10,80 | 14,50 | 97,48 |
| Conformer không PE (ablation) | 22,35 | 13,33 | 7,25 | 1,77 | 0,41 | 0,17 | 12,75 | 19,89 | 99,69 |

### 3.8b — Train từ đầu, cấu hình 10 ms, ep37, 1 seed, greedy: thay thế từ Việt theo thành phần âm tiết, từ dính

| Hệ thống | Cặp hợp lệ (số) | Sai thanh | Sai âm đầu | Sai vần | Chỉ thanh | Chỉ vần | Chỉ âm đầu | ≥ 2 thành phần | h không phải âm tiết (% S Việt) | Từ dính / 1.000 từ |
|---|---|---|---|---|---|---|---|---|---|---|
| Conformer relPE | 597 | 46,6 | 57,1 | 84,8 | 4,0 | 26,8 | 7,9 | 61,3 | 31,5 | 23,38 |
| B1 Mamba | 570 | 48,8 | 57,0 | 83,7 | 4,6 | 24,6 | 8,4 | 62,5 | 28,4 | 8,98 |
| ConExtBiMamba | 636 | 47,8 | 60,7 | 83,0 | 4,6 | 24,2 | 8,8 | 62,4 | 32,9 | 15,94 |
| Conformer không PE (ablation) | 794 | 53,4 | 59,4 | 84,0 | 5,2 | 22,2 | 7,2 | 65,5 | 32,7 | 34,29 |

### 3.8c — Train từ đầu, cấu hình 10 ms, ep37, 1 seed, greedy: CI 95% (bootstrap khối video, 10000 lần, seed 42)

| Hệ thống | WER | S | D | I | S: thanh | WER từ ngoại | Sai thanh (% cặp) | Từ dính / 1.000 |
|---|---|---|---|---|---|---|---|---|
| Conformer relPE | [14,73; 19,46] | [8,90; 11,67] | [4,03; 5,90] | [1,45; 2,27] | [0,13; 0,32] | [85,77; 104,83] | [41,4; 51,6] | [18,92; 28,55] |
| B1 Mamba | [12,30; 16,65] | [8,33; 11,49] | [2,15; 3,32] | [1,46; 2,21] | [0,17; 0,34] | [86,01; 104,49] | [43,8; 54,0] | [6,50; 12,02] |
| ConExtBiMamba | [14,60; 19,77] | [9,55; 13,11] | [3,25; 4,71] | [1,47; 2,33] | [0,18; 0,39] | [87,82; 107,96] | [44,1; 51,4] | [12,51; 19,82] |
| Conformer không PE (ablation) | [19,94; 24,86] | [11,83; 14,90] | [6,20; 8,43] | [1,38; 2,17] | [0,28; 0,54] | [92,19; 107,77] | [48,9; 57,7] | [29,87; 38,82] |

### 3.8a — Train từ đầu, cấu hình 40 ms, ep37, 3 seed (tb ± sd), greedy — bảng chính

| Hệ thống | WER | S | D | I | S: thanh | S: dấu phụ | S: khác | WER từ Việt | WER từ ngoại |
|---|---|---|---|---|---|---|---|---|---|
| Conformer relPE | 12,64 ± 0,46 | 7,84 ± 0,26 | 2,71 ± 0,13 | 2,08 ± 0,14 | 0,17 ± 0,02 | 0,15 ± 0,01 | 7,52 ± 0,23 | 10,11 ± 0,46 | 92,35 ± 2,00 |
| B1 Mamba | 12,66 ± 0,14 | 7,86 ± 0,11 | 3,08 ± 0,14 | 1,72 ± 0,14 | 0,23 ± 0,05 | 0,11 ± 0,01 | 7,52 ± 0,07 | 10,27 ± 0,19 | 87,97 ± 1,89 |
| ConExtBiMamba | 12,91 ± 0,18 | 8,00 ± 0,17 | 2,92 ± 0,15 | 2,00 ± 0,11 | 0,21 ± 0,01 | 0,14 ± 0,01 | 7,65 ± 0,16 | 10,38 ± 0,13 | 92,78 ± 1,67 |

### 3.8b — Train từ đầu, cấu hình 40 ms, ep37, 3 seed (tb ± sd), greedy — bảng chính: thay thế từ Việt theo thành phần âm tiết, từ dính

| Hệ thống | Cặp hợp lệ (số) | Sai thanh | Sai âm đầu | Sai vần | Chỉ thanh | Chỉ vần | Chỉ âm đầu | ≥ 2 thành phần | h không phải âm tiết (% S Việt) | Từ dính / 1.000 từ |
|---|---|---|---|---|---|---|---|---|---|---|
| Conformer relPE | 470 | 40,2 ± 2,0 | 55,0 ± 1,8 | 84,4 ± 1,6 | 3,8 ± 0,4 | 29,7 ± 2,1 | 9,4 ± 0,6 | 57,2 ± 1,4 | 24,2 ± 2,0 | 11,27 ± 1,04 |
| B1 Mamba | 470 | 47,5 ± 1,7 | 59,7 ± 0,2 | 82,4 ± 0,7 | 5,1 ± 1,0 | 23,3 ± 0,8 | 9,5 ± 0,4 | 62,1 ± 1,4 | 25,6 ± 2,9 | 10,37 ± 1,18 |
| ConExtBiMamba | 483 | 44,6 ± 3,0 | 59,6 ± 1,6 | 82,7 ± 1,0 | 4,4 ± 0,2 | 24,5 ± 1,9 | 10,1 ± 0,4 | 61,0 ± 2,4 | 24,8 ± 0,8 | 11,11 ± 0,68 |

### 3.8c — Train từ đầu, cấu hình 40 ms, ep37, 3 seed (tb ± sd), greedy — bảng chính: CI 95% (bootstrap khối video, 10000 lần, seed 42)

| Hệ thống | WER | S | D | I | S: thanh | WER từ ngoại | Sai thanh (% cặp) | Từ dính / 1.000 |
|---|---|---|---|---|---|---|---|---|
| Conformer relPE | [10,80; 14,70] | [6,62; 9,21] | [2,24; 3,27] | [1,72; 2,48] | [0,10; 0,26] | [82,09; 102,32] | [35,8; 43,9] | [8,87; 14,27] |
| B1 Mamba | [10,75; 14,77] | [6,63; 9,23] | [2,53; 3,67] | [1,36; 2,14] | [0,15; 0,32] | [76,02; 99,48] | [43,2; 51,8] | [8,41; 12,54] |
| ConExtBiMamba | [10,99; 15,01] | [6,74; 9,39] | [2,45; 3,43] | [1,62; 2,40] | [0,13; 0,29] | [81,81; 103,43] | [41,9; 47,5] | [8,94; 13,50] |

### 3.8a — Train từ đầu, cấu hình 40 ms, ep37, 3 seed (tb ± sd), beam 10 + từ điển + LM 3-gram — bảng phụ

| Hệ thống | WER | S | D | I | S: thanh | S: dấu phụ | S: khác | WER từ Việt | WER từ ngoại |
|---|---|---|---|---|---|---|---|---|---|
| Conformer relPE | 10,59 ± 0,34 | 5,83 ± 0,27 | 3,40 ± 0,03 | 1,36 ± 0,06 | 0,17 ± 0,01 | 0,14 ± 0,01 | 5,52 ± 0,28 | 8,49 ± 0,36 | 76,80 ± 1,44 |
| B1 Mamba | 11,17 ± 0,07 | 6,28 ± 0,10 | 3,70 ± 0,08 | 1,19 ± 0,03 | 0,24 ± 0,01 | 0,11 ± 0,01 | 5,93 ± 0,11 | 9,06 ± 0,15 | 77,43 ± 2,68 |
| ConExtBiMamba | 10,95 ± 0,13 | 6,31 ± 0,06 | 3,33 ± 0,15 | 1,31 ± 0,05 | 0,24 ± 0,03 | 0,14 ± 0,02 | 5,94 ± 0,02 | 8,79 ± 0,07 | 79,10 ± 2,53 |

### 3.8b — Train từ đầu, cấu hình 40 ms, ep37, 3 seed (tb ± sd), beam 10 + từ điển + LM 3-gram — bảng phụ: thay thế từ Việt theo thành phần âm tiết, từ dính

| Hệ thống | Cặp hợp lệ (số) | Sai thanh | Sai âm đầu | Sai vần | Chỉ thanh | Chỉ vần | Chỉ âm đầu | ≥ 2 thành phần | h không phải âm tiết (% S Việt) | Từ dính / 1.000 từ |
|---|---|---|---|---|---|---|---|---|---|---|
| Conformer relPE | 440 | 40,8 ± 2,1 | 57,7 ± 2,4 | 83,8 ± 0,4 | 4,0 ± 0,3 | 27,5 ± 2,9 | 9,4 ± 0,8 | 59,0 ± 2,6 | 4,0 ± 0,5 | 0,00 ± 0,00 |
| B1 Mamba | 477 | 46,9 ± 1,1 | 61,4 ± 1,2 | 82,6 ± 0,4 | 5,3 ± 0,3 | 21,7 ± 0,6 | 9,4 ± 0,5 | 63,6 ± 0,3 | 4,3 ± 0,7 | 0,03 ± 0,06 |
| ConExtBiMamba | 480 | 47,0 ± 2,3 | 60,5 ± 1,3 | 82,0 ± 1,3 | 5,1 ± 0,6 | 23,1 ± 1,1 | 9,5 ± 1,5 | 62,3 ± 0,9 | 3,8 ± 0,7 | 0,00 ± 0,00 |

### 3.8c — Train từ đầu, cấu hình 40 ms, ep37, 3 seed (tb ± sd), beam 10 + từ điển + LM 3-gram — bảng phụ: CI 95% (bootstrap khối video, 10000 lần, seed 42)

| Hệ thống | WER | S | D | I | S: thanh | WER từ ngoại | Sai thanh (% cặp) | Từ dính / 1.000 |
|---|---|---|---|---|---|---|---|---|
| Conformer relPE | [8,93; 12,46] | [4,96; 6,80] | [2,68; 4,20] | [1,04; 1,72] | [0,10; 0,25] | [65,75; 86,54] | [36,9; 44,6] | [0,00; 0,00] |
| B1 Mamba | [9,44; 13,08] | [5,25; 7,42] | [3,01; 4,44] | [0,92; 1,50] | [0,15; 0,35] | [65,69; 87,65] | [42,4; 51,3] | [0,00; 0,10] |
| ConExtBiMamba | [9,23; 12,86] | [5,26; 7,47] | [2,74; 3,99] | [1,01; 1,63] | [0,15; 0,34] | [67,76; 89,14] | [43,6; 50,2] | [0,00; 0,00] |

### 3.8a — Nhóm pre-train (mốc tham chiếu, khác điều kiện, không xếp hạng chung)

| Hệ thống | WER | S | D | I | S: thanh | S: dấu phụ | S: khác | WER từ Việt | WER từ ngoại |
|---|---|---|---|---|---|---|---|---|---|
| Parakeet-CTC-0.6B-vi zero-shot | 13,78 | 7,27 | 2,05 | 4,46 | 0,15 | 0,09 | 7,03 | 8,79 | 170,85 |
| Parakeet-CTC-0.6B-vi fine-tune (nhãn giả) | 9,43 | 5,45 | 2,28 | 1,70 | 0,14 | 0,09 | 5,23 | 7,05 | 84,59 |
| PhoWhisper-small zero-shot | 14,86 | 8,16 | 1,91 | 4,78 | 0,34 | 0,13 | 7,70 | 10,10 | 164,58 |

### 3.8b — Nhóm pre-train (mốc tham chiếu, khác điều kiện, không xếp hạng chung): thay thế từ Việt theo thành phần âm tiết, từ dính

| Hệ thống | Cặp hợp lệ (số) | Sai thanh | Sai âm đầu | Sai vần | Chỉ thanh | Chỉ vần | Chỉ âm đầu | ≥ 2 thành phần | h không phải âm tiết (% S Việt) | Từ dính / 1.000 từ |
|---|---|---|---|---|---|---|---|---|---|---|
| Parakeet-CTC-0.6B-vi zero-shot | 365 | 43,3 | 62,7 | 81,1 | 4,4 | 23,8 | 12,1 | 59,7 | 24,6 | 14,78 |
| Parakeet-CTC-0.6B-vi fine-tune (nhãn giả) | 291 | 40,9 | 58,4 | 81,1 | 4,8 | 25,4 | 10,7 | 59,1 | 23,2 | 6,28 |
| PhoWhisper-small zero-shot | 571 | 50,1 | 58,1 | 81,8 | 6,1 | 22,1 | 8,2 | 63,6 | 1,2 | 0,77 |

### 3.8c — Nhóm pre-train (mốc tham chiếu, khác điều kiện, không xếp hạng chung): CI 95% (bootstrap khối video, 10000 lần, seed 42)

| Hệ thống | WER | S | D | I | S: thanh | WER từ ngoại | Sai thanh (% cặp) | Từ dính / 1.000 |
|---|---|---|---|---|---|---|---|---|
| Parakeet-CTC-0.6B-vi zero-shot | [11,50; 16,24] | [5,89; 8,79] | [1,55; 2,57] | [3,66; 5,32] | [0,08; 0,24] | [154,65; 194,42] | [36,8; 49,9] | [11,19; 18,83] |
| Parakeet-CTC-0.6B-vi fine-tune (nhãn giả) | [7,73; 11,27] | [4,30; 6,71] | [1,83; 2,78] | [1,39; 2,03] | [0,07; 0,21] | [74,07; 94,37] | [35,2; 46,9] | [4,40; 8,40] |
| PhoWhisper-small zero-shot | [12,32; 17,62] | [6,61; 9,82] | [1,56; 2,28] | [3,72; 5,98] | [0,20; 0,50] | [149,47; 177,57] | [44,9; 54,6] | [0,09; 1,93] |

### 3.8a — Nhãn giả Zipformer so với bản hiệu đính (tham khảo)

| Hệ thống | WER | S | D | I | S: thanh | S: dấu phụ | S: khác | WER từ Việt | WER từ ngoại |
|---|---|---|---|---|---|---|---|---|---|
| Nhãn giả Zipformer | 1,77 | 1,16 | 0,41 | 0,20 | 0,04 | 0,00 | 1,12 | 0,65 | 36,99 |

### 3.8b — Nhãn giả Zipformer so với bản hiệu đính (tham khảo): thay thế từ Việt theo thành phần âm tiết, từ dính

| Hệ thống | Cặp hợp lệ (số) | Sai thanh | Sai âm đầu | Sai vần | Chỉ thanh | Chỉ vần | Chỉ âm đầu | ≥ 2 thành phần | h không phải âm tiết (% S Việt) | Từ dính / 1.000 từ |
|---|---|---|---|---|---|---|---|---|---|---|
| Nhãn giả Zipformer | 22 | 59,1 | 31,8 | 59,1 | 18,2 | 13,6 | 22,7 | 45,5 | 24,1 | 0,00 |

### 3.8c — Nhãn giả Zipformer so với bản hiệu đính (tham khảo): CI 95% (bootstrap khối video, 10000 lần, seed 42)

| Hệ thống | WER | S | D | I | S: thanh | WER từ ngoại | Sai thanh (% cặp) | Từ dính / 1.000 |
|---|---|---|---|---|---|---|---|---|
| Nhãn giả Zipformer | [1,24; 2,34] | [0,75; 1,63] | [0,25; 0,58] | [0,08; 0,34] | [0,00; 0,09] | [27,34; 47,60] | [42,9; 80,0] | [0,00; 0,00] |

### 3.8a — Phụ lục: v1 10 ms ep28 (đối chiếu số từ dính 15/10)

| Hệ thống | WER | S | D | I | S: thanh | S: dấu phụ | S: khác | WER từ Việt | WER từ ngoại |
|---|---|---|---|---|---|---|---|---|---|
| Conformer không PE v1 ep28 | 26,48 | 16,51 | 7,98 | 1,99 | 0,43 | 0,14 | 15,94 | 23,93 | 106,92 |
| B1 Mamba v1 ep28 | 17,04 | 12,03 | 2,71 | 2,31 | 0,33 | 0,18 | 11,52 | 14,24 | 105,35 |
| ConExtBiMamba v1 ep28 | 20,18 | 13,94 | 3,88 | 2,36 | 0,25 | 0,19 | 13,49 | 17,37 | 108,46 |

### 3.8b — Phụ lục: v1 10 ms ep28 (đối chiếu số từ dính 15/10): thay thế từ Việt theo thành phần âm tiết, từ dính

| Hệ thống | Cặp hợp lệ (số) | Sai thanh | Sai âm đầu | Sai vần | Chỉ thanh | Chỉ vần | Chỉ âm đầu | ≥ 2 thành phần | h không phải âm tiết (% S Việt) | Từ dính / 1.000 từ |
|---|---|---|---|---|---|---|---|---|---|---|
| Conformer không PE v1 ep28 | 1047 | 54,9 | 62,1 | 85,6 | 4,3 | 20,2 | 6,2 | 69,2 | 30,1 | 42,89 |
| B1 Mamba v1 ep28 | 744 | 50,1 | 57,7 | 84,1 | 4,6 | 25,0 | 6,9 | 63,6 | 26,4 | 8,79 |
| ConExtBiMamba v1 ep28 | 850 | 48,4 | 61,5 | 85,6 | 3,1 | 22,1 | 7,5 | 67,3 | 29,6 | 18,35 |

### 3.8c — Phụ lục: v1 10 ms ep28 (đối chiếu số từ dính 15/10): CI 95% (bootstrap khối video, 10000 lần, seed 42)

| Hệ thống | WER | S | D | I | S: thanh | WER từ ngoại | Sai thanh (% cặp) | Từ dính / 1.000 |
|---|---|---|---|---|---|---|---|---|
| Conformer không PE v1 ep28 | [23,70; 29,37] | [14,71; 18,31] | [6,84; 9,29] | [1,55; 2,48] | [0,31; 0,58] | [99,73; 114,40] | [50,3; 58,9] | [37,70; 48,47] |
| B1 Mamba v1 ep28 | [14,79; 19,54] | [10,47; 13,78] | [2,16; 3,32] | [1,80; 2,85] | [0,22; 0,45] | [95,36; 115,75] | [45,3; 55,0] | [6,86; 11,04] |
| ConExtBiMamba v1 ep28 | [17,73; 22,75] | [12,13; 15,83] | [3,29; 4,56] | [1,93; 2,82] | [0,16; 0,35] | [101,55; 116,13] | [44,6; 52,0] | [15,08; 21,77] |

Ghi chú: "Cặp hợp lệ" = thay thế có từ ref nhóm Việt và cả hai phía là âm tiết Việt hợp lệ (số trung bình seed); các cột "Sai …" là % trên số cặp đó, đa nhãn (một cặp có thể sai nhiều thành phần); "Chỉ …" và "≥ 2" loại trừ, cộng = 100%. CI của hàng 3 seed chỉ phản ánh dao động mẫu test.

## Top 10 cặp nhầm lẫn (ref → hyp: số lần) — seed 42 / 1 seed

- **Conformer relPE (10ms)**: bar→ba 6 [sub_other], đó→nó 5 [onset], nó→đó 5 [onset], trong→cho 4 [onset+rhyme], trung→chung 3 [onset], nó→nói 3 [rhyme], vào→và 3 [rhyme], đội→độ 3 [rhyme], voi→vo 3 [rhyme], tắt→tác 2 [rhyme]
- **B1 Mamba (10ms)**: đó→nó 5 [onset], trong→cho 4 [onset+rhyme], bar→ba 4 [sub_other], trung→chung 3 [onset], nó→nói 3 [rhyme], nó→đó 3 [onset], thật→thực 3 [rhyme], nft→nt 3 [sub_other], friendzone→fanzon 3 [sub_other], tắt→tác 2 [rhyme]
- **ConExtBiMamba (10ms)**: mà→và 4 [onset], friendzone→fanzon 4 [sub_other], bar→ba 4 [sub_other], là→và 3 [onset], đó→nó 3 [onset], và→vào 3 [rhyme], sau→xong 3 [onset+rhyme], nó→nói 3 [rhyme], lại→là 3 [rhyme+tone], vào→và 3 [rhyme]
- **Conformer không PE (ablation) (10ms)**: làm→là 6 [rhyme], đó→nó 5 [onset], vào→và 5 [rhyme], nó→đó 5 [onset], bar→ba 5 [sub_other], là→lại 3 [rhyme+tone], mà→và 3 [onset], và→mà 3 [onset], đội→độ 3 [rhyme], thật→thực 3 [rhyme]
- **Conformer relPE (40ms)**: nó→đó 5 [onset], bar→ba 4 [sub_other], trung→chung 3 [onset], đan→đang 3 [rhyme], bà→ba 3 [tone], thật→thực 3 [rhyme], trong→cho 3 [onset+rhyme], và→mà 3 [onset], nấm→nắm 3 [rhyme], các→cái 2 [rhyme]
- **B1 Mamba (40ms)**: mà→và 3 [onset], thật→thực 3 [rhyme], với→cái 3 [onset+rhyme], bar→ba 3 [sub_other], vuông→dung 2 [onset+rhyme], trung→chung 2 [onset], gojek→câuje 2 [sub_other], các→cái 2 [rhyme], như→nhưng 2 [rhyme], làm→là 2 [rhyme]
- **ConExtBiMamba (40ms)**: nó→đó 5 [onset], trong→cho 4 [onset+rhyme], bar→ba 4 [sub_other], đó→nó 3 [onset], thật→thực 3 [rhyme], viết→biết 2 [onset], trung→chung 2 [onset], các→cái 2 [rhyme], làm→là 2 [rhyme], anh→thanh 2 [onset]
- **Conformer relPE (40lm)**: đó→nó 3 [onset], đan→đang 3 [rhyme], bà→ba 3 [tone], thật→thực 3 [rhyme], trong→cho 3 [onset+rhyme], bar→ba 3 [sub_other], mà→và 2 [onset], trung→chung 2 [onset], gojek→tre 2 [sub_other], các→cái 2 [rhyme]
- **B1 Mamba (40lm)**: mà→và 4 [onset], bar→ba 4 [sub_other], đó→nó 3 [onset], vuông→dung 2 [onset+rhyme], trung→chung 2 [onset], các→cái 2 [rhyme], như→nhưng 2 [rhyme], mất→mắc 2 [rhyme], singapore→sinh 2 [sub_other], làm→là 2 [rhyme]
- **ConExtBiMamba (40lm)**: đó→nó 4 [onset], trong→cho 4 [onset+rhyme], bar→ba 4 [sub_other], các→cái 3 [rhyme], nó→đó 3 [onset], làm→là 3 [rhyme], mà→và 2 [onset], trung→chung 2 [onset], và→là 2 [onset], brief→brip 2 [sub_other]
- **Parakeet-CTC-0.6B-vi zero-shot (pretrain)**: bar→ba 5 [sub_other], nó→đó 4 [onset], thật→thực 4 [rhyme], friendzone→dôn 4 [sub_other], và→vào 3 [rhyme], marketing→tinh 3 [sub_other], và→mà 3 [onset], trung→chung 2 [onset], res→rít 2 [sub_other], ruby→rui 2 [sub_other]
- **Parakeet-CTC-0.6B-vi fine-tune (nhãn giả) (pretrain)**: nó→đó 4 [onset], trung→chung 3 [onset], trong→cho 3 [onset+rhyme], thật→thực 3 [rhyme], các→cái 2 [rhyme], như→nhưng 2 [rhyme], anh→ảnh 2 [tone], ông→ônga 2 [sub_other], và→mà 2 [onset], đan→đang 2 [rhyme]
- **PhoWhisper-small zero-shot (pretrain)**: cảm→cám 5 [tone], ấy→ý 4 [rhyme], thật→thực 4 [rhyme], và→mà 4 [onset], friendzone→rôn 4 [sub_other], đấy→ấy 3 [onset], dna→ây 3 [sub_other], đó→nó 3 [onset], nft→thi 3 [sub_other], web→goép 3 [sub_other]
- **Nhãn giả Zipformer (label)**: trung→chung 3 [onset], gojek→trách 2 [sub_other], tét→test 2 [sub_other], sex→sách 2 [sub_other], friendzone→john 2 [sub_other], show→sho 1 [sub_other], curieous→cua 1 [sub_other], curieous→crush 1 [sub_other], university→u 1 [sub_other], upenn→pan 1 [sub_other]
- **Conformer không PE v1 ep28 (v1)**: nó→đó 7 [onset], làm→là 6 [rhyme], nó→nói 5 [rhyme], và→mà 5 [onset], là→và 4 [onset], và→vào 4 [rhyme], đó→nó 4 [onset], về→với 4 [rhyme+tone], là→lại 3 [rhyme+tone], trong→cho 3 [onset+rhyme]
- **B1 Mamba v1 ep28 (v1)**: làm→là 7 [rhyme], mà→là 4 [onset], đó→nó 4 [onset], về→với 3 [rhyme+tone], vào→và 3 [rhyme], nói→nó 3 [rhyme], mà→và 2 [onset], đi→thì 2 [onset+tone], ngay→nghe 2 [rhyme], tắt→tác 2 [rhyme]
- **ConExtBiMamba v1 ep28 (v1)**: làm→là 8 [rhyme], mà→và 6 [onset], vào→và 5 [rhyme], đó→nó 4 [onset], nó→đó 4 [onset], lại→là 4 [rhyme+tone], và→mà 4 [onset], bar→ba 4 [sub_other], mà→là 3 [onset], tắt→t 3 [sub_other]
