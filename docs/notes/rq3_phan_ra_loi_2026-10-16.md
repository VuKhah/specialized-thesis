# RQ3 — Phân rã lỗi trên clean-test (Bảng 3.8, 2026-10-16)

Việc L6 (`review_fable_2026-10-15.md` mục 2.2): phân tích trên dự đoán **đã có**, 0 GPU. Bảng **mô tả** — không có
kiểm định so cặp mới; mọi so sánh "A tốt hơn B" chỉ dẫn lại so cặp chính thức ở `clean_test_results_2026-10-15.md`.

Chạy lại: `python -m src.evaluation.error_breakdown` → `reports/results/error_analysis/clean_test_2026-10-16/`.

## 1. Quy tắc (chốt 16/10 TRƯỚC khi tính bất kỳ số phân rã nào)

Mục này viết và lưu trước lần chạy đầu tiên của `error_breakdown.py`; sau khi có số không sửa (muốn đổi thì thêm mục mới
ghi ngày, báo cả hai cách).

**Đầu vào và chuẩn hoá — y hệt bảng chính** (`score_clean_test.py`, lần chấm chung 16/10):
ref = `corrected_text` 203 câu; token hoá bằng `ref_tokens` / `hyp_tokens` (normalize_text → bỏ từ đệm → `canon_tone` →
`alt_spellings`); từ vấp dở #153 tuỳ chọn (chọn phương án như `score_sentence`); căn chỉnh Levenshtein mức từ của jiwer
trên chuỗi token chuẩn. Nguồn giả thuyết: đúng các nguồn `score_clean_test.main()` nạp (hàm chung `load_hypotheses`).
Ràng buộc kiểm: với mọi hệ thống, (S + D + I) / số từ ref == `wer` trong `clean_test_scored_2026-10-16_chung/results.json`,
và lỗi nhóm Việt / ngoại == `wer_vi` / `wer_fx` (sai số ≤ 1e-9).

**S / D / I.** Đếm theo đoạn căn chỉnh của jiwer; tỉ lệ = số lỗi / số từ ref (cùng mẫu số với WER) → S + D + I = WER.

**Nhóm Việt / ngoại** — giữ nguyên quy tắc `count_errors`: S và D theo nhãn của từ ref; I kề một đoạn thay thế → nhóm
của từ ref liền kề; I đứng riêng → theo dạng chữ từ chèn (âm tiết Việt hợp lệ hoặc có dấu Việt → Việt).

**Loại thay thế** (giữ thứ tự cố định của `error_analysis.classify_substitution`, mỗi cặp đúng một loại):
(a) `thanh`: bỏ 5 dấu thanh thì r == h; (b) `dấu phụ`: không thuộc (a), bỏ mọi dấu (kể cả mũ/trăng/móc, đ→d) thì
r == h; (c) `khác`: còn lại. Cộng: thanh + dấu phụ + khác = S.

**Theo thành phần âm tiết** — chỉ cho cặp thay thế mà **từ ref thuộc nhóm Việt và cả r lẫn h là âm tiết Việt hợp lệ**
(`is_valid_syllable`). Cặp còn lại của nhóm Việt (h không hợp lệ: từ dính, chuỗi Latin…) đếm riêng là "h không phải âm
tiết". Mỗi âm tiết tách thành (âm đầu, vần, thanh):
- thanh ∈ {ngang, huyền, sắc, hỏi, ngã, nặng} (dấu thanh trong NFD);
- bỏ dấu thanh, viết hoa, tách âm đầu = **âm đầu dài nhất trong danh sách `ONSETS` của `vi_syllable.py` sao cho phần còn
  lại là vần hợp lệ trong `RHYMES`**; không có âm đầu nào thoả → âm đầu rỗng; riêng G + vần bắt đầu bằng I (*gì*,
  *gìn*, *giếng*) đổi thành âm đầu GI (chữ G trước I luôn là GI viết gộp; sửa trước khi xem số, khi kiểm nhất quán
  trong code bắt được cặp *ghi*/*gì*). Hệ quả: *gì* → GI + I, *giá* → GI + A, *quá* → QU + A — quy ước chính tả,
  không phải âm vị học;
- âm đầu chuẩn hoá các biến thể chính tả cùng âm trước khi so: NGH → NG, GH → G, K → C (*kê*/*cơ* không tính sai âm đầu).
  QU, GI, D, R giữ nguyên (khác chữ = khác cách viết, kể cả khi trùng âm theo phương ngữ).
- So từng thành phần; một cặp có thể sai nhiều thành phần (đa nhãn). Báo: số cặp sai âm đầu / vần / thanh (đa nhãn, tổng
  có thể > số cặp) và tỉ lệ trên số cặp hợp lệ; thêm phân loại loại trừ: chỉ thanh · chỉ vần · chỉ âm đầu · ≥ 2 thành phần.
  Mọi cặp loại (a) là "chỉ thanh" (kiểm trong code); chiều ngược lại có thể lệch vài cặp viết sai chính tả âm đầu cùng
  âm (vd. *nghẹn* → *ngèn*: loại (c) nhưng "chỉ thanh") — bổ sung khi chạy kiểm, trước khi xem số. Cặp không khác thành phần nào sau chuẩn hoá âm
  đầu (vd. *kê*/*cê*) xếp riêng "chỉ khác chính tả âm đầu" (dự kiến ~0).

**Từ dính** — định nghĩa của `glued_words_2026-10-15.md` / `vi_syllable.sticky_words`: token hyp có dấu/chữ riêng tiếng
Việt nhưng không phải âm tiết hợp lệ. Đếm trên token hyp sau chuẩn hoá chấm (đã bỏ từ đệm, `canon_tone`, alt_spellings);
báo số tuyệt đối và trên 1.000 từ ref. Từ dính là thuộc tính của hyp, không phải một loại lỗi tách rời — nó nằm trong S
(hoặc I) và trong loại "h không phải âm tiết".

**Gộp seed / CI.** 40 ms: báo trung bình ± sd 3 seed (42/43/44) cho mọi tỉ lệ; greedy là bảng chính, beam + LM bảng
phụ. 10 ms: 1 seed. CI 95%: bootstrap theo khối video (29 video, seed 42, 10.000 lần, cùng chỉ số lấy mẫu cho mọi hệ
thống); với 40 ms là CI của **trung bình 3 seed** dưới lấy lại video (chỉ phản ánh dao động mẫu test, không gồm dao động
seed — sd seed báo riêng). CI cho: WER, S, D, I, thanh, dấu phụ, khác, WER Việt, WER ngoại, từ dính/1.000 từ.

**Trình bày.** Bốn nhóm tách bảng, không xếp hạng chung: train từ đầu 10 ms ep37 · train từ đầu 40 ms ep37 (greedy;
beam + LM phụ) · pre-train (Parakeet zero-shot / fine-tune, PhoWhisper) · nhãn giả Zipformer. v1 10 ms ep28 chỉ ở phụ lục
(đối chiếu số từ dính cũ).

**Ghi nhận thứ tự (khai trung thực).** Ba chỗ trong quy tắc trên được thêm khi chạy kiểm nhất quán trong code, trước khi
in bất kỳ số tổng hợp nào: luật G + I → GI (assert bắt cặp *ghi*/*gì*), chiều ngược của kiểm "chỉ thanh ⇔ loại (a)"
(assert bắt cặp *nghẹn*/*ngèn*), và loại "chỉ khác chính tả âm đầu" (kết quả: 0 cặp ở mọi hệ thống). Không đổi gì sau
khi thấy bảng.

**Người dùng duyệt 17/10:** giữ quy tắc tách âm tiết trên (kể cả 3 chỉnh sửa lúc chạy kiểm, khai ở Ch2); số từ dính
dùng `sticky_words` cho mọi hệ thống, bộ số 15/10 chỉ ghi phụ lục (định nghĩa khác, thứ tự không đổi — mục 5).

## 2. Kiểm khớp bảng chính

`error_breakdown` dừng nếu lệch: trên cả 29 hệ thống, |(S + D + I)/từ ref − `wer`|, |… − `wer_vi`|, |… − `wer_fx`| lớn
nhất = **0** (khớp chính xác, `summary.json` → `meta.max_abs_diff_vs_results`); từng câu cũng khớp `score_sentence`
(assert). `score_clean_test.py` được tách hàm `load_hypotheses()` (không đổi logic); chạy lại vào thư mục tạm cho
`results.json` và `per_sentence.tsv` **trùng từng byte** với `clean_test_scored_2026-10-16_chung/`.

## 3. Bảng 3.8 (dạng chép vào Word)

Đơn vị: % trên số từ ref (10.352 từ: Việt 10.033 · ngoại 319; từ vấp dở #153 có thể làm mẫu số 10.351), nên S + D + I =
WER và thanh + dấu phụ + khác = S. WER từ Việt/ngoại: mẫu số là số từ ref của nhóm (ngoại > 100% khi chèn nhiều). 40 ms:
trung bình ± sd 3 seed (42/43/44). CI 95% từng ô: `reports/results/error_analysis/clean_test_2026-10-16/table_3_8.md`
(bảng 3.8c) và `summary.csv`.

**Bảng 3.8a — S/D/I, loại thay thế, nhóm từ, từ dính**

| Hệ thống | WER | S | D | I | S: thanh | S: dấu phụ | S: khác | WER Việt | WER ngoại | Từ dính /1.000 từ |
|---|---|---|---|---|---|---|---|---|---|---|
| *Train từ đầu 10 ms (ep37, 1 seed)* | | | | | | | | | | |
| Conformer relPE | 16,94 | 10,20 | 4,90 | 1,84 | 0,22 | 0,14 | 9,84 | 14,44 | 95,60 | 23,4 |
| B1 Mamba | 14,32 | 9,81 | 2,70 | 1,82 | 0,25 | 0,18 | 9,37 | 11,74 | 95,60 | 9,0 |
| ConExtBiMamba | 17,05 | 11,24 | 3,94 | 1,87 | 0,28 | 0,15 | 10,80 | 14,50 | 97,48 | 15,9 |
| Conformer không PE (ablation) | 22,35 | 13,33 | 7,25 | 1,77 | 0,41 | 0,17 | 12,75 | 19,89 | 99,69 | 34,3 |
| *Train từ đầu 40 ms (ep37, 3 seed, greedy)* | | | | | | | | | | |
| Conformer relPE | 12,64 ± 0,46 | 7,84 ± 0,26 | 2,71 ± 0,13 | 2,08 ± 0,14 | 0,17 ± 0,02 | 0,15 ± 0,01 | 7,52 ± 0,23 | 10,11 ± 0,46 | 92,35 ± 2,00 | 11,3 ± 1,0 |
| B1 Mamba | 12,66 ± 0,14 | 7,86 ± 0,11 | 3,08 ± 0,14 | 1,72 ± 0,14 | 0,23 ± 0,05 | 0,11 ± 0,01 | 7,52 ± 0,07 | 10,27 ± 0,19 | 87,97 ± 1,89 | 10,4 ± 1,2 |
| ConExtBiMamba | 12,91 ± 0,18 | 8,00 ± 0,17 | 2,92 ± 0,15 | 2,00 ± 0,11 | 0,21 ± 0,01 | 0,14 ± 0,01 | 7,65 ± 0,16 | 10,38 ± 0,13 | 92,78 ± 1,67 | 11,1 ± 0,7 |
| *40 ms + beam 10/từ điển/LM 3-gram (bảng phụ)* | | | | | | | | | | |
| Conformer relPE | 10,59 ± 0,34 | 5,83 ± 0,27 | 3,40 ± 0,03 | 1,36 ± 0,06 | 0,17 ± 0,01 | 0,14 ± 0,01 | 5,52 ± 0,28 | 8,49 ± 0,36 | 76,80 ± 1,44 | 0,0 ± 0,0 |
| B1 Mamba | 11,17 ± 0,07 | 6,28 ± 0,10 | 3,70 ± 0,08 | 1,19 ± 0,03 | 0,24 ± 0,01 | 0,11 ± 0,01 | 5,93 ± 0,11 | 9,06 ± 0,15 | 77,43 ± 2,68 | 0,0 ± 0,1 |
| ConExtBiMamba | 10,95 ± 0,13 | 6,31 ± 0,06 | 3,33 ± 0,15 | 1,31 ± 0,05 | 0,24 ± 0,03 | 0,14 ± 0,02 | 5,94 ± 0,02 | 8,79 ± 0,07 | 79,10 ± 2,53 | 0,0 ± 0,0 |
| *Pre-train (mốc tham chiếu, khác điều kiện, không xếp hạng chung)* | | | | | | | | | | |
| Parakeet-CTC-0.6B-vi zero-shot | 13,78 | 7,27 | 2,05 | 4,46 | 0,15 | 0,09 | 7,03 | 8,79 | 170,85 | 14,8 |
| Parakeet-CTC-0.6B-vi fine-tune (nhãn giả) | 9,43 | 5,45 | 2,28 | 1,70 | 0,14 | 0,09 | 5,23 | 7,05 | 84,59 | 6,3 |
| PhoWhisper-small zero-shot | 14,86 | 8,16 | 1,91 | 4,78 | 0,34 | 0,13 | 7,70 | 10,10 | 164,58 | 0,8 |
| *Tham khảo* | | | | | | | | | | |
| Nhãn giả Zipformer | 1,77 | 1,16 | 0,41 | 0,20 | 0,04 | 0,00 | 1,12 | 0,65 | 36,99 | 0,0 |

**Bảng 3.8b — Thay thế từ Việt theo thành phần âm tiết** (chỉ cặp mà cả ref và hyp là âm tiết Việt hợp lệ; % trên số cặp;
ba cột "Sai …" đa nhãn; "Chỉ …" + "≥ 2" = 100%; cột cuối: % thay thế từ Việt có hyp không phải âm tiết — từ dính, chuỗi Latin)

| Hệ thống | Số cặp | Sai thanh | Sai âm đầu | Sai vần | Chỉ thanh | Chỉ âm đầu | Chỉ vần | ≥ 2 thành phần | h không phải âm tiết (% S Việt) |
|---|---|---|---|---|---|---|---|---|---|
| *Train từ đầu 10 ms (ep37, 1 seed)* | | | | | | | | | |
| Conformer relPE | 597 | 46,6 | 57,1 | 84,8 | 4,0 | 7,9 | 26,8 | 61,3 | 31,5 |
| B1 Mamba | 570 | 48,8 | 57,0 | 83,7 | 4,6 | 8,4 | 24,6 | 62,5 | 28,4 |
| ConExtBiMamba | 636 | 47,8 | 60,7 | 83,0 | 4,6 | 8,8 | 24,2 | 62,4 | 32,9 |
| Conformer không PE (ablation) | 794 | 53,4 | 59,4 | 84,0 | 5,2 | 7,2 | 22,2 | 65,5 | 32,7 |
| *Train từ đầu 40 ms (ep37, 3 seed, greedy)* | | | | | | | | | |
| Conformer relPE | 470 | 40,2 ± 2,0 | 55,0 ± 1,8 | 84,4 ± 1,6 | 3,8 ± 0,4 | 9,4 ± 0,6 | 29,7 ± 2,1 | 57,2 ± 1,4 | 24,2 ± 2,0 |
| B1 Mamba | 470 | 47,5 ± 1,7 | 59,7 ± 0,2 | 82,4 ± 0,7 | 5,1 ± 1,0 | 9,5 ± 0,4 | 23,3 ± 0,8 | 62,1 ± 1,4 | 25,6 ± 2,9 |
| ConExtBiMamba | 483 | 44,6 ± 3,0 | 59,6 ± 1,6 | 82,7 ± 1,0 | 4,4 ± 0,2 | 10,1 ± 0,4 | 24,5 ± 1,9 | 61,0 ± 2,4 | 24,8 ± 0,8 |
| *40 ms + beam 10/từ điển/LM 3-gram (bảng phụ)* | | | | | | | | | |
| Conformer relPE | 440 | 40,8 ± 2,1 | 57,7 ± 2,4 | 83,8 ± 0,4 | 4,0 ± 0,3 | 9,4 ± 0,8 | 27,5 ± 2,9 | 59,0 ± 2,6 | 4,0 ± 0,5 |
| B1 Mamba | 477 | 46,9 ± 1,1 | 61,4 ± 1,2 | 82,6 ± 0,4 | 5,3 ± 0,3 | 9,4 ± 0,5 | 21,7 ± 0,6 | 63,6 ± 0,3 | 4,3 ± 0,7 |
| ConExtBiMamba | 480 | 47,0 ± 2,3 | 60,5 ± 1,3 | 82,0 ± 1,3 | 5,1 ± 0,6 | 9,5 ± 1,5 | 23,1 ± 1,1 | 62,3 ± 0,9 | 3,8 ± 0,7 |
| *Pre-train (tham chiếu)* | | | | | | | | | |
| Parakeet-CTC-0.6B-vi zero-shot | 365 | 43,3 | 62,7 | 81,1 | 4,4 | 12,1 | 23,8 | 59,7 | 24,6 |
| Parakeet-CTC-0.6B-vi fine-tune (nhãn giả) | 291 | 40,9 | 58,4 | 81,1 | 4,8 | 10,7 | 25,4 | 59,1 | 23,2 |
| PhoWhisper-small zero-shot | 571 | 50,1 | 58,1 | 81,8 | 6,1 | 8,2 | 22,1 | 63,6 | 1,2 |
| *Tham khảo* | | | | | | | | | |
| Nhãn giả Zipformer | 22 | 59,1 | 31,8 | 59,1 | 18,2 | 22,7 | 13,6 | 45,5 | 24,1 |

Phụ lục (v1 10 ms ep28), CI 95% từng ô và top 10 cặp nhầm lẫn mỗi hệ thống: `table_3_8.md`. Cặp nhầm lẫn đầy đủ:
`confusions.csv` (cột `syllable_diff`).

## 4. Nhận xét (chỉ từ số trên; không có kiểm định so cặp mới)

1. **Thay thế là phần lớn lỗi, lỗi thuần thanh điệu rất nhỏ.** Ở mọi hệ train từ đầu S chiếm ~60–70% WER; thay thế chỉ
   khác dấu thanh (loại a) chỉ 0,17–0,28% số từ (≈ 2–3% số thay thế; 4–5% số cặp âm tiết hợp lệ). Trong các cặp âm tiết
   hợp lệ, vần sai ở 82–85% cặp và ~57–65% cặp sai ≥ 2 thành phần — phần lớn là thay bằng một từ khác hẳn. Top cặp nhầm
   lẫn của mọi mô hình là hư từ ngắn (đó/nó, mà/và, vào/và, làm/là, trong/cho, trung/chung, thật/thực). Cột "Sai thanh"
   40–50% vì thanh thường sai **kèm** vần/âm đầu, không phải sai riêng.
2. **10 ms: phần chênh B1 − relPE (−2,62 điểm, so cặp chính thức) nằm chủ yếu ở D.** D 4,90 vs 2,70 (chênh 2,2 điểm),
   S chỉ chênh 0,4, I gần bằng; từ dính 23,4 vs 9,0/1.000 từ. Conformer không PE có D cao nhất (7,25) và từ dính nhiều
   nhất (34,3). (Phân rã mô tả, 1 seed; chưa kiểm định riêng từng thành phần.) **Suy luận, chưa kiểm:** xoá và từ dính
   cùng cơ chế CTC bỏ mảnh phụ âm đầu (`glued_words_2026-10-15.md` mục 1) — mảnh bị bỏ làm âm tiết sau dính vào âm tiết
   trước (1 S + 1 D).
3. **40 ms: ba encoder có hồ sơ lỗi gần nhau** (S 7,8–8,0 · D 2,7–3,1 · I 1,7–2,1 · từ dính 10–11/1.000), CI từng thành
   phần chồng lấn rộng — nhất quán với kết luận WER "không phân biệt được ở 3 seed". Từ 10 ms sang 40 ms, từ dính giảm
   ở relPE (23,4 → 11,3) và ConExt (15,9 → 11,1), B1 gần như giữ (9,0 → 10,4); D của relPE giảm 4,90 → 2,71. Hai cấu
   hình khác nhau 4 biến (note gap 9.4 điều 5) → không quy riêng cho độ phân giải.
4. **Từ ngoại:** 319 từ (3,1% từ ref) nhưng chiếm ~21–22% số lỗi của nhóm 40 ms greedy và ~14–21% ở 10 ms; WER ngoại
   88–100% — mọi mô hình train từ đầu gần như sai hết tiếng Anh/tên riêng. Pre-train zero-shot chèn gấp ~2 lần
   (I 4,5–4,8% so với 1,7–2,1%), ~240 từ chèn tính cho nhóm ngoại (phiên tiếng Anh thành nhiều âm tiết Việt); Parakeet
   fine-tune trên nhãn giả còn ~50. 65% lỗi của chính nhãn giả Zipformer nằm ở từ ngoại.
5. **Beam + từ điển + LM** (bảng phụ) xoá từ dính (≈ 11 → 0/1.000), giảm tỉ lệ "h không phải âm tiết" 24–26% → ~4% thay
   thế từ Việt, giảm S ~1,6–2,0 điểm và I ~0,5–0,7 điểm, nhưng **tăng D 0,4–0,7 điểm ở cả ba** (relPE 2,71 → 3,40;
   B1 3,08 → 3,70; ConExt 2,92 → 3,33). Không tách được phần do từ điển hay do LM (chưa đo; suy luận: từ điển âm tiết
   loại đoạn không ghép được thành từ có trong từ điển).

## 5. Hạn chế

- Phân rã theo thành phần âm tiết dựa trên **chính tả**, không phải âm vị (gi/d/r, s/x, tr/ch giữ là khác; quy ước tách
  âm đầu mục 1). Cặp thay thế ghép 1-1 theo vị trí trong đoạn của jiwer: cặp "≥ 2 thành phần" phần lớn là từ khác hẳn,
  không nên đọc như "lỗi âm vị kép".
- Nhóm từ ref theo bảng từ gán tay (một người); từ chèn đứng riêng xếp nhóm theo dạng chữ.
- CI hàng 3 seed chỉ phản ánh dao động mẫu test (29 video); sd seed tính trên 3 giá trị — ước thô. 10 ms 1 seed.
- **Số từ dính khác note 15/10.** Trên cùng predictions 08/10 (v1 ep28), `vi_syllable.sticky_words` đếm Conformer 444 ·
  ConExt 190 · Parakeet 153 · B1 91 · PhoWhisper 8; note `glued_words` ghi 387 · 159 · 137 · 86 · 5. Script cũ
  (`D:/Model/kltn_decoding_2026-10-15/diagnosis/a1_text.py`, ngoài repo) dùng định nghĩa khác (regex riêng + tần suất
  trong nhãn train); thứ tự không đổi. Bảng này dùng `sticky_words` (định nghĩa có trong repo) cho mọi hệ thống.
- Clean-test chỉ dùng để báo cáo; không chọn gì theo bảng này.

## 6. L3 — mô hình "nhại" lỗi nhãn giả (quy tắc chốt 2026-10-17, TRƯỚC khi tính)

Bản tối giản người dùng duyệt 17/10 (giới hạn 1 ngày). Mô tả, không kiểm định so cặp. Chạy:
`python -m src.evaluation.label_echo` → `reports/results/label_echo_2026-10-17/`.

- **Token:** y hệt bảng chính — ref = `ref_tokens(corrected_text)`, nhãn giả và giả thuyết = `hyp_tokens(...)` (bỏ từ
  đệm, `canon_tone`, `alt_spellings`). Từ vấp dở #153 giữ trong ref (không áp phương án tuỳ chọn).
- **Vị trí lỗi nhãn:** căn chỉnh jiwer ref → nhãn giả; mỗi từ ref nằm trong đoạn `substitute` (kèm từ nhãn giả cùng vị
  trí) hoặc `delete` là một vị trí lỗi nhãn. Chèn của nhãn giả **không tính** (không có vị trí ref để đối chiếu — khai
  hạn chế).
- **Kết cục của hệ thống tại vị trí đó** (căn chỉnh jiwer ref → giả thuyết, độc lập): *nhại* = cùng loại và cùng từ với
  nhãn giả (thay bằng đúng từ nhãn giả dùng, hoặc cùng xoá); *đúng* = giả thuyết khớp ref; *sai khác* = còn lại.
  Ba loại cộng 100%. Tách theo nhóm từ ref Việt / ngoại.
- **Mốc so:** tỉ lệ lỗi của cùng hệ thống trên các từ ref mà nhãn giả **đúng** — để biết vị trí lỗi nhãn có khó hơn
  hẳn không.
- **Đối chứng:** Parakeet zero-shot và PhoWhisper chưa từng thấy nhãn giả VietSuperSpeech (theo card công bố; khả năng
  trùng dữ liệu pre-train đã khai) → tỉ lệ nhại của chúng là mức "trùng hợp" khi từ khó; Parakeet fine-tune và nhóm
  train từ đầu học trên nhãn giả. Nhại vượt mức đối chứng mới được đọc là "học theo lỗi nhãn" — viết là mô tả, không
  kết luận nhân quả.
- 40 ms: trung bình ± sd 3 seed; greedy là chính, beam + LM phụ (LM cũng học nhãn giả). Không chọn gì theo bảng này.

### 6.1 Kết quả (17/10, quy tắc trên không sửa sau khi có số)

162 vị trí lỗi nhãn / 10.352 từ ref (Việt 63 · ngoại 99) — khớp S + D của nhãn giả ở Bảng 3.8a (1,16% + 0,41%).
% trên số vị trí lỗi nhãn; 40 ms trung bình ± sd 3 seed. Cột cuối: tỉ lệ lỗi của hệ thống ở các từ nhãn giả **đúng**.

| Nhóm | Hệ thống | Nhại | Đúng | Sai khác | Nhại (từ Việt, n = 63) | Nhại (từ ngoại, n = 99) | Lỗi nơi nhãn đúng |
|---|---|---|---|---|---|---|---|
| 10 ms | Conformer relPE | 19,8 | 15,4 | 64,8 | 31,7 | 12,1 | 14,0 |
| | B1 Mamba | 16,7 | 9,9 | 73,5 | 33,3 | 6,1 | 11,3 |
| | ConExtBiMamba | 18,5 | 13,6 | 67,9 | 31,7 | 10,1 | 14,1 |
| | Conformer không PE | 15,4 | 10,5 | 74,1 | 22,2 | 11,1 | 19,5 |
| 40 ms greedy | Conformer relPE | 18,7 ± 3,6 | 18,5 | 62,8 | 30,7 | 11,1 | 9,4 |
| | B1 Mamba | 21,8 ± 2,0 | 16,3 | 61,9 | 38,1 | 11,4 | 9,8 |
| | ConExtBiMamba | 20,6 ± 3,4 | 16,3 | 63,2 | 38,1 | 9,4 | 9,8 |
| 40 ms beam+LM | Conformer relPE | 23,7 ± 0,9 | 22,0 | 54,3 | 38,6 | 14,1 | 8,1 |
| | B1 Mamba | 22,8 ± 2,2 | 17,1 | 60,1 | 36,5 | 14,1 | 8,8 |
| | ConExtBiMamba | 23,7 ± 1,6 | 18,5 | 57,8 | 38,1 | 14,5 | 8,5 |
| Pre-train, học nhãn giả | Parakeet fine-tune | 18,5 | 19,1 | 62,3 | 30,2 | 11,1 | 6,6 |
| Đối chứng, chưa thấy nhãn giả | Parakeet zero-shot | 13,0 | 16,0 | 71,0 | 20,6 | 8,1 | 8,1 |
| | PhoWhisper zero-shot | 8,6 | 15,4 | 75,9 | 14,3 | 5,1 | 8,9 |

Nhận xét (mô tả; n nhỏ, không kiểm định):
1. **Vị trí lỗi nhãn là chỗ khó cho mọi hệ thống:** chỉ 10–22% được sửa đúng, so với lỗi 7–20% ở vị trí nhãn đúng — 61%
   vị trí là từ ngoại.
2. **Hệ học trên nhãn giả nhại nhiều hơn đối chứng:** 15–24% so với 9–13% (zero-shot); ở từ Việt 22–39% so với 14–21%.
   Parakeet fine-tune (18,5%) cao hơn chính nó lúc zero-shot (13,0%). Phù hợp với việc học theo lỗi nhãn, nhưng chỉ là
   mô tả: 63 vị trí từ Việt, đối chứng khác mô hình/dữ liệu pre-train.
3. **Phần lớn "nhại" là cùng xoá** (B1 40 ms seed 42: 24/34; Parakeet FT: 21/30 — `echo_examples.csv`), thường là hư
   từ/từ đệm nói lướt (*mình, cái, a, nên*) mà người hiệu đính nghe ra còn Zipformer bỏ — cùng xoá có thể do âm khó
   nghe chứ không hẳn do học nhãn. Thay thế nhại điển hình: *trung → chung*, *mad → med*, *gone → gon*.
4. **Không có khác biệt đáng nói giữa encoder** ở cùng cấu hình (40 ms greedy 18,7–21,8%, sd seed 2–3,6 điểm). Beam +
   LM tăng nhại ~2–5 điểm (LM cũng học nhãn giả).
5. Hạn chế: chèn của nhãn giả không tính; một người hiệu đính (neo nhãn — mục Hạn chế chung) làm số vị trí lỗi nhãn
   có thể thấp hơn thật.
