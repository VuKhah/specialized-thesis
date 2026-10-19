# Kết quả clean-test bản hiệu đính (2026-10-15) — quy tắc chấm, bảng, nhận xét

Chạy lại: `python -m src.evaluation.score_clean_test` → `reports/results/clean_test_scored_2026-10-15/`
(`results.json`, `per_sentence.tsv`). Output 40 ms trên clean-test: `D:/Model/kltn_decoding_2026-10-15/decoding/`
(`cleantest_decode.py` → `cleantest_hyps.tsv`; greedy khớp 203/203 câu với output dump trên Kaggle).

## 1. Quy tắc chấm (chốt TRƯỚC khi tính điểm)

Người dùng duyệt: cho phép cách viết thay thế, WER chia 2 nhóm; từ đệm/lặp từ người dùng giao AI chọn theo chuẩn.

- **Cách viết thay thế:** `configs/scoring/alt_spellings.tsv` (13 dòng: startup, makeup, livestream, setup, checkin,
  playlist, vali, ok/ô kê, show/sô, upenn/u penn, pg/p g, email, website). Tiêu chí: cùng một từ, cùng cách đọc, cả
  hai dạng có trong nhãn train hoặc là cách Việt hoá thông dụng. Phiên âm sai (*con sumer*) không đưa vào.
- **Từ đệm phi từ vựng** (ờ, ơ, ừm, ưm, ầm, hừm, hmm, uh, um + chỗ gán `từ đệm` trong bảng từ): bỏ ở cả ref và hyp.
  Căn cứ: Hub-5NE 1997 eval plan (LDC) — từ đệm bị xoá hoặc khớp đều không tính lỗi; bộ chuẩn hoá Whisper bỏ hẳn.
  Từ có nghĩa (ừ, à, á, ạ, ồ) chấm như từ thường. Hạn chế: *ờ* miền Nam có lúc nghĩa là "ừ" — không tách được tự động.
- **Lặp từ:** chấm như từ thường (người nói thật sự nói hai lần; không chuẩn nào bỏ).
- **Từ vấp dở** (#153 *in* trước *independence*, người dùng xác nhận): tuỳ chọn — bị xoá không tính lỗi (Hub-5NE 1997).
- **Nhóm từ:** ref theo `data/processed/clean_test_word_tags.tsv` (tiếng anh / tên riêng / viết tắt → "ngoại").
  Từ chèn kề một từ ref bị thay thế → tính cho nhóm của từ đó (*marketing → ma két tinh*); chèn đứng riêng → theo dạng
  chữ (âm tiết Việt hợp lệ hoặc có dấu → Việt). Lỗi hai nhóm cộng lại = lỗi chung.

## 2. Bảng (203 câu, 29 video; từ ref: Việt 10.033 · ngoại 319)

CI 95%: bootstrap theo khối video, 10.000 lần. "Trước mốc": 107 câu hiệu đính trước 14/10 23:17 (chưa thấy output
mô hình).

| Nhóm | Hệ thống | WER | CI 95% | Việt | Ngoại | Trước mốc | Không bỏ từ đệm |
|---|---|---|---|---|---|---|---|
| Train từ đầu 40 ms, greedy (ep37) | Conformer relPE | 0,1211 | [0,1030; 0,1409] | 0,0960 | 0,912 | 0,1120 | 0,1214 |
| | B1 Mamba | 0,1264 | [0,1072; 0,1480] | 0,1031 | 0,862 | 0,1159 | 0,1269 |
| | ConExtBiMamba | 0,1309 | [0,1110; 0,1530] | 0,1052 | 0,940 | 0,1198 | 0,1312 |
| 40 ms + beam 10 / từ điển / LM 3-gram | Conformer relPE | 0,1024 | [0,0869; 0,1197] | 0,0813 | 0,765 | 0,0926 | 0,1029 |
| | ConExtBiMamba | 0,1081 | [0,0905; 0,1277] | 0,0871 | 0,768 | 0,0961 | 0,1085 |
| | B1 Mamba | 0,1115 | [0,0942; 0,1304] | 0,0903 | 0,777 | 0,1008 | 0,1120 |
| Train từ đầu v1 10 ms (ep28) | B1 Mamba | 0,1704 | [0,1479; 0,1954] | 0,1424 | 1,054 | 0,1591 | 0,1710 |
| | ConExtBiMamba | 0,2019 | [0,1774; 0,2275] | 0,1738 | 1,085 | 0,1898 | 0,2024 |
| | Conformer | 0,2648 | [0,2370; 0,2937] | 0,2393 | 1,069 | 0,2499 | 0,2653 |
| Pre-train zero-shot | Parakeet-CTC-0.6B-vi | 0,1378 | [0,1150; 0,1624] | 0,0879 | 1,709 | 0,1175 | 0,1389 |
| | PhoWhisper-small | 0,1490 | [0,1235; 0,1766] | 0,1014 | 1,646 | 0,1375 | 0,1532 |
| Tham khảo | Nhãn giả Zipformer | 0,0177 | [0,0124; 0,0234] | 0,0065 | 0,370 | 0,0131 | 0,0177 |

So cặp (A − B, CI 95% khối video):
- 40 ms greedy: B1 − ConExt −0,0044 [−0,0097; +0,0007] · B1 − relPE +0,0054 [+0,0009; +0,0101] · ConExt − relPE
  +0,0098 [+0,0048; +0,0152].
- 40 ms + LM: B1 − ConExt +0,0034 [−0,0012; +0,0080] · B1 − relPE +0,0091 [+0,0048; +0,0133] · ConExt − relPE
  +0,0057 [+0,0009; +0,0106].
- v1 10 ms: B1 − ConExt −0,0315 [−0,0379; −0,0250] · B1 − Conformer −0,0944 [−0,1061; −0,0830] · ConExt − Conformer
  −0,0629 [−0,0735; −0,0524].
- Parakeet zero-shot − PhoWhisper zero-shot −0,0111 [−0,0211; −0,0014].

Chưa có: Parakeet fine-tune (val_unseen 8,0% nhãn giả) trên clean-test — `best.pt` trên Kaggle, cần GPU.

## 3. Nhận xét

1. **Nhãn giả khá tốt** (lệch bản hiệu đính 1,8% WER; từ Việt 0,65%) nhưng sai 37% từ ngoại → công hiệu đính chủ yếu
   ở tiếng Anh/tên riêng. WER val/val_unseen (nhãn giả) vẫn dùng được để xếp hạng; con số tuyệt đối lấy clean-test.
2. **40 ms greedy:** relPE tốt nhất, chênh có ý nghĩa với cả B1 (0,54 điểm) và ConExt (0,98 điểm); B1 vs ConExt chưa
   tách được. Với LM: relPE < ConExt < B1, ConExt vs B1 vẫn chưa tách được. ⚠️ relPE tốn GPU-giờ hơn hai mô hình Mamba;
   quy tắc so theo trục GPU-giờ chưa chốt. Trên nửa dev val_unseen (nhãn giả) thứ tự greedy là relPE < ConExt < B1 →
   B1 vs ConExt **đổi chỗ giữa hai tập** — thêm bằng chứng là hai mô hình này ngang nhau.
3. **v1 10 ms:** B1 tốt hơn rõ (ConExt +3,2 điểm, Conformer +9,4 điểm). Thứ hạng **đảo** khi sang 40 ms
   (Conformer từ cuối lên đầu) → kết luận "Mamba hơn Conformer" phụ thuộc cấu hình (độ phân giải thời gian, PE).
   Phải báo cả hai, không chọn cái có lợi.
4. **Từ ngoại:** WER 0,76–1,71 — gần như mọi mô hình sai hết tiếng Anh. Pre-train sai nhiều nhất (phiên tiếng Anh thành
   nhiều âm tiết Việt → nhiều lỗi chèn). Trên **từ Việt** Parakeet zero-shot tốt nhất (0,088 < 0,096 của relPE greedy);
   khoảng cách WER chung của Parakeet chủ yếu do từ ngoại.
5. **Beam + từ điển + LM** hạ 1,8–2,3 điểm chung, 13–15 điểm trên từ ngoại (từ điển lấy từ nhãn train có tiếng Anh).
   Đây là cải thiện của **hệ thống giải mã**, không phải của encoder (note `glued_words_2026-10-15.md` mục 4).
6. **Quy tắc từ đệm** đổi WER ≤ 0,0005 (PhoWhisper 0,004 vì hay xuất *ờ*) → không ảnh hưởng kết luận.
7. **Không thấy thiên lệch do thấy output mô hình:** trên 107 câu trước mốc, WER mọi hệ thống thấp hơn ~1 điểm và
   thứ hạng giữ nguyên — kể cả mô hình 40 ms (chưa từng hiện cho người dùng) → do độ khó câu, không do thiên lệch.
8. **Hạn chế còn lại:** 1 seed mỗi cấu hình (khác biệt 0,4–1 điểm giữa ba encoder 40 ms có thể nằm trong dao động
   seed — chưa đo); 29 video → CI rộng ±2 điểm; một người hiệu đính, chưa đo độ đồng thuận.

## Bổ sung 2026-10-16 — 10 ms ep37 (đủ 4 encoder) + Parakeet fine-tune

Kernel `scripts/kaggle/dump_greedy_10ms/` (A, 5,5 phút GPU, commit `edd7dd4` = commit train), batch 1, fp16 (Parakeet
fp32); output `reports/results/dump_greedy_10ms_2026-10-16/`; chấm bằng cùng quy tắc mục 1 →
`reports/results/clean_test_scored_2026-10-16/`. Kiểm nạp đúng: 100 câu val_unseen trùng eval lúc train B1 97 · ConExt
99 · relPE 98 · Conformer không PE **75** — bản không PE (conv v1 chưa sửa padding) phụ thuộc padding của batch lúc
eval; WER 100 câu gần như không đổi (0,1816 batched → 0,1790 batch 1), nên không phải lỗi nạp.

| Hệ thống (10 ms, ep37) | WER | CI 95% | Việt | ngoại |
|---|---|---|---|---|
| Conformer relPE | 0,1694 | [0,1473; 0,1946] | 0,1444 | 0,9560 |
| B1 Mamba | **0,1432** | [0,1230; 0,1665] | 0,1174 | 0,9560 |
| ConExtBiMamba | 0,1705 | [0,1460; 0,1977] | 0,1450 | 0,9748 |
| Conformer không PE (ablation) | 0,2235 | [0,1994; 0,2486] | 0,1989 | 0,9969 |
| Parakeet fine-tune `best.pt` (step 9379, chọn trên val) | 0,0943 | [0,0773; 0,1127] | 0,0705 | 0,8459 |

So cặp (bootstrap theo video), đọc theo note gap 9.6 điều 6 (10 ms 1 seed: |hiệu| > 2 điểm **và** CI không chứa 0):
- B1 − relPE −0,0262 [−0,0325; −0,0199] → **B1 tốt hơn** (1 seed).
- ConExt − relPE +0,0012 [−0,0063; +0,0087] → **không phân biệt được**.
- B1 − ConExt −0,0273 [−0,0336; −0,0211] → B1 tốt hơn.
- relPE − không PE −0,0542 [−0,0639; −0,0445] → PE giúp Conformer ở 10 ms.
- Parakeet FT − Parakeet zero-shot −0,0436 [−0,0525; −0,0349] (nhóm pre-train, tham chiếu).

Nhận xét: thứ hạng **đảo** giữa hai cấu hình — 10 ms: B1 > relPE ≈ ConExt; 40 ms (seed 42): relPE > B1 ≈ ConExt. Theo
9.4 điều 5: "phù hợp với giả thuyết độ phân giải", không "chứng minh" (1 seed ở 10 ms; cấu hình 40 ms gộp 4 khác biệt).
Cả ba encoder đều tốt hơn ở 40 ms. Thứ hạng trên clean-test khớp thứ hạng val_unseen nhãn giả (B1 0,1252 · relPE 0,1501
· ConExt 0,1506 · không PE 0,2068).

### Quy tắc bổ sung 2026-10-16 — vị trí dấu thanh (quyết định #11)
`canon_tone` trong `src/evaluation/score_clean_test.py`: vần mở oa/oe/uy có đúng một dấu thanh → đặt dấu ở nguyên âm
sau (hóa → hoá, khỏe → khoẻ, thủy → thuỷ), cả ref và hyp; bỏ qua sau `q`. Thêm **sau** lần chấm 15/10 và sau khi đã đo
ảnh hưởng (chỉ 3 cặp xuất hiện: hoá/hóa, thoả/thỏa, khoá/khóa): PhoWhisper −4 lỗi, ConExt v1 −1, mọi hệ thống khác 0.
Bảng `clean_test_scored_2026-10-16/` hiện **chưa** áp quy tắc này; áp ở lần chấm chung khi đủ 9 run 40 ms.

## Lần chấm chung 2026-10-16 (~18:10) — 40 ms đủ 3 seed (L3), áp `canon_tone`

Đầu vào: 6 run seed 43/44 (A ConExt · B relPE · C B1; commit `0090bd75`, config `*_sub4.yaml` y nguyên, chỉ `--seed`).
Kiểm toàn vẹn (`collect_seeds.py`): cả 6 returncode 0, 38/38 epoch, `finished`, 0 step không hữu hạn, log không có
`!!!`/`LỖI`/Traceback. Kernel `dump-seeds-v2a-s4` (A, 2,6′ GPU, không lỗi) → log-prob clean-test; greedy 100 câu
val_unseen trùng eval ep37 lúc train 593/600 câu, mỗi câu lệch đúng 1 từ (≤ 7/5.342 từ — sai số fp16/padding batch);
greedy giải lại từ npz bằng `lm_decode` trùng kernel 203/203 × 6. Beam+LM: `lm_decode decode`, lmw 0,8 / ws 1,0 như cũ
(`reports/results/decode_v2a_s4_seeds/cleantest_hyps.tsv`). Chấm: `python -m src.evaluation.score_clean_test` →
`reports/results/clean_test_scored_2026-10-16_chung/`. `canon_tone` chỉ đổi 2 hệ thống so với bản sáng 16/10
(ConExt v1 0,2019 → 0,2018; PhoWhisper 0,1490 → 0,1486); mọi số khác giữ nguyên.

**Bảng 3.2 — 40 ms, ep37, 3 seed (42 / 43 / 44).** WER clean-test greedy; val, val_unseen là nhãn giả (eval lúc train).

| Encoder 40 ms | clean-test (tb ± sd) | seed 42 · 43 · 44 | beam+LM (tb ± sd) | val | val_unseen | GPU-h |
|---|---|---|---|---|---|---|
| Conformer relPE | 0,1264 ± 0,0046 | 0,1211 · 0,1287 · 0,1295 | 0,1059 ± 0,0034 | 0,1110 ± 0,0040 | 0,1101 ± 0,0034 | 5,16 ± 0,20 |
| B1 Mamba | 0,1266 ± 0,0014 | 0,1264 · 0,1253 · 0,1281 | 0,1117 ± 0,0007 | 0,1114 ± 0,0014 | 0,1123 ± 0,0012 | 5,93 ± 0,11 |
| ConExtBiMamba | 0,1291 ± 0,0018 | 0,1309 · 0,1272 · 0,1293 | 0,1095 ± 0,0013 | 0,1117 ± 0,0010 | 0,1114 ± 0,0007 | 4,89 ± 0,04 |

**So cặp — bootstrap hai tầng video × seed (note gap 9.6 điều 4).** Điều 5 (chỉ mô tả): số cặp seed cùng dấu / 9.

| Cặp (A − B) | greedy: hiệu [CI 95%] | cùng dấu | beam+LM: hiệu [CI 95%] | cùng dấu | Kết luận |
|---|---|---|---|---|---|
| B1 − ConExt | −0,0025 [−0,0086; +0,0039] | 8/9 | +0,0022 [−0,0026; +0,0074] | 9/9 | không phân biệt được ở 3 seed |
| B1 − relPE | +0,0002 [−0,0082; +0,0098] | 6/9 | +0,0058 [−0,0015; +0,0131] | 9/9 | không phân biệt được ở 3 seed |
| ConExt − relPE | +0,0027 [−0,0060; +0,0126] | 6/9 | +0,0036 [−0,0035; +0,0112] | 8/9 | không phân biệt được ở 3 seed |

Nhận xét (từ số trên, không thêm phép thử):
- **Kết quả seed 42 không đứng vững.** Seed 42 cho relPE "tốt hơn" B1 và ConExt với CI cặp không chứa 0 (bảng 15/10);
  relPE seed 42 hoá ra là run tốt nhất trong 9 run (0,1211, các seed khác 0,1287/0,1295). Đây đúng là đe doạ #2 của
  review 15/10: một seed + CI chỉ theo mẫu test đánh giá thấp dao động. Ở 3 seed, cả ba encoder 40 ms **không phân
  biệt được**; sd seed của relPE (0,46 điểm) lớn gấp ~3 lần hai encoder Mamba (0,14–0,18 điểm). Val/val_unseen (nhãn
  giả, tập lớn hơn) cho cùng bức tranh: ba trung bình cách nhau ≤ 0,2 điểm.
- **Beam+LM:** B1 − relPE dương ở cả 9 cặp seed nhưng CI hai tầng chứa 0 → theo luật chốt trước vẫn là "không phân biệt
  được"; số cặp cùng dấu chỉ được nêu như mô tả (điều 5), không làm căn cứ kết luận.
- **Nhận xét "thứ hạng đảo giữa hai cấu hình" ở bổ sung 16/10 phía trên dựa trên seed 42 — thay bằng:** 10 ms (1 seed):
  B1 tốt hơn relPE và ConExt (> 2 điểm, CI không chứa 0), relPE và ConExt không phân biệt được; 40 ms (3 seed): ba encoder
  không phân biệt được. Lợi thế của B1 thấy ở 10 ms không còn thấy ở 40 ms — "phù hợp với giả thuyết độ phân giải",
  không "chứng minh" (cấu hình 40 ms gộp 4 khác biệt; 10 ms chỉ 1 seed, độ lệch seed ở 10 ms chưa đo).
- Hạn chế phải khai: 3 seed nên phần dao động seed ước thô; 29 video.

## Bổ sung 2026-10-17 — kiểm định dấu theo video (mô tả, không đổi tiêu chí kết luận)

`python -m src.evaluation.video_sign_test` → `reports/results/video_sign_2026-10-17/` (đọc `per_sentence.tsv` của lần
chấm chung; tổng lỗi/từ khớp `results.json`). Số video A có WER thấp hơn B / B thấp hơn A / hoà, trên 29 video; p hai
phía phép thử dấu chính xác. 40 ms: WER mỗi video là trung bình 3 seed (bỏ qua dao động seed — chỉ mô tả).

| Cấu hình | Cặp (A − B) | A thấp hơn | B thấp hơn | Hoà | p | Kết luận chính thức (CI) |
|---|---|---|---|---|---|---|
| 10 ms | B1 − relPE | 26 | 3 | 0 | < 0,001 | B1 tốt hơn |
| 10 ms | ConExt − relPE | 13 | 14 | 2 | 1,00 | không phân biệt được |
| 10 ms | B1 − ConExt | 28 | 0 | 1 | < 0,001 | B1 tốt hơn |
| 10 ms | relPE / B1 / ConExt − không PE | 29 | 0 | 0 | < 0,001 | A tốt hơn |
| 40 ms greedy | B1 − ConExt | 20 | 9 | 0 | 0,061 | không phân biệt được |
| 40 ms greedy | B1 − relPE | 15 | 13 | 1 | 0,85 | không phân biệt được |
| 40 ms greedy | ConExt − relPE | 12 | 16 | 1 | 0,57 | không phân biệt được |
| 40 ms beam+LM | B1 − ConExt | 13 | 16 | 0 | 0,71 | không phân biệt được |
| 40 ms beam+LM | B1 − relPE | 8 | 19 | 2 | 0,052 | không phân biệt được |
| 40 ms beam+LM | ConExt − relPE | 9 | 19 | 1 | 0,087 | không phân biệt được |

Nhận xét: phép thử dấu đồng thuận với bootstrap ở mọi cặp. Ở 10 ms, lợi thế của B1 không do vài video kéo — 26/29 video
so với relPE. Ở 40 ms không cặp nào có p < 0,05; beam + LM nghiêng về relPE (19/29 video) nhưng chưa qua ngưỡng nào —
vẫn viết "không phân biệt được".

**Dao động seed (mô tả):** sd 3 seed của relPE 40 ms lớn gấp ~3 lần hai encoder Mamba trên cả clean-test (0,46 so với
0,14–0,18 điểm), val (0,40 so với 0,10–0,14) và val_unseen (0,34 so với 0,07–0,12). Theo từng video, khoảng chênh giữa 3
seed có trung vị 1,9 điểm (relPE) · 1,4 (B1) · 1,7 (ConExt). Chỉ 3 seed → ước thô; không suy ra "Mamba ổn định hơn".
