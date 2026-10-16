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
