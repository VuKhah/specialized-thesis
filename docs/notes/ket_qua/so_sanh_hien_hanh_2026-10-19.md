# So với mô hình hiện hành — phương án A + B (chốt 2026-10-19)

Bối cảnh: GVHD 17/10 (`Thao_Luan_17_10_26.txt`) hỏi "so với mấy model hiện thời thế nào?". Review 15/10 mục 3 từng
loại "eval ngoài miền"; người dùng 19/10 mở lại vì câu hỏi này và chọn **A + B**:
- **A** (0 GPU): so nhóm train từ đầu với pre-train trên cùng clean-test, có CI; kèm số công bố của mô hình hiện hành
  trên bộ chuẩn (ghi rõ khác tập test).
- **B** (chỉ eval): chấm mọi checkpoint bảng chính + pre-train trên **test GigaSpeech 2 tiếng Việt** (người gán nhãn
  chuyên nghiệp, YouTube, tách kênh với train) → trục **ngoài kênh**, bổ sung cho clean-test (**trong kênh**: VietSuperSpeech
  100% từ kênh Vietcetera, `data_audit_2026-10-19.md`).

## A. Kết quả (clean-test 203 câu, lần chấm chung 16/10)

`python -m src.evaluation.compare_pretrained` → `reports/results/compare_pretrained_2026-10-19/pairs.csv`. Bootstrap hai
tầng video × seed (cùng hàm với tiêu chí 9.6; script tự kiểm tái lập CI cặp B1–relPE đã công bố). Bên mình: tb 3 seed ep37.

| Train từ đầu 40 ms (tb ± sd 3 seed) | vs Parakeet zero-shot 0,1378 | vs PhoWhisper-small 0,1486 | vs Parakeet fine-tune 0,0943 |
|---|---|---|---|
| relPE greedy 0,1264 ± 0,0046 | −1,15 [−2,72; +0,46] **không phân biệt được** | −2,22 [−4,02; −0,49] thấp hơn | +3,21 [+2,06; +4,35] cao hơn |
| B1 greedy 0,1266 ± 0,0014 | −1,12 [−2,49; +0,34] **không phân biệt được** | −2,20 [−3,87; −0,55] thấp hơn | +3,23 [+2,33; +4,23] cao hơn |
| ConExt greedy 0,1291 ± 0,0018 | −0,87 [−2,18; +0,50] **không phân biệt được** | −1,94 [−3,57; −0,36] thấp hơn | +3,48 [+2,65; +4,39] cao hơn |
| relPE beam+LM 0,1059 ± 0,0034 | −3,19 [−4,63; −1,85] thấp hơn | −4,26 [−5,96; −2,72] thấp hơn | +1,17 [+0,27; +1,99] cao hơn |
| B1 beam+LM 0,1117 ± 0,0007 | −2,61 [−3,91; −1,30] thấp hơn | −3,69 [−5,37; −2,14] thấp hơn | +1,74 [+1,01; +2,51] cao hơn |
| ConExt beam+LM 0,1095 ± 0,0013 | −2,83 [−4,14; −1,56] thấp hơn | −3,90 [−5,55; −2,35] thấp hơn | +1,53 [+0,80; +2,25] cao hơn |

(Δ và CI 95% tính bằng điểm WER.)

Cách viết (luật CLAUDE.md: kết luận chỉ khi CI không chứa 0):
- Mô hình ~12–14M train từ đầu trên ~165 h, greedy, **không phân biệt được** với Parakeet-CTC-0.6B zero-shot và **thấp
  hơn** PhoWhisper-small zero-shot; thêm beam + LM thì thấp hơn cả hai. Parakeet fine-tune trên cùng nhãn giả vẫn tốt
  nhất (thấp hơn mọi hệ train từ đầu).
- **Điều kiện khác, phải nêu cạnh bảng:** clean-test là **trong kênh** (cùng kênh Vietcetera với dữ liệu train của
  nhóm train từ đầu, khác video), pre-train zero-shot chưa từng thấy kênh này → lợi thế miền thuộc về nhóm mình; nhóm
  mình học nhãn giả của Zipformer (mô hình 6.000 h); số tham số chênh ~40–50×; LM train trên văn bản cùng kênh. Vì vậy
  câu "ngang Parakeet" chỉ đúng **trong kênh** — phần B kiểm điều này.

### Số công bố của mô hình hiện hành (khác tập test — chỉ để định vị, không so trực tiếp)

| Bộ test | Mô hình | WER % | Nguồn |
|---|---|---|---|
| GigaSpeech 2 vi test | Whisper large-v3 | 17,94 | Yang và cs., arXiv:2406.11546, bảng kết quả vi |
| | Conformer (ESPnet, train trên GigaSpeech 2) | 15,60 | như trên, Bảng 5 |
| | Zipformer-L (train trên GigaSpeech 2) | 12,83 | như trên |
| | Azure Speech / Google Chirp v2 (thương mại) | 11,86 / 13,28 | như trên |
| | Zipformer-30M-RNNT-6000h (mô hình sinh nhãn VietSuperSpeech) | 7,56 | model card `hynt/Zipformer-30M-RNNT-6000h` |
| VIVOS | PhoWhisper-small / large | 6,33 / 4,67 | Le và cs., arXiv:2406.02555 (README VinAIResearch/PhoWhisper) |
| VLSP 2020 T1 | PhoWhisper-small / large / Zipformer-30M | 15,93 / 13,75 / 12,29 | như trên; model card Zipformer |

PhoWhisper đã train trên phần train của VIVOS và VLSP 2020 → số trên hai bộ này là trong miền của nó. Zipformer-30M có `Sub-GigaSpeech2-Vi` trong dữ liệu train (model card).
Bài VietSuperSpeech (arXiv:2603.01894) **không công bố WER baseline** và không có test do người kiểm.

## B. Kế hoạch eval GigaSpeech 2 vi test (chưa chạy)

Dữ liệu: `speechcolab/gigaspeech2`, `data/vi/test.tar.gz` + `test.tsv` (~11 h theo trang HF; bài báo ghi 10 h), **gated**,
điều khoản chỉ nghiên cứu phi thương mại, SpeechColab không giữ bản quyền audio → **không commit audio/transcript**.

1. **Người dùng:** chấp nhận điều khoản trên trang HF bằng tài khoản của mình, tạo token đọc, tự chạy `hf auth login`
   (máy hiện chưa có token — 19/10 thử bị 401 GatedRepoError).
2. Tải `test.tsv` trước, đọc quy ước chép (hoa/thường, chữ số, dấu câu, từ đệm, tiếng Anh), số câu, phân bố độ dài.
   **Chốt quy tắc chấm vào note này TRƯỚC khi giải mã** (người dùng duyệt). Mặc định đề xuất: `normalize_text` chung,
   không `alt_spellings`/bỏ từ đệm trừ khi quy ước chép buộc phải có; CI bootstrap theo khối = video/kênh nếu `test.tsv`
   có thông tin đó, không thì theo câu (ghi rõ).
3. Tải `test.tar.gz` vào `data/raw/gigaspeech2_vi/` (gitignore), kiểm md5; tạo Kaggle Dataset **private** cho audio.
4. Kernel eval (tài khoản A): mọi checkpoint bảng chính — 9 run 40 ms ep37 (3 encoder × 3 seed), 4 checkpoint 10 ms ep37,
   Parakeet zero-shot + fine-tune, PhoWhisper-small — greedy, dump log-prob 40 ms để beam+LM chạy CPU local như L4. Đo
   GPU-giờ thật; **ước 1–3 GPU-h (suy luận, chưa đo)**. Báo người dùng ngay khi đẩy. Xong trước mốc đóng băng 25/10.
5. Chấm một lần duy nhất theo quy tắc mục 2; so cặp như clean-test; viết kết quả cạnh bảng A, tách "trong kênh" /
   "ngoài kênh". Kết quả bất lợi cho nhóm train từ đầu vẫn báo nguyên.

Rủi ro: mô hình 10 ms trên câu dài (nếu `test.tsv` có câu > 30 s) cần kiểm VRAM; Parakeet fine-tune 2,27 GB đã có trong
dataset kernel L2; nếu điều khoản không cho tải lên Kaggle private → chạy phần Conformer/pre-train trên CPU local, phần
Mamba phải có GPU (hỏi người dùng).

## B.2 Dữ liệu và quy tắc chấm đề xuất (19/10, CHỜ NGƯỜI DÙNG DUYỆT — chưa giải mã gì)

Đã tải (`data/raw/gigaspeech2/`, gitignore): `test.tsv` (md5 khớp file `md5` của repo), `test.tar.gz` 1,47 GB.

Đặc điểm `test.tsv` (đọc trước khi chạy bất kỳ mô hình nào):
- 4.422 câu, 155.509 từ (tb 35,2 từ/câu); id dạng `<nhóm>-<số>`, **30 nhóm** (130–148 câu/nhóm).
- Quy ước chép **trùng VietSuperSpeech**: chữ HOA, NFC 100%, 0 chữ số (số đọc thành chữ: "HAI KHÔNG HAI MỐT"), 0 dấu câu,
  có ghi từ đệm (À 860, Ờ 682, Á 668, HA 183, Ơ 164…). `normalize_text` không đổi gì ngoài hoa → thường.
- 2,02% từ không phải âm tiết Việt (VIDEO, OK, LEAGUE, BITCOIN, ARSENAL, METAVERSE…); chủ đề rộng: bóng đá, tiền mã hoá,
  lớp học, y tế, tin tức — khác hẳn phỏng vấn Vietcetera.

Quy tắc đề xuất (giữ nguyên quy tắc clean-test để hai trục so được):
1. **Chuẩn hoá:** `normalize_text` + `canon_tone` + `configs/scoring/alt_spellings.tsv` + bỏ từ đệm `FILLERS` ở cả ref và
   hyp — đúng `score_clean_test.py`, không thêm/bớt mục nào. Không có quy tắc từ vấp dở.
2. **Nhóm từ Việt/ngoại:** không có bảng gán tay → tự động: từ ref là âm tiết Việt hợp lệ (`is_valid_syllable`) → Việt,
   còn lại → ngoại. Khác cách gán của clean-test → chỉ là cột phụ, ghi rõ.
3. **CI:** bootstrap khối theo 30 nhóm id (giả định mỗi nhóm = một bản ghi/video — kiểm bằng tên file trong tar trước khi
   chấm); 40 ms 3 seed: bootstrap hai tầng như clean-test.
4. ~~Không chọn gì trên tập này~~ **Sửa 19/10 (người dùng: nguyên tắc để giữ tiến độ, không để chặn tối ưu):** được tối ưu giải mã (lm_weight, word_score, beam, LM rộng từ văn bản GigaSpeech 2 train) — **chỉ chỉnh trên dev GigaSpeech 2** (10 h, cùng phân bố, người gán nhãn); test chỉ để báo. Checkpoint vẫn ep37 mọi run. Báo cả cấu hình cố định (như clean-test) lẫn cấu hình tối ưu trên dev.
5. ~~Chấm một lần duy nhất~~ **Sửa 19/10:** chấm test cho mỗi cấu hình cuối đã chốt trên dev; không chọn cấu hình theo số test. Kết quả bất lợi vẫn báo nguyên.

## B.3 Kết quả tạm (19/10 ~17:10 — thiếu beam+LM, PhoWhisper, Parakeet zero-shot)

Người dùng xem kết quả lúc này → **quy tắc B.2 khoá** (không đổi sau khi đã thấy số). `python -m src.evaluation.score_gs2`
→ `reports/results/gs2_scored_2026-10-19/`. Kiểm nạp đúng: greedy clean-test của 14 hệ thống trùng 203/203 với các lần
chạy 15–16/10; greedy dựng từ top-k trùng greedy GPU 4.422/4.422.

| Hệ thống | Clean-test (trong kênh) | GS2 (ngoài kênh) | Δ |
|---|---|---|---|
| Parakeet-0.6B fine-tune | 0,0943 | **0,1215** | +2,7 điểm |
| 40 ms greedy, tb 3 seed: relPE / B1 / ConExt | 0,1264 / 0,1266 / 0,1291 | 0,2777 / 0,2827 / 0,2790 | ≈ +15 điểm |
| 10 ms: B1 / ConExt / relPE / không PE | 0,1432 / 0,1705 / 0,1694 / 0,2235 | 0,2926 / 0,3349 / 0,3417 / 0,4164 | +15 → +19 điểm |

- Nhóm train từ đầu (một kênh, 165 h) sụt ~15 điểm ngoài kênh; Parakeet fine-tune trên cùng nhãn giả chỉ sụt 2,7 → khoảng
  cách với pre-train + fine-tune tăng từ ~3 điểm lên ~16 điểm. Số tuyệt đối 27–29% (40 ms), không phải 40–50% như lo ngại.
- 40 ms, 3 seed (hai tầng): relPE vs B1 và relPE vs ConExt **không phân biệt được**; B1 − ConExt +0,36 điểm, CI
  [+0,02; +0,72] (sát 0). Thứ hạng giữ như clean-test (không phân biệt được), relPE dao động seed lớn nhất (0,271–0,286).
- 10 ms (1 seed, CI chỉ gồm dao động mẫu test, **không** gồm dao động seed): B1 < ConExt < relPE < không PE, cùng
  chiều clean-test; B1 − relPE −4,9 điểm. ConExt − relPE −0,69 điểm nhỏ hơn sd seed relPE ở 40 ms (0,46 điểm × 3 seed)
  → không nên diễn giải.
- Từ dính (greedy) 32–36‰ ở 40 ms, 26–74‰ ở 10 ms, so với ~10–11‰ trên clean-test → tăng ~3 lần ngoài kênh (dự đoán
  trước khi chấm). WER từ ngoại > 1 (chèn nhiều), relPE 40 ms cao nhất (1,30–1,33).
- Số công bố (Whisper large-v3 17,94; Zipformer GS2 12,83) dùng chuẩn hoá khác → chỉ định vị, không so trực tiếp.

### B.3b Beam + LM, cấu hình cố định (như clean-test: LM 3-gram Vietcetera, lmw 0,8, ws 1,0) — 19/10 ~17:50

| 40 ms, tb 3 seed | greedy | beam+LM | clean-test beam+LM |
|---|---|---|---|
| relPE | 0,2777 | **0,2246** | 0,1059 |
| ConExt | 0,2790 | 0,2325 | 0,1095 |
| B1 | 0,2827 | 0,2381 | 0,1117 |

- LM hạ ~5,3 điểm (≈ 19% tương đối) — nhiều hơn trong kênh (~2 điểm) dù perplexity LM trên GS2 gấp 3 (234 vs 74):
  phần lớn từ việc xoá từ dính (33‰ → ~0‰). Dự đoán trước khi chấm: 22–26% → khớp.
- Với LM, thứ hạng ngoài kênh tách ra: relPE < B1 (−1,34 điểm, CI [−2,09; −0,45]) và ConExt < B1 (−0,56, CI [−0,81;
  −0,29]); relPE vs ConExt không phân biệt được. Trên clean-test cả 3 cặp + LM không phân biệt được. Đây là hiệu ứng
  của **hệ thống giải mã** (LM giúp mỗi encoder một khác), không phải kết luận về encoder (kết luận encoder theo greedy).

## C. Kiểm dữ liệu train của mô hình pre-train (19/10 tối, trước khi chọn mô hình fine-tune thêm)

Câu hỏi của người dùng: mô hình dự kiến fine-tune có thể đã "thấy" audio VietSuperSpeech/Vietcetera hoặc GigaSpeech 2 không.

| Mô hình | Dữ liệu train công bố | Nguồn | Khả năng đã thấy dữ liệu đánh giá |
|---|---|---|---|
| PhoWhisper-small | 844 h: CMV-Vi 3,04 h, VIVOS 13,94 h, VLSP 2020 T1 240,91 h, dữ liệu riêng 585,90 h (26K người, 63 tỉnh, câu 11–24 âm tiết) | bài báo arXiv 2406.02555 Bảng 1 | Thấp ở tầng fine-tune (dữ liệu riêng mô tả như đọc theo kịch bản, câu ngắn); **không kiểm được** tầng Whisper gốc (680K h web, OpenAI không công bố nguồn) |
| wav2vec2-base-vi (nguyenvulebinh) | tự giám sát 13K h "Vietnamese youtube audio", không liệt kê kênh | model card HF | **Không loại trừ được**: Vietcetera và kênh GigaSpeech 2 đều trên YouTube. Chỉ audio, không nhãn |
| Parakeet-CTC-0.6B-vi (đang dùng) | ~2.000 h, 10 tập: CV 20, VietMed-L, LSVSC, FLEURS, InfoRe 2, FOSD, VAIS, VLSP 2020, VSV-1100, MS-SNSD; GigaSpeech 2 chỉ dùng để eval | model card HF | VSV-1100 (1.100 h) ghi nguồn "nhiều nguồn mạng xã hội", không liệt kê → **không loại trừ được** |

Kết luận: không mô hình nào chứng minh được là sạch hoàn toàn. Đây là hạn chế chung khi so với mô hình pre-train, cần ghi
khi báo. PhoWhisper có mô tả dữ liệu rõ nhất. Chưa chọn mô hình fine-tune thêm; chờ kết quả GigaSpeech 2 (người dùng 19/10).
