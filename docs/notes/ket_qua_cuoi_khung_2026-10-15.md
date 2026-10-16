# Khung kết quả cuối của khóa luận — bảng, hình, kết luận viết sẵn theo nhánh (2026-10-15)

Bản kiểm kê để dựng khung Chương 3 ngay, trước khi có đủ số. Căn cứ: `review_fable_2026-10-15.md` (phương án) và
`review_fable_scorecard_2026-10-15.md`. Ba chỗ còn chờ số: **Conformer relPE 10 ms**, **ep37 của nhóm 10 ms**,
**seed 43/44 ở 40 ms** — mỗi kết luận liên quan đều có hai nhánh. Số đã có lấy từ
`reports/results/clean_test_scored_2026-10-15/` và `reports/results/rq2_final_2026-10-13/`.

## 1. Bảng (9)

### RQ1 — độ chính xác (nhóm train từ đầu; WER clean-test 203 câu đã hiệu đính; greedy; CI 95% bootstrap theo video)

| Bảng | Nội dung | Trạng thái số (15/10) |
|---|---|---|
| **3.1** Cấu hình 10 ms, ep37, 1 seed | Conformer relPE (baseline chuẩn) · B1 · ConExt · dòng ablation Conformer không PE. Cột: WER chung / từ Việt / từ ngoại, val_unseen (nhãn giả), GPU-h tới ep37, tham số encoder + tổng | relPE và ep37 **chưa có**; hiện chỉ có ep28 v1 (B1 0,1704 · ConExt 0,2019 · Conformer không PE 0,2648) |
| **3.2** Cấu hình 40 ms, ep37, 3 seed | trung bình ± độ lệch chuẩn; số từng seed ở phụ lục; cùng các cột như 3.1 | seed 42 có (relPE 0,1211 · B1 0,1264 · ConExt 0,1309); seed 43/44 chưa chạy |
| **3.3** So cặp | hiệu WER từng cặp ở mỗi cấu hình, CI 95% cặp theo video, kết luận "tốt hơn" / "không phân biệt được" theo tiêu chí chốt trước | 40 ms seed 42: B1−ConExt −0,0044 [−0,0097; +0,0007]; B1−relPE +0,0054 [+0,0009; +0,0101]; ConExt−relPE +0,0098 [+0,0048; +0,0152]. 10 ms ep28: B1−ConExt −0,0315; B1−Conformer −0,0944; ConExt−Conformer −0,0629 (CI đều không chứa 0) |
| **3.4** Bảng phụ giải mã | beam 10 + từ điển âm tiết + LM 3-gram (lmw 0,8 / ws 1,0 chốt trên nửa tune) cho nhóm 40 ms, mọi seed; ghi rõ là cải thiện của hệ giải mã, LM học nhãn giả | seed 42: relPE 0,1024 · ConExt 0,1081 · B1 0,1115 |
| **3.5** Tham chiếu pre-train | Parakeet zero-shot · Parakeet fine-tune (7 GPU-h, nhãn giả) · PhoWhisper zero-shot; WER chung và từ Việt; **không xếp hạng chung** | zero-shot có (Parakeet 0,1378, từ Việt **0,0879**; PhoWhisper 0,1490); fine-tune chưa chấm clean-test |

### RQ2 — hiệu quả

| Bảng | Nội dung | Trạng thái |
|---|---|---|
| **3.6** Chi phí huấn luyện | s/step, GPU-h tới ep37, VRAM đỉnh, step không hữu hạn cho 7 cấu hình (và Parakeet FT) | đủ. 40 ms: ConExt 4,86 h · relPE 5,28 h · B1 6,00 h; 10 ms 38 epoch: Conformer không PE 8,4 · ConExt 12,7 · B1 21,3 (relPE 10 ms điền sau) |
| **3.7** Chi phí suy luận theo độ dài | độ trễ, RTF, VRAM tại 15 / 60 / 240 / 3.600 s; điểm OOM; điểm giao từng cặp; bảng điều kiện số học (Mamba fp32, attention fp16, Parakeet fp32) | đủ (`rq2_final/results.csv`): ConExt 40 ms nhanh hơn relPE 40 ms từ ~45 s, B1 từ ~60 s; relPE OOM ~656 s (40 ms) / ~165 s (10 ms, cài bias tường minh); Conformer không PE chạy hết 3.600 s (157 s) |

### RQ3 — đặc điểm lỗi và chất lượng nhãn

| Bảng | Nội dung | Trạng thái |
|---|---|---|
| **3.8** Phân rã lỗi | S/D/I, từ dính, từ Việt vs từ ngoại, theo âm tiết (âm đầu / vần / thanh) cho từng mô hình/cấu hình | cần chạy lại `error_analysis.py` trên bản hiệu đính; từ dính đã đếm (ep28: Conformer 387 · ConExt 159 · Parakeet 137 · B1 86 · PhoWhisper 5) |
| **3.9** Chất lượng nhãn L1–L4 | L1 nhãn giả vs hiệu đính (chung / Việt / ngoại); L2 mô hình vs Zipformer; L3 tỉ lệ "nhại" lỗi nhãn; L4 thứ hạng trên nhãn giả vs nhãn thật; độ lệch gõ mù 20 câu | L1: 0,0177 chung · 0,0065 Việt · 0,370 ngoại; L2: không mô hình nào vượt; L4: thứ hạng không đổi. L3 và gõ mù chưa làm |

Phụ lục: v1 ep28 + avg ep20–29 + Conformer dài 69 epoch; số từng seed; cấu hình đầy đủ; quy tắc 9.4 nguyên văn.

## 2. Hình (5)

1. WER val theo epoch cho mọi run 10 ms và 40 ms, đánh dấu pha giảm LR (ep30–37).
2. WER theo GPU-giờ tích luỹ (thay cho trục "cùng GPU-giờ"; không train bù).
3. Độ trễ và VRAM theo độ dài audio, log-log, 7 cấu hình + Parakeet, đánh dấu OOM.
4. Đánh đổi WER – RTF trên câu 10–15 s: mỗi mô hình một điểm, thanh CI.
5. Forest plot hiệu WER từng cặp với CI, hai cấu hình cạnh nhau.

## 3. Kết luận — viết sẵn theo nhánh

**KL1 — Mamba vs Conformer phụ thuộc cấu hình độ phân giải.**
- 40 ms, seed 42: Conformer relPE dẫn 0,5–1,0 điểm, CI cặp không chứa 0. Sau 3 seed:
  - dấu giữ ở cả 3 cặp seed **và** CI của hiệu trung bình-seed không chứa 0 → "Conformer relPE tốt hơn X ± Y điểm";
  - thiếu một điều kiện → "không phân biệt được ở 3 seed".
- 10 ms: hai encoder Mamba vượt Conformer **không PE** 6–9 điểm (CI rõ). Với Conformer relPE 10 ms:
  - relPE vẫn thua Mamba → "lợi thế Mamba ở 10 ms là thật";
  - relPE bắt kịp/vượt → "lợi thế ở 10 ms chủ yếu do baseline không PE yếu" — và **đó là kết luận chính**, không
    chuyển sang so với bản không PE.

**KL2 — Hai cách đưa Mamba vào encoder.** B1 tốt hơn ConExt 3,2 điểm ở 10 ms (CI [2,5; 3,8]). Ở 40 ms không phân
biệt được, thứ hạng đổi chỗ giữa clean-test và nửa dev. Viết "phù hợp với giả thuyết độ phân giải", **không** "chứng
minh" (cấu hình 40 ms trộn 4 biến: +1,84M tham số, front-end Conv2d, trường tiếp nhận ×4, train từ đầu vs rẽ nhánh).

**KL3 — Lợi thế tính toán của Mamba có ngưỡng độ dài.** Trên câu 10–15 s: ConExt 40 ms ≈ Conformer relPE 40 ms
(~20 ms/câu), B1 chậm hơn ~40%. Mamba nhanh hơn từ ~45 s (ConExt) và ~60–90 s (B1). Ở 1 giờ audio: B1 40 ms 1,6 s,
ConExt 1,2 s; relPE OOM ~656 s **do cách cài bias vị trí tường minh** (bản không PE chạy hết nhưng 157 s). Chi phí
train ở 40 ms lệch ≤ 25%. VRAM không phải lợi thế của Mamba ở đây (khối fp32). Số đã đủ, không chờ gì.

**KL4 — Nhãn giả.** Nhãn Zipformer lệch bản hiệu đính 1,8% WER: 0,65% trên từ Việt, 37% trên từ ngoại. Thứ hạng mô
hình trên nhãn giả và nhãn hiệu đính không đổi → nhãn giả đủ để xếp hạng, sai độ lớn chủ yếu ở tiếng Anh/tên riêng.
Không mô hình nào vượt "thầy". Nhánh gõ mù 20 câu: lệch > ~1 điểm so với bản hiệu đính → ghi "L1 bị đánh giá thấp
do neo nhãn"; một người hiệu đính luôn khai.

**KL5 — Nhóm pre-train là mốc, không phải đối thủ.** Parakeet zero-shot thua nhóm 40 ms ở WER chung nhưng **tốt nhất
trên từ Việt** (0,088); khoảng cách do từ ngoại bị phiên âm thành âm tiết Việt. Parakeet fine-tune điền sau khi chấm,
kèm điều kiện khác (2.000 h pre-train, 600M, subsampling 8×, nhãn giả khi fine-tune, khả năng rò rỉ).

**Khuyến nghị thực dụng (Chương 4), luôn kèm điều kiện "một bộ dữ liệu, ~12M, CTC, T4, 1–3 seed":**
- Câu ngắn, cần train nhanh, GPU nhỏ → Conformer relPE 40 ms.
- Audio dài (≥ ~1 phút) hoặc giới hạn VRAM/độ trễ suy luận → ConExtBiMamba 40 ms.
- Có sẵn mô hình lớn, không cần train, chấp nhận chậm 5–10× → Parakeet.

## 4. Mục hạn chế phải khai (đưa nguyên vào Ch3 và Kết luận)

Nhãn giả cho train/val; một người hiệu đính + neo nhãn (số từ gõ mù); 203 câu/29 video; 1 seed ở 10 ms; không dò LR;
cấu hình 40 ms trộn 4 biến; nhánh 10 ms rẽ từ v1 qua commit khác; Conformer không PE là baseline phi chuẩn; T4 /
fp16 / khối Mamba fp32; audio ghép nhân tạo cho RQ2; nhóm pre-train khác điều kiện, có thể rò rỉ; tách theo video ≠
theo người nói; thứ tự quyết định sau khi thấy v1; sự cố thấy WER relPE sớm (13/10).
