# ConExtBiMamba — vì sao kém B1, cách thu hẹp khoảng cách (2026-10-12)

Người dùng (2026-10-12): trọng tâm là ConExtBiMamba, muốn mô hình Mamba đuổi sát các mô hình khác.
Ghi chú này là **phân tích + đề xuất của AI, CHƯA CHỐT**. Mọi phương án tốn GPU phải được người dùng duyệt;
phương án đụng chốt đang treo (subsampling, Q11) phải chờ GVHD.

## 1. Hiện trạng (số đã có, không cần chạy thêm)

Clean-test 203 câu (nhãn giả, `reports/results/final_compare_2026-10-08/`), WER + CI 95% bootstrap theo video:

| Mô hình | WER clean-test | CI 95% | WER val (best) | GPU-h train |
|---|---|---|---|---|
| Parakeet-0.6B zero-shot | 14,0% | 11,7-16,5 | — | (pre-train) |
| PhoWhisper-small zero-shot | 15,5% | 12,9-18,3 | — | (pre-train) |
| Mamba B1 (30 ep) | 17,1% | 14,8-19,6 | 15,6% (ep28) | 16,8 |
| **ConExtBiMamba (30 ep)** | **20,3%** | 17,8-22,8 | 18,8% (ep28) | 10,0 |
| Conformer-12M (30 ep) | 26,5% | 23,7-29,4 | 25,2% (ep28) | 6,6 |
| Conformer-12M dài (69 ep) | — | — | 19,4% (ep66) | 14,8 |

ConExtBiMamba − B1 = +3,2 điểm (CI cặp 2,6-3,8; p(B1 tốt hơn) = 1,0) — khoảng cách có thật.

## 2. Chẩn đoán từ `metrics.jsonl` (train full, `reports/results/train_full/`)

1. **Underfit, không phải overfit.** Ở ep29 ConExtBiMamba có train loss 2,08 và loss val 0,77, cả hai đều cao hơn B1
   (train loss 1,86, loss val 0,67). Mô hình chưa khớp được tập train tốt bằng B1. Đây không phải trường hợp khớp
   tốt tập train nhưng tổng quát kém.
2. **Chưa hội tụ.** WER val của cả ba mô hình vẫn giảm ở ep28. LR giữ **hằng 3e-4** tới cuối, không giảm.
   Conformer dài cho thấy mức "để lại": 30 → 66 epoch, WER val giảm 25,2 → 19,4 (−5,8 điểm).
3. **ConExtBiMamba nhanh hơn B1 ~1,7×/step** (0,40 so với 0,68 s/step). Với cùng 30 epoch, nó chỉ dùng
   **10,0 GPU-h, trong khi B1 dùng 16,8 GPU-h**. Bảng hiện tại vì vậy so ở cùng số epoch, nhưng không cùng
   lượng tính toán.
4. Ổn định: không có step loss không hữu hạn; AMP bỏ 1-5 step/epoch, như B1. Ổn định không phải nguyên nhân.
5. Lệch so với công thức của bài gốc (arXiv:2405.12609, ESPnet, LibriSpeech-100): bài train **120 epoch**, LR
   đỉnh 2e-3, warmup 15k, có subsampling, và chuyển nguyên siêu tham số của Conformer sang mô hình Mamba
   ("directly transferred the optimal parameters from attention-based models"). Ở dự án: 30 epoch, LR hằng 3e-4,
   không subsampling. Khởi tạo A dùng S4D-Real mặc định; Gaussian-noise-diagonal tốt hơn 0,2 điểm trong Bảng XIV
   của bài (6,1/6,2 → 5,9/6,0).

**Giả thuyết, chưa kiểm:** ConExtBiMamba chỉ có 6 cặp Mamba trên chuỗi 10 ms không subsampling (≤ 1.500 khung).
B1 có 14 cặp. Phần lớn tham số của ConExtBiMamba nằm ở FFN và conv, nên khả năng mô hình hóa chuỗi dài yếu hơn
B1 khi chuỗi dài. Bài gốc chạy có subsampling 4×, tức chuỗi ngắn hơn 4 lần. Không có tài liệu nào so ConExtBiMamba
ở 10 ms/khung.

## 3. Phương án, xếp theo chi phí

| # | Việc | GPU-h ước | Công bằng giữa các mô hình | Căn cứ |
|---|---|---|---|---|
| P0 | **Trung bình trọng số** 5/10 epoch cuối cho **cả ba** mô hình (đã có `epochs/epochNN.pt` 30 file mỗi mô hình), eval val + clean-test | ~0,5-1 (eval Mamba cần CUDA) | Áp như nhau cho cả ba | Thông lệ ESPnet (công cụ của bài gốc) là trung bình các checkpoint tốt nhất. SWA (Izmailov và cs., arXiv:1803.05407): trung bình dọc quỹ đạo LR hằng giúp tổng quát hóa. Đây cũng là "mức 0" đã đề xuất trong `subsample_speed_2026-10-08.md` mục 8 |
| P1 | **So cùng ngân sách GPU-giờ:** train tiếp ConExtBiMamba với đúng công thức hiện tại (LR hằng) từ `best.pt` ep28 (có trạng thái optimizer) tới ~**51 epoch** ≈ 16,8 GPU-h, bằng B1 | ~6,5-7 (1 phiên A) | Công bằng theo trục tài nguyên: B1 30 ep ≈ 16,8 h · ConExt ~51 ep ≈ 16,8 h · **Conformer dài 69 ep ≈ 14,8 h đã có sẵn** | Không đổi công thức nên không có "tinh chỉnh riêng". Đề tài nhận xét theo đánh đổi WER-tốc độ-tài nguyên (RQ), nên so cùng GPU-giờ là một mức so sánh hợp lệ, đặt cạnh mức cùng epoch |
| P2 | Pha **giảm LR** (cosine về ~0) ~5-10 epoch cho **cả ba** mô hình, sau P0/P1 | 6-9 | Áp như nhau | Mức 2 của mục 8 ghi chú 08/10; bài gốc có lịch giảm LR |
| P3 | **Subsampling 4×** cho cả ba (cấu hình gần bài gốc) | 12-20 | Như nhau | ⏸ **Chờ GVHD (Q11)**, chốt 2026-09-26 đang treo. Không tự mở |
| — | Không đề xuất | — | — | Khởi tạo A Gaussian (bài: chỉ +0,2 điểm, phải train lại 10 h); đổi số lớp/ffn (không có bằng chứng, phải train lại, phá khớp tham số ~12M); chỉnh LR riêng cho ConExtBiMamba (tinh chỉnh một phía, thiên lệch) |

**Không hứa trước kết quả.** Chưa biết P1 đuổi được B1 tới đâu. Đường cong ep20 → 28 giảm ~0,27 điểm/epoch và đang
chậm lại; ví dụ Conformer dài cho thấy còn dư địa vài điểm, nhưng mỗi kiến trúc mỗi khác. Nếu P1 đưa ConExtBiMamba
xuống khoảng B1 với cùng GPU-giờ, kết luận sẽ là "ConExtBiMamba hiệu quả tài nguyên ngang/hơn B1"; nếu không, đó
vẫn là một kết quả có giá trị.

## 4. Luật công bằng (nhắc lại vì mục tiêu là "đuổi sát")

- Mọi thay đổi công thức (giảm LR, trung bình checkpoint, subsampling) phải áp cho **cả ba** mô hình train từ đầu
  (CLAUDE.md: hai mức so sánh, chỉ encoder khác nhau).
- So cùng GPU-giờ (P1) là một **mức so sánh thứ hai**, báo cáo **song song** với cùng epoch, không thay thế.
- Chốt tiêu chí so sánh và các mốc eval **trước** khi chạy, không chọn mốc sau khi đã thấy số.

## 5. Đề xuất thứ tự (AI)

P0 trước (rẻ, trả lời luôn câu hỏi "LR hằng để lại bao nhiêu"), sau đó P1 (1 phiên, rẻ nhất trong các phương án
có train). P2/P3 tùy kết quả P0/P1 và ý kiến GVHD.

## 6. Chuỗi phương án "công thức v2" (2026-10-12, người dùng: không chờ GVHD — AI đề xuất, CHƯA CHỐT)

Nguồn đã kiểm phiên này:
- **ConMamba** (`xi-j/Mamba-ASR`, `hparams/CTC/conmamba_large.yaml`): CNN 2 lớp stride 2 (4×), LR 1e-3 Noam warmup
  7.500, 500 epoch, **trung bình 10 checkpoint**, clip 5,0, speed perturb 0,95/1/1,05.
- **2405.12609** (ESPnet, Bảng I-II): LR đỉnh 2e-3, warmup 15k, 120 epoch, có subsampling; ConExtBiMamba hơn Conformer
  ở LibriSpeech-100/960, SEAME, ASRU (vd. 5,9/6,0 so với 6,3/6,5). Trong bài, ExtBiMamba xếp chồng thuần kém xa
  (38,5/37,7, Bảng XVI) — dự án thấy ngược lại (B1 > ConExt) ở 10 ms/khung: **giả thuyết** độ phân giải quyết định.
- **Fast Conformer** (arXiv:2305.05084, Bảng 2/4): 4× → 8× không làm tăng WER (RNNT 5,19 → 5,11; CTC Conformer 5,74
  so với Fast Conformer 5,64), encoder nhanh 2,8×. Không có bài nào so 1× (10 ms) với 4× cho Conformer/Mamba mà AI tìm được.
- **InterCTC** (Lee & Watanabe, arXiv:2102.03216): thêm CTC ở lớp giữa, trọng số 0,3, gần như không tốn khi train,
  không tốn khi suy luận. Conformer 12 lớp: WSJ 15,2/12,4 → 13,4/10,8; TED-LIUM2 10,5/9,8 → 9,7/9,1; AISHELL-1
  5,4/6,0 → 5,1/5,6; bài nói vẫn có lợi với mô hình nhỏ (Transformer 6 lớp WSJ 21,1 → 18,3).
- Mã hóa vị trí cho Conformer: Conformer chuẩn (Gulati 2020) dùng PE tương đối; **không tìm được ablation bỏ PE** ở
  Conformer — chỉ có 2405.12609 Bảng XIV (thêm PE cho ConExtBiMamba không đổi WER).

| Bước | Việc | GPU-h | Ghi chú |
|---|---|---|---|
| S0 | Trung bình 10 checkpoint cuối của 3 mô hình v1 + eval val/val_unseen/clean-test | ~1 | Số cuối của nhóm v1; không đổi gì khác |
| S1 | Code v2 (CPU test): Conv2d subsampling 4× chung (như phép đo 08/10), PE cho Conformer, lịch LR warmup → đỉnh 1e-3 → giảm, InterCTC 0,3 lớp giữa, trung bình checkpoint | 0 | Áp cho **cả ba**; mỗi thay đổi có nguồn ở trên |
| S2 | Trial v2: 3 mô hình × ~2 epoch/1 shard — LR đỉnh 1e-3 có ổn định với Mamba không, s/step thật (đọc dữ liệu có thành nút cổ chai?) | ~2 | Mamba từng tràn fp16 → khối Mamba vẫn fp32 |
| S3 | Train v2 cả ba, ~60 epoch (ước từ s/step 08/10 × 1,3: Conformer ~5 h, ConExt ~7 h, B1 ~10 h) | ~22 | Hai tài khoản song song; báo cáo cả cùng epoch lẫn GPU-giờ |
| S4 | Eval + RQ2 (v1 và v2) | ~2-3 | v1 thành ablation "10 ms vs 40 ms" — đóng góp riêng |
| S5 | (tùy) ConExt v2 train thêm tới cùng GPU-giờ của B1 v2 | ~3-4 | Chỉ khi S3 cho thấy ConExt nhanh hơn rõ |

Rủi ro: (1) LR 1e-3 có thể làm Mamba mất ổn định — S2 kiểm, lùi 5e-4 nếu cần; (2) không bảo đảm ConExt vượt B1 —
v2 trả lời giả thuyết độ phân giải dù kết quả thế nào; (3) đổi công thức giữa chừng → phải viết rõ v1/v2 trong khóa luận.

## 7. Đánh giá lại v1 và kế hoạch "đúng khoa học nhất" (2026-10-12, sau trao đổi với người dùng)

**Bối cảnh mới (người dùng):** bỏ ràng buộc "chỉ đổi encoder"; có thể thêm tài khoản Kaggle thứ ba (~90 GPU-h/tuần);
ưu tiên tính đúng khoa học. Mục 6 (chuỗi "v2" đổi mọi thứ cùng lúc) **bị thay** bởi mục này.

### 7.1 Đánh giá v1
- **Lỗi thật:** (1) Conformer không có PE nào (torchaudio dùng MHSA thường) — lệch Conformer chuẩn, baseline yếu có
  chủ đích hay không đều bị hỏi; (2) LR hằng 30 epoch, không pha giảm, không trung bình → cả ba chưa hội tụ.
- **Không phải lỗi:** không subsampling — chốt có chủ đích vì RQ2 (pilot: điểm giao tốc độ 60 s ở 1× so với 240 s ở 4×);
  12M, CTC, phạm vi đề tài.
- **Thiếu về phương pháp:** 1 seed (CI bootstrap hiện chỉ phản ánh biến thiên do mẫu test, không do train); LR 3e-4 chọn
  một lần, không mô hình nào được dò LR → không biết có mô hình nào bị thiệt vì LR.

### 7.2 Nguyên tắc (có nguồn)
| Nguyên tắc | Nguồn | Áp vào |
|---|---|---|
| LR hằng + pha giảm cuối (WSD) ≈ cosine; pha giảm ~20% tổng step, dạng 1-sqrt tốt hơn tuyến tính; SWA trên LR hằng thu hẹp nhưng không bằng pha giảm | Hägele và cs., arXiv:2405.18392 (LLM, không phải ASR — giả định chuyển được) | Tái dùng các lần v1 làm pha "hằng", thêm pha giảm |
| Ngân sách dò siêu tham số ngang nhau cho mọi kiến trúc; baseline dò kỹ có thể đảo kết luận | Melis và cs., ICLR 2018 (arXiv:1707.05589) | Lưới LR chung cho cả ba, chọn theo val |
| Biến thiên do khởi tạo/thứ tự dữ liệu đáng kể; một seed đánh giá thấp phương sai | Bouthillier và cs., MLSys 2021 (arXiv:2103.03098) | 3 seed cho bảng chính |
| Conformer chuẩn có PE tương đối | Gulati và cs., 2020 | Conformer bảng chính = bản có rel-PE; bản không PE thành ablation |
| Subsampling 4× là cấu hình của tài liệu Mamba/Conformer | ConMamba yaml; 2405.12609; Fast Conformer | Nhánh phụ 40 ms (thiết kế giai thừa độ phân giải × encoder) |
| Mọi lựa chọn trên val; clean-test (đã hiệu đính) chỉ dùng một lần cuối | thông lệ | Hiệu đính 203 câu là đường găng |

### 7.3 Kế hoạch theo pha (ước GPU-h 2×T4; s/epoch từ log v1, nhánh 40 ms từ benchmark 08/10 × 1,3)
| Pha | Việc | GPU-h | Cổng quyết định (chốt trước khi có số) |
|---|---|---|---|
| G0 | Code + test CPU: lịch WSD (pha giảm 1-sqrt, rẽ nhánh từ checkpoint), script trung bình checkpoint, Conformer rel-PE, tùy chọn subsampling 4× trong `features`, tài khoản C trong launcher | 0 | test CPU cả ba encoder (Mamba bằng khối giả) |
| P1 | v1: trung bình 10 checkpoint cuối (SWA) + eval | ~2 | — (số tham chiếu) |
| P2 | Dò LR proxy 10 epoch (8 hằng + 2 giảm) × lưới {3e-4, 6e-4, 1e-3} × {B1, ConExt, Conformer-relPE}; 3e-4 của B1/ConExt rẽ nhánh từ v1 ep8 | ~28 | Mỗi mô hình lấy LR có WER val tốt nhất; LR mất ổn định (loss không hữu hạn, step bỏ > 1%) bị loại. Hạn chế: proxy ngắn có thể thiên về LR cao |
| P3 | Bảng chính 10 ms: 3 mô hình × 3 seed × 40 epoch (32 hằng + 8 giảm, 1-sqrt) + trung bình checkpoint pha giảm. Seed 1 của B1/ConExt tái dùng v1 nếu LR chọn = 3e-4 (từ `best.pt` ep28, có optimizer) | ~110 (+~40 nếu không tái dùng được) | Báo cáo trung bình ± độ lệch 3 seed + bootstrap video; cả theo epoch và GPU-giờ |
| P4 | Nhánh 40 ms (subsampling 4×): 3 mô hình × 1 seed × 40 epoch, LR như P3 | ~15 | Trả lời: ConExt thua B1 có do độ phân giải không; đánh đổi WER-tốc độ theo độ phân giải (RQ2) |
| P5 | Ablation Conformer không PE (v1) + pha giảm; eval clean-test đã hiệu đính mọi mô hình; đo RQ2 cuối | ~6 | — |
| Tổng | | ~160 (+dự phòng lỗi khởi động ~20%) | ~2 tuần với 3 tài khoản |

Không làm: InterCTC (cải tiến thêm, không sửa lỗi v1, đổi mục tiêu train); chỉnh riêng cho ConExt ngoài lưới chung;
nhánh 40 ms nhiều seed (ưu tiên seed cho bảng chính).

## 8. Bộ thay đổi v2 đã cài (2026-10-12, người dùng: "mọi thay đổi có lợi triển khai hết trong lần này")

**Đưa vào (áp cho cả ba encoder trừ khi ghi khác):**
| # | Thay đổi | Căn cứ | Ghi chú |
|---|---|---|---|
| 1 | Lịch WSD: warmup ~1 epoch → hằng → giảm 1-sqrt 8/40 epoch cuối | Hägele 2024 | warmup 2840 step (v1: 1000) là lựa chọn dự án cho LR đỉnh tới 1e-3 |
| 2 | Dò LR lưới chung {3e-4, 6e-4, 1e-3} | Melis 2018 | proxy 10 epoch (8 + 2 giảm) |
| 3 | 3 seed bảng chính | Bouthillier 2021 | |
| 4 | InterCTC w = 0,3 lớp ⌊L/2⌋ + self-conditioning | Lee & Watanabe 2021 (Conformer WSJ 12,4 → 10,8); Nozaki & Komatsu 2021 (> 20% tương đối so CTC, WSJ); cách cài ESPnet | chung CTC head, intermediate chuẩn hóa; +0,26M tham số mỗi mô hình |
| 5 | Conformer PE tương đối (Gulati 2020, cài theo ESPnet) | bài gốc | ablation `pos_enc: none` cùng công thức |
| 6 | Conformer xoá padding trước depthwise conv | đo: v1 lệch 0,13-0,38 câu riêng/trong batch; ConExt đã xoá từ v1 | eval không còn phụ thuộc cách ghép batch |
| 7 | Trung bình checkpoint | ConMamba (10), ESPnet | chốt trước: N epoch cuối của pha giảm |
| 8 | Nhánh 40 ms (subsampling 4×) | ConMamba, 2405.12609, Fast Conformer | 1 seed |

**Không đưa vào (lý do):** khởi tạo A Gaussian (bài chỉ +0,2 điểm, đặc tả nhiễu không đủ để cài đúng); đổi
weight decay / batch (không có bằng chứng trực tiếp, phải mở rộng lưới dò); stochastic depth (InterCTC paper:
Conformer TED-LIUM2 tệ hơn); InterCTC nhiều lớp (bài chính dùng một lớp L/2); LM n-gram khi giải mã (đổi
phần giải mã, không cần train lại — để sau, áp cho mọi mô hình, báo cáo bảng riêng).

**Hệ quả:** v1 không tái dùng được làm seed 1 (khác InterCTC/warmup) → mọi lần train v2 từ đầu, cùng công thức.
v1 giữ làm mốc "công thức đơn giản" + ablation 10 ms/LR hằng.

**Ngân sách cập nhật (GPU-h, ước từ log v1; Conformer relpos chưa đo):** preflight ~1 · P1 ~2 · P2 ~34 ·
P3 9 lần từ đầu ~136 (B1 67, ConExt 40, Conformer 29) · P4 ~17 · P5 ablation không PE ~10 + eval ~4 ·
ConExt cùng GPU-giờ ~9 → **~213, +20% dự phòng ≈ 255**. 3 tài khoản tới 31/10 ≈ 262 → vừa khít.
