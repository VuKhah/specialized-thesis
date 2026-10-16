# Rà soát độc lập toàn dự án và phương án cuối (Fable 5.1, 2026-10-15)

Vai trò: người phản biện độc lập, chỉ đọc repo và tra nguồn; không chạy GPU, không sửa code. Người dùng và GVHD
chốt. Số liệu dự án trích từ `reports/results/` và các note ghi ngày; nguồn ngoài ghi arXiv/link và mức đã đọc
(toàn văn / abstract) ở mục 7. Chỗ không có bằng chứng ghi rõ **(suy luận)**.

## 0. Tóm tắt 10 dòng

1. Dự án đã có một kết quả **thật và đáng viết**: ở ~12M tham số, CTC, tiếng Việt hội thoại, Mamba **không mặc
   nhiên tốt hơn** Conformer. Ở 40 ms (cấu hình chuẩn) Conformer relPE dẫn 0,5–1,0 điểm (CI cặp không chứa 0, 1 seed);
   ở 10 ms không subsampling hai encoder Mamba bỏ xa Conformer *không PE* 3–9 điểm. Lợi thế tính toán của Mamba chỉ
   xuất hiện từ ~45–90 s audio, không có trên câu hội thoại 10–15 s; chi phí train ở 40 ms ba mô hình lệch ≤ 25%.
2. Hai lỗ hổng **0 GPU-giờ** phải vá trước mọi việc khác: (a) run **Conformer relPE 10 ms** (tài khoản B, 3 phiên,
   ~18 GPU-h đã tiêu) **không có trong `reports/`** và chưa chấm — bảng 10 ms đang thiếu baseline chuẩn; (b) nhóm 10 ms
   chấm clean-test bằng **ep28 v1** trong khi quy tắc 9.4 chốt checkpoint bảng chính là **ep37** — ba checkpoint ep37
   10 ms chưa chấm. Chỉ cần tải output + 1 kernel dump log-prob (~0,2 GPU-h).
3. Mối đe doạ lớn nhất còn lại là **1 seed ở 40 ms** (chênh 0,4–1,0 điểm). Khuyến nghị **thí nghiệm GPU duy nhất**:
   thêm **2 seed × 3 mô hình ở 40 ms**, công thức y nguyên, ≈ 32 GPU-h, 6 run mỗi run ≤ 6 h (vừa 1 phiên), đủ chạy
   trong tuần 17–24/10 với 3 tài khoản hiện có. Không cần thêm tài khoản.
4. **Không làm**: CR-CTC, BPE3000, train dài 40 ms, dò LR, train bù cùng GPU-giờ, seed ở 10 ms, eval ngoài miền. Lý do
   từng cái ở mục 3.
5. Trục GPU-giờ: chỉ **báo cột GPU-giờ**, không train bù. Ở 40 ms relPE 5,28 h · ConExt 4,86 h · B1 6,00 h → không có
   mô hình nào được lợi vì tính toán nhiều hơn. Đóng mục ⛔ này.
6. Bảng chính RQ1 = greedy; bảng phụ = beam 10 + từ điển + LM 3-gram cho 3 mô hình 40 ms (cả 3 seed), ghi rõ là cải
   thiện của hệ giải mã. Chấm Parakeet fine-tune trên clean-test (~0,2 GPU-h) để nhóm pre-train đủ hai dòng.
7. Hướng chất lượng nhãn gộp vào RQ3, dùng số đã có (nhãn giả lệch 1,8% WER; từ Việt 0,65%; từ ngoại 37%; thứ hạng
   trên nhãn giả và nhãn hiệu đính **không đổi**). Gõ lại mù ~20 câu để đo neo nhãn (việc tay ~1 h).
8. Đề nghị **đóng băng GPU sớm hơn: 25/10** thay vì 01/11 (dự phòng tới 01/11 nếu run seed lỗi). Mỗi tuần sớm là một
   tuần viết.
9. Lịch: 16–17/10 vá lỗ hổng + gửi đề cương; 17–20/10 chạy 6 run seed; gặp GVHD tuần 19–25/10; 25/10 đóng băng;
   08/11 đóng băng số liệu; 15/11 nháp Ch3; 27/11 bản đủ; 30/11–06/12 nộp.
10. Quyết định người dùng cần chốt: 12 mục ở mục 4, mỗi mục kèm khuyến nghị.

## 1. Đánh giá hiện trạng

### 1.1 Điểm mạnh (giữ nguyên, đây là vốn của khóa luận)

- **Thiết kế có kiểm soát thật sự**: cùng front-end, tokenizer, CTC head, `train.py`, dữ liệu, công thức; chỉ encoder
  khác; tham số khớp 12,20–12,29M (`ARCHITECTURE.md` mục 4). Hai mức so sánh tách bạch (train từ đầu / pre-train).
- **Tập test độc lập theo video + bootstrap theo khối video** (`src/evaluation/bootstrap.py`, Liu & Peng
  arXiv:1912.09508) — đúng vì 561/562 video `validation` gốc trùng train (`dataset_discrepancy.md`).
- **Clean-test 203 câu đã hiệu đính tay**, quy tắc chấm chốt trước khi tính (`clean_test_results_2026-10-15.md` mục 1),
  có kiểm thiên lệch do thấy output mô hình (107 câu trước mốc: thứ hạng không đổi).
- **Quy tắc phân tích viết trước khi có số** (`conextbimamba_gap_2026-10-12.md` mục 9.4) và khai thứ tự thời gian các
  quyết định, kể cả sự cố AI thấy WER relPE sớm (13/10 12:58). Hiếm ở KLTN, đáng đưa vào Ch2.
- **RQ2 đo đủ đường cong tới 3.600 s, có làm ấm từng độ dài, chia đôi tìm OOM, đủ 7 cấu hình + Parakeet**
  (`reports/results/rq2_final_2026-10-13/`). Đây là phần đứng vững nhất (phản biện thử 14/10 cũng đồng ý).
- Mọi run có `metrics.jsonl` (s/step, VRAM, wall time cộng dồn, step không hữu hạn) → tính được GPU-giờ thật.

### 1.2 Số liệu gốc dùng trong bản này

WER clean-test 203 câu / 29 video, bản hiệu đính, greedy, CI 95% bootstrap theo video
(`reports/results/clean_test_scored_2026-10-15/results.json`):

| Nhóm | Hệ thống | WER | CI | WER từ Việt | GPU-h train (2×T4, wall) |
|---|---|---|---|---|---|
| 40 ms, ep37 | Conformer relPE | 0,1211 | [0,1030; 0,1409] | 0,0960 | 5,28 |
| | B1 Mamba | 0,1264 | [0,1072; 0,1480] | 0,1031 | 6,00 |
| | ConExtBiMamba | 0,1309 | [0,1110; 0,1530] | 0,1052 | 4,86 |
| 10 ms, **ep28 v1** (chưa phải ep37) | B1 Mamba | 0,1704 | [0,1479; 0,1954] | 0,1424 | 16,7 (30 ep) |
| | ConExtBiMamba | 0,2019 | [0,1774; 0,2275] | 0,1738 | 10,0 |
| | Conformer **không PE** | 0,2648 | [0,2370; 0,2937] | 0,2393 | 6,6 |
| Pre-train zero-shot | Parakeet-CTC-0.6B-vi | 0,1378 | [0,1150; 0,1624] | **0,0879** | — |
| | PhoWhisper-small | 0,1490 | [0,1235; 0,1766] | 0,1014 | — |
| Tham khảo | Nhãn giả Zipformer | 0,0177 | [0,0124; 0,0234] | 0,0065 | — |

So cặp 40 ms greedy: B1 − ConExt −0,0044 [−0,0097; +0,0007] (**chứa 0**); B1 − relPE +0,0054 [+0,0009; +0,0101];
ConExt − relPE +0,0098 [+0,0048; +0,0152]. Với beam + LM: relPE 0,1024 · ConExt 0,1081 · B1 0,1115; ConExt − B1 vẫn
chứa 0.

GPU-giờ lấy từ `wall_seconds_total` trong `reports/results/train_full/*/s1/checkpoints/*/metrics.jsonl`
(40 ms: 17.502 / 18.997 / 21.615 s; 10 ms 38 epoch: Conformer không PE 8,4 h · ConExt 12,7 h · B1 21,3 h).

RQ2 (`rq2_final/results.csv`, trọng số ngẫu nhiên, T4, batch 1, độ trễ trung vị, giây):

| Độ dài | 15 s | 60 s | 240 s | 3.600 s | OOM |
|---|---|---|---|---|---|
| B1 10 ms | 0,026 | 0,075 | 0,317 | 5,21 | không |
| ConExt 10 ms | 0,021 | 0,051 | 0,212 | 3,40 | không |
| Conformer không PE 10 ms | 0,015 | 0,062 | 0,644 | 156,7 | không |
| Conformer relPE 10 ms | 0,031 | 0,393 | OOM | — | ~165 s |
| B1 40 ms | 0,029 | 0,029 | 0,112 | 1,65 | không |
| ConExt 40 ms | 0,020 | 0,022 | 0,075 | 1,18 | không |
| Conformer relPE 40 ms | 0,022 | 0,035 | 0,423 | OOM | ~656 s |
| Parakeet fp32 (80 ms) | 0,095 | 0,414 | 2,64 | OOM | ~585 s |

Điểm giao (Mamba nhanh hơn từ đây): so với relPE 40 ms, ConExt ~45 s, B1 ~60 s; so với Conformer không PE 10 ms,
ConExt ~45 s, B1 ~90 s. Trên câu 10–15 s: ConExt 40 ms ≈ relPE 40 ms (~20 ms), B1 chậm hơn ~40%.

### 1.3 Mối đe doạ tới tính đúng của kết luận — xếp hạng

| # | Mối đe doạ | Bằng chứng trong repo | Hậu quả nếu bỏ qua | Cách xử lý | GPU-h |
|---|---|---|---|---|---|
| **1** | **Bảng 10 ms thiếu Conformer relPE và dùng sai checkpoint.** Run `v2a-conformer-relpos` 10 ms (B, 3 phiên, đẩy s3 14/10 01:10) không có thư mục trong `reports/results/train_full/`; không số WER nào trong Plan/TODO/notes. Ba nhánh 10 ms (B1, ConExt, Conformer không PE) có ep37 (val_unseen nhãn giả 0,1252 / 0,1506 / 0,2068) nhưng clean-test chỉ chấm ep28 v1 | `TODO.md` khối ▶▶▶▶ mục 2; `clean_test_results_2026-10-15.md` dòng "v1 10 ms (ep28)"; quy tắc 9.4 mục 9 (ep37) | "Mamba > Conformer ở 10 ms" chỉ so với baseline phi chuẩn; bảng 10 ms vi phạm chính quy tắc đã chốt; giả thuyết độ phân giải không kiểm được | Tải output B s3 (kiểm `latest.pt` ep37, returncode); 1 kernel dump log-prob 4 checkpoint ep37 10 ms trên clean-test + nửa dev; chấm lại. Nếu output B mất: xem mục 5 | ~0,2 |
| **2** | **1 seed ở 40 ms**, chênh 0,4–1,0 điểm; CI bootstrap chỉ phản ánh biến thiên mẫu test, không phản ánh biến thiên khởi tạo/thứ tự dữ liệu | `clean_test_results` nhận xét 8; `phan_bien_2026-10-14.md` #2; Bouthillier arXiv:2103.03098 (khởi tạo + lấy mẫu "impact markedly") | Thứ hạng 40 ms có thể là nhiễu seed; không nói được "relPE tốt hơn" lẫn "ngang nhau" | 2 seed thêm × 3 mô hình 40 ms, công thức y nguyên; tiêu chí viết trước (mục 2.3) | ~32 |
| **3** | **Nhãn giả và neo nhãn khi hiệu đính**: ô hiệu đính điền sẵn nhãn giả; 80/113 câu đầu giữ y nhãn giả; bản hiệu đính cách nhãn giả chỉ 1,5–1,8% | `Plan.md` nhật ký 14/10 khuya; `label_quality_direction.md` mục 4 | L1 (nhãn sai bao nhiêu) bị đánh giá thấp; mọi WER tuyệt đối hơi lạc quan như nhau cho mọi mô hình (không đổi thứ hạng) | Gõ lại mù ~20 câu không xem nhãn giả, báo độ lệch; khai một người hiệu đính | 0 |
| **4** | **"10 ms → 40 ms" trộn 4 biến**: +1,84M tham số subsampling, front-end Conv2d học được, trường tiếp nhận ×4, 40 ms train từ đầu còn 10 ms rẽ nhánh từ v1 (RNG khác, commit khác) | `phan_bien` #3, #12; `ARCHITECTURE.md` 8-v2a | Viết "độ phân giải quyết định" là vượt bằng chứng | Gọi là **cấu hình 40 ms**; khai 4 khác biệt; kết luận theo nhánh đã viết sẵn ở 9.4 mục 5 | 0 |
| **5** | **29 video, 203 câu**: CI ±2 điểm tuyệt đối; tách theo video ≠ theo người nói (MC chương trình lớn) | `dataset_discrepancy.md` cuối; `phan_bien` #11 | Chênh < 1 điểm chỉ tách được nhờ so cặp | Báo số khối; thêm kiểm định dấu theo video; khai | 0 |
| **6** | **RQ2 trên audio ghép + trọng số ngẫu nhiên; relPE OOM là thuộc tính cài đặt** (bias [B,H,T,2T−1] tường minh; Conformer không PE qua SDPA chạy tới 3.600 s); số học không đồng nhất (Mamba fp32) | `rq2_final`; `rq2_pilot` nhận xét 3–4; `phan_bien` #5, #7 | Nói "Conformer không chạy được audio dài" là sai; VRAM không phải lợi thế Mamba ở đây | Báo cả đường không PE làm cận dưới; gọi đúng tên OOM; bảng điều kiện số học | 0 |
| **7** | **Zero-shot / fine-tune vs train trên cùng miền**: Parakeet pre-train ~2.000 h (10 bộ, card không nêu VietSuperSpeech), fine-tune trên nhãn giả | `survey_model_candidates.md` mục 1; `Plan.md` 12/10 tối | Xếp hạng chung với nhóm train từ đầu là sai điều kiện | Chỉ tham chiếu; ghi dữ liệu/cỡ/subsampling 8×/khả năng rò rỉ; Parakeet FT phải chấm clean-test trước khi kết luận | ~0,2 |
| **8** | **Không dò LR** (3e-4 chung, chọn một lần); công bố Mamba dùng 1e-3–2,5e-3 với batch lớn | `conextbimamba_gap` 7.1, 9.5; Melis arXiv:1707.05589 | Có thể một mô hình bị thiệt vì LR | Khai là hạn chế; không mở vòng dò (mục 3) | 0 |
| 9 | **Từ dính của CTC + BPE1000** (12–14% số lỗi ở 40 ms) là lỗi của hệ giải mã/tokenizer, không của encoder; beam + LM xoá gần hết | `glued_words_2026-10-15.md` mục 1, 3.1 | Dễ gán nhầm cho encoder | Phân tích RQ3; bảng phụ giải mã | 0 |
| 10 | **Chỉnh sửa sau khi thấy số** (HARKing): pha giảm, PE, 40 ms, chọn ConExt làm "mô hình chính" đều sau v1 | 9.4 mục 1; `Plan.md` 12/10 | Hội đồng hỏi | Đã khai; đưa phụ lục E (nguyên văn 9.4) | 0 |
| 11 | `normalize_text` thống nhất dấu thanh (hóa/hoá) còn ⛔ từ 12/10; không rõ bản chấm 15/10 đã xử lý chưa | `TODO.md` ▶▶▶▶ mục 4; `phan_bien` #8 | Nhóm pre-train có thể bị phạt oan vài từ | Kiểm `alt_spellings` + `text_normalize.py` trước lần chấm cuối; nếu đổi thì chấm lại toàn bộ cùng lúc | 0 |

Mối đe doạ 1 là **lỗi quy trình đang mở**, không phải hạn chế: phải vá trước khi bàn bất kỳ run mới nào.

## 2. Phương án khuyến nghị duy nhất

### 2.1 Luận điểm chính và cách phát biểu RQ cuối

**Luận điểm** (viết sẵn hai nhánh cho chỗ còn thiếu số):

> Ở quy mô ~12M tham số, CTC, dữ liệu ~165 h tiếng Việt hội thoại, **Mamba hai chiều không mặc nhiên tốt hơn
> Conformer**. Ở cấu hình chuẩn 40 ms (subsampling 4×), Conformer PE tương đối đạt WER thấp hơn hai encoder Mamba
> [kết luận chính thức sau 3 seed: "thấp hơn X ± Y điểm" hoặc "không phân biệt được"]; hai encoder Mamba không phân
> biệt được với nhau. Ở 10 ms không subsampling, hai encoder Mamba vượt Conformer không PE 3–9 điểm [và so với
> Conformer PE: điền sau khi chấm]. Về tính toán, Mamba có chi phí suy luận gần tuyến tính theo độ dài và chỉ nhanh
> hơn Conformer từ ~45–90 s audio; trên câu hội thoại 10–15 s không có lợi thế, chi phí train ở 40 ms ba mô hình lệch
> ≤ 25%. Nhãn giả Zipformer của VietSuperSpeech đủ tốt để xếp hạng mô hình (thứ hạng không đổi giữa nhãn giả và nhãn
> hiệu đính) nhưng sai 37% ở từ tiếng Anh/tên riêng, và mọi mô hình train trên nó đều kém "thầy" ở từ ngoại.

Đây là phát hiện thật, bất lợi cho giả thuyết ban đầu ("Mamba tốt hơn"), nhưng **có giá trị thực dụng**: người làm ASR
tiếng Việt hội thoại trên GPU nhỏ biết khi nào Mamba đáng dùng (audio dài, tài nguyên suy luận) và khi nào không
(câu ngắn, train nhanh). Khớp bối cảnh công bố: Speech Slytherin (arXiv:2407.09732: Mamba chỉ hiệu quả hơn trên ngưỡng
độ dài phụ thuộc độ phân giải token), Miyazaki (arXiv:2406.16808: "comparable or better"), Zevallos (Interspeech 2025:
ngang nhau ở ngữ cảnh ngắn). Dự án là lần đầu đo điều này trên tiếng Việt và ở hai độ phân giải khung (theo phạm vi
tra cứu đã khai — chưa quét ACL Anthology/VLSP).

**RQ cuối (đề xuất, thay bản làm việc ở `Plan.md` mục 1):**

- **RQ1 — Độ chính xác.** Trong cùng pipeline CTC, ~12M tham số, cùng công thức huấn luyện, hai encoder Mamba (B1,
  ConExtBiMamba) đạt WER thế nào so với Conformer có PE tương đối trên tiếng Việt hội thoại, ở **hai cấu hình độ phân
  giải khung 10 ms và 40 ms**? Thứ hạng có giữ nguyên giữa hai cấu hình không? (Phương án A của đề cương nháp.)
- **RQ2 — Hiệu quả.** Chi phí suy luận (độ trễ, RTF, VRAM) của từng encoder thay đổi thế nào theo độ dài audio
  (5–3.600 s, audio ghép), và chi phí huấn luyện (GPU-giờ) khác nhau ra sao? Từ độ dài nào Mamba có lợi?
- **RQ3 — Đặc điểm lỗi và chất lượng nhãn.** Các encoder khác nhau ở loại lỗi nào (từ dính CTC, tiếng Anh/tên riêng,
  xoá/chèn, thanh điệu)? Nhãn giả VietSuperSpeech sai bao nhiêu so với bản hiệu đính, mô hình học trò có vượt thầy,
  và xếp hạng đo trên nhãn giả có giữ nguyên khi đo trên nhãn thật?
- Nhóm pre-train (Parakeet zero-shot/fine-tune, PhoWhisper zero-shot): **mốc tham chiếu**, không phải RQ, không xếp
  hạng chung.

**Về "mô hình chính = ConExtBiMamba"** (người dùng chốt 12/10): khuyến nghị **bỏ khái niệm mô hình chính** trong cách
viết, chuyển thành "khảo sát hai cách đưa Mamba vào encoder". Lý do: ConExt thua B1 3,2 điểm ở 10 ms và không phân
biệt được với B1 ở 40 ms; giữ nhãn "chính" buộc phải biện minh hồi tố. Thứ tự quyết định vẫn khai trong Ch2.

### 2.2 Thực nghiệm: làm gì, không làm gì

**Làm (theo thứ tự, tổng ≈ 33 GPU-h, 3 tài khoản hiện có là đủ):**

| # | Việc | Phục vụ | GPU-h | Tài khoản / lúc | Ghi chú |
|---|---|---|---|---|---|
| L1 | Tải output B `v2a-conformer-relpos` s3 (kiểm ep37, returncode 0, `latest.pt`); chép vào `reports/`; tải 3 `latest.pt`/ep37 của nhánh 10 ms nếu chưa có cục bộ | RQ1 10 ms | 0 | ngay 16/10 | Nếu output không còn → mục 5 |
| L2 | Một kernel dump log-prob trên clean-test + nửa dev val_unseen cho 4 checkpoint ep37 10 ms (relPE, B1, ConExt, Conformer không PE) + giải mã Parakeet fine-tune `best.pt` trên clean-test; chấm bằng `score_clean_test.py` | RQ1 10 ms; nhóm pre-train | ~0,3 | A hoặc C, 17/10 sau reset | Conformer chạy được CPU local nếu kernel kẹt |
| L3 | **2 seed thêm (43, 44) × 3 mô hình 40 ms**, config `configs/v2a/*_sub4.yaml` y nguyên, chỉ `--seed`; ep37; dump log-prob clean-test + nửa dev ngay cuối kernel | RQ1 40 ms (đe doạ #2) | ~32 (6 run: ConExt 2×4,9 · relPE 2×5,3 · B1 2×6,0) | A: ConExt ×2; B: relPE ×2; C: B1 ×2; đẩy 17–18/10, mỗi run 1 phiên; C từng chạy 2 kernel song song | Tiêu chí mục 2.3 chốt **trước** khi đẩy |
| L4 | Beam 10 + từ điển + LM 3-gram (lmw 0,8 / ws 1,0 đã chốt trên nửa tune) cho 9 run 40 ms; bootstrap | Bảng phụ RQ1 | 0 (CPU) | 21–23/10 | Không chỉnh lại tham số LM theo seed |
| L5 | Gõ lại mù ~20 câu clean-test (người dùng), đo WER(bản mù, bản hiệu đính) | RQ3 chất lượng nhãn (#3) | 0 | tuần 17–24/10 | chọn câu ngẫu nhiên seed 42, trải đều video |
| L6 | Phân tích RQ3 trên dự đoán đã có: từ dính theo mô hình/độ phân giải, WER từ Việt/từ ngoại, L1–L4 (L3 "nhại" bằng căn chỉnh 3 chuỗi, mở rộng `error_analysis.py`), theo âm tiết (âm đầu/vần/thanh) | RQ3 | 0 | 26/10–08/11 | không thêm run |
| L7 | Bảng điều kiện số học RQ2; tham số encoder + tổng; gọi đúng tên OOM relPE | RQ2 | 0 | khi viết Ch3 | — |

**Không làm (chi tiết lý do ở mục 3):** CR-CTC; BPE3000; "train dài 40 ms ~80 epoch"; dò LR; train bù cùng GPU-giờ;
seed ở 10 ms; eval ngoài miền (VIVOS); đo lại RQ2 với trọng số đã train; thêm tài khoản Kaggle.

### 2.3 Tiêu chí chốt trước cho 3 seed (đề xuất — người dùng duyệt rồi mới đẩy kernel)

1. Seed 42 (đã có), 43, 44; không thay seed nếu một run "xấu"; run lỗi kỹ thuật (nonfinite, không đủ 38 epoch) thì chạy
   lại **cùng seed**, khai.
2. Checkpoint ep37 cho mọi run (như 9.4 mục 9). Không chọn gì trên clean-test.
3. Báo **trung bình ± độ lệch chuẩn 3 seed** trên clean-test, val, val_unseen; kèm số từng seed (phụ lục).
4. "A tốt hơn B" khi **cả hai**: (a) hiệu trung bình 3 seed có cùng dấu ở cả 3 cặp seed (seed i của A so seed i của B);
   (b) CI 95% bootstrap theo video của hiệu WER trung bình-seed không chứa 0. Thiếu một điều kiện → "không phân biệt
   được ở 3 seed". Không dùng chữ "tương đương"/"nhỉnh hơn".
5. Độ lệch chuẩn seed được báo cạnh mọi chênh lệch giữa mô hình; chênh < 2 × độ lệch chuẩn lớn nhất → nói rõ "nằm trong
   dao động seed" dù (4) có đạt.
6. Nhóm 10 ms vẫn 1 seed: chỉ phát biểu khi |hiệu| > 2 điểm **và** CI cặp không chứa 0 (hiện B1 − ConExt 3,2 và
   Mamba − Conformer không PE 6–9 điểm thoả; relPE 10 ms điền sau). Ghi hạn chế "1 seed, độ lệch seed ước từ nhóm
   40 ms" **(suy luận: độ lệch seed ở 10 ms có thể khác)**.
7. Bảng phụ beam + LM: tham số LM giữ nguyên lmw 0,8 / ws 1,0 cho mọi seed/mô hình.

### 2.4 Cách báo cáo

- **Bảng 1 (RQ1, 10 ms, ep37, 1 seed):** Conformer relPE (baseline chuẩn) · B1 · ConExt · dòng ablation Conformer không
  PE; WER clean-test + CI, WER từ Việt/từ ngoại, val_unseen (nhãn giả), GPU-h. Phụ lục: v1 ep28, avg ep20–29, Conformer
  dài 69 epoch (số tham chiếu).
- **Bảng 2 (RQ1, 40 ms, ep37, 3 seed):** trung bình ± sd; CI cặp theo 2.3; cột GPU-h; cột tham số encoder/tổng.
- **Bảng 3 (phụ):** beam 10 + từ điển + LM cho Bảng 2; ghi rõ LM học nhãn giả Zipformer, từ điển lấy từ nhãn train;
  đây là cải thiện của hệ giải mã, không của encoder; RTF giải mã CPU ~0,025.
- **Bảng 4 (tham chiếu pre-train):** Parakeet zero-shot, Parakeet fine-tune (7 GPU-h, nhãn giả), PhoWhisper zero-shot;
  WER chung và WER từ Việt (Parakeet zero-shot 0,088 là tốt nhất trên từ Việt — phải nói); không xếp hạng chung.
- **RQ2:** hình độ trễ/RTF/VRAM theo độ dài (log-log) cho 7 cấu hình + Parakeet; bảng điểm giao theo từng cặp; bảng điều
  kiện số học; GPU-h train. Viết "trên T4, batch 1, trọng số ngẫu nhiên (chi phí không phụ thuộc trọng số), relPE cài
  bằng bias tường minh".
- **RQ3:** phân rã lỗi theo nhóm từ và theo âm tiết; từ dính theo mô hình; L1–L4; ví dụ câu.
- **Hạn chế phải khai (mục riêng trong Ch3 + Kết luận):** nhãn giả cho train/val, một người hiệu đính + neo nhãn (số
  đo từ L5), 203 câu/29 video, 1 seed ở 10 ms, không dò LR, cấu hình 40 ms trộn 4 biến, 10 ms rẽ nhánh từ v1 qua
  commit khác, Conformer không PE là baseline phi chuẩn, T4/fp16/khối Mamba fp32, audio ghép, nhóm pre-train khác điều
  kiện và có thể rò rỉ, tách theo video ≠ người nói, thứ tự quyết định sau khi thấy v1, sự cố thấy WER relPE sớm.

## 3. Phương án đã cân nhắc và loại

| Phương án | GPU-h | Vì sao loại |
|---|---|---|
| **CR-CTC** (arXiv:2410.05101, ICLR 2025) cho 3 mô hình 40 ms | 17–36 | Đổi mục tiêu huấn luyện cho cả nhóm → thành một "hệ" khác, không trả lời RQ nào; bài chỉ thử Zipformer, chưa có số cho Mamba; cần chọn α/time-mask → thêm vòng chỉnh; cải thiện tuyệt đối (−3% đến −22% tương đối) không đổi câu hỏi "Mamba vs Conformer". Để "hướng phát triển" |
| **BPE3000** rẽ nhánh 8 epoch | 4,5–6 + code | Từ dính chỉ 12–14% số lỗi, lợi ≤ 1,3–1,5 điểm; đổi tokenizer = đổi hệ; rẽ nhánh với head khởi tạo lại không còn so được với công thức chung. Từ dính báo ở RQ3 + bảng phụ LM là đủ |
| **Train dài 40 ms (~80 epoch)** | ~33 | Vẫn 1 seed; ba mô hình đều đã qua pha giảm về LR 0; không đổi câu hỏi thứ hạng; cùng chi phí với 2 seed nhưng ít thông tin hơn |
| **Dò LR lưới {6e-4, 1e-3} ở 40 ms** | ~16/giá trị | Melis (1707.05589) đúng về nguyên tắc, nhưng mở vòng xoắn: LR nào tốt hơn cho mô hình nào thì lại cần seed ở LR đó; Mamba từng tràn fp16 ở 3e-4. Khai "không dò LR" là hạn chế |
| **Train bù cùng GPU-giờ** (ConExt tới 51 epoch; relPE "nếu nhanh hơn") | 7–9 | Ở 40 ms ba mô hình lệch ≤ 25% GPU-h, relPE rẻ hơn B1 → không có mô hình nào được lợi vì tính toán. Chỉ báo cột GPU-h và đường WER theo GPU-h từ eval mỗi epoch |
| **Seed ở 10 ms** | ~42/seed | B1 10 ms 21 GPU-h/run; chênh ở 10 ms (3–9 điểm) đã lớn hơn nhiều so với dao động seed hợp lý **(suy luận)** |
| **Eval ngoài miền (VIVOS)** | ~0,5 | Chỉ eval, rẻ, nhưng là trục mới (đọc vs hội thoại, chữ hoa/thường), kéo thêm thảo luận; không phục vụ RQ. Hướng phát triển |
| **Đo lại RQ2 với trọng số đã train** | ~1 | Chi phí không phụ thuộc trọng số; đã có 7 cấu hình + Parakeet |
| **Thêm tài khoản Kaggle (tới 10)** | — | Kế hoạch trên vừa 3 tài khoản; thêm tài khoản chỉ có nghĩa nếu mở lại các mục đã loại |
| **Pitch/thanh điệu, InterCTC, self-conditioned CTC, head AED/transducer** | nhiều | Đã bác ở 12–13/10 cùng lý do: thêm biến, không phục vụ RQ |
| **RQ riêng cho chất lượng nhãn (RQ4)** | 0 | Số liệu L1–L4 chỉ từ 203 câu, trả lời gọn trong RQ3 đủ; tách RQ làm loãng |

## 4. Quyết định người dùng cần chốt (kèm khuyến nghị)

| # | Quyết định | Khuyến nghị |
|---|---|---|
| 1 | Luận điểm + RQ cuối (mục 2.1); 10/40 ms đặt ở RQ1 (phương án A đề cương) | Chốt như 2.1; đưa vào đề cương mới gửi GVHD trước buổi tuần 19–25/10 |
| 2 | Bỏ nhãn "mô hình chính = ConExtBiMamba", viết thành khảo sát hai encoder Mamba | **Bỏ**; khai thứ tự quyết định ở Ch2 |
| 3 | Vá lỗ hổng 10 ms (L1, L2) trước mọi run mới | **Bắt buộc**, 16–17/10 |
| 4 | 2 seed thêm × 3 mô hình 40 ms, tiêu chí 2.3 | **Làm**, ~32 GPU-h, 17–20/10 |
| 5 | CR-CTC (c), BPE3000 (b) | **Không**; chọn (a) bảng phụ giải mã |
| 6 | Trục GPU-giờ | Chỉ báo cột GPU-h + đường WER theo GPU-h; đóng ⛔; khai sự cố 13/10 |
| 7 | Chấm Parakeet fine-tune trên clean-test | **Có**, gộp vào kernel L2 |
| 8 | Gõ lại mù 20 câu đo neo nhãn | **Có** (~1 h việc tay) |
| 9 | Chất lượng nhãn: RQ riêng hay gộp RQ3 | **Gộp RQ3** |
| 10 | Mốc đóng băng GPU | **25/10** (dự phòng tới 01/11 chỉ cho run seed lỗi) |
| 11 | Kiểm `normalize_text` hóa/hoá + `alt_spellings` trước lần chấm cuối; nếu đổi thì chấm lại toàn bộ một lần | Kiểm ngay; chấm lại một lần duy nhất khi đủ 9 run + 4 checkpoint 10 ms |
| 12 | Commit các file chưa track (`configs/scoring/`, `score_clean_test.py`, `word_tags.py`, 4 note mới, bảng từ `clean_test_word_tags.tsv` — thêm ngoại lệ gitignore như manifest) | Commit sau khi rà repo public (không khoá/username/tên người thứ ba) |

## 5. Lịch từ nay đến nộp và rủi ro

### 5.1 Lịch

| Tuần | Ngày | GPU | Viết / việc tay | GVHD |
|---|---|---|---|---|
| 8 | 16–18/10 | L1 tải output; 17/10 (reset) L2 kernel dump (~0,3 h); đẩy 6 run seed (A/B/C) 17–18/10 | Chốt 12 quyết định mục 4; hoàn thiện đề cương mới (RQ 2.1, bảng dữ liệu, kế hoạch) → chuyển Word → gửi; commit | Gửi đề cương + xin lịch |
| 9 | 19–25/10 | Run seed xong ~20/10; dump log-prob; L4 giải mã CPU; chấm một lần đủ bảng; **đóng băng GPU 25/10** | L5 gõ mù 20 câu; sửa Ch1 theo `ch1_kiem_trich_dan.md` | **Gặp lần 2**: đề cương, Bảng 1–2 (có CI), RQ2, kế hoạch viết |
| 10 | 26/10–01/11 | (dự phòng seed lỗi) | L6 RQ3 + L1–L4; hình RQ2; Ch2 nháp (từ `ch2_nhap.md`, `muc_luc_de_xuat.md`) | — |
| 11 | 02–08/11 | — | **Đóng băng số liệu 08/11**; bảng/hình cuối; Ch3 phần RQ1–RQ2 | Gặp lần 3 (≤ 2 tuần) |
| 12 | 09–15/11 | — | Ch3 RQ3 + Thảo luận + Hạn chế; **nháp Ch3 15/11** | — |
| 13 | 16–22/11 | — | Ch1 hoàn thiện, Ch2 hoàn thiện, Ch4 demo (thêm 3 mô hình 40 ms vào demo CPU — Conformer relPE chạy CPU; Mamba qua `mamba_ref.py`) | Gặp lần 4 |
| 14 | 23–29/11 | — | Mở đầu, Kết luận, TLTK (đối chiếu nguồn), Phụ lục; **bản đủ gửi GVHD 27/11** | Nhận góp ý |
| 15 | 30/11–06/12 | — | Sửa, slide 15 phút, chạy thử demo 5 phút, in, CD (sao lưu `best.pt`/ep37 lên Drive trước) | **Nộp** |

Thời gian viết thuần: ~5 tuần sau 01/11, cộng Ch2/Ch1 làm xen từ tuần 10. Mục lục `muc_luc_de_xuat.md` ước 66–80
trang là khả thi nếu Ch3 bắt đầu ngay khi có bảng (tuần 11).

### 5.2 Rủi ro và dự phòng

| Rủi ro | Dấu hiệu | Dự phòng |
|---|---|---|
| Output relPE 10 ms (B s3) đã mất/không đủ ep37 | L1 không thấy `latest.pt` ep37 | Chạy lại từ đầu ~18 GPU-h (B, 3 phiên) **chỉ nếu** xong trước 25/10; không thì bảng 10 ms ghi "Conformer không PE (phi chuẩn); Conformer PE chỉ có ở 40 ms" và kết luận 10 ms hạ xuống "so với Conformer không PE" |
| Kernel seed kẹt cấp máy (đã xảy ra 3 lần) | RUNNING, 0 log > 15′, quota tăng | Huỷ, hoàn quota, đẩy lại cùng seed; dự phòng tới 01/11 |
| Chỉ xong 1 seed thêm | 24/10 còn run dở | Báo 2 seed (mean ± range), tiêu chí 2.3 (4a) đổi thành "cùng dấu ở 2 cặp"; khai |
| Seed làm thứ hạng 40 ms đổi/không phân biệt | sd ≥ chênh | Đó **là kết quả**: viết "không phân biệt được ở 3 seed"; luận điểm mục 2.1 vẫn đứng (Mamba không mặc nhiên tốt hơn) |
| GVHD muốn thêm thí nghiệm | sau buổi tuần 9 | Trình bảng chi phí mục 3; xin giữ phạm vi; nếu bắt buộc, chỉ nhận việc eval (0–1 GPU-h) |
| Viết chậm | 15/11 chưa có nháp Ch3 | Cắt theo thứ tự: L3 "nhại" (L6) → bảng phụ LM → phân tích theo âm tiết; không cắt Bảng 1–2, RQ2, Hạn chế |
| Đề cương mới chưa duyệt khi nộp | — | Quyển vẫn đóng đề cương đã ký (quy định Mẫu 5); phần lệch ghi ở Mở đầu/Hạn chế như đã làm với Kaggle/Colab |
| Kết luận "Conformer tốt hơn" bị xem là "đề tài Mamba thất bại" | hỏi đáp | Trả lời bằng RQ2 (lợi thế độ dài có thật, đo được) + RQ3 + giá trị thực dụng; kết quả âm có kiểm soát vẫn là đóng góp |

## 6. Việc tuyệt đối không làm trong phần còn lại

- Không chọn checkpoint/seed/tham số LM theo clean-test. Không chấm lại clean-test sau khi đã in bảng cuối trừ một lần
  chấm chung khi đủ run (mục 4 #11).
- Không đổi công thức huấn luyện cho riêng một mô hình.
- Không viết "chứng minh", "tương đương", "nhỉnh hơn" khi CI chứa 0 hoặc nằm trong dao động seed.
- Không bỏ kết quả bất lợi (ConExt thua B1 ở 10 ms; Conformer relPE dẫn ở 40 ms; Mamba chậm hơn trên câu ngắn; Parakeet
  zero-shot tốt nhất trên từ Việt).

## 7. Nguồn đã kiểm trong phiên này

| Nguồn | Mức đọc | Dùng cho |
|---|---|---|
| Bouthillier và cs., *Accounting for Variance in Machine Learning Benchmarks*, arXiv:2103.03098 (MLSys 2021) | abstract (trang arXiv) | khởi tạo/lấy mẫu "impact markedly" → cần nhiều seed; **không** nêu số seed cụ thể |
| Liu & Peng, *Statistical Testing on ASR Performance via Blockwise Bootstrap*, arXiv:1912.09508 | abstract | bootstrap theo khối cho dữ liệu phụ thuộc (câu cùng video) |
| Gillick & Cox, *Some statistical issues in the comparison of speech recognition algorithms*, ICASSP 1989 | trích dẫn gián tiếp (chưa mở toàn văn) | kiểm định cặp (MAPSSWE) — tuỳ chọn bổ sung cho bootstrap |
| Jiang và cs., *Speech Slytherin*, arXiv:2407.09732 | abstract | Mamba chỉ hiệu quả hơn trên ngưỡng độ dài, phụ thuộc độ phân giải token |
| Miyazaki và cs., *Exploring the Capability of Mamba in Speech Applications*, Interspeech 2024, arXiv:2406.16808 | abstract | Mamba "comparable or better" — cách đọc an toàn là "ngang" |
| Zevallos và cs., *Assessing the Performance and Efficiency of Mamba ASR in Low-Resource Scenarios*, Interspeech 2025 (ISCA Archive `zevallos25_interspeech`) | tóm tắt từ tìm kiếm | ConMamba ≈ Conformer ở ngữ cảnh ngắn, lợi ở ngữ cảnh dài, train nhanh hơn 40–45% (điều kiện khác dự án) |
| Yao và cs., *CR-CTC*, arXiv:2410.05101, ICLR 2025 | abstract (số bảng đã kiểm ở `glued_words` 3.2) | lý do loại CR-CTC |
| Zhang và cs., arXiv:2405.12609 (ExtBiMamba/ConExtBiMamba) | đã đọc toàn văn ở các phiên trước (theo notes) | đối chiếu kết quả |
| Hägele và cs., arXiv:2405.18392 (WSD) | theo notes | lịch pha giảm |
| Melis và cs., arXiv:1707.05589 | theo notes | rủi ro không dò LR |
| arXiv:2608.10670 (*Seeds Before Objectives*, ASR Garhwali, 5 seed) và arXiv:2512.13576 (3 seed, ±0,06 WER) | chỉ kết quả tìm kiếm, chưa mở | ví dụ thực hành nhiều seed trong ASR; **không** dùng làm số tham chiếu cho dự án |

Không tìm được số công bố về độ lệch chuẩn WER theo seed cho mô hình ~12M trên ~170 h → mọi phát biểu về "dao động
seed" trước khi có 3 seed là **suy luận**; đó chính là lý do phải đo.
