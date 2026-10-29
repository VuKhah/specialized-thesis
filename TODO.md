# TODO — Mamba vs Conformer ASR tiếng Việt

Viết lại gọn **2026-10-15** sau bản rà soát `docs/notes/review/review_fable_2026-10-15.md`. Bản cũ (lịch sử train full,
checklist kiểm kernel, lệnh đẩy từng phiên) lưu nguyên văn ở `docs/notes/luu_tru/todo_luu_tru_2026-10-15.md`. **Nguồn sự thật
duy nhất cho việc cần làm / đang chặn / đã xong.** Kế hoạch và quyết định ở `Plan.md`; kiến trúc ở `ARCHITECTURE.md`.

## ▶ 2026-10-28 — nháp Ch1–Ch4 đã có (`docs/khoa_luan/ch{1,2,3,4}/`), chờ người dùng đọc

- [ ] Người dùng đọc `ch1.md`–`ch4.md`; trả lời chỗ treo ở `nguon_ch{1,2,3,4}.md` mục 1 (mỗi tệp một bảng).
- [ ] ⛔ BPE 1.000: duyệt câu đề xuất ở Bảng 2.2 ("giá trị thử ban đầu, không dò; giữ để CTC head không đổi") hoặc cho lý do khác.
- [ ] ⛔ Duyệt/bác 10 sửa chữ của bản rà 23/10 đã áp trong văn Ch3 (`ch3/nguon_ch3.md` mục 3); sửa note gốc theo đó.
- [ ] Khoá trích dẫn chưa có PDF: Dai2019, Watanabe2018, NVIDIA2025, Ma2024, VietSuperSpeech, NIST1997, Ali2025, BenKish2025,
      Alabi2026, Laptev2023, Jung2022, Le2021 (+ nguồn shallow fusion cho (1.7)).
- [ ] Viết tiếp: Mở đầu → Kết luận → Tóm tắt. Hình 1.1, 2.1–2.3, 3.1: có PNG 29/10 (`reports/figures/{kien_truc,clean_test}/`) — người dùng xem; 1.1 giữ hay bỏ.
- [ ] Ch1: rà độc lập (do tác tử viết). ~~Kiểm ví dụ tách BPE *nghỉ ngơi*~~ xong 29/10: khớp tokenizer thật (`nguon_ch1.md` #5).
- [x] 29/10: CI Conformer PE + LM − Parakeet FT trong kênh = +1,17 [+0,27; +1,99] → văn Ch3 3.3.4, Ch4 4.2, hỏi đáp đã sửa
      thành "kém 1,2 điểm (CI…)"; Hình 3.1 forest plot `reports/figures/clean_test/fig_forest_cap_encoder.png`. Người dùng duyệt câu + hình.
- [ ] Slide (deck 20/10, giữ link): dựng lại theo sườn chốt 27/10 **sau khi** người dùng duyệt chương. Sửa: #17 "40 ms không phân
      biệt được cả ngoài kênh" (sai: ngoài kênh ConExt > B1); #16/#17 "audio dài chưa đo" (đã đo 3.4.4); #8 ghi chú quy +5,4 điểm cho
      PE (ablation khác 4 điểm); #5 tiêu đề "Ba cách đặt Mamba"; #6 lý do fp32 (NaN fp16, T4 không bf16), ký hiệu h_t → s_t; #11
      "kiểm trực tiếp Jiang" → "phù hợp", "~4 lần" → "3–4,5 lần"; #10 "ngang Parakeet"; "độ phân giải khung" → theo [CHỌN] 2.
      Thêm: ngoại suy độ dài, bên trong encoder (Δ, M3), ứng dụng + demo, ba điểm mới; forest plot → slide dự phòng.
- [ ] Docstring lỗi thời `src/models/conformer_relpos_encoder.py` dòng 15–16 (ablation thật là torchaudio) — người dùng quyết sửa.

## ▶ PHIÊN SAU BẮT ĐẦU Ở ĐÂY (đóng phiên 2026-10-27 ~23:50)

Tổng kết phiên 26–27/10: `Plan.md` mục 7, mục cuối "2026-10-27 — TỔNG KẾT PHIÊN". Trạng thái mọi bổ sung (bảng):
`docs/notes/review/danh_gia_bo_sung_2026-10-26.md` mục 4 "Trạng thái 27/10". **Không còn thí nghiệm nào đang chạy.**
Số liệu đóng băng **08/11** (`Plan.md` mục 2).

**Kết quả phiên (đã ghi note, đủ số):**
- A ngữ cảnh tiêu đề: (iii) tên file và (iv) tiêu đề có dấu + mô tả — B-WER giảm (8/9, 9/9 run) nhưng CI chứa 0 → không phân
  biệt được; WER không đổi. `docs/notes/ket_qua/bo_sung_A_ket_qua_2026-10-27.md`.
- B hộp đen Mamba: Δ nhỏ ở khung lặng (40 ms), không reset ở ranh giới từ, Conformer nhìn 2–3 s mọi lớp, Mamba hẹp dần.
  `bo_sung_B_ket_qua_2026-10-27.md`.
- Sản phẩm phiên âm podcast dài → phụ đề: P1 tự cắt +0,20 [+0,07; +0,37]; P3 AUROC 0,940; P2 1 giờ ≈ 2,6 phút CPU.
  `san_pham_podcast_ket_qua_2026-10-27.md`; hướng dẫn `docs/huong_dan_san_pham.md`; hình `reports/figures/san_pham/`.
- Audio dài nguyên chuỗi: cả ba encoder kém đi khi cửa sổ dài (mọi CI ≠ 0); Mamba vs Conformer không phân biệt được;
  Mamba RTF/RAM không đổi, Conformer 10–13 GB ở 5,5′. `audio_dai_nguyen_chuoi_ket_qua_2026-10-27.md`.
- Bản ghi Vietcetera: không công bố (`transcriptSSR` chỉ là id). `docs/notes/du_lieu/vietcetera_transcript_2026-10-27.md`.
- Viết/bảo vệ: thông điệp mở đầu + sườn slide CHỐT (`Plan.md` mục 4); mục lục cập nhật (`docs/khoa_luan/muc_luc_de_xuat.md`
  mục 5b, ~64,5 tr); khung 3.4.4/3.6/4.3 (`docs/khoa_luan/ch3_khung_bo_sung_2026-10-27.md`, số cuối); bảng hỏi đáp 24 câu
  (`docs/khoa_luan/hoi_dap_bao_ve.md`).

**Người dùng (chặn văn xuôi / quyết định):**
- [ ] Bài tập văn phong Phần B: ~6 đề (2, 5, 8, 10, 14, 15) — `docs/khoa_luan/bai_tap_van_phong.md` từ dòng 152.
- [ ] 5 mục [CHỌN]/[GVHD] cuối `muc_luc_de_xuat.md` + chọn Hình 3.7 (`reports/figures/mamba_blackbox/fig_a_*` hay `fig_b_*`).
- [ ] Điền bảng hỏi đáp: Câu 10 (dùng AI), Câu 12 (lý do BPE1000), Câu 13 (Colab → Kaggle).
- [ ] ⛔ Duyệt **commit** (≈ 2 ngày việc chưa commit, danh sách dưới) — AI quét khoá + username Kaggle trước khi push.
- [ ] ⛔ Gặp thầy + bản 1–2 trang: sửa câu "kém 0,6B chỉ 1–3,5 điểm" (báo cáo 17/10) bằng số GS2; báo A/B/sản phẩm/audio
      dài; hỏi tên đề tài (đơn 08/10) + đề cương mới + "Con-Former"/"Conformer". AI soạn được ngay khi người dùng bảo.
- [ ] ⛔ Gộp demo (người dùng "tạm để y"): gộp trọn (AI khuyên) · chỉ gộp lõi · giữ nguyên.

**AI làm được ngay, 0 GPU (đánh giá cuối phiên: KHÔNG cần thêm thí nghiệm nào; còn thiếu là phần cho quyển + thủ tục):**
1. [x] CI chênh trong kênh Conformer PE + LM vs Parakeet tinh chỉnh — xong 29/10: +1,17 [+0,27; +1,99] (`ch3/nguon_ch3.md` #6).
       Còn: thông điệp 3 trong `Plan.md` mục 4 và slide vẫn ghi "~1,2 điểm" không CI — sửa khi dựng lại deck.
2. [x] 29/10: Hình 3.1, 2.1–2.3, 1.1 có PNG; Bảng 4.2 và dòng CPU Bảng 4.1 đã có trong nháp Ch4 (kiểm khớp note). Dòng cũ: Hình 2.1–2.3 kiến
       trúc (vẽ bản in); Hình 1.1 CTC (vẽ hoặc bỏ); dựng Bảng 4.2 (P1–P3 + A); cập nhật dòng CPU của Bảng 4.1 bằng số P2.
3. [x] 29/10 lượt 3: không thấy công bố phân tích Δ/attention ẩn trên encoder ASR Mamba; nguồn liên quan (LongMamba ICLR 2025) ở `ch3/nguon_ch3.md` mục 5. Dòng cũ: Quét nguồn thêm (~15′) trước khi viết 3.6: phân tích Δ / attention ẩn trên encoder ASR Mamba (mới 2 lượt).
4. [ ] Slide: dựng lại theo sườn đã chốt (artifact cũ 20/10, 14 slide) + hình mới — sau Ch3/Ch4 hoặc trước buổi gặp thầy.
5. [ ] Viết văn xuôi theo khung — sau bài tập văn phong.

**Tuỳ chọn (người dùng quyết, 0 GPU):** người hiệu đính thứ hai cho ~30 câu clean-test để đo độ đồng thuận (gỡ câu hỏi
"một mình em sửa nhãn"; tốn ~1–2 giờ của một người khác; vẫn chỉ để báo cáo). Không khuyên: thêm fine-tune, dấu câu phụ
đề, sửa luật tự cắt, đẩy demo công khai.

**Chưa commit (trước khi push quét khoá + username Kaggle):** code `src/evaluation/{bias_title,lm_decode,long_audio,
long_audio_score}.py`, `src/demo/{app,longform}.py`, `src/analysis/`, `scripts/fig_{mamba_m3,san_pham_quy_trinh,pareto_cpu,
audio_dai}.py`, `requirements.txt`, `ARCHITECTURE.md`, `TODO.md`, `Plan.md`; note `docs/notes/ket_qua/{bo_sung_*,san_pham_podcast_*,
audio_dai_nguyen_chuoi_*}`, `docs/notes/du_lieu/vietcetera_transcript_2026-10-27.md`, `docs/huong_dan_san_pham.md`; kết quả
`reports/results/{bias_title,mamba_blackbox,longform,long_audio}_2026-10-27/` (long_audio `hyps/` ~10 MB — xem có nên track);
hình `reports/figures/{mamba_blackbox,san_pham,audio_dai}/`. **Không commit:** `data/processed/*` (trang Vietcetera, chuỗi
audio, demo, ảnh chụp có thanh dấu trang); `docs/khoa_luan/`, `docs/notes/review/` (đã gitignore).

**Treo cũ (chưa đổi):** (C) thêm mô hình fine-tune — AI: không. Chi tiết các khối cũ dưới đây.

**(Đã xong — giữ để tra) Kế hoạch nương theo từ 26/10** (người dùng chọn A + B; 0 GPU, không train; ý tưởng + nguồn ở `Plan.md` 26/10).

1. [ ] **A — Tiêu đề tập podcast làm ngữ cảnh cho tên riêng/từ ngoại** (~1,5–2 ngày, CPU).
   - Trước khi chạy, ghi quy tắc vào note mới `docs/notes/ket_qua/`: danh sách thưởng = từ trong tiêu đề tập; điểm thưởng
     chốt trước (không chọn trên clean-test); đo WER, B-WER, U-WER (Le 2021); 9 run 40 ms, beam/LM giữ cấu hình đã khoá;
     bootstrap theo video; lợi nhỏ vẫn báo nguyên.
   - **Người dùng:** chép 29 tiêu đề có dấu của 29 video giữ riêng (~15′).
   - Code: mở rộng `src/evaluation/lm_decode.py` nhận danh sách từ theo video (thêm vào từ điển + điểm thưởng trong LM);
     log-prob 40 ms: `checkpoints/v2a_s4_seeds/logprobs/` (seed 43/44), `D:/Model/kltn_checkpoints/logprobs_v2a_s4` (seed 42).
   - Demo: thêm ô "tiêu đề tập".
2. [ ] **B — Mở hộp đen Mamba** (~1–1,5 ngày, CPU).
   - Quy tắc chốt trước: Δ trung bình quanh ranh giới từ (căn chỉnh CTC) so với giữa từ; Δ ở từ đệm/lặng so với từ thường;
     khoảng cách trung bình của attention ẩn (Ali 2025) và attention Conformer theo lớp, 10 ms và 40 ms. Câu chọn theo quy
     tắc cố định (vd. ngẫu nhiên seed 42 từ val_unseen), không chọn theo hình.
   - Code: lấy Δ, B, C từ `src/models/mamba_ref.py`; attention từ khối MHSA Conformer; hình cho Ch2 + slide.
3. [ ] **Viết (đường găng):** thứ tự khối 23/10 (bài tập văn phong → 2.2.1 → Ch2); Ch3 từ `ch3_khung.md`.

Treo, chờ người dùng (chưa làm):
- [ ] ⛔ Gặp thầy tuần này + bản 1–2 trang: bảng trong/ngoài kênh, 5 nhận xét đánh đổi, 3 điểm mới, cập nhật câu "kém 0,6B
      chỉ 1–3,5 điểm" của báo cáo 17/10 bằng số GS2 (ngoài kênh 8–16 điểm). Tác tử A khuyên làm trước mọi bổ sung.
- [ ] ⛔ (1) WER audio dài · (2) công cụ phiên âm dài + SRT · (3) tốc độ CPU/Pareto · (C) độ tin cậy + tô màu · thêm mô hình
      fine-tune.
- Tác tử B/C/D dừng 26/10 trước khi báo cáo (hết token): chưa có tổng quan audio dài, số đo CPU, phản biện.

## ▷ Khối 2026-10-23

Chi tiết: `Plan.md` mục 7, "2026-10-23".

- [ ] **Người dùng:** viết tay đề ở `docs/khoa_luan/bai_tap_van_phong.md` Phần B (≥ 5–6 đề, đủ các loại đoạn) → AI rút
      giọng, ghi skill `van-phong-khoa-luan` mục 1/10, viết lại để so → viết 2.2.1 → trọn Ch2.
- [ ] **Người dùng:** 5 mục [CHỌN]/[GVHD] cuối `docs/khoa_luan/muc_luc_de_xuat.md` → AI sửa "Phần NỘI DUNG" đề cương mới cho khớp.
- [ ] **Hỏi người dùng:** áp các sửa chữ của bản rà phương pháp 23/10 (GS2 B1 − ConExt; mô tả CI hai tầng; câu nhân quả
      "không lấy mẫu xuống"; sd seed; "kiểm trực tiếp Jiang"; khai lmw/ws dò trên val_unseen; GPU-giờ là giờ tường).
- [ ] Mẫu thử 2.2.1: kiểm ô [CẦN KIỂM] (Jiang dùng chung phép chiếu vào–ra?) bằng hình 2 bài gốc.

## ▷ Khối 2026-10-20

20/10: GS2 xong toàn bộ; khung Ch3 cập nhật + trích dẫn đã kiểm; bản tổng hợp kết quả + phân tích đánh đổi + bộ slide
phác (artifact riêng tư https://claude.ai/artifact/XTrMQr4CvYq1WDVwDYUmcQ). Chi tiết: `Plan.md` mục 7, "2026-10-20 (đóng phiên)".

- [ ] **Người dùng:** điền cột Chốt `docs/notes/ket_qua/tong_hop_ket_qua_2026-10-20.md` (6 dòng [CHỌN]) + mục 5 `danh_gia_danh_doi_2026-10-20.md`.
- [ ] **Người dùng:** giữ/bỏ giới hạn "clean-test đã được nhìn trong giai đoạn thiết kế" (`ch3_dulieu_phuongphap_nhap.md` mục 6).
- [ ] **Slide:** người dùng xem slide tham khảo (Hung-yi Lee Mamba, sigport Conformer…, `Plan.md`) rồi chốt hướng; 17 slide ~19–20′ (Mẫu 3: 15′, tối đa 20′).
- [ ] Hình 3.2 forest plot PNG cho Ch3 (0 GPU).
- [ ] ⛔ Thêm mô hình fine-tune? (AI: không; người dùng chưa chốt). GPU A khoá; đóng băng GPU 25/10.

## ▷ Khối 2026-10-17 (phần lớn đã xong)

17/10 người dùng chốt 8 mục treo (`Plan.md` mục 4, dòng 17/10); **đóng băng GPU giữ 25/10**. Đã làm: Hình 3.5 vẽ lại
(40 ms tb 3 seed), khuyến nghị Ch4 viết lại (`ket_qua_cuoi_khung` cuối mục 3), gitignore `data/Test_micro/`, ghi duyệt
quy tắc âm tiết + `sticky_words` vào note rq3. Dashboard GVHD đã có hình cho Bảng 3.1–3.3 (forest plot cặp), 3.4a/b,
3.6, 3.8 — số khớp chấm chung; **chưa cập nhật Hình 3.5 mới lên dashboard**.

- [x] **A — so với pre-train trên clean-test (19/10)**: `python -m src.evaluation.compare_pretrained`; bảng ở `docs/notes/ket_qua/so_sanh_hien_hanh_2026-10-19.md` mục A.
- [x] **B — eval GigaSpeech 2 vi test (xong 20/10):** đủ mọi hệ thống; `score_gs2` → `gs2_scored_2026-10-19/`; note `so_sanh_hien_hanh` B.3c; khung Ch3 3.3.6 + Bảng 3.5b + 3.5.2. Parakeet zs 12,66% (E). Quyết định thêm mô hình fine-tune: người dùng chờ số Parakeet (đã có) — hỏi lại.
- ⛔ **GPU tài khoản A tạm khoá (người dùng 19/10)** — không đẩy kernel GPU lên A cho tới khi người dùng mở lại.
- [x] **Trùng nội dung clean-test ↔ train (19/10)**: 11/203 câu — người dùng chốt bỏ qua, giữ 203 câu, không khai báo.
- [ ] Việc tay: gửi đề cương mới + xin lịch GVHD 19–25/10 (gấp nhất).
- [x] Hình WER val_unseen theo GPU-giờ (gộp hình 1 + 2 khung Ch3) — `scripts/fig_wer_gpu_hours.py`, note RQ2 cuối.
- [x] Kiểm định dấu theo video — `src/evaluation/video_sign_test.py`; đồng thuận bootstrap mọi cặp (note clean-test cuối).
- [x] L3 "nhại" — `src/evaluation/label_echo.py`, note rq3 mục 6 (học nhãn giả nhại 15–24% vs đối chứng 9–13%).
- [ ] Câu "lần đầu đo trên tiếng Việt" (chỉ có trong review/khung, chưa vào đề cương): khi viết Ch1/Kết luận dùng "theo phạm vi tra cứu" hoặc quét VLSP/ACL 1–2 h.
      26/10 quét nhanh (~20′, 6 lượt Anh + Việt): không thấy công bố Mamba cho ASR tiếng Việt; Zevallos 2025 (9 ngôn ngữ, kiểm
      PDF) và MLMA arXiv:2510.18684 (6 ngôn ngữ châu Âu) không có tiếng Việt; bài VietSuperSpeech arXiv:2603.01894 (HTML v1)
      không báo WER nào, kiểm nhãn tay không có cỡ mẫu/tỉ lệ lỗi. Chưa quét kỷ yếu VLSP → giữ "theo phạm vi tra cứu".
- [x] Nhận xét sd seed relPE ~3× Mamba — note clean-test bổ sung 17/10.
- [ ] Commit (chờ người dùng bảo): gồm `error_analysis/`, seed, Plan/TODO, note, script; 20/10 thêm `eval_gs2.py`, `decode_cpu.py`, `launch_cpu.py`, `score_gs2.py` + output `gs2_eval_2026-10-19/`, `gs2_scored_2026-10-19/` (rà username/dữ liệu gated trước khi add).
- [ ] **Khung Ch3 đã soạn** `docs/khoa_luan/ch3_khung.md` (17/10, gitignore): người dùng rà văn phong + 3 chỗ [CHỌN]/[CẦN KIỂM] → chuyển Word.
- [ ] Hình cho Word: hình RQ1 trên dashboard là d3 → cần bản PNG (script matplotlib từ cùng CSV, hoặc chụp màn hình).

## ▷ Khối 2026-10-16 ~18:20 (mục 2–3 đã chốt 17/10)

1. **L3 xong, đã chấm chung** (16/10 17:11–18:10): 6/6 run seed sạch; kernel `dump-seeds-v2a-s4` (A, 2,6′); beam+LM;
   `reports/results/clean_test_scored_2026-10-16_chung/`. **40 ms: cả 3 cặp "không phân biệt được ở 3 seed"** (greedy và
   beam+LM); relPE seed 42 là run tốt nhất trong 9 run. Số + nhận xét: `clean_test_results_2026-10-15.md` mục "Lần chấm
   chung 2026-10-16". **Báo người dùng/GVHD** — đổi KL1 40 ms so với bản seed 42.
2. **Cần người dùng xem lại** (không tự sửa): khuyến nghị Ch4 "câu ngắn → Conformer relPE 40 ms" (dựa WER seed 42) —
   ghi chú ở `ket_qua_cuoi_khung` KL1; đề cương mới + 4 Google Doc trên Drive còn câu "relPE 40 ms tốt hơn" nếu có.
3. **16/10 tối xong (2 agent song song, đã kiểm lại):** Bảng 3.6 (40 ms tb ± sd 3 seed + CSV từng seed) và **Hình 3.4
   1×2** (người dùng yêu cầu; `scripts/rq2_final_tables.py`, Hình 3.5 chỉ vẽ khi `--fig35`, file giữ nguyên byte — người
   dùng "tạm để"); **Bảng 3.8** phân rã lỗi (`src/evaluation/error_breakdown.py` → `reports/results/error_analysis/
   clean_test_2026-10-16/`, note `rq3_phan_ra_loi_2026-10-16.md`; khớp WER bảng chính 29/29 hệ thống, lệch 0).
   `score_clean_test.py` tách `load_hypotheses()` — kết quả trùng byte bản `_chung`. **Chờ người dùng duyệt:** (a) quy
   tắc tách âm tiết (note rq3 mục 1, có 3 chỉnh sửa phát sinh khi assert, trước khi in số); (b) số từ dính ep28: định
   nghĩa trong repo `sticky_words` cho 444/190/153/91/8, khác note `glued_words` 15/10 (387/159/137/86/5, script ngoài
   repo) — chọn bộ nào; (c) commit `reports/results/error_analysis/` không; (d) các câu agent sửa thêm trong
   `rq2_ket_qua_cuoi_2026-10-16.md` (điểm 5 "≤ 21%", dòng hạn chế).
4. **Sao lưu checkpoint:** staging `D:/Model/kltn_drive_backup/checkpoints_kltn_2026-10-16/` (39 file, 3,4 GB: Parakeet FT + 13 ep37
   bảng chính + 7 phụ lục v1 + LM + config, README, SHA256SUMS); thư mục Drive `checkpoints_kltn_2026-10-16` đã tạo —
   **người dùng kéo thả nội dung vào**, rồi kiểm đủ 39 file. Parakeet FT đã tải về 16/10 (bản gốc `D:/Model/kltn_checkpoints/parakeet_ft/`).
4b. **Demo + dashboard (16/10 tối):** demo 7 mô hình ep37 + beam+LM chạy (`python -m src.demo.app`, mở ẩn danh vì IDM);
   dashboard GVHD https://claude.ai/artifact/N9y2mpE5kgL15XNcvcAL3S (người dùng bật Share). `data/Test_micro/*.txt|md` chưa
   gitignore — hỏi người dùng trước khi commit.
5. Kế tiếp không cần GPU: L6 (RQ3, `error_analysis.py` trên bản hiệu đính, gồm 6 run seed), L7, khung Ch3. Sao lưu
   checkpoint ep37 mọi run lên Drive (seed: `checkpoints/v2a_s4_seeds/dataset/`, 6 × ~56 MB; log-prob ~130 MB/run ở
   `checkpoints/v2a_s4_seeds/logprobs/`).
6. Việc tay người dùng: đề cương mới → Word → GVHD + xin lịch 19–25/10; duyệt mục lục mới; kéo thả PDF lên Drive.
7. Chưa commit: kết quả seed (`reports/results/train_full/v2a-s4-*/seed4*/`, `decode_v2a_s4_seeds/`,
   `clean_test_scored_2026-10-16_chung/`), `score_clean_test.py` (OUT), 2 note, `Plan.md`, `TODO.md`.

## ▷ Khối cũ (ghi 2026-10-15 khuya — phần lớn đã xong 16/10)

1. Đọc cùng người dùng: `review_fable_2026-10-15.md` (phương án), `review_fable_scorecard_2026-10-15.md` (điểm +
   việc kế tiếp), `ket_qua_cuoi_khung_2026-10-15.md` (bảng/kết luận viết sẵn). 12 mục ⛔ đã chốt 16/10.
2. Sau khi chốt: L1 (tải output, 0 GPU) → L2 (kernel dump log-prob sau reset quota 17/10) → L3 (6 run seed) theo
   mục 🟡. Mỗi lần đẩy kernel: báo ngay, ghi `Plan.md`.
3. Việc tay song song (không chờ GPU): đề cương mới → Word → gửi GVHD + xin lịch tuần 19–25/10; dựng khung Ch3.
4. Chưa commit: `Plan.md`, `TODO.md`, `CLAUDE.md`, `configs/scoring/`, `src/evaluation/score_clean_test.py`,
   `src/evaluation/vi_syllable.py`, `src/data/word_tags.py`, `src/data/correct_clean_test.py`, manifest clean-test,
   `reports/results/clean_test_scored_2026-10-15/`, `scripts/kaggle/{dump_logprobs,whisper_advisor}/`, các note mới
   (`clean_test_entities`, `clean_test_results`, `de_cuong_moi_nhap`, `glued_words`, `phan_bien`, `prompt_review_fable`,
   `review_fable*`, `ket_qua_cuoi_khung`, 2 file lưu trữ). Rà repo public trước khi commit (hỏi người dùng).

## ⛔ Chờ người dùng chốt

**12 quyết định sau rà soát: ĐÃ CHỐT 2026-10-16** theo khuyến nghị review mục 4 (ghi `Plan.md` mục 4). Khác review:
tiêu chí 3 seed = bootstrap hai tầng (`conextbimamba_gap_2026-10-12.md` mục **9.6**, thay điều 4a/5 của review 2.3);
10 ms giữ 1 seed, không cổng seed. #11 chốt 16/10: `canon_tone` (vị trí dấu thanh) trong scorer, áp ở lần chấm chung.

Mục mở cũ còn giữ: ngày gặp GVHD gần nhất/kế tiếp (ghi `Plan.md` mục 6); tên đề tài trên bìa (`Conformer` hay
`Con-Former`); hình thức nộp (biểu mẫu 2018); dạng "thiết kế" Ch2; bản `.docx` đề cương nào là bản đã nộp (Q4 cũ).

## 🟡 Sẵn sàng làm ngay (theo thứ tự; chi tiết review mục 2.2)

- [x] **L1 — tải output còn thiếu (0 GPU-h).** ✅ 16/10: B relPE 10 ms s3 COMPLETE, 38/38, returncode 0, 0 nonfinite,
      19,2 GPU-h, encoder 12,73M; file nhỏ ở `reports/results/train_full/v2a-conformer-relpos/s3/`; `epoch37.pt` có trên Kaggle
      (cột size của `kaggle kernels files` sai, chưa kiểm dung lượng). **Còn:** kiểm 3 nhánh 10 ms còn `epoch37.pt`. Bản cũ: B `train-full-v2a-conformer-relpos-s3` (10 ms, đẩy 14/10 01:10): kiểm
      status, `session_summary.json` (returncode 0, `latest.epoch` = 37, `finished`), tải file nhỏ
      `kaggle kernels output <kernel> -p reports/results/train_full/v2a-conformer-relpos/s3 --file-pattern '.*\.(json|jsonl|log|csv)$'`
      (tài khoản B: đặt `KAGGLE_CONFIG_DIR` trỏ tới thư mục khoá của B, ngoài repo). Kiểm 3 nhánh 10 ms (`v2a-mamba` B, `v2a-conextbimamba` A,
      `v2a-conformer` A) còn `latest.pt`/`epochs/epoch37.pt` trong output. **Không in WER relPE** cho tới khi quyết định
      #6 chốt. Nếu output B mất → báo người dùng (review 5.2: chạy lại ~18 GPU-h chỉ nếu xong trước 25/10).
- [x] **L2 — kernel dump log-prob (~0,3 GPU-h, sau reset 17/10).** ✅ 16/10 xong (5,5′; số ở `clean_test_results` bổ sung 16/10) ▶ 16/10 08:34 đẩy A `dump-greedy-v2a-10ms`
      (`scripts/kaggle/dump_greedy_10ms/`, dataset `kltn-v2a-10ms-ep37`; chỉ greedy, không log-prob — 10 ms quá lớn) →
      tải `greedy.json` vào `reports/results/dump_greedy_10ms_2026-10-16/` → `python -m src.evaluation.score_clean_test`
      (đã thêm 5 hệ thống, ghi `clean_test_scored_2026-10-16/`). Bản cũ: Mở rộng `scripts/kaggle/dump_logprobs/` cho 4
      checkpoint ep37 10 ms (relPE, B1, ConExt, Conformer không PE) trên clean-test + nửa dev val_unseen; cùng kernel
      giải mã Parakeet fine-tune `best.pt` (2,27 GB, output `finetune-parakeet-main` A) trên clean-test. Rồi
      `python -m src.evaluation.score_clean_test` (bổ sung hệ thống mới) → cập nhật `clean_test_results_2026-10-15.md`.
- [x] **L3 — 6 run seed 40 ms (~32 GPU-h).** ✅ 16/10 xong: 6/6 sạch, GPU-h ConExt 4,93/4,88 · relPE 4,93/5,28 · B1 5,98/5,80; chấm chung 18:10. ▶ 16/10 08:36 đẩy B relPE seed43/44 + C B1 seed43/44 (`launch.py --tag
      seed<N> --commit 0090bd75 --extra --seed <N> --experiment_name v2a_s4_<model>_seed<N>`); A ConExt ×2 đẩy 08:49 → **6/6 đang chạy**. Tiêu chí đã ghi note gap 9.6. Bản cũ: Trước khi đẩy: ghi tiêu chí review 2.3 vào
      `conextbimamba_gap_2026-10-12.md` mục 9.6 (người dùng duyệt). Config `configs/v2a/model_*_sub4.yaml` y nguyên,
      chỉ `--seed 43` / `44` + `experiment_name` riêng (kiểm `launch.py --help`: `--tag`, `--extra`); ep37; cuối kernel
      dump log-prob clean-test + nửa dev. Phân bổ: A ConExt ×2 (≈4,9 h/run) · B relPE ×2 (≈5,3) · C B1 ×2 (≈6,0);
      2 kernel song song mỗi tài khoản. Theo dõi 10′ đầu; kẹt cấp máy → huỷ, đẩy lại cùng seed. Tải output nhỏ vào
      `reports/results/train_full/v2a-s4-*/seed43|44/` ngay khi xong.
- [x] **RQ2 hoàn chỉnh (16/10):** Bảng 3.6, 3.7, điểm giao, điều kiện số học, Hình 3.4, 3.5 — `scripts/rq2_final_tables.py`,
      note `docs/notes/ket_qua/rq2_ket_qua_cuoi_2026-10-16.md`. Chạy lại khi seed về (GPU-h, WER 40 ms trung bình 3 seed).
- [x] **Quy trình chấm seed** ✅ 16/10 chạy đủ 5 bước (OUT → `clean_test_scored_2026-10-16_chung/`). Bản gốc (viết sẵn 16/10):
      1. `python scripts/kaggle/train_full/collect_seeds.py` — trạng thái, tải file nhỏ, kiểm toàn vẹn (không in WER).
      2. `... collect_seeds.py --ckpt --push` — tải 6 `epoch37.pt` vào `checkpoints/v2a_s4_seeds/`, tạo dataset
         `<A>/kltn-v2a-s4-seeds-ep37` (đợi status `ready`).
      3. `kaggle kernels push -p scripts/kaggle/dump_seeds_40ms -t 3600` (A, ~10′ GPU) → tải output (npz clean-test
         ~130 MB/run) vào `checkpoints/v2a_s4_seeds/logprobs/`; đối chiếu `val_unseen_check` với eval lúc train.
      4. `python -m src.evaluation.lm_decode decode --npz v2a_<m>_seed<N>=<lp>.npz ... --out
         reports/results/decode_v2a_s4_seeds/cleantest_hyps.tsv` (LM ở `checkpoints/lm/`, dựng lại bằng `build --text`).
      5. **Lần chấm chung duy nhất:** sửa `OUT` trong `score_clean_test.py` sang thư mục ngày chấm →
         `python -m src.evaluation.score_clean_test` (áp `canon_tone`; in bootstrap hai tầng `paired_seeds`).
      Đã kiểm 16/10: `lm_decode` tái lập 203/203 câu × 6 cột của 15/10; đường hai tầng với seed giả (nhân bản seed 42)
      cho sd 0, CI trùng bootstrap cặp.
- [x] **L4 — giải mã beam + LM (CPU)** ✅ 16/10 đủ 9 run (seed 42 từ 15/10, seed 43/44 bằng `src.evaluation.lm_decode`); bảng 3.4 trong note clean-test. Bản cũ: cho 9 run 40 ms, lmw 0,8 / ws 1,0 giữ nguyên; bootstrap; bảng 3.2–3.4.
      Code: `D:/Model/kltn_decoding_2026-10-15/decoding/` (ngoài repo — cân nhắc đưa phần dùng lại vào `src/evaluation/`).
- [x] ~~**L5 — gõ mù 20 câu**~~ **BỎ (người dùng 16/10)** — chỉ khai hạn chế neo nhãn bằng số đã có (`Plan.md` mục 4). Bản cũ: (người dùng, ~1 h): chọn ngẫu nhiên seed 42 trải đều video; so với bản hiệu đính → số
      "neo nhãn" cho Bảng 3.9.
- [ ] **L6 — RQ3 (0 GPU):** chạy lại `src/evaluation/error_analysis.py` trên bản hiệu đính cho mọi hệ thống; thêm
      căn chỉnh 3 chuỗi (nhãn giả / hiệu đính / dự đoán) cho L3 "nhại"; phân rã theo âm tiết; giới hạn 3 ngày.
- [ ] **L7 — bảng điều kiện số học RQ2, tham số encoder + tổng, gọi đúng tên OOM relPE** khi viết Ch3. 20/10: phần cấu hình + tham số đã vào Bảng 3.0 `ch3_khung.md`; còn điều kiện số học RQ2.
- [x] **Chấm một lần duy nhất** ✅ 16/10 → `clean_test_scored_2026-10-16_chung/` (bảng khoá; còn sao lưu checkpoint lên Drive). Bản cũ: khi đủ 9 run 40 ms + 4 checkpoint 10 ms + Parakeet FT (và sau #11); khoá bảng; sao
      lưu checkpoint ep37 mọi run lên Drive (cần cho CD `SETUP`).

Checklist kiểm mỗi kernel vừa xong: `kaggle kernels status` → tải file nhỏ → `session_summary.json` (returncode 0,
epoch/step tăng, commit đúng) → `metrics.jsonl` (`nonfinite_loss_steps` ≈ 0, WER val giảm, `wall_seconds_total`) →
log không có `!!!`/`LỖI` → báo người dùng + quota còn. Lỗi **log rỗng, quota không tăng** → đẩy lại cùng phiên;
lỗi **sau khi đã train** → đẩy phiên kế (resume), không đẩy lại cùng phiên.

## 📘 Khóa luận (bản viết) — mốc ở `Plan.md` mục 2, quy cách mục 6

- [ ] **Đề cương mới** (`docs/notes/de_cuong_gvhd/de_cuong_moi_nhap.md` → sửa RQ theo review 2.1 sau khi chốt #1 → Word Mẫu 4) →
      gửi GVHD + xin lịch gặp tuần 19–25/10. Không chờ số seed.
- [ ] **Khung Ch3 trong Word ngay** theo `ket_qua_cuoi_khung_2026-10-15.md`: 9 bảng, 5 hình để trống có tên/cột/chú
      thích; mục Hạn chế chép danh sách sẵn.
- [ ] Ch3 viết thật từ tuần 11 (số đã khoá) → Ch2 từ `docs/khoa_luan/ch2_nhap.md` + `muc_luc_de_xuat.md` → sửa Ch1
      theo `docs/khoa_luan/ch1_kiem_trich_dan.md` ([10] ghép số hai cấu hình phải sửa) → Ch4 demo (thêm 3 mô hình
      40 ms; Conformer relPE chạy CPU, Mamba qua `mamba_ref.py`) → Mở đầu, Kết luận, TLTK (đối chiếu từng nguồn), Phụ lục.
- [ ] Mốc: nháp Ch3 15/11 · bản đủ gửi GVHD 27/11 · nộp 30/11–06/12. Gặp GVHD ≤ 2 tuần/lần, ghi bảng `Plan.md` mục 6.
- [ ] Việc tay cũ còn mở: sao lưu `docs/khoa_luan/*.docx` lên Drive; điền bìa (Bộ môn, Khóa, logo); TLTK ghi dần.

## ⚠️ Rủi ro / repo public (không phải việc làm ngay)

- `notebooks/02_dataset_eda.ipynb` nhúng audio YouTube trong repo public và lịch sử git (người dùng chọn bỏ qua
  2026-09-19); file `.docx` và file TLCN có tên+MSSV người thứ ba còn trong lịch sử git — chỉ viết lại lịch sử khi
  người dùng yêu cầu rõ.
- `docs/de_cuong/De_Cuong_Chi_Tiet_Mamba_ASR.docx` trên đĩa khác bản đã commit — không sửa, xác nhận bản nào đã nộp.
- Kernel kẹt cấp máy Kaggle đã xảy ra 3 lần (05–06/10, 12/10, 13/10): huỷ → quota hoàn.
- Dữ liệu pre-train Parakeet có thể trùng nguồn VietSuperSpeech (card không nêu) — chỉ khai, không kiểm được.

## ✅ Đã xong (mốc chính; chi tiết ở file lưu trữ và `Plan.md` mục 4)

- Dữ liệu: A1 lọc ngôn ngữ, 29 video giữ riêng, 5 Kaggle Dataset, tokenizer BPE1000, CMVN (09–10/2026).
- Pipeline DDP/AMP/resume/rẽ nhánh chạy thật trên Kaggle; khối Mamba fp32; bucketing; speed perturbation.
- Train: v1 30 epoch ×3 (+ Conformer dài 69 ep) → nhánh pha giảm 10 ms ×3 (ep37) → 40 ms ×3 seed 42 (ep37);
  Conformer relPE 10 ms (B, output chưa tải — L1). Parakeet fine-tune 7 GPU-h (val_unseen nhãn giả 12,6 → 8,0%).
- Đánh giá: zero-shot Parakeet/PhoWhisper; trung bình checkpoint v1; RQ2 pilot + RQ2 cuối (7 cấu hình + Parakeet);
  benchmark subsampling/fp16; hiệu đính clean-test 203/203 + bảng loại từ 241/241; quy tắc chấm + chấm 15/10; bảng
  giải mã beam + LM 40 ms (nửa dev + clean-test); chẩn đoán từ dính; phản biện thử 14/10; rà soát Fable 15/10.
- Demo Gradio CPU (Conformer v1 + B1 v1).
