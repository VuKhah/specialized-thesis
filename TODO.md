# TODO — Mamba vs Conformer ASR tiếng Việt

Viết lại gọn **2026-10-15** sau bản rà soát `docs/notes/review/review_fable_2026-10-15.md`. Bản cũ (lịch sử train full,
checklist kiểm kernel, lệnh đẩy từng phiên) lưu nguyên văn ở `docs/notes/luu_tru/todo_luu_tru_2026-10-15.md`. **Nguồn sự thật
duy nhất cho việc cần làm / đang chặn / đã xong.** Kế hoạch và quyết định ở `Plan.md`; kiến trúc ở `ARCHITECTURE.md`.

## ▶ PHIÊN SAU BẮT ĐẦU Ở ĐÂY (ghi 2026-10-17)

17/10 người dùng chốt 8 mục treo (`Plan.md` mục 4, dòng 17/10); **đóng băng GPU giữ 25/10**. Đã làm: Hình 3.5 vẽ lại
(40 ms tb 3 seed), khuyến nghị Ch4 viết lại (`ket_qua_cuoi_khung` cuối mục 3), gitignore `data/Test_micro/`, ghi duyệt
quy tắc âm tiết + `sticky_words` vào note rq3. Dashboard GVHD đã có hình cho Bảng 3.1–3.3 (forest plot cặp), 3.4a/b,
3.6, 3.8 — số khớp chấm chung; **chưa cập nhật Hình 3.5 mới lên dashboard**.

- [x] **A — so với pre-train trên clean-test (19/10)**: `python -m src.evaluation.compare_pretrained`; bảng ở `docs/notes/ket_qua/so_sanh_hien_hanh_2026-10-19.md` mục A.
- [ ] **B — eval GigaSpeech 2 vi test (chốt 19/10, trước 25/10):** đã có greedy 10/40 ms + Parakeet FT + beam+LM cố định (`so_sanh_hien_hanh` B.3/B.3b). Còn: tải output A `eval-gs2-40ms-dev`, D `eval-gs2-parakeet-zs`, D `eval-gs2-phowhisper` (đang chạy/xếp hàng 19/10 18:45) → dò LM trên dev bằng CPU B/C/D (`scripts/kaggle/decode_gs2_cpu/launch_cpu.py`) → final test → `score_gs2` (cột `_lmtuned`) → cập nhật note. ⛔ người dùng: thêm 1–2 mô hình fine-tune?
- [x] **Trùng nội dung clean-test ↔ train (19/10)**: 11/203 câu — người dùng chốt bỏ qua, giữ 203 câu, không khai báo.
- [ ] Việc tay: gửi đề cương mới + xin lịch GVHD 19–25/10 (gấp nhất).
- [x] Hình WER val_unseen theo GPU-giờ (gộp hình 1 + 2 khung Ch3) — `scripts/fig_wer_gpu_hours.py`, note RQ2 cuối.
- [x] Kiểm định dấu theo video — `src/evaluation/video_sign_test.py`; đồng thuận bootstrap mọi cặp (note clean-test cuối).
- [x] L3 "nhại" — `src/evaluation/label_echo.py`, note rq3 mục 6 (học nhãn giả nhại 15–24% vs đối chứng 9–13%).
- [ ] Câu "lần đầu đo trên tiếng Việt" (chỉ có trong review/khung, chưa vào đề cương): khi viết Ch1/Kết luận dùng "theo phạm vi tra cứu" hoặc quét VLSP/ACL 1–2 h.
- [x] Nhận xét sd seed relPE ~3× Mamba — note clean-test bổ sung 17/10.
- [ ] Commit (chờ người dùng bảo): gồm `error_analysis/`, seed, Plan/TODO, note, script.
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
- [ ] **L7 — bảng điều kiện số học RQ2, tham số encoder + tổng, gọi đúng tên OOM relPE** khi viết Ch3.
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
