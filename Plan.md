# Plan.md — Kế hoạch, trạng thái & quyết định

Cập nhật lần cuối: **2026-10-15** (dọn lại mục 1–6 sau bản rà soát độc lập
`docs/notes/review/review_fable_2026-10-15.md`; nhật ký trước 2026-10-11 chuyển nguyên
văn sang `docs/notes/luu_tru/plan_nhat_ky_luu_tru_den_2026-10-08.md`). Đây là nơi **nối
tiếp giữa các phiên**: kế hoạch theo tuần, trạng thái, quyết định đã chốt/đang
mở, nhật ký phiên. Việc chi tiết ở [`TODO.md`](TODO.md); kiến trúc ở
[`ARCHITECTURE.md`](ARCHITECTURE.md); luật cho AI ở [`CLAUDE.md`](CLAUDE.md).

> Đọc nhanh khi mở phiên mới: mục 3 (đang ở đâu) → mục 5 (quyết định mở) → mục 6
> (khóa luận, lịch gặp GVHD) → **mục cuối của nhật ký (mục 7)**.

## 1. Mục tiêu, đội hình, câu hỏi nghiên cứu

**Khảo sát Mamba làm encoder ASR tiếng Việt hội thoại** trên VietSuperSpeech
(sau lọc A1 + giữ riêng 29 video: train 45.442 câu / 165,18 h; val 4.824; val_unseen
3.011; clean-test 203 câu đã hiệu đính tay), nhận xét theo đánh đổi **độ chính xác
– tốc độ – tài nguyên** (GVHD lần 1, 2026-10-01). Đề cương đã nộp sẽ được thay
bằng **đề cương mới** (nháp `docs/notes/de_cuong_gvhd/de_cuong_moi_nhap.md`, chưa duyệt);
`docs/de_cuong/*.docx` cũ giữ nguyên.

**Đội hình:**

| # | Mô hình | Vai trò | Cấu hình |
|---|---|---|---|
| 1 | Mamba hai chiều B1 (ExtBiMamba chồng, 14 lớp × 2 khối) | Encoder Mamba, cách 1 | train từ đầu ~12M, 10 ms và 40 ms |
| 2 | ConExtBiMamba (khung Conformer, MHSA → ExtBiMamba) | Encoder Mamba, cách 2 (chốt 12/10 là "mô hình chính" — ⛔ #2 đề nghị bỏ nhãn này) | như trên |
| 3 | Conformer-12M PE tương đối (relPE) | Baseline chuẩn | như trên; bản **không PE** (v1) là ablation ở 10 ms |
| 4 | Parakeet-CTC-0.6B-vi | Tham chiếu pre-train (zero-shot + fine-tune 7 GPU-h) | pipeline riêng |
| 5 | PhoWhisper-small | Tham chiếu pre-train (zero-shot) | pipeline riêng |
| — | Mamba một chiều | Chỉ trích dẫn (arXiv:2405.12609) | không train |

**Hai mức so sánh — không trộn khi kết luận:** nhóm train từ đầu (#1–#3) có kiểm
soát, chỉ encoder khác; nhóm pre-train (#4–#5) chỉ dùng chung `normalize_text`
và tập đánh giá.

**Công thức chung nhóm train từ đầu (`configs/v2a/`):** CTC, BPE1000, log-mel
80 + CMVN + SpecAugment, speed perturb, AdamW LR 3e-4 warmup 1000, batch 16,
DDP 2×T4 + AMP (khối Mamba fp32), 30 epoch LR hằng + 8 epoch giảm 1-sqrt = 38
epoch; **checkpoint bảng chính = ep37**. Cấu hình 10 ms: không subsampling, rẽ
nhánh từ v1 ep29 (relPE train từ đầu). Cấu hình 40 ms: `Conv2dSubsampling4`,
train từ đầu, seed 42 (⛔ #4: thêm seed 43, 44).

**Câu hỏi nghiên cứu** — bản đề xuất của review 15/10 mục 2.1, **⛔ #1 chờ chốt**
(bản làm việc cũ 04/10 ở nhật ký lưu trữ):
- **RQ1 — Độ chính xác.** Cùng pipeline CTC, ~12M, cùng công thức: hai encoder
  Mamba đạt WER thế nào so với Conformer relPE ở hai cấu hình 10 ms và 40 ms?
  Thứ hạng có giữ nguyên giữa hai cấu hình không?
- **RQ2 — Hiệu quả.** Chi phí suy luận (độ trễ, RTF, VRAM) theo độ dài audio
  5–3.600 s (audio ghép nhân tạo) và chi phí huấn luyện (GPU-giờ); từ độ dài nào
  Mamba có lợi?
- **RQ3 — Đặc điểm lỗi và chất lượng nhãn.** Loại lỗi theo encoder (từ dính
  CTC, tiếng Anh/tên riêng, xoá/chèn, thanh điệu); nhãn giả sai bao nhiêu, mô
  hình có vượt "thầy", xếp hạng trên nhãn giả có giữ khi đo trên nhãn thật?

Bảng, hình, kết luận viết sẵn theo nhánh: `docs/notes/ket_qua/ket_qua_cuoi_khung_2026-10-15.md`.
Quy tắc phân tích chốt trước khi có số: `docs/notes/ket_qua/conextbimamba_gap_2026-10-12.md` mục 9.4.

## 2. Lộ trình còn lại (đề xuất review 15/10 mục 5.1 — ⛔ #10 mốc đóng băng GPU)

Tuần tính từ thứ Hai; "Tuần 6 ≈ 2026-10-01". Quota: 3 tài khoản A/B/C × 30 GPU-h/tuần, reset thứ Sáu (17/10, 24/10…).

| Tuần | Ngày | GPU | Viết / việc tay | GVHD |
|---|---|---|---|---|
| 8 | 16–18/10 | L1 tải output; 17/10 L2 kernel dump (~0,3 h); đẩy 6 run seed 40 ms (A/B/C) | chốt 12 ⛔; đề cương mới → Word → gửi; commit; khung Ch3 | gửi đề cương + xin lịch |
| 9 | 19–25/10 | seed xong ~20/10; dump log-prob; giải mã CPU; **chấm một lần**; **đóng băng GPU 25/10** (đề xuất) | gõ mù 20 câu; sửa Ch1 | **gặp lần 2** |
| 10 | 26/10–01/11 | (dự phòng seed lỗi; mốc cũ 01/11) | RQ3 + L1–L4; hình RQ2; Ch2 nháp | — |
| 11 | 02–08/11 | — | **đóng băng số liệu 08/11**; bảng/hình cuối; Ch3 RQ1–RQ2 | gặp lần 3 |
| 12 | 09–15/11 | — | Ch3 RQ3 + thảo luận + hạn chế; **nháp Ch3 15/11** | — |
| 13 | 16–22/11 | — | Ch1, Ch2 hoàn thiện; Ch4 demo | gặp lần 4 |
| 14 | 23–29/11 | — | Mở đầu, Kết luận, TLTK, Phụ lục; **bản đủ 27/11** | góp ý |
| 15 | 30/11–06/12 | — | sửa, slide, chạy thử demo, in, CD (sao lưu checkpoint lên Drive trước) | **nộp** |

Cắt giảm nếu chậm (theo thứ tự): L3 "nhại" (RQ3) → bảng phụ LM → phân rã theo âm tiết. Không cắt Bảng 3.1–3.3,
RQ2, mục Hạn chế.

## 3. Trạng thái hiện tại (2026-10-16 tối)

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Dữ liệu, pipeline, Kaggle Dataset | ✅ | 3 tài khoản A/B/C đều train được |
| v1 30 epoch ×3 + Conformer dài 69 ep | ✅ | phụ lục; avg ep20–29 là số tham chiếu |
| 10 ms ×4 ep37 (B1, ConExt, relPE, không PE) | ✅ train + chấm clean-test 16/10 | B1 0,1432 · relPE 0,1694 · ConExt 0,1705 · không PE 0,2235 (`clean_test_results` bổ sung 16/10) |
| 40 ms ×3 seed 42 (ep37) | ✅ train + chấm clean-test | relPE 0,1211 · B1 0,1264 · ConExt 0,1309 (greedy); + LM 0,1024 / 0,1115 / 0,1081 |
| Seed 43/44 ở 40 ms | ✅ 6/6 + chấm chung 16/10 | 3 seed tb: relPE 0,1264±0,0046 · B1 0,1266±0,0014 · ConExt 0,1291±0,0018 → **không phân biệt được** (cả LM); `clean_test_scored_2026-10-16_chung/` |
| Parakeet zero-shot / fine-tune | ✅ / ✅ chấm clean-test | zero-shot 0,1378; FT 0,0943 |
| PhoWhisper zero-shot | ✅ | 0,1490 |
| RQ2 cuối | ✅ 7 cấu hình + Parakeet | `reports/results/rq2_final_2026-10-13/` |
| Clean-test hiệu đính + quy tắc chấm | ✅ 203/203, chấm 15/10 | `clean_test_results_2026-10-15.md`; gõ mù 20 câu ⛔ #8 |
| RQ3 / chất lượng nhãn | 🟡 từ dính đã đếm; L1, L2, L4 có số; L3 chưa | 0 GPU |
| Đề cương mới | 🟡 soạn lại 16/10, chờ người dùng chuyển Word + gửi | `docs/de_cuong/de_cuong_moi_2026-10-16.md` |
| Khóa luận (quyển) | 🔴 Ch1 nháp; Ch2/Ch3 chỉ là note | đường găng |
| Demo | 🟡 CPU, v1 | thêm 40 ms ở Ch4 |

## 4. Quyết định đã chốt

Theo thứ tự thời gian. Quyết định bị thay thế giữ lại để truy vết, đánh dấu **(thay bởi …)**.

| Ngày | Quyết định | Lý do / nơi ghi |
|---|---|---|
| 2026-09-14 | Hạ tầng thật là **Kaggle** (2x T4), không phải Colab; giữ nguyên `.docx` đã nộp | Lệch có chủ đích. `docs/notes/ky_thuat/mamba_ssm_install_log.md` |
| 2026-09-14 | **Mamba-1/S6**, pin `mamba-ssm` `v2.3.1`; Mamba-3 chỉ nhắc như hướng phát triển | `docs/notes/khao_sat/mamba_versions.md` |
| 2026-09-14 | Tokenizer BPE `vocab_size=1000`; clean-test seed 42 | Sau A1 train lại tokenizer; clean-test đổi theo video giữ riêng (10-04) |
| 2026-09-16 | Prefetch chỉ 67.405 file cần dùng | Repo HF có 118.259 file |
| 2026-09-19 | Tài liệu nhị phân lưu Drive, không lên GitHub; khóa luận Word Mẫu 5 ở `docs/khoa_luan/` (gitignore), viết song song | `docs/CONVENTIONS.md` mục 7; mục 6 |
| 2026-09-26 | ~~Giữ không subsampling~~ **(thay bởi 10-12: hai cấu hình 10 ms / 40 ms)**; 2 tài khoản Kaggle, train nền | `training_plan_kaggle.md` |
| 2026-09-28 | Duyệt D1-D11: 4 shard theo video, val = `validation` trừ clean-test, train full lấy kết quả theo epoch, pin HF `cbf624ae9b` | `training_plan_kaggle.md` mục 5 |
| 2026-09-30 | DDP 2 GPU + SyncBN + AMP, batch 16; dataset tar; khuyến nghị tác giả Mamba (miễn wd `A_log`/`D`, `norm_f`, chia `out_proj`) | `training_plan_kaggle.md`; mã mamba-ssm |
| 2026-10-02 | Định hướng **khảo sát**; **Mamba hai chiều B1**; chuẩn hóa văn bản chung trước WER; tách nhóm zero-shot/fine-tune | `gvhd_buoi_1.md`, `mamba_bidirectional.md` |
| 2026-10-03 | **A1 lọc C ∩ L**; **RQ2 audio ghép nhân tạo**, WER chỉ trên đoạn gốc; không train Mamba một chiều; ~12M; đội hình B1 + ConExt + Conformer + Parakeet (dự bị PhoWhisper) | `survey_model_candidates.md` mục 6; `dataset_discrepancy.md`; `lineup_preparation.md` |
| 2026-10-04 | **Giữ CTC**; CMVN toàn cục + SpecAugment; dropout 0,1 Mamba; không B1 + FFN; **test độc lập theo video** (29 video, clean-test 203, val_unseen 3.011); train thử để chốt giữ 2 hay 3 (giữ cả 3) | `frontend_decoder_survey.md`; `make_heldout.py` |
| 2026-10-05 | Khối Mamba fp32; speed perturbation 0,9/1/1,1; `out_proj` chia √28; bucketing | `mamba_precision_survey.md`, `audio_preprocessing_survey.md` |
| 2026-10-06 | Không early stopping; train cố định; lưu trọng số mỗi epoch | icefall/ESPnet |
| 2026-10-11 | Chỉ fine-tune Parakeet (~7 GPU-h, LR 1e-4 theo config NeMo, fp32); PhoWhisper zero-shot; chấm clean-test sau hiệu đính | `lineup_preparation.md` mục 4 |
| 2026-10-12 | **Phạm vi "A + 40 ms"**: công thức v1 + pha giảm 1-sqrt 8 epoch (38 ep) không InterCTC; (1) avg v1, (2) pha giảm rẽ nhánh 10 ms ×3, (3) Conformer relPE từ đầu, (4) 40 ms ×3 × 1 seed, (5) RQ2 lại; đóng băng GPU 01/11; cổng seed 2 | `conextbimamba_gap_2026-10-12.md` mục 9 |
| 2026-10-12 | Checkpoint bảng chính = **ep37** cho mọi run; Conformer dài để phụ lục; quy tắc phân tích 9.4 viết trước khi có số; **mô hình chính = ConExtBiMamba** (⛔ #2 đề nghị bỏ nhãn) | Người dùng; note gap 9.4 |
| 2026-10-13 | Thêm hướng **chất lượng nhãn** (L1–L4, 0 GPU); **không đề xuất thí nghiệm GPU mới**; seed 2 không ưu tiên (xem lại ở ⛔ #4) | `label_quality_direction.md`; người dùng |
| 2026-10-14 | Hiệu đính: cố vấn Whisper large-v3 (ngoài đội hình), bảng tên chuẩn; hiện output mô hình trong công cụ từ 23:17 (mốc `OURS_VISIBLE_FROM`) | `clean_test_correction_guide.md`, `clean_test_entities.md` |
| 2026-10-15 | **Quy tắc chấm clean-test**: cách viết thay thế (`configs/scoring/alt_spellings.tsv`), bỏ từ đệm phi từ vựng cả hai phía, lặp từ chấm thường, WER tách từ Việt / từ ngoại theo bảng từ; LM giải mã chỉ từ `train_transcripts_filtered.txt` | `clean_test_results_2026-10-15.md` mục 1; `glued_words_2026-10-15.md` |
| 2026-10-16 | **12 quyết định sau rà soát Fable 15/10, chốt theo khuyến nghị** (review mục 4): (1) luận điểm + RQ như review 2.1, 10/40 ms ở RQ1; (2) **bỏ nhãn "mô hình chính"** (thay quyết định 12/10), viết thành khảo sát hai encoder Mamba; (3) vá bảng 10 ms (L1, L2) trước mọi run mới; (4) **seed 43/44 × 3 mô hình 40 ms** (~32 GPU-h), tiêu chí **9.6** (bootstrap hai tầng thay điều 4a/5 của review 2.3); (5) không CR-CTC/BPE3000, chỉ bảng phụ giải mã; (6) **trục GPU-giờ: chỉ báo cột GPU-h + đường WER theo GPU-h, không train bù** (đóng 9.4 điều 10; khai sự cố 13/10); (7) chấm Parakeet FT trên clean-test (gộp L2); (8) gõ mù ~20 câu; (9) chất lượng nhãn gộp RQ3; (10) **đóng băng GPU 25/10** (dự phòng 01/11 chỉ cho run seed lỗi); (11) kiểm `normalize_text`/`alt_spellings`, đổi thì chấm lại toàn bộ một lần; (12) commit file chưa track sau rà repo public. **10 ms: 1 seed, không cổng seed** (người dùng loại đề xuất cổng của AI) | Người dùng ~08:20, trước khi xem WER relPE 10 ms; `conextbimamba_gap_2026-10-12.md` mục 9.6 |
| 2026-10-16 | **Không gõ mù 20 câu** (thay #8 vừa chốt): mức neo nhãn không đo; khai hạn chế bằng số đã có (ô hiệu đính điền sẵn nhãn giả; 80/113 câu đầu giữ y nhãn giả; bản hiệu đính cách nhãn giả 1,5–1,8% WER; một người hiệu đính) | Người dùng ~08:45 |
| 2026-10-16 | **#11: thống nhất vị trí dấu thanh ở vần mở oa/oe/uy** (hóa→hoá, thủy→thuỷ; bỏ qua sau `q`) — quy tắc chung trong `score_clean_test.py` (`canon_tone`), **không** đổi `normalize_text`; chốt trước kết quả seed 43/44, áp ở lần chấm chung khi đủ run. Đo trên output hiện có: PhoWhisper −4 lỗi (0,1490 → 0,1486), ConExt v1 −1, còn lại 0 | Người dùng ~09:05; khai ở Ch2 (quy tắc thêm sau lần chấm 15/10) |
| 2026-10-17 | **8 mục treo sau chấm chung, chốt theo đề xuất AI:** (1) quy tắc tách âm tiết giữ nguyên như note rq3 mục 1, khai 3 chỉnh sửa lúc chạy kiểm; (2) từ dính dùng `sticky_words` cho mọi hệ thống, số note 15/10 chỉ ghi phụ lục; (3) commit `reports/results/error_analysis/`; (4) nhận câu sửa note RQ2 ("≤ 21%" theo tb 3 seed); (5) **viết lại khuyến nghị Ch4**: câu ngắn chọn theo triển khai (WER 40 ms không phân biệt được), audio dài → ConExt 40 ms; (6) "không phân biệt được" luôn kèm CI 95% của hiệu, không viết "chênh ≤ X"; (7) Hình 3.5 vẽ lại với 40 ms trung bình 3 seed; (8) gitignore `data/Test_micro/`. Thêm việc 0 GPU (giới hạn ~2 ngày): kiểm định dấu theo video (mô tả, không đổi tiêu chí 9.4/9.6); hình WER theo GPU-giờ; L3 "nhại" bản tối giản (≤ 1 ngày); hạ câu "lần đầu" thành "theo phạm vi tra cứu" hoặc quét VLSP/ACL. **Không đổi mốc đóng băng GPU: giữ 25/10** (người dùng bác đề xuất đóng 17/10) | Người dùng 17/10; đề xuất trong phiên 17/10 |
| 2026-10-19 | **So với mô hình hiện hành: A + B** (mở lại mục "eval ngoài miền" review 15/10 mục 3 vì GVHD 17/10 hỏi "so với mấy model hiện thời"). A: bảng cùng clean-test có CI hai tầng + số công bố (khác tập test); B: chỉ eval mọi checkpoint bảng chính + pre-train trên test GigaSpeech 2 vi, ước 1–3 GPU-h, trước 25/10; quy tắc chấm chốt trước khi giải mã. `docs/notes/ket_qua/so_sanh_hien_hanh_2026-10-19.md` | Người dùng 19/10 |
| 2026-10-19 | **Bỏ ràng buộc "tham số giải mã cố định / chấm một lần" cho GigaSpeech 2** (người dùng: nguyên tắc để giữ tiến độ, không để chặn tối ưu). Được tối ưu giải mã + LM rộng (văn bản GigaSpeech 2 train); giữ một điều: chỉ chỉnh trên **dev** GigaSpeech 2, test chỉ để báo; báo cả cấu hình cố định lẫn tối ưu. `so_sanh_hien_hanh` B.2 mục 4–5 | Người dùng 19/10 |
| 2026-10-19 | **11/203 câu clean-test trùng nội dung với train (tập tổng hợp/đăng lại): bỏ qua** — giữ 203 câu, không sửa, không khai báo; ảnh hưởng 0,1–0,4 điểm, thứ hạng không đổi. `data_audit_2026-10-19.md` mục 3 | Người dùng 19/10 |
| 2026-10-20 | **Không khai "thứ tự quyết định sau khi thấy v1" và "sự cố thấy WER relPE sớm 13/10"** trong khóa luận: các lần chốt với AI là để định nghĩa lệnh chạy, không phải thời điểm ra quyết định nghiên cứu. Giới hạn về thiết kế (40 ms trộn 4 biến, nhánh 10 ms rẽ từ v1, Conformer không PE phi chuẩn) vẫn giữ | Người dùng 20/10 |
| 2026-10-26 | **Bổ sung A (tiêu đề tập podcast làm ngữ cảnh khi giải mã) + B (trực quan hoá Δ và attention ẩn của Mamba)**, 0 GPU, không train, quy tắc chốt trước khi chạy; WER audio dài, công cụ SRT, tốc độ CPU, độ tin cậy, thêm fine-tune: treo | Người dùng 26/10 sau câu GVHD "chỉ có vậy thôi à?"; `TODO.md` khối ▶ |
| 2026-10-27 | **Dồn sức "giải pháp/sản phẩm"**: công cụ phiên âm podcast dài → phụ đề (`src/demo/longform.py`) + tốc độ CPU + độ tin cậy; sau đó duyệt thêm hình Pareto, kiểm bản ghi Vietcetera, **WER audio dài nguyên chuỗi** (hạn 31/10). Gộp demo: treo. Thêm mô hình fine-tune: vẫn treo | Người dùng 27/10 |
| 2026-10-27 | **Thông điệp mở đầu + sườn slide theo khung "báo cáo về AI"** (`docs/notes/review/danh_gia_bo_sung_2026-10-26.md`, mục "Khung báo cáo về Trí tuệ nhân tạo"): câu hỏi trung tâm "Mamba có thay được attention cho ASR tiếng Việt hội thoại không, trả giá gì, bên trong học khác ra sao"; slide: bài toán → hai mô hình chuỗi → thiết kế có kiểm soát → độ chính xác → chi phí + ngoại suy độ dài → bên trong mô hình (B) → nhãn nhiễu + ngoài phân bố → ứng dụng (1 slide + demo) → kết luận về AI; CI chi tiết ở slide dự phòng. Không đổi tên đề tài vì khung này (chờ GVHD trả lời đơn đổi tên 08/10) | Người dùng 27/10: "giữ thông điệp mở đầu và sườn slide" |

## 5. Quyết định đang mở / rủi ro đã biết

12 quyết định sau rà soát 15/10: **đã chốt 2026-10-16** (mục 4).

| Trạng thái | Vấn đề | Chi tiết |
|---|---|---|
| 🔴 | Đề cương mới chưa gửi/duyệt; đã quá 2 tuần chưa gặp GVHD | mục 6; Mẫu 3 |
| 🔴 | Quyển chưa có Ch2/Ch3 | mục 2 lịch viết |
| ✅ | Output Conformer relPE 10 ms còn đủ (38/38, returncode 0, `epoch37.pt` trên Kaggle B) — 16/10 | `reports/results/train_full/v2a-conformer-relpos/s3/` |
| ⚠️ | Kernel kẹt cấp máy Kaggle (3 lần) | huỷ → quota hoàn |
| ⚠️ | Dữ liệu pre-train Parakeet có thể trùng VietSuperSpeech; card không nêu | chỉ khai |
| ⚠️ | Notebook EDA nhúng audio; `.docx`/file TLCN trong lịch sử git | chỉ xử lý khi người dùng yêu cầu |
| ⚠️ | `docs/de_cuong/*.docx` trên đĩa khác bản commit; bản nào đã nộp (Q4 cũ) | người dùng xác nhận |

## 6. Khóa luận (bản viết) và làm việc với GVHD

Viết theo biểu mẫu khoa (Mẫu 0/3/5, 2018, `docs/bieu_mau/` trên máy + Drive). Mẫu 3 dặn *"vừa làm vừa viết"*, Mẫu 5
yêu cầu **50–100 trang** → viết song song theo chương. File Word `docs/khoa_luan/KhoaLuan_Mamba_vs_Conformer_ASR.docx`
(gitignore, sao lưu Drive) là **bản làm việc duy nhất**; script sinh đã bỏ, đừng dựng lại. Mục lục đề xuất:
`docs/khoa_luan/muc_luc_de_xuat.md` (Ch2 = vì sao thiết kế, Ch3 = đã đo gì; ~66–80 trang). Nháp hiện có:
`ch1/`, `ch1_kiem_trich_dan.md`, `ch2_nhap.md`, `ch3_dulieu_phuongphap_nhap.md`.

| Phần | Lấy từ | Trạng thái |
|---|---|---|
| Mở đầu | đề cương mới | ⬜ khung |
| Ch1 Cơ sở lý thuyết | nháp ~9 trang; sửa theo `ch1_kiem_trich_dan.md` | 🔄 |
| Ch2 Phương pháp và thiết kế | `ch2_nhap.md`, `ARCHITECTURE.md`, note gap 9.4 | ⬜ |
| Ch3 Thực nghiệm | `ket_qua_cuoi_khung_2026-10-15.md`, `reports/` | ⬜ dựng khung ngay |
| Ch4 Demo và tổng kết | `src/demo/` | ⬜ |
| Kết luận, TLTK, Phụ lục | — | ⬜; TLTK ghi dần, đối chiếu nguồn |

**Hình thức (Mẫu 5):** A4, Times New Roman, dãn 1,5; lề trên/dưới/phải 2 cm, trái 3 cm; số trang giữa dưới; chương
14 HOA đậm giữa · mục 1 13 HOA đậm · mục 2 13 đậm · mục 3 13 nghiêng · nội dung 13 · bảng 12 · tên bảng 11 đậm trên
bảng · tên hình 11 đậm dưới hình · TLTK 11. Thứ tự quyển: bìa chính → bìa phụ → nhận xét GVHD → GVPB → cám ơn →
**đề cương có chữ ký** → mục lục → danh mục → tóm tắt → Mở đầu/Nội dung/Kết luận → TLTK → Phụ lục.

**Khi nộp (Mẫu 3, 5):** GVHD duyệt trước khi in; 2 cuốn bìa mềm + 2 CD (`<Tên phần mềm>/{SETUP,SOURCE}`,
`THESIS/{DOC,PDF}`, `REF`, `SOFT` — `SETUP` cần checkpoint ep37 khớp báo cáo → sao lưu Drive trước); bảo vệ ~15′ +
demo 5′ + hỏi đáp ~10′; sau bảo vệ sửa theo hội đồng, nộp lại.

**Nhật ký gặp GVHD** (≤ 2 tuần/lần, GVHD ký logbook; người dùng cung cấp, AI không tự điền):

| Ngày gặp | Đã báo cáo | GVHD dặn | Hẹn buổi sau |
|---|---|---|---|
| Tuần 6 (≈ 2026-10-01), lần 1 | Tiến độ Tuần 4–5, câu hỏi A1–A4/B1/C1–C4 | Đổi sang khảo sát, Mamba trọng tâm, thêm mô hình pre-train + baseline, nhận xét theo đánh đổi; A1 tự quyết; **"đừng ôm đồm"**; sẽ sửa đề cương. `gvhd_buoi_1.md` | Tuần 9 (19–25/10) — chưa có ngày |
| (thư 08/10: báo cáo lần 2, xin đổi tên đề tài, xin báo cáo sớm) | người dùng tự gửi | ⛔ đã trả lời chưa? | — |

**Hỏi ở buổi tới (giới hạn 3–4 câu):** duyệt đề cương mới (RQ, đội hình, hai cấu hình); dạng "thiết kế" Ch2 cho đề tài
nghiên cứu; hình thức nộp hiện hành; tên đề tài/bìa. Trình kèm Bảng 3.1–3.2 có CI, hình RQ2, lịch viết. Không hỏi
"có nên làm thêm X".

## 7. Nhật ký phiên (chỉ thêm vào cuối, không sửa mục cũ)

Mỗi mục: làm gì · chốt gì · phiên sau bắt đầu từ đâu. Các mục từ đầu dự án tới 2026-10-08 chuyển nguyên văn sang
`docs/notes/luu_tru/plan_nhat_ky_luu_tru_den_2026-10-08.md` (2026-10-15, không sửa chữ nào); trước 2026-09-19 xem `git log`.

### 2026-10-11 — fine-tune Parakeet: chốt, code, kernel
- **Người dùng chốt (bỏ treo phần fine-tune; GVHD buổi 1 đã nêu "Conformer pre-train + fine-tune"):** chỉ fine-tune
  Parakeet (PhoWhisper giữ zero-shot); trần ~7 GPU-h cho lần chính (≈ Conformer-12M 6,6 h); chạy ngay, chấm clean-test
  lại sau hiệu đính (fine-tune trên nhãn giả → WER so nhãn giả lạc quan). Subsampling 4× / seed 2 vẫn ⏸ chờ GVHD (Q11).
- **Code:** `src/training/finetune_parakeet.py` (test CPU bằng mô hình giả + dữ liệu thật: warmup/cosine, tích lũy
  gradient, eval đầu = zero-shot, best/last, eval_*.jsonl có video) + kernel `scripts/kaggle/finetune_parakeet/` (trial:
  phủ tokenizer, khớp giải mã NeMo, probe fp32/fp16, 2 LR × 20′; main ≤ 420′). Thiết lập + nguồn:
  `lineup_preparation.md` mục 4. NeMo/DDP chưa chạy thật.
- **Khung quyết định "train tiếp" (AI đề xuất, chờ kết quả):** Parakeet-FT bỏ xa Mamba (CI không chứa 0) → kể chuyện
  đánh đổi, không train thêm để đuổi; khoảng cách nhỏ → trung bình checkpoint rồi pha decay LR (6-9 GPU-h); trung bình
  checkpoint lợi ≥ ~0,5 điểm → pha decay cho cả ba.
- **Hiệu đính:** tên riêng/tiếng Anh viết đúng chính tả thật (nhãn train viết đúng: MARKETING 585, FACEBOOK 337, GOOGLE
  148 lần; phiên âm 0 lần); mô hình đoán theo âm khi từ hiếm (Bachelor → "ba sholo"/"ba chơ dô"). Ghi chú `tên riêng`
  để tách WER (RQ3). Tiến độ 11/203.
- **Phiên sau bắt đầu từ:** push (nếu người dùng duyệt) → kernel trial A (~1,5 h, cần duyệt + xác nhận quota) → chốt
  cấu hình main; hỏi thầy Dũng đã trả lời thư chưa; đề cương mới.
- **Nhận xét chuyên môn của AI (người dùng yêu cầu ghi nhận, 2026-10-11):** điểm mạnh — thực nghiệm chặt (test giữ riêng theo
  video, bootstrap khối video, ~12M chỉ khác encoder, kernel ghim commit, quyết định có nguồn); kết quả có chuyện (Mamba B1
  12M ≈ PhoWhisper-small zero-shot 244M trên clean-test nhãn giả, nhanh hơn ~60×). Rủi ro theo thứ tự: (1) **baseline
  Conformer lệch chuẩn** (không subsampling, không PE, LR hằng) — đe dọa kết luận chính "Mamba > Conformer", quan trọng
  hơn fine-tune; (2) clean-test còn nhãn giả (11/203) — đường găng, việc tay; (3) thời gian viết (~8 tuần, mới nháp Ch1);
  (4) đề cương mới chưa có, gặp GVHD đã quá 2 tuần (Mẫu 3); (5) dàn trải. Đề xuất: đề cương + hẹn GVHD tuần này, hỏi trọng
  tâm "có cần Conformer chuẩn (subsampling 4× + PE) + Mamba cùng điều kiện" (~12-20 GPU-h); hiệu đính xong trong 7-10
  ngày; fine-tune Parakeet như đã chốt; đóng băng thực nghiệm ~đầu tháng 11; viết Ch3 ngay từ `docs/notes/`.
- Người dùng duyệt push + chạy kernel trial fine-tune Parakeet (A, ~1,5 h).

### 2026-10-11 (đóng phiên)
- **Đã làm:** fine-tune Parakeet — chốt (chỉ Parakeet, ~7 GPU-h, chấm lại sau hiệu đính), code + kernel, commit `9b1dfcb`;
  ghi nhận xét chuyên môn, commit `50b6da4` (đã push). **Đẩy kernel trial `<A>/finetune-parakeet-trial` 21:51**
  (A, quota A đã reset — 0,0 h lúc đẩy; trần 150′). Lúc đóng phiên: RUNNING, chưa có dòng log (máy vừa cấp).
- **Chốt:** "train" = lần fine-tune chính của Parakeet (không phải train lại nhóm train từ đầu — vẫn ⏸ GVHD, Q11).
- **Phiên sau bắt đầu từ:** kiểm trial (status + output) → trình người dùng số liệu (phủ tokenizer, khớp giải mã, probe
  fp32/fp16 s/step + VRAM, WER 400+400 câu trước/sau 2 LR) + **đề xuất cấu hình lần chính** → người dùng duyệt mới đẩy
  `--mode main`. Song song: hỏi thầy Dũng đã trả lời chưa; đề cương mới; hiệu đính (11/203). Ghi chú phiên này chưa commit.

### 2026-10-12 — trial fine-tune Parakeet lỗi cấp máy; chốt LR 1e-4
- **Trial `<A>/finetune-parakeet-trial` (đẩy 11/10 21:51):** RUNNING ~10,3 h, log 0 dòng (kể cả `print` đầu tiên),
  output rỗng; người dùng Cancel 08:12 → **quota A hoàn về 0,00 h**. Chẩn đoán không tốn GPU: cùng script chạy CPU
  (`finetune-parakeet-cpucheck`) → clone/import/log OK, dừng đúng chỗ thiếu CUDA; kernel GPU 16 s (`gpu-check-asr`,
  gắn 5 dataset) → 2× T4, torch 2.11+cu128 OK. Kết luận: lỗi cấp máy phía Kaggle (như 05-06/10), không phải code.
  Thêm: API stream log (`kaggle kernels logs -f`) trả 500 cho mọi kernel → khi chạy chỉ canh được status + quota;
  `kaggle kernels push -t` **không** cắt kernel.
- **Chốt (người dùng): LR fine-tune Parakeet = 1e-4** (bỏ so 1e-5/5e-5). Đối chiếu tài liệu: config fine-tune
  chính thức NeMo (1e-4, AdamW 0,9/0,98, wd 1e-3, cosine → 5e-6, batch 16/GPU), docs NeMo "1e-4 to 1e-5". Code:
  `--lr` mặc định 1e-4, `--min_lr_ratio` 0,05; trial chỉ còn 1 lần 20′ ở 1e-4. Nguồn: `lineup_preparation.md` mục 4.
  Lần chính phải tính `--total_steps` từ s/step trial (cosine của `lr_factor` theo step, không theo thời gian).
- Người dùng nêu trọng tâm là ConExtBiMamba, mục tiêu để các mô hình Mamba đuổi sát mô hình khác.
- **Trial v2 đẩy 08:37** (commit `54a3efa`, A). Commit + push được người dùng duyệt.
- **Phân tích ConExtBiMamba** (người dùng yêu cầu): `docs/notes/ket_qua/conextbimamba_gap_2026-10-12.md` — kém B1 3,2 điểm
  clean-test là underfit + chưa hội tụ (LR hằng), không phải overfit; nhanh hơn B1 1,7×/step nên 30 ep chỉ 10,0 GPU-h
  so với B1 16,8. Đề xuất (chưa chốt): P0 trung bình checkpoint cả ba; P1 train tiếp ConExt tới ~51 ep = cùng GPU-giờ
  B1 (Conformer dài 69 ep 14,8 h đã có) — mức so sánh thứ hai song song; P2 giảm LR cả ba; P3 subsampling ⏸ GVHD.
- **Trial v2 xong 09:13** (33′, 0,56 GPU-h): tokenizer phủ 100% (0 unk); giải mã khác NeMo chỉ ở khoảng trắng trước dấu
  câu; batch 8/GPU OOM → 4; probe fp32 1,08 / fp16 0,73 s/step, VRAM 12,6 GB; chạy ngắn LR 1e-4 DDP fp32 2,07 s/step,
  0 step bỏ; WER 400 câu: val 12,1 → 9,6%, val_unseen 11,0 → 7,5% sau 551 step (nhãn giả — một phần có thể là học
  cách viết của Zipformer). Output nhỏ ở scratchpad phiên (không lưu repo).
- **Lần chính đẩy 09:39** `<A>/finetune-parakeet-main` (A, commit `54a3efa`, người dùng duyệt fp32): batch 4×2×2=16,
  LR 1e-4 → 5e-6 cosine, warmup 800, total 10.000 step (~3,5 epoch, ước từ 2,07 s/step), eval đủ val+val_unseen mỗi
  2.500 step (~6′/lần). Ước xong ~16:40. Giữ `find_unused_parameters=True` (chưa sửa).
- **Phiên sau / khi xong:** tải output nhỏ (`--file-pattern '.*\.(json|jsonl|log|csv)$'`), best.pt 2,27 GB chỉ tải khi
  cần eval clean-test; chờ người dùng chốt 4 câu hỏi chuỗi v2 (`conextbimamba_gap_2026-10-12.md` mục 6).
- **Người dùng duyệt kế hoạch v2 + "mọi thay đổi có lợi triển khai hết lần này"; bỏ ràng buộc chỉ đổi encoder;
  sẽ cấp tài khoản Kaggle C.** G0 (code) xong, test CPU 10 config OK (chưa commit): `conformer_relpos_encoder.py`
  (PE tương đối qua SDPA bias, xoá padding conv, `pos_enc: none`), `subsampling.py`, InterCTC + self-conditioning
  (`ctc_model.py`, `layer_hook` trong encoder), WSD + ghi đè CLI (`train.py`), `average_checkpoints.py`,
  `configs/v2/` (7 file), launcher (v2 + C). Smoke `train.py` dữ liệu thật CPU OK (10 ms batch 4, 40 ms batch 16;
  10 ms batch 16 vượt RAM máy local). Danh sách + ngân sách ~255 GPU-h: `conextbimamba_gap_2026-10-12.md` mục 8.

### 2026-10-12 (đóng phiên ~11:55)
- **Đã làm:** chẩn đoán trial Parakeet 10 h (lỗi cấp máy Kaggle, quota hoàn) → chốt LR 1e-4 (config fine-tune NeMo) →
  trial OK → **lần chính đang chạy**; phân tích ConExtBiMamba (underfit + chưa hội tụ; cùng độ dốc học với B1 —
  ngoại suy power-law kiểm trên Conformer dài sai ≤ 0,5 điểm) → **công thức v2** (người dùng duyệt, bỏ chờ GVHD, bỏ
  ràng buộc chỉ đổi encoder, "mọi thay đổi có lợi một lần") → G0 code + test CPU, commit `1384e70`; kernel preflight
  `fcb188f`. Ghi nhận AI: hôm nay 2 lần push trước khi quét khoá (quét bù sạch; memory đã cập nhật).
- **Đang chạy lúc đóng phiên:**
  - A `<A>/finetune-parakeet-main` (đẩy 09:39, commit `54a3efa`; quota A 2,82 h lúc 11:55; ước xong ~16:40,
    tổng ~7-7,5 h). Báo động nếu quota A > 8,3 h mà chưa xong.
  - B `<B>/preflight-v2-asr` (đẩy 11:53, commit code `1384e70`; quota B 0,04 h lúc 11:55; ước 75-90′,
    trần 120′). Báo động nếu quota B > 2,5 h.
- **Phiên sau bắt đầu từ:**
  1. Kiểm 2 kernel trên (status + quota). Tải output nhỏ:
     `kaggle kernels output <kernel> -p <dir> --file-pattern '.*\.(json|jsonl|log|csv)$'` (B: `KAGGLE_CONFIG_DIR=<thư mục khoá B>`).
     Log stream (`logs -f`) đang lỗi 500 phía Kaggle → chỉ đọc được sau khi kernel xong.
  2. Preflight: đọc `preflight_v2_summary.json` — K1 (tham số, padding_diff), **Đ2 VRAM xấu nhất của conformer_relpos
     (rủi ro OOM chính)**, K3 7 pipeline, Đ4 s/step thật → **chốt lại ngân sách** (ước hiện ~213 + 20% ≈ 255 GPU-h,
     Conformer relpos/40 ms chưa đo).
  3. Parakeet: đọc `parakeet_ft/metrics.jsonl` (eval mỗi 2.500 step, đủ val + val_unseen), trình người dùng.
  4. ⛔ Chờ người dùng: username + khoá tài khoản C (điền `ACCOUNTS` ở `scripts/kaggle/train_full/launch.py`,
     chia sẻ 6 dataset A → C); chọn ngân sách (a) tài khoản 4 / (b) bỏ ConExt cùng GPU-giờ / (c) 40 ms chỉ 2 mô hình.
  5. Sau đó theo thứ tự: P1 trung bình checkpoint v1 (cần kernel eval — chưa viết) → P2 dò LR (9 lần proxy 10 epoch,
     `launch.py --model v2-... --tag p2lr<lr> --extra --lr <lr> --epochs 10 --decay_epochs 2 --experiment_name ...`)
     → P3 3 seed × 40 epoch → P4 40 ms → P5. Mỗi lần đẩy kernel GPU vẫn phải hỏi.
  6. Việc tay song song: hiệu đính clean-test (11/203 — đường găng của mọi kết luận); đề cương mới; hẹn GVHD.

### 2026-10-12 (chiều) — rà soát lộ trình, chốt phạm vi "A + 40 ms"
- **Preflight v2 (B) COMPLETE, đạt hết** (K1/K3/K5, 0 nonfinite; VRAM xấu nhất Conformer relPE 5,1 GiB). s/step (CÓ
  InterCTC): B1 0,69 · ConExt 0,81 · Conformer relPE 1,06 · không PE 0,53 · sub4 0,19/0,28/0,26. Output ở scratchpad phiên.
- **Rà soát 3 agent** (phương pháp, tiến độ, tài liệu): đường găng là hiệu đính clean-test + viết + GVHD; mục 8 v2 quá tay
  và ước thấp; InterCTC không phục vụ RQ. Đính chính nguồn + nguồn mới Miyazaki 2406.16808: note gap mục 9.3.
- **Chốt (người dùng): phạm vi "A + 40 ms"** — Plan mục 4, note gap mục 9. Người dùng yêu cầu chia việc cho agent; sau khi
  phương án chạy sẽ dọn lại các file hướng dẫn (CLAUDE/Plan/TODO/notes).
- **Đang làm (agent, chưa commit):** config công thức chung + rẽ nhánh từ checkpoint v1 + launcher; trung bình checkpoint +
  kernel eval; công cụ + kernel RQ2 cuối; nháp đề cương mới `docs/notes/de_cuong_gvhd/de_cuong_moi_nhap.md`.
- **Phiên sau:** đọc báo cáo agent → test lại → xin duyệt đẩy kernel (bước 1 trước, rẻ); Parakeet main xong ~16:40 → tải
  output nhỏ; hiệu đính clean-test (mục tiêu ≥ 80/203 trước 18/10); gửi đề cương + hẹn GVHD trước ~17/10.
- **Người dùng chốt: mô hình chính = ConExtBiMamba** (Plan mục 1, 4; nháp đề cương đã sửa).

### 2026-10-12 (tối) — Parakeet fine-tune xong; agent bị ngắt vì hạn mức API
- **Parakeet main COMPLETE** (397′ train, returncode 0; dừng ở step 9.379/10.000 do trần thời gian, LR đã ~6e-6):
  WER val 12,1 → 8,2% · val_unseen 12,6 → 8,0% (0 → 2.500 → 5.000 → 7.500 → 9.379: val_unseen 12,6 / 9,0 / 8,4 / 8,1 /
  8,0); 0 step không hữu hạn; 2,46 s/step; VRAM 12,6 GB. **Đo trên nhãn giả Zipformer** → một phần cải thiện có thể là học
  cách viết của nhãn; chấm clean-test sau hiệu đính mới kết luận. Quota A 7,33 h. Output nhỏ ở scratchpad phiên; best.pt chưa tải.
- **Agent:** xong — nháp đề cương, RQ2 cuối (`rtf.py`, `concat_audio.py`, `scripts/kaggle/rq2_final/`), công cụ RQ3
  (`error_analysis.py`), kiểm trích dẫn Ch1 (`docs/khoa_luan/ch1_kiem_trich_dan.md`: [10] ghép số hai cấu hình — phải sửa),
  nháp Ch2, Ch3 (dữ liệu + phương pháp) trong `docs/khoa_luan/`. Ba agent bị ngắt 429 (hạn mức phiên) → nháp Ch3 đã đủ;
  agent công thức chung (`configs/v2a/`, rẽ nhánh v1) và trung bình checkpoint được cho làm tiếp.
- **Bài học quản lý agent:** tối đa ~2-3 agent song song, phạm vi hẹp, báo cáo ngắn; agent bị ngắt thì làm tiếp bằng chính
  agent đó (giữ ngữ cảnh), không giao lại từ đầu.
- **Chờ người dùng:** commit + push để đẩy kernel; sửa `normalize_text` thống nhất dấu thanh (hóa/hoá); N trung bình pha
  giảm (AI nghiêng: cả 8 epoch); 4 chỗ đề cương; lý do thật chọn ConExt làm mô hình chính (ghi vào Ch2).
- **Đẩy kernel (AI, theo quyền người dùng giao):** commit `cf1c3ec` (trung bình checkpoint, RQ2, RQ3, tài liệu) và
  `edd7dd4` (configs/v2a + rẽ nhánh v1) đã push. Dataset riêng tư `<A>/kltn-v1-avg20-29` (3 file trung bình ep20-29).
  **Đang chạy:** A `<A>/avg-eval-v1-asr` (đẩy 19:01, ước 25-35′); B `<B>/train-full-v2a-mamba-s1` (đẩy 19:10,
  nhánh pha giảm B1 từ `train-full-mamba-s2` ep29, ước ~4,5 GPU-h). Sau avg_eval: A đẩy nhánh ConExt (~2,7 h) rồi
  Conformer v1 (~1,8 h); relPE 38 epoch và nhánh 40 ms xếp sau.
- **Chờ người dùng:** quy tắc chọn checkpoint bảng chính (AI đề xuất: trung bình cả 8 epoch pha giảm ep30-37, áp
  y hệt mọi run kể cả relPE và 40 ms — không dùng best.pt vì nhánh rẽ chỉ chọn trong ep30-37 còn run từ đầu chọn
  trong ep0-37). Segfault relPE 10 ms ở CPU local (nghi hết RAM; preflight GPU đã qua) — theo dõi ở run GPU.
- **Người dùng chốt: Conformer dài (69 ep, LR hằng) để ở phụ lục**, không train thêm (tạm thời).
- **Người dùng chốt: checkpoint bảng chính = cuối pha giảm (ep37)** cho mọi run (note gap 9.4 mục 9). Người dùng muốn báo cáo tốc độ Conformer PE và, nếu Conformer nhanh hơn, train thêm cho cùng GPU-giờ — AI đề xuất quy tắc ngân sách định trước (note 9.4 mục 10).

### 2026-10-12 (đóng phiên ~21:50)
- **Đã làm trong phiên:** rà soát lộ trình 3 góc → chốt phạm vi **"A + 40 ms"**; mô hình chính = **ConExtBiMamba**; quy
  tắc phân tích chốt trước (note gap 9.4); checkpoint bảng chính = **cuối pha giảm (ep37)**; Conformer dài để phụ lục.
  Code (commit `cf1c3ec`, `edd7dd4`, đã push): `configs/v2a/`, rẽ nhánh từ v1 (`--branch_from`), trung bình checkpoint +
  kernel `avg_eval`, `rtf.py` + `concat_audio.py` + kernel `rq2_final`, `error_analysis.py`. Tra cứu cải tiến ConExt (9.5):
  code tác giả dùng subsampling 4×; 4 lệch bài không phải lỗi. Nháp: đề cương (`docs/notes/de_cuong_gvhd/de_cuong_moi_nhap.md`, chưa
  commit — chờ người dùng quyết có đưa lên repo public không), `docs/khoa_luan/` (gitignore): `ch1_kiem_trich_dan.md`
  ([10] ghép số hai cấu hình — phải sửa trong Word), `ch2_nhap.md`, `ch3_dulieu_phuongphap_nhap.md`, **`muc_luc_de_xuat.md`**.
- **Kernel lúc đóng phiên:**
  - B `<B>/train-full-v2a-mamba-s1` RUNNING (đẩy 19:10; xong ~23:45). WER val/val_unseen theo epoch pha giảm:
    ep30 0,1445/0,1427 · ep31 0,1421/0,1398 · ep32 0,1378/0,1350 · ep33 0,1343/0,1322 (v1 best 0,1562/0,1547; nhãn giả;
    số chính thức = ep37). Quota B 3,75 h.
  - A `<A>/avg-eval-v1-asr` **version 3 QUEUED** từ 20:57 (chưa cấp máy, quota A 7,33 h không đổi). Version 1 chạy
    81′ không 0 dòng log (sự cố cấp máy Kaggle, người dùng cancel, quota hoàn); version 2 QUEUED rồi cancel.
- **Phiên sau bắt đầu từ:**
  1. **Nhắc người dùng review `docs/khoa_luan/muc_luc_de_xuat.md`** (2 chỗ chọn: tên Ch2; Bảng 2 10/40 ms ở RQ1 hay RQ4).
  2. Kiểm B1: COMPLETE? tải output nhỏ (`fetch.py --model v2a-mamba --session 1 --account B` hoặc
     `kaggle kernels output ... --file-pattern '.*\.(json|jsonl|log|csv)$'`), lấy WER ep37 val/val_unseen.
  3. Kiểm avg_eval A: nếu đã chạy → trong 10′ đầu phải có log, không thì báo người dùng cancel; nếu vẫn QUEUED → chạy trên
     B (cần người dùng share `<A>/kltn-v1-avg20-29` + `<A>/kltn-best-ckpt` cho `<B>`; sửa `id`/sources).
     Đẩy lại avg_eval: điền `COMMIT = "cf1c3ec"` tạm trong `avg_eval.py` rồi `git checkout` lại (không commit hash).
  4. Đẩy tiếp (AI có quyền quyết, báo sau): B → `v2a-conformer-relpos` (đo s/step thật 15′ đầu → báo cáo tốc độ +
     đề xuất quy tắc trục cùng GPU-giờ, người dùng chốt **trước khi xem WER**); A → `v2a-conextbimamba` rồi `v2a-conformer`
     (nguồn rẽ nhánh là output kernel A). Nhánh 40 ms (`v2a-s4-*`) sau đó. Theo dõi 10′ đầu mỗi kernel.
  5. Chờ người dùng: sửa `normalize_text` thống nhất dấu thanh (hóa/hoá) trước lần chấm cuối; 4 chỗ đề cương; lý do thật
     chọn ConExt (ghi Ch2); có đánh giá ngoài miền (VIVOS/Common Voice, chỉ eval) sau 01/11 không; tài khoản C.
  6. Việc tay song song: hiệu đính clean-test (11/203 — mục tiêu ≥ 80 trước 18/10); gửi đề cương + hẹn GVHD trước ~17/10;
     sửa Ch1 theo `ch1_kiem_trich_dan.md`.
- **Quản lý agent:** ≤ 2-3 agent song song, phạm vi hẹp, báo cáo ngắn; agent bị ngắt → làm tiếp bằng chính agent đó.

### 2026-10-13 (sáng ~04:20) — B1 pha giảm xong; huỷ avg_eval kẹt
- **B `train-full-v2a-mamba-s1` COMPLETE** (279′, commit `edd7dd4`, rẽ nhánh từ `train-full-mamba-s2` ep29): **ep37 WER
  val 0,1271 / val_unseen 0,1252** (v1 best 0,1562/0,1547; nhãn giả). Giảm đều ep30→37; 0 step không hữu hạn; 0,69 s/step;
  VRAM đỉnh 3,6 GiB. Output nhỏ ở scratchpad phiên (`b1_v2a/`). Quota B 5,70 h.
- **A `avg-eval-v1-asr` v3 kẹt** (RUNNING ~5,9 h, 0 log, output 0 B; quota A lên 13,22 h) → người dùng cho huỷ, AI bấm
  Stop Session qua Chrome → `CANCEL_ACKNOWLEDGED`, **quota A hoàn về 7,33 h**. Lần thứ 2 kernel này kẹt ở cấp máy A.
- **Đẩy lại (người dùng duyệt "chạy lại và đẩy tiếp theo kế hoạch"):** A `avg-eval-v1-asr` v4 (04:20, `cf1c3ec`) **COMPLETE
  ~14′**; B `train-full-v2a-conformer-relpos-s1` (04:20, `edd7dd4`, phiên 1/≈5); A `train-full-v2a-conextbimamba-s1` (04:38).
- **Trung bình checkpoint v1 (ep20-29, LR hằng) — số tham chiếu, nhãn giả** (best.pt chấm lại khớp số cũ → cùng đường đo):

  | | clean_test | val | val_unseen |
  |---|---|---|---|
  | B1 best ep28 → avg20-29 | 0,1711 → 0,1556 | 0,1562 → 0,1388 | 0,1547 → 0,1361 |
  | ConExt best ep28 → avg | 0,2028 → 0,1833 | 0,1875 → 0,1671 | 0,1845 → 0,1635 |
  | Conformer best ep28 → avg | 0,2646 → 0,2392 | 0,2519 → 0,2285 | 0,2481 → 0,2250 |

  Output nhỏ ở scratchpad phiên (`avg_eval/`). Thứ hạng không đổi; khoảng B1–ConExt ~2,7 điểm (best: 3,0).
- **relPE s/step:** 1,42 ở step 100 là trung bình cộng dồn gồm khởi động — **đính chính:** nhịp ổn định step 750→850 ~0,57 s/step (50 step / ~28,5 s; cộng dồn 0,79 và đang giảm) → ước ~17-18 GPU-h / 38 epoch, **nhanh hơn B1 (0,69) ~15-20%**. Chốt số sau epoch 0 (`s_per_step` trong metrics). Quy tắc trục cùng GPU-giờ: chờ người dùng, chốt trước khi xem WER relPE.
- **Đính chính tốc độ:** ConExt v1 train 0,40 s/step (log train full), không phải 0,81 (đó là preflight có InterCTC).
  Nhánh ConExt v2a nhịp ổn định ~0,40; relPE ~0,57-0,60 (cộng dồn epoch 1: 0,566). Thứ tự train: Conformer v1 0,26 <
  ConExt 0,40 < relPE ~0,58 < B1 0,69. Giải thích + lập luận cho ConExt (Pareto WER–tốc độ audio dài; rủi ro: Conformer
  v1 không PE nên khoảng WER có thể co lại với relPE) đã trình người dùng — dựa trên `rq2_pilot_2026-10-05.md`,
  `subsample_speed_2026-10-08.md`. Người dùng hỏi vì sao giữ v1: v1 = 30 epoch đầu (pha hằng) của run 38 epoch; chỉ
  relPE và 40 ms train từ đầu (kiến trúc/front-end khác).

### 2026-10-13 (đóng phiên ~05:10)
- **Đã làm:** B1 v2a xong (ep37 0,1271/0,1252); huỷ avg_eval kẹt (quota hoàn); avg_eval v4 xong (bảng ở trên); đẩy relPE
  (B) + nhánh ConExt (A). AI chưa xem WER relPE (giữ đến khi chốt quy tắc cùng GPU-giờ).
- **Kernel lúc đóng phiên (05:08):**
  - A `<A>/train-full-v2a-conextbimamba-s1` RUNNING (đẩy 04:38), ep30 step 2450, 0 step lỗi, VRAM 3,2 GiB → xong
    ~07:50-08:20. Quota A 8,06 h.
  - B `<B>/train-full-v2a-conformer-relpos-s1` RUNNING (đẩy 04:20), ep1 step 850 → hết trần 540′ ~13:20 (~16-17
    epoch). Quota B 6,49 h.
- **Phiên sau bắt đầu từ:**
  1. **Nhắc người dùng review `docs/khoa_luan/muc_luc_de_xuat.md`** (tên Ch2; Bảng 2 10/40 ms ở RQ1 hay RQ4) — đã nhắc 13/10, chưa trả lời.
  2. Kiểm ConExt: COMPLETE → tải output nhỏ (`kaggle kernels output ... --file-pattern '.*\.(json|jsonl|log|csv)$'`), WER ep37
     → A đẩy `v2a-conformer` (`launch.py --model v2a-conformer --session 1 --account A --push`) → rồi `v2a-s4-conextbimamba`
     → `v2a-s4-mamba` → `v2a-s4-conformer-relpos` (thứ tự AI đề xuất, ConExt trước vì là mô hình chính).
  3. relPE: hết phiên 1 (~13:20) → đẩy phiên 2 (`--session 2 --account B`), rồi phiên 3. Lọc dòng WER khi đọc log relPE.
  4. ⛔ Chờ người dùng: (a) quy tắc trục cùng GPU-giờ — AI đề xuất bảng chính cùng 38 epoch + cột GPU-h + đường WER theo
     GPU-h từ điểm eval mỗi epoch, không train bù; (b) có cho AI tự huỷ kernel kẹt (RUNNING, quota tăng, 0 log mới > 15′)
     khi vắng không; (c) muốn chạy tự động thì người dùng gõ `/loop ...` (AI không tự thức được); (d) các mục chờ cũ:
     `normalize_text` hóa/hoá, 4 chỗ đề cương, lý do chọn ConExt (Ch2), eval ngoài miền, tài khoản C, commit nháp đề cương.
  5. Việc tay: hiệu đính clean-test (11/203, mục tiêu ≥ 80 trước 18/10); gửi đề cương + hẹn GVHD trước ~17/10; sửa Ch1.

### 2026-10-13 (trưa ~11:50) — ConExt v2a xong; đẩy Conformer v2a
- **A `train-full-v2a-conextbimamba-s1` COMPLETE** (170′, `edd7dd4`, rẽ nhánh `conextbimamba_ctc` ep29): **ep37 WER val
  0,1525 / val_unseen 0,1506** (v1 best 0,1875/0,1845; nhãn giả). Giảm đều ep30→37; 0 step không hữu hạn (amp bỏ ≤ 2
  step/epoch); 0,40 s/step; VRAM đỉnh 3,2 GiB. Khoảng B1–ConExt sau pha giảm: ~2,5 điểm (val_unseen 0,1252 vs 0,1506).
  Output nhỏ ở scratchpad phiên (`conext_v2a/`).
- **Đẩy A `train-full-v2a-conformer-s1`** (11:40, `edd7dd4`, rẽ nhánh `<A>/train-full-conformer-s1` latest ep29 —
  launcher chọn đúng `latest.pt`, bỏ `best.pt` ep28). Theo dõi 7′: cài wheel OK, đang giải nén audio, không lỗi khởi động.
  Quota A 10,52 h lúc đó. Ước ~8 epoch × 2841 step × ~0,26 s ≈ 1,7 h.
- B relPE s1 vẫn RUNNING (hết trần ~13:20). Chưa xem WER relPE.
- **Phiên sau bắt đầu từ:** kiểm Conformer v2a (COMPLETE → tải output nhỏ, WER ep37) → A đẩy `v2a-s4-conextbimamba` →
  `v2a-s4-mamba` → `v2a-s4-conformer-relpos`; B relPE hết phiên 1 → đẩy `--session 2 --account B`. Các mục ⛔ chờ người
  dùng giữ nguyên như mục đóng phiên 05:10.
- **Tài khoản C** (12:40): CLI OK, launcher đọc username từ `kaggle.json` (không ghi tên vào repo public; commit `0090bd7`).
  Kernel `gpu-check-asr` trên C: **2x T4 được cấp**, torch 2.11+cu128 thấy 2 GPU, matmul fp16 60 s/GPU hữu hạn
  (23,5 / 19,5 TFLOPS), GPU CHECK OK, tốn 0,04 h. Còn chờ: người dùng chia sẻ 6 dataset A → C; chốt job nào chạy trên C.
- **B relPE phiên 1 COMPLETE** (511′, dừng epoch 17 step 833/2841, `latest.pt` có, 0 nonfinite, 0,568 s/step, 5,1 GiB).
  **Sự cố minh bạch:** khi đọc `session_summary.json` AI in cả `best_wer` (val, tới ep16) → **AI đã thấy 1 số WER relPE
  lúc 12:58, TRƯỚC khi chốt quy tắc trục cùng GPU-giờ**. Ghi để khai đúng thứ tự thời gian quyết định; từ nay lọc trường này.
- **Đẩy B `train-full-v2a-conformer-relpos-s2`** (12:58, `edd7dd4`): resume đúng epoch 17 step 833 (13:08). Ước hết trần
  ~22:00 ở ~ep34-35; phiên 3 ~2 h. Quota B 14,37 h đã dùng.
- A Conformer v2a ep34 lúc 12:57, 0,259 s/step (chậm ep30 chỉ là khởi động) → xong ~13:45. Ước nhánh 40 ms ~20-22 GPU-h
  (ngoại suy preflight có InterCTC) > A còn ~17,5 h → cần C.
- **A `train-full-v2a-conformer-s1` COMPLETE** (117′): **ep37 WER val 0,2094 / val_unseen 0,2068** (v1 best ep28
  0,2519/0,2481; avg20-29 0,2285/0,2250); giảm đều ep30→37, 0 nonfinite, 0,259 s/step (ep30 0,334 = khởi động), 2,9 GiB.
  Nhóm rẽ nhánh ep37 val_unseen: B1 0,1252 · ConExt 0,1506 · Conformer 0,2068 (1 seed, nhãn giả). Output: scratchpad `conformer_v2a/`.
- **C:** 8 dataset A đọc được từ C (6 bắt buộc + 2 eval). Kernel kiểm môi trường `train-full-envc-v2a-s4-conextbimamba-s1`
  (pipeline thật, `--limit_train 1024 --limit_eval 64`, phiên 50′, tag `envc`) đẩy ~13:30.
- **Đẩy A `train-full-v2a-s4-conextbimamba-s1`** (13:46, `0090bd7` — chỉ khác `edd7dd4` ở launcher/tài liệu). Quota A 12,37 h.
- **C kiểm môi trường COMPLETE (14:07):** pipeline thật chạy hết 38 epoch trên 1024 câu (returncode 0, `finished`, lưu
  checkpoint ep37, 0 nonfinite, lịch WSD về lr 0, DDP 2x T4, mamba-ssm 2.3.1). Tốn 0,60 h. **Chép audio chậm: 20′**
  (A/B ~4-5′; mỗi shard 214-328 s) — có thể do lần đầu gắn dataset trên C; theo dõi lần sau. → **C sẵn sàng train.**
- 14:02: A s4-ConExt ep0 0,22 s/step, 1,2 GiB (ước 38 epoch ~7,3 h, có thể vừa 1 phiên); B relPE s2 ep18, 0,61 s/step.
- **Người dùng: "chia bớt việc cho C"** → đẩy C `train-full-v2a-s4-mamba-s1` + `train-full-v2a-s4-conformer-relpos-s1`
  (~14:15, `0090bd7`), **chạy song song** — Kaggle nhận 2 phiên GPU cùng lúc trên C (cả hai RUNNING). Nhánh 40 ms giờ đủ
  3 mô hình: ConExt (A), B1 + relPE (C).
- 15:03: C s4-mamba ep2, 0,187 s/step · C s4-relPE ep2, 0,154 s/step; cả hai 1,3 GiB, 0 nonfinite; chép audio lần này
  nhanh (đã vào ep2 sau ~25′ phiên). Ước ~6 h / ~5 h → **mỗi run vừa 1 phiên**, xong ~20:00-21:00. Quota C 1,80 h.
- Trong lúc chờ (người dùng: "việc gì giúp được thì cứ làm"): chép output v2a + avg_eval vào `reports/results/`;
  `scripts/plot_v2a_branch.py` → `reports/results/v2a_branch_2026-10-13/` (hình WER theo epoch + bảng); sửa nháp Ch3
  cho khớp quy tắc 9 (bảng chính = ep37, không trung bình pha giảm), điền trạng thái Bảng 3.9, tài khoản thứ ba.

### 2026-10-13 (đóng phiên ~15:10)
- **Đã làm:** ConExt v2a + Conformer v2a xong → nhóm rẽ nhánh đủ 3 mô hình (ep37 val_unseen B1 0,1252 · ConExt 0,1506 ·
  Conformer 0,2068); tài khoản C kết nối + kiểm môi trường OK (commit `0090bd7` đã push); đẩy relPE phiên 2 (B), s4-ConExt
  (A), s4-mamba + s4-relPE (C, song song); gom output vào `reports/results/`; script + hình nhóm rẽ nhánh; sửa nháp Ch3.
- **Sự cố minh bạch:** AI đã thấy 1 số WER relPE (val best tới ep16) lúc 12:58, trước khi chốt quy tắc GPU-giờ.
- **Chưa commit:** `Plan.md`, `TODO.md`, `scripts/plot_v2a_branch.py`, `reports/results/{train_full/v2a-*, avg_eval_v1_2026-10-13,
  v2a_branch_2026-10-13}`. Vẫn không track: `docs/notes/de_cuong_gvhd/de_cuong_moi_nhap.md`, `reports/results/error_analysis/` (chưa kiểm).
- **Kernel lúc đóng phiên (15:05):**
  - A `<A>/train-full-v2a-s4-conextbimamba-s1` (đẩy 13:46): 0,22 s/step → xong ~21:30-22:00 (1 phiên). Quota A ~13 h.
  - B `<B>/train-full-v2a-conformer-relpos-s2` (đẩy 12:58, resume ep17 step 833): ~0,6 s/step → hết trần ~22:00 ở
    ~ep34. Quota B ~15,5 h.
  - C `<C>/train-full-v2a-s4-conformer-relpos-s1` (~19:30) + `train-full-v2a-s4-mamba-s1` (~20:30), 1 phiên mỗi run.
- **Phiên sau bắt đầu từ:**
  1. Kiểm 3 run 40 ms (A, C×2): COMPLETE → tải output nhỏ (`--file-pattern '.*\.(json|jsonl|log|csv)$'`), chép vào
     `reports/results/train_full/v2a-s4-*/s1/`. **WER 40 ms là khám phá** (Plan 3.2.4).
  2. B relPE hết phiên 2 → kiểm `latest.pt` + returncode → đẩy `--model v2a-conformer-relpos --session 3 --account B`.
     **Không in `best_wer`/WER relPE** khi kiểm (lọc `session_summary.json`, log).
  3. Sau khi relPE + 40 ms xong: chấm clean-test checkpoint ep37 cả nhóm (kernel eval) → RQ2 cuối (`rq2_final`).
  4. ⛔ Chờ người dùng: quy tắc trục GPU-giờ (giờ cần ghi kèm sự cố đã thấy số relPE); review `muc_luc_de_xuat.md`;
     xác nhận sửa Ch3 theo quy tắc 9 (ep37, không trung bình pha giảm); commit các file trên; các mục cũ (normalize_text
     hóa/hoá, 4 chỗ đề cương, lý do chọn ConExt, eval ngoài miền, commit nháp đề cương, `<B>` trong launch.py public).
  5. Việc tay: hiệu đính clean-test (11/203), gửi đề cương + hẹn GVHD trước ~17/10.

### 2026-10-13 (tối ~19:25) — ConExt 40 ms xong; đẩy RQ2 cuối (A)
- A s4-ConExt COMPLETE 38/38: ep37 WER val 0,1128 / val_unseen 0,1115 (nhãn giả, khám phá). Output → reports.
- Đẩy A `rq2-final-asr` 19:12 (`0090bd7`, trọng số ngẫu nhiên); 10′ đầu chạy bình thường (đang đo ConExt tới 1.800 s).
- B relPE s2 + C×2 vẫn RUNNING. **Phiên sau:** kiểm RQ2; C xong → tải output; B s2 hết trần → đẩy s3; hỏi người dùng về
  cổng seed 2 trước khi giao việc mới cho C.

### 2026-10-13 (đóng phiên ~21:00) — 40 ms xong cả ba; RQ2 cuối; chốt hướng chất lượng nhãn
- **Kernel:** C s4-relPE COMPLETE (38/38, 330′) · C s4-mamba COMPLETE (38/38, 369′, ep37 val 0,1128 / val_unseen
  0,1134) · A `rq2-final-asr` COMPLETE (50′, nhóm train từ đầu + Parakeet returncode 0). Output chép vào `reports/`.
  B relPE s2 RUNNING tới ~22:10.
- **Thảo luận phạm vi với người dùng:** seed 2 (~50 GPU-h) không ưu tiên; "train dài 40 ms" người dùng thiên về, chưa chốt
  epoch; dashboard mô phỏng chương 3 (số giả) https://claude.ai/artifact/BjsgBGGfZyamnsw9RBXx8y; AI đề xuất đặc trưng pitch
  → người dùng bác (thêm run + bảng). Người dùng nhắc: AI đổi đề xuất liên tục, tốn thời gian → từ nay không đề xuất thêm
  thí nghiệm GPU; độ sâu đến từ nhận xét.
- **Chốt (người dùng):** thêm hướng **chất lượng nhãn VietSuperSpeech** (nhãn Zipformer, không kiểm tay) — L1-L4 ở
  `docs/notes/du_lieu/label_quality_direction.md`; 0 GPU-h, dựa trên bộ hiệu đính clean-test.
- Đã che username tài khoản C trong Plan/TODO trước khi commit (repo public).
- **Phiên sau bắt đầu từ:** TODO khối "▶▶▶▶" (kiểm B relPE s3 đã đẩy chưa; chờ người dùng số epoch train dài 40 ms).


### 2026-10-14 (tối ~18:40) — công cụ hiệu đính: tô chữ theo audio
- Người dùng thấy công cụ khó dùng, muốn chữ được tô theo audio. Đã thêm vào `src/data/correct_clean_test.py`: forced
  alignment CTC (Viterbi tự viết, khớp hệt `torchaudio.forced_align` trên 8 câu; hàm torchaudio bị gỡ từ 2.9) cho
  **bản hiệu đính đang gõ**, dựa trên log-prob Conformer v1 avg20-29 chạy CPU (~0,7 s/câu lần đầu, ~50 ms mỗi lần căn
  lại); trình phát `<audio>` riêng + JS tô chữ, bấm chữ để nhảy tới. Kiểm trong Chrome: tự phát, tô đúng, cập nhật khi
  gõ; không bấm Lưu nên manifest không đổi. Hướng dẫn mục 1, 2, 5, 8 đã cập nhật (ghi phương pháp: model chỉ cấp
  thời gian, không cấp chữ).
- **Phiên sau bắt đầu từ:** như mục trước; người dùng dùng thử công cụ mới khi hiệu đính tiếp (11/203).

### 2026-10-14 (khuya ~23:30) — kiểm bản hiệu đính, cố vấn Whisper, bảng tên riêng
- Người dùng hỏi cách kiểm 113/203 câu đã hiệu đính (tiếng Anh, tên công ty, nói vấp). Đã thử chấm bằng Conformer v1
  (CTC NLL + bỏ từng từ): không phân biệt được ở tiếng Anh/tên riêng (model muốn bỏ cả *singapore*), nhiễu ở tiếng Việt.
  Thêm nữa, model nằm trong đội hình → **bỏ** cách này. Quét chữ: 80/113 câu giữ y nhãn giả, còn sót garble Zipformer.
- **Người dùng chốt:** không chờ GVHD cho phần này. Không chép lại mù (tốn thời gian). Muốn đối chiếu với model khác,
  nhưng tự mình hiệu đính.
- Kernel `whisper-advisor-asr` (A, 1×T4, 16′): Whisper large-v3 beam 5 cho đủ 203 câu → `data/processed/whisper_advisor.json`
  (gitignore). Whisper vs nhãn giả WER 0,156. Bản hiệu đính vs nhãn giả **0,015**, vs Whisper 0,099. Whisper **bịa 6 câu**
  ("subscribe kênh Ghiền Mì Gõ"), viết sai OPEX/Curious/Micah, bỏ lặp từ, ra chữ số.
- Công cụ: thêm ô Whisper, khung đối chiếu (tô chữ theo việc Zipformer/Whisper có cùng chữ đó không), chặn câu bịa (trùng
  < 0,3). Hướng dẫn mục 2a. Bảng tên chuẩn có nguồn: `docs/notes/du_lieu/clean_test_entities.md`.
- Phương pháp khai trong khóa luận: cố vấn chỉ dùng model ngoài đội hình (tiền lệ Faria et al. 2022, arXiv 2206.06192
  mục 2.1). PhoWhisper (dự bị) fine-tune từ Whisper → có thể hưởng lợi. Bản hiệu đính rất gần Zipformer (0,015) → neo
  nhãn có lợi cho Zipformer ở câu hỏi L2 ("học trò vượt thầy").
- Chưa chốt (người dùng): cách chấm từ đệm/lặp từ/cách viết thay thế; tách N-WER/CS-WER (ViMedCSS, PIER).
- **Phiên sau bắt đầu từ:** người dùng mở lại công cụ (khởi động lại để nạp ô Whisper), sửa các câu ở
  `clean_test_entities.md` mục 1, 3 và danh sách garble ở TODO, rồi làm tiếp 90 câu.
- (23:17) **Người dùng chốt, dù đã được báo rủi ro:** hiện output 5 mô hình đang so sánh trong công cụ (khung gập,
  `predictions.tsv` 08/10). Mỗi lần lưu ghi `corrected_at`; mốc `OURS_VISIBLE_FROM` = 2026-10-14T23:17. Lúc chấm, báo
  thêm WER trên các câu làm trước mốc (113 câu) để đo ảnh hưởng. Đã khởi động lại công cụ.
- (~23:50) Bố trí lại công cụ (người dùng thấy phải cuộn nhiều): 2 cột, trái dính màn hình (audio, theo dõi lời, ô
  hiệu đính, ghi chú + Lưu), phải chia tab (Đối chiếu / Zipformer / Whisper / Mô hình đề tài / Hướng dẫn); Ctrl+Enter =
  Lưu & tiếp. Đã kiểm khởi động + config; **chưa xem giao diện thật trong trình duyệt** (Chrome mất kết nối).
- Người dùng thấy mô hình đề tài sai 7–10 từ/câu. Đã giải thích: câu TB 51 từ; ep28 WER(nhãn giả) Parakeet 0,140 ·
  PhoWhisper 0,155 · B1 0,171 · ConExt 0,203 · Conformer 0,265; bản 40 ms tốt hơn (val_unseen ~0,11) nhưng chưa có
  output clean-test từng câu.

### 2026-10-14 (đóng phiên ~23:55)
- Chưa commit: công cụ (`correct_clean_test.py`), kernel `scripts/kaggle/whisper_advisor/`, `clean_test_entities.md`,
  hướng dẫn hiệu đính, manifest (113/203), TODO/Plan, cùng 2 file nháp có từ trước (`de_cuong_moi_nhap.md`,
  `phan_bien_2026-10-14.md`).
- **Phiên sau bắt đầu từ:** (1) hỏi người dùng có commit không; (2) người dùng góp ý giao diện mới; (3) người dùng
  hiệu đính tiếp, trước hết là các lỗi khách quan + garble ở TODO và `clean_test_entities.md`; (4) quyết định còn mở:
  cách chấm từ đệm/lặp từ/cách viết thay thế, tách N-WER/CS-WER, có chấm clean-test bản 40 ms để xem output không
  (tốn GPU, hỏi trước).


### 2026-10-15 — quy ước tiếng Anh/ghi chú, từ ghép, lỗi "từ dính", giải mã beam + LM (dừng vì hết token)
- Trả lời người dùng: cách viết tiếng Anh/tên riêng + từ khóa ô Ghi chú (hướng dẫn mục 2a, 3, 4). Phát hiện *make up* /
  *makeup*: nhãn train tự mâu thuẫn (MAKE UP 80 / MAKEUP 11…) → ứng viên danh sách cách viết thay thế lúc chấm (TODO).
- Người dùng phát hiện ConExt ra "từ dính" (*nghỉơi*), muốn kết quả tốt nhất, **sẵn sàng train lại/tiếp nếu đáng**,
  giao quyền đẩy Kaggle "nếu cần thiết". 3 agent: chẩn đoán, huấn luyện (xong), giải mã (đang chạy khi dừng phiên).
  Toàn bộ kết quả + số: **`docs/notes/ket_qua/glued_words_2026-10-15.md`**; file agent: `D:/Model/kltn_decoding_2026-10-15/`.
- Kaggle (A): dataset `<A>/kltn-v2a-s4-ep37` (3 checkpoint ep37 40 ms + `sets.json`), kernel
  `dump-logprobs-v2a-s4` 4′ GPU → `D:/Model/kltn_checkpoints/logprobs_v2a_s4/`. Kernel `scripts/kaggle/dump_logprobs/`.
- Phát hiện rò rỉ: `train_transcripts.txt` chứa 179/203 câu clean-test → LM chỉ train từ `train_transcripts_filtered.txt`.
- Người dùng hỏi "beam + LM có làm mô hình tốt hơn thật không" → đã giải thích (note mục 4).
- **Phiên sau bắt đầu từ:** (1) đọc `glued_words_2026-10-15.md`; agent giải mã có thể đã dừng theo phiên → kiểm
  `D:/Model/kltn_decoding_2026-10-15/decoding/` (chép lúc ~17:00; bản mới hơn có thể còn ở scratchpad phiên cũ
  `C:/Users/Dell/AppData/Local/Temp/claude/d--workspaces-specialized-thesis/7b468c40-7e92-4585-8ab2-e1b492d53849/scratchpad/decoding/`);
  nếu chưa có bảng 3 mô hình 40 ms trên dev thì chạy `run_exp.py` với log-prob 40 ms (lex + lm3c, lmw 0,8, ws 1,0,
  beam 10; chỉnh chỉ trên nửa tune); (2) trình bảng greedy / beam+từ điển / +LM cho 3 mô hình → người dùng chọn: chỉ
  bảng phụ giải mã, hay thêm train BPE3000 (~4,5 GPU-h), hay CR-CTC (~18–36 GPU-h) — GPU đóng băng 01/11, quota reset
  17/10; (3) bật lại trang hiệu đính; chưa commit gì.

### 2026-10-15 (tối ~19:50) — hoàn tất bảng giải mã 40 ms
- Phiên trước đã chạy 16/20 dòng (bản mới ở scratchpad cũ) → chép hết về `D:/Model/kltn_decoding_2026-10-15/decoding/`
  (kể cả log-prob Conformer v1 1,9 GB); chạy nốt 4 dòng beam 30 (CPU). Bootstrap theo video (14 video).
- Kết quả: note `glued_words_2026-10-15.md` mục 3.1. Thứ hạng relPE < ConExt < B1 giữ nguyên, khoảng cách giãn sau LM;
  LM lợi 1,4–1,8 điểm, xoá từ dính. Lmw/ws chọn theo trung bình 3 mô hình trên nửa tune.
- Bật lại trang hiệu đính (cổng 7861). Chưa commit.
- **Phiên sau bắt đầu từ:** người dùng chọn (a)/(b)/(c) ở TODO khối 19:50; quy tắc trục GPU-giờ (relPE) vẫn mở.
- Người dùng hỏi chi tiết (c): đã kiểm bảng bài CR-CTC (v3) — nửa batch + nửa epoch = cùng chi phí; Aishell-1 S −22%,
  GigaSpeech S −3%; ghi note mục 3.2 + TODO. Đếm tiếng Anh clean-test: 3,4% số từ; nhãn giả lệch bản hiệu đính 2,1% WER.
- Người dùng báo hiệu đính xong 203/203, nhiều câu quên tag `tiếng Anh` → 33 câu thiếu tag (phát hiện tự động).
- Bảng loại từ clean-test xong 241/241 (`src/data/word_tags.py` → `data/processed/clean_test_word_tags.tsv`; nhãn: tiếng
  anh 188 · tên riêng 31 · viết tắt 19 · tiếng việt 2 · từ đệm 1). Manifest sửa theo lời người dùng: #11 u penn, #86 mad /
  visual artist và creative director / suboi, #99 nft. **Phiên sau:** chốt quy tắc chấm (cách viết thay thế, từ đệm) →
  tính lại WER 5 mô hình trên bản hiệu đính + WER tách nhóm từ; quyết (c) CR-CTC; gitignore bảng từ; commit (hỏi).

### 2026-10-15 (tối ~21:00) — chấm lại clean-test đã hiệu đính
- Quy tắc chốt trước khi chấm (người dùng: được viết thay thế, WER chia 2 nhóm; từ đệm/lặp từ giao AI theo chuẩn):
  `configs/scoring/alt_spellings.tsv` (13 dòng, lấy từ nhãn train); từ đệm phi từ vựng bỏ cả hai phía (Hub-5NE 1997
  + chuẩn hoá Whisper), lặp từ chấm như thường, từ vấp dở tuỳ chọn (#153 *in*). Script `src/evaluation/score_clean_test.py`
  → `reports/results/clean_test_scored_2026-10-15/`. Giải mã clean-test 40 ms: `D:/Model/kltn_decoding_2026-10-15/decoding/cleantest_decode.py`.
- WER (203 câu, 29 video): 40 ms greedy relPE 0,1211 · B1 0,1264 · ConExt 0,1309; + beam/LM 0,1024 · 0,1115 · 0,1081;
  v1 10 ms B1 0,1704 · ConExt 0,2019 · Conformer 0,2648; zero-shot Parakeet 0,1378 · PhoWhisper 0,1490; nhãn giả 0,0177.
- **Phiên sau:** trang báo cáo; Parakeet fine-tune trên clean-test (cần GPU — hỏi); quyết (c) CR-CTC; gitignore bảng từ; commit (hỏi).
- 21:30 Ghi nhận xét clean-test vào `docs/notes/ket_qua/clean_test_results_2026-10-15.md`. Người dùng: **có thể thêm tài khoản
  Kaggle, tối đa 10**; **ý nghĩa khóa luận phải thật**; sẽ nhờ Fable 5.1 rà soát toàn dự án và đề xuất phương án cuối —
  prompt `docs/notes/review/prompt_review_fable_2026-10-15.md` (đầu ra dự kiến `docs/notes/review/review_fable_2026-10-15.md`).
  **Phiên sau:** đọc bản rà soát của Fable (nếu có) cùng người dùng trước khi làm bất cứ việc GPU nào.

### 2026-10-15 (khuya) — rà soát độc lập toàn dự án (Fable 5.1, chỉ đọc)
- **Làm:** đọc CLAUDE/TODO/Plan/ARCHITECTURE, 9 note, `reports/results/` (clean-test đã chấm, RQ2 cuối, metrics mọi
  run), tra 10 nguồn → **`docs/notes/review/review_fable_2026-10-15.md`**. Không chạy GPU, không sửa code.
- **Phát hiện cần xử lý trước mọi run mới (0 GPU-h):** (1) run **Conformer relPE 10 ms** (B, s3 đẩy 14/10) **không có
  trong `reports/`**, chưa chấm → bảng 10 ms đang thiếu baseline chuẩn; (2) nhóm 10 ms chấm clean-test bằng **ep28 v1**
  trong khi quy tắc 9.4 chốt **ep37** — 3 checkpoint ep37 10 ms chưa chấm; (3) ở 40 ms GPU-h đo được relPE 5,28 h ·
  ConExt 4,86 h · B1 6,00 h → trục GPU-giờ không đe doạ thứ hạng, chỉ cần báo cột.
- **Khuyến nghị (chưa chốt):** thí nghiệm GPU duy nhất = 2 seed thêm × 3 mô hình 40 ms (~32 GPU-h, 3 tài khoản đủ,
  17–20/10, tiêu chí viết trước ở note mục 2.3); không CR-CTC/BPE3000/train dài/dò LR/train bù; đóng băng GPU 25/10;
  12 quyết định ở note mục 4; lịch tới nộp ở mục 5.
- **Phiên sau bắt đầu từ:** người dùng đọc note, chốt mục 4 → L1 (tải output B s3 + checkpoint ep37 10 ms) → L2 kernel
  dump log-prob sau reset 17/10 → đẩy 6 run seed (sau khi duyệt tiêu chí 2.3).

### 2026-10-15 (khuya, tiếp) — chấm điểm, khung kết quả cuối, dọn file hướng dẫn
- **Làm:** `docs/notes/review/review_fable_scorecard_2026-10-15.md` (62/100 hôm nay → 84/100 nếu làm đúng phương án; việc 72 giờ
  tới); `docs/notes/ket_qua/ket_qua_cuoi_khung_2026-10-15.md` (9 bảng, 5 hình, 5 kết luận viết sẵn theo nhánh). **Dọn file hướng
  dẫn** (người dùng yêu cầu, đã ghi TODO từ 12/10): `CLAUDE.md` viết lại theo hiện trạng (hai cấu hình 10/40 ms, luật
  clean-test chỉ để báo cáo + ep37, không đề xuất GPU mới, tải output kernel ngay, cạm bẫy `python -I`); `TODO.md` viết
  lại gọn (bản cũ nguyên văn `docs/notes/luu_tru/todo_luu_tru_2026-10-15.md`); `Plan.md` mục 1–6 viết lại, nhật ký trước
  2026-10-11 chuyển nguyên văn sang `docs/notes/luu_tru/plan_nhat_ky_luu_tru_den_2026-10-08.md`.
- **Không chốt gì mới:** 12 quyết định ⛔ vẫn mở (TODO mục ⛔). Không chạy GPU, không sửa code, chưa commit.
- **Phiên sau bắt đầu từ:** TODO khối ▶ (chốt ⛔ → L1 → L2 → L3; đề cương + khung Ch3 song song).

### 2026-10-16 (sáng ~08:30) — đọc rà soát Fable, chốt 12 quyết định, L1 relPE 10 ms
- **Đọc cùng người dùng** review + scorecard Fable 15/10. AI nêu: điều 4a (ghép seed i–i) và điều 5 (2 × sd) của tiêu chí
  2.3 tùy ý/không nguồn; "gõ mù" do chính người hiệu đính gõ không thật sự mù; tóm tắt review ghi "3–9 điểm" (đúng là 6–9).
- **Chốt (người dùng):** cả 12 mục theo khuyến nghị (`Plan.md` mục 4); tiêu chí 3 seed = **bootstrap hai tầng** (note gap
  **9.6**); **10 ms 1 seed, không cổng seed** (AI đề xuất cổng "|B1 − relPE| < 2 điểm → thêm seed", người dùng loại). Chốt
  ~08:20, **trước** khi xem WER relPE 10 ms; lúc kiểm toàn vẹn AI đã thấy `loss_val` 0,5647 (khai ở 9.6 điều 8).
- **L1:** B `train-full-v2a-conformer-relpos-s3` COMPLETE, 38/38, returncode 0, 0 nonfinite, 19,2 GPU-h, encoder **12,73M**
  (lớn hơn ~4% so với 12,20–12,29M — khai); file nhỏ → `reports/results/train_full/v2a-conformer-relpos/s3/`.
- **Số khám phá (nhãn giả, val_unseen, ep37, bootstrap cặp 29 video):** B1 0,1252 · ConExt 0,1506 · relPE 0,1501 ·
  không PE 0,2068. B1 − relPE −2,49 [−2,73; −2,27] điểm; ConExt − relPE +0,05 [−0,15; +0,25] (chứa 0); relPE − không PE
  −5,68. Ngược thứ hạng 40 ms (relPE dẫn). Số chính thức chờ clean-test ep37 (L2).
- **Phiên sau bắt đầu từ:** kiểm 3 nhánh 10 ms còn `epoch37.pt` → L2 (mở rộng `scripts/kaggle/dump_logprobs/` cho 4
  checkpoint 10 ms + Parakeet FT, đẩy kernel, báo) → L3 6 run seed 40 ms (cần code bootstrap hai tầng trước khi chấm, CPU).

### 2026-10-16 (sáng ~09:00) — L2 xong, 6 run seed đã đẩy, commit, đề cương mới
- **Commit `e4c30b8`** (người dùng yêu cầu; chưa push): rà repo public — không khoá/token/username C/tên người thứ ba;
  username A/B trong 6 file mới thay bằng `<A>`/`<B>` (còn trong `launch.py` và note cũ đã commit).
- **Người dùng chốt: không gõ mù** (thay #8) → khai hạn chế neo nhãn bằng số đã có.
- **L2** (A `dump-greedy-v2a-10ms`, 5,5′): clean-test 10 ms ep37 B1 0,1432 · relPE 0,1694 · ConExt 0,1705 · không PE
  0,2235; Parakeet FT 0,0943. B1 − relPE −2,62 [−3,25; −1,99] → B1 tốt hơn (9.6 điều 6); ConExt ≈ relPE (CI chứa 0).
  **Thứ hạng đảo giữa 10 ms và 40 ms.** Chi tiết: `clean_test_results_2026-10-15.md` mục bổ sung 16/10. Hai lỗ hổng
  review #1 đóng.
- **L3:** 6/6 run seed 43/44 đã đẩy (B relPE, C B1 08:36; A ConExt 08:49), commit `0090bd75` như seed 42; B/C qua 7′
  đầu RUNNING. Ước xong ~14:00–15:00.
- **#11:** `normalize_text` không thống nhất vị trí dấu thanh; trên clean-test chỉ 3 cặp (hoá/hóa, thoả/thỏa, khoá/khóa),
  phạt oan PhoWhisper 4 từ, ConExt v1 1 từ (< 0,05% WER) — chờ người dùng: (a) giữ quy tắc + khai, (b) thêm vào
  `alt_spellings` rồi chấm lại một lần khi đủ run.
- Code: `two_level_bootstrap` (`src/evaluation/bootstrap.py`, khớp bootstrap cặp khi 1 seed); scorer thêm 5 hệ thống.
- **Đề cương mới** `docs/de_cuong/de_cuong_moi_2026-10-16.md` viết lại toàn bộ (thay nháp 12/10), đã có số 10 ms.
- **Phiên sau bắt đầu từ:** tải output 6 run seed khi xong (file nhỏ → `reports/results/train_full/v2a-s4-*/seed43|44/`),
  dump log-prob/greedy clean-test (mở rộng `dump_logprobs/`), chấm bằng `two_level_bootstrap`, giải mã beam + LM CPU;
  người dùng chuyển đề cương sang Word, gửi GVHD; khung Ch3.
- **(~10:30) RQ2 hoàn chỉnh** (0 GPU): `scripts/rq2_final_tables.py` → Bảng 3.6/3.7, điểm giao, `numerics.md`, Hình 3.4/3.5;
  note `rq2_ket_qua_cuoi_2026-10-16.md` (thay KL3). Đính chính: điểm giao "~45 s" cũ là chỗ đường cong tách — cùng
  cấu hình 40 ms ConExt nhanh hơn relPE ở mọi độ dài nhưng chỉ 1–2 ms tới 45 s, ≥ 1,5× từ 60 s; B1 từ ~53 s (≥ 1,5× từ
  90 s). Phát hiện: VRAM 1 giờ audio Mamba 40 ms 8,7 GiB > 10 ms 5,7–6,0 GiB (suy luận: Conv2d subsampling).
  Pipeline chấm seed viết sẵn (`collect_seeds.py`, `dump_seeds_40ms/`, `lm_decode.py`, scorer hai tầng) — `lm_decode`
  tái lập 203/203 câu × 6 cột của 15/10. Chưa commit.

### 2026-10-16 (đóng phiên ~10:50)
- **Đã làm thêm sau 09:00:** pipeline chấm seed (`collect_seeds.py`, kernel `dump_seeds_40ms/`, `lm_decode.py` đưa giải
  mã beam + LM vào repo — tái lập 203/203 × 6 cột của 15/10; scorer đọc seed + bootstrap hai tầng, kiểm bằng seed
  giả); RQ2 hoàn chỉnh (`scripts/rq2_final_tables.py`, note `rq2_ket_qua_cuoi_2026-10-16.md`); mục lục đề xuất viết
  lại + nhận định Ch1 (`docs/khoa_luan/muc_luc_de_xuat.md`, Drive); 3 agent phản biện RQ2 (thiết kế hình, phương pháp,
  VRAM) → kết luận RQ2 thu hẹp phạm vi; VRAM 40 ms > 10 ms giải thích bằng activation Conv2d subsampling (đếm numel).
- **Người dùng chốt:** quy tắc `canon_tone` (#11); **bác Hình 3.4 dạng 2×2**; Hình 3.5 sẽ hỏi lại.
- **Xếp hạng clean-test (16/10, greedy, 40 ms mới seed 42):** Parakeet FT 9,4 › Conformer PE 40 ms 12,1 › B1 40 ms 12,6
  ≈ ConExt 40 ms 13,1 ≈ Parakeet zero-shot 13,8 › B1 10 ms 14,3 ≈ PhoWhisper 14,9 › Conformer PE 10 ms 16,9 ≈ ConExt
  10 ms 17,1 › không PE 22,4 (nhóm theo bootstrap cặp theo video). Parakeet FT − Conformer PE 40 ms −2,7 [−3,5; −1,9].
- **Người dùng phản hồi:** bị rối vì AI liệt kê quá nhiều giới hạn mà không nói rõ cái đã có → trả lời bằng bảng
  "câu hỏi → trả lời", giới hạn tách riêng và ngắn.
- **Kernel:** 6 run seed RUNNING lúc 10:48; ước xong ~13:50–14:45.
- **Phiên sau bắt đầu từ:** TODO khối ▶ (16/10 ~10:50).

### 2026-10-16 (đóng phiên ~11:10)
- **Đã làm:** commit + push `e81ee17..ecb3d2d` (quét khoá sạch trước khi push); tạo thư mục Drive `2026-10-16` với 4
  Google Doc (đề cương điều chỉnh, mục lục + nhận định Ch1, danh mục TLTK 26 mục, tóm tắt kết quả); tải 23 PDF tham
  khảo vào `docs/khoa_luan/tai_lieu_tham_khao/` (gitignore), kiểm tên bài/tác giả qua trang đầu → sửa tác giả trong
  đề cương local; chọn 4 bài gần đề tài nhất (Zhang 2024, Miyazaki 2024, Jiang 2024, Zevallos 2025) vào `4_bai_gan_nhat/`.
- **Chưa xong:** PDF chưa lên Drive (connector cần nội dung trong lệnh gọi) → người dùng kéo thả. Doc đề cương trên
  Drive còn dấu [đối chiếu] (không sửa được Doc sau khi tạo).
- **Kernel:** 11:10 cả 6 run seed RUNNING; ước xong ~13:50–14:45.
- **Phiên sau bắt đầu từ:** TODO khối ▶ (16/10 ~11:10) — chấm seed khi xong; Hình 3.4/3.5 chờ người dùng.

### 2026-10-16 (phiên chiều–tối, đóng ~18:20)
- **Đã làm:** gom 6 run seed 40 ms (`collect_seeds.py`): cả 6 returncode 0, 38/38 epoch, 0 nonfinite, commit `0090bd75`,
  log sạch; file nhỏ vào `reports/results/train_full/v2a-s4-*/seed43|44/`. Tải 6 `epoch37.pt` → dataset A
  `kltn-v2a-s4-seeds-ep37`; **đẩy kernel `dump-seeds-v2a-s4` (A) 17:44**, xong 17:48 (2,6′ GPU, 0 lỗi); greedy val_unseen
  khớp eval lúc train 593/600 câu (lệch 1 từ/câu, sai số fp16/padding). Giải mã beam+LM CPU (7,5′) → greedy khớp kernel
  203/203 × 6. **Lần chấm chung duy nhất** (áp `canon_tone`) → `reports/results/clean_test_scored_2026-10-16_chung/`
  (`OUT` mới trong `score_clean_test.py`, giữ thư mục sáng 16/10).
- **Kết quả (theo tiêu chí chốt trước, note gap 9.6):** 40 ms 3 seed — cả 3 cặp, greedy và beam+LM, CI hai tầng chứa 0 →
  "không phân biệt được ở 3 seed". relPE seed 42 (0,1211) là run tốt nhất trong 9 run; sd seed relPE 0,46 điểm, gấp ~3
  lần Mamba. Kết luận "relPE tốt hơn ở 40 ms" của seed 42 **không đứng vững**. Ghi ở `clean_test_results_2026-10-15.md`
  mục "Lần chấm chung" và nhánh đã xảy ra ở `ket_qua_cuoi_khung` KL1 (gồm nhánh 10 ms: B1 > relPE, ConExt ≈ relPE).
- **Không làm:** chưa chạy lại `rq2_final_tables.py` (sẽ vẽ lại Hình 3.4/3.5 đang chờ người dùng); không sửa khuyến nghị
  Ch4 — đánh dấu để người dùng xem.
- **Quota:** A dùng thêm ~0,1 GPU-h (kernel dump).
- **Phiên sau bắt đầu từ:** TODO khối ▶ (16/10 ~18:20).

### 2026-10-16 (tối, ~19:00–20:00)
- **Đã làm:** tải Parakeet FT `best.pt` (step 9379) về `D:/Model/kltn_checkpoints/parakeet_ft/`; dựng bản sao lưu
  `D:/Model/kltn_drive_backup/checkpoints_kltn_2026-10-16/` (39 file, 3,4 GB, README + SHA256SUMS khớp); tạo thư mục Drive
  `checkpoints_kltn_2026-10-16` — người dùng kéo thả (Chrome giới hạn 10 MB/lần, máy không có Drive for Desktop).
  Theo yêu cầu người dùng, 2 agent song song: Bảng 3.6 (3 seed) + Hình 3.4 lưới 1×2 (Hình 3.5 giữ nguyên); Bảng 3.8
  (`error_breakdown.py`, khớp WER bảng chính 29/29). Đã tự chạy lại: phân rã trùng ngoài cột CI, scorer trùng byte.
- **Chờ người dùng:** 4 điểm ở TODO ▶ mục 3; Hình 3.5; khuyến nghị Ch4.
- **Phiên sau bắt đầu từ:** TODO khối ▶.

### 2026-10-16 (tối muộn, ~19:30–22:30)
- **Demo (`src/demo/app.py`):** nạp 7 checkpoint ep37 từ `checkpoints/final/` (hard link tới bản sao lưu; 13/13 sha256 khớp
  bản sao lưu + file gốc Kaggle; greedy trùng chấm chính thức 21/21); giải mã beam + LM (mặc định, trùng chính thức 15/15);
  mặc định 3 mô hình 40 ms, "Xem thêm" 10 ms, "Hiện lỗi" tắt; dòng tham chiếu đọc từ lần chấm chung; câu ngẫu nhiên + độ khó;
  thu âm sửa (bỏ `format="wav"`, đọc webm qua ffmpeg). Lỗi mất bản thu phía người dùng do extension IDM (Chrome sạch chạy đúng).
- **Thử audio ngoài bộ kiểm tra** (`data/Test_micro/`, 2 câu, chưa gitignore phần txt/md): cùng kênh 19–26% (LM), kênh khác
  28–39%; gộp log-prob nhiều checkpoint (150 câu val_unseen) chỉ giúp greedy, làm beam + LM tệ đi → không dùng.
- **Dashboard cho GVHD** https://claude.ai/artifact/N9y2mpE5kgL15XNcvcAL3S (riêng tư — người dùng tự bật chia sẻ): bảng bối
  cảnh, Bảng 3.1–3.3 (3.3 chỉ greedy + nhóm pre-train), Hình 3.4a toàn chiều rộng cắt 20 ph + điểm giao, 3.6–3.7, 3.8; mỗi
  hình: giả thuyết + nhãn + mã nguồn [Miy24]… (đọc abstract PDF local) + căn cứ + câu viết được; bảng số liệu luôn mở;
  khung "Khuyến nghị" 6 ý; danh mục 8 tài liệu. Dữ liệu dựng bằng script scratchpad `dash/build_data.py` (ngoài repo).
- `ket_qua_cuoi_khung` Bảng 3.3 thay số cuối. Bản sao lưu checkpoint + Parakeet FT: 39 file, chờ người dùng kéo thả Drive.
- **Chờ người dùng:** gitignore `data/Test_micro/`; câu chữ "chênh ≤ X điểm"; bộ số từ dính; quy tắc âm tiết; Hình 3.5;
  khuyến nghị Ch4; commit.
- **Phiên sau bắt đầu từ:** TODO khối ▶.

### 2026-10-17
- **Đã làm:** rà lại dự án + review Fable (bản rà soát + bảng điểm); người dùng chốt 8 mục treo theo đề xuất, **bác đóng
  băng GPU sớm (giữ 25/10)** — `Plan.md` mục 4 dòng 17/10. Hình 3.5 vẽ lại (40 ms tb 3 seed, chấm từng seed); khuyến nghị
  Ch4 viết lại; gitignore `data/Test_micro/`. Kiểm dashboard GVHD: số khớp chấm chung, đã có forest plot cặp, 3.4a/b, 3.6,
  3.8 — chỉ thiếu hình WER theo GPU-giờ → `scripts/fig_wer_gpu_hours.py` (`fig_wer_gpu_hours.png`, GPU-h/WER ep37 khớp
  Bảng 3.6). Kiểm định dấu theo video (`src/evaluation/video_sign_test.py`): đồng thuận bootstrap mọi cặp; 10 ms B1 <
  relPE ở 26/29 video; 40 ms không cặp nào p < 0,05. L3 "nhại" (`src/evaluation/label_echo.py`, quy tắc ghi note rq3 mục
  6 trước khi tính): 162 vị trí lỗi nhãn; hệ học nhãn giả nhại 15–24% so với đối chứng zero-shot 9–13%, phần lớn là cùng
  xoá hư từ; không khác biệt giữa encoder.
- **Chưa làm:** hình RQ1 dạng PNG cho Word (dashboard là d3); cập nhật Hình 3.5 / hình GPU-giờ lên dashboard; commit.
- **Khung Ch3** (người dùng yêu cầu, văn phong bài báo — thiết kế cuối, hạn chế viết thành phạm vi, không kể lịch sử
  các lần chạy): `docs/khoa_luan/ch3_khung.md`, đủ 9 bảng + 4 hình, số từ lần chấm chung. Thêm L4 vào `label_echo.py`
  (Spearman thứ hạng nhãn giả vs hiệu đính 0,99; chỉ B1/relPE 40 ms đổi chỗ, chênh 0,05 điểm).
- **Phiên sau bắt đầu từ:** TODO khối ▶ (17/10) — còn lại chủ yếu là viết (khung Ch3) + việc tay đề cương/GVHD.

### 2026-10-19
- **Đã làm:** rà code + hướng đi, kiểm tính đúng của xử lý dữ liệu bằng dữ liệu thật → `docs/notes/du_lieu/data_audit_2026-10-19.md`.
  Sạch: split không giao nhau, video giữ riêng không lọt train/val, manifest khớp HF, tokenizer 0 unk/khứ hồi đúng,
  CTC khả thi mọi câu ở 10/40 ms × speed, duration khớp file. **Mới:** vài video giữ riêng là tập tổng hợp/đặc biệt dùng
  lại audio tập thường trong train → 11/203 câu clean-test trùng nội dung (8-gram ≥ 50%); bỏ 11 câu đổi WER +0,1–0,4
  điểm, thứ hạng không đổi, nhóm train từ đầu được lợi nhiều hơn pre-train.
- **Không làm:** không đổi tập chính thức, không chạy lại bootstrap 192 câu, không commit.
- **Tiếng Anh "dịch mù" (đối chiếu `Argon_feedback.md`):** A1 không xoá câu chêm (48,2/50,3 h loại là cả video khách nước ngoài; 40,9% câu train còn từ tiếng Anh); Zipformer viết đúng từ phổ biến, sai ở tiếng Anh hiếm (8.223 loại ≤ 2 lần) — lỗi nhãn, không phải pipeline. Không đổi pipeline; note mục 4.
- **So với mô hình hiện hành (nhận xét GVHD 17/10):** đưa phương án A (0 GPU, bảng cùng clean-test + số công bố) / B (+ eval test GigaSpeech 2 vi, ~1–3 GPU-h ước) / C (VIVOS) — **chờ người dùng chốt**, review 15/10 từng loại eval ngoài miền. Kiểm thêm: VietSuperSpeech train **100% một kênh Vietcetera** (60.656/60.656 đoạn) — chưa ghi thành hạn chế ở đâu; GigaSpeech 2 tách test theo kênh, 19 chủ đề/8 định dạng (không có mục "phỏng vấn/podcast" riêng) → đo khả năng ra ngoài kênh, không phải cùng miền. "Trà tắc": từng âm tiết có trong nhãn train (trà 238, tắc 122), cụm 0 lần; Test_micro sai cả từ phổ biến (hồng 182 lần → "học") → chủ yếu lệch kênh/giọng, không chỉ thiếu từ vựng.
- **Chốt A + B.** A xong: `src/evaluation/compare_pretrained.py` → `reports/results/compare_pretrained_2026-10-19/` (40 ms greedy không phân biệt được với Parakeet zero-shot, thấp hơn PhoWhisper-small; + LM thấp hơn cả hai; Parakeet FT tốt nhất). B chờ người dùng mở quyền GigaSpeech 2 trên HF (máy chưa có token).
- **B (GigaSpeech 2) 19/10 chiều:** 4 kernel (A 40 ms + Parakeet FT, A 10 ms — B tạo dataset 403 do cache upload dùng chung giữa tài khoản; C PhoWhisper — bản 1 kẹt cấp máy 0 log, người dùng huỷ, đẩy lại 16:45; D Parakeet zero-shot). Kết quả tạm `so_sanh_hien_hanh` B.3: train từ đầu sụt ~15 điểm ngoài kênh, Parakeet FT 2,7; thứ hạng encoder giữ như clean-test. Quy tắc B.2 khoá khi người dùng xem số.
- **Phiên sau bắt đầu từ:** B bước 1–2 (note `so_sanh_hien_hanh` mục B) nếu người dùng đã đăng nhập HF; song song: người dùng chốt cách báo cáo trùng nội dung (`data_audit` mục 3).
- **B tối 19/10:** beam+LM cố định (local, LM Vietcetera) trên test GS2: relPE/ConExt/B1 40 ms tb 3 seed 22,5/23,3/23,8% (greedy 27,8/27,9/28,3%), xoá gần hết từ dính; LM ngoài kênh perplexity 234 vs 74 → chưa tới ~20%. Người dùng **bỏ quy tắc hạn chế**, giữ "chỉ chỉnh trên dev" (Plan mục 4). Dựng pipeline CPU Kaggle B/C/D (trừ A): `scripts/kaggle/decode_gs2_cpu/` (KenLM + flashlight trên venv uv py3.12, cổng kiểm khớp LM Python, LM rộng = LM Vietcetera + văn bản train GS2 trừ câu dev/test, lưới LMW × WS trên 2.000 câu dev). Thêm tài khoản D, E (GPU kiểm OK). `score_gs2` đọc thêm cột `_lmtuned` (`40ms_tuned/<enc>/test_hyps.tsv`).
- **Đang chạy khi đóng phiên (18:45):** A `eval-gs2-40ms-dev` (RUNNING — top-k dev 3 encoder seed 42); D `eval-gs2-parakeet-zs`, D `eval-gs2-phowhisper` (QUEUED).
- **Phiên sau bắt đầu từ:** kiểm 3 kernel trên → tải output (`gs2_eval_2026-10-19/<mode>/`, `tk_*_gs2_dev.npz` → `checkpoints/gs2_topk/`) → `score_gs2` → `launch_cpu.py bundle/upload/push --stage tune` → chọn cấu hình chung theo WER dev tb → `push --stage final` → chấm. Quyết định chờ người dùng: thêm 1–2 mô hình fine-tune (gợi ý chờ số GS2). (11 câu trùng nội dung: đã chốt bỏ qua.)
- **Cuối phiên 19/10 (sau 19:00):** kernel A `eval-gs2-40ms-dev` COMPLETE (30 phút, không lỗi) → `reports/results/gs2_eval_2026-10-19/40ms_dev/` + 9 file `tk_*_gs2_dev.npz` trong `checkpoints/gs2_topk/`. D `eval-gs2-parakeet-zs`, `eval-gs2-phowhisper` vẫn QUEUED. Người dùng: bỏ qua 11 câu trùng nội dung; chờ số GS2 rồi mới chọn mô hình fine-tune thêm (đã kiểm dữ liệu train pre-train, `so_sanh_hien_hanh` mục C). Gỡ username Kaggle khỏi file track (`scripts/kaggle/accounts.py` + `accounts.local.json` gitignore; lịch sử git cũ giữ nguyên). Sắp `docs/notes/` vào thư mục con; `review/`, `luu_tru/`, `de_cuong_gvhd/`, `Argon_feedback.md` chỉ giữ local. Commit `426d2f9`, `cd89d96`, `95f2840`.
- **Phiên sau bắt đầu từ:** kiểm 2 kernel D (QUEUED lâu → cân nhắc chuyển sang E, cần upload `gs2-vi-test` cho E) → `launch_cpu.py bundle` → `upload` → `push --stage tune` (B/C/D CPU) → chọn cấu hình chung theo WER dev → `push --stage final` → tải `test_hyps.tsv` vào `gs2_eval_2026-10-19/40ms_tuned/<enc>/` → `score_gs2`.
- **Phiên 19/10 (tối, tiếp):** D `eval-gs2-parakeet-zs`, `eval-gs2-phowhisper` vẫn QUEUED. `launch_cpu.py bundle` xong (B/C/D mỗi bên 6 file, 0,70 GB) → `upload` lần lượt. **Người dùng: tạm khoá GPU tài khoản A** (không đẩy kernel GPU lên A tới khi mở lại).
- **22:51:** người dùng huỷ 2 kernel D (CANCEL_ACKNOWLEDGED) → đẩy lại trên D: `eval-gs2-parakeet-zs` v3, `eval-gs2-phowhisper` v2 (`launch_gs2.py --mode phowhisper --account D`); `watch_kernel.py` canh. Đã dựng `kaggle_ds/E/` (`gs2-vi-test`, hard link) — tải lên sau khi upload B/C/D xong (cache upload dùng chung).
- **22:54:** `eval-gs2-parakeet-zs` v3 ERROR sau 1,6′: `install_nemo` (zero_shot.py) pip-cài giữa tiến trình đã nạp numpy → `import nemo` gãy ở `numpy.ma`. Sửa trong file kernel: `preinstall_nemo()` đầu `main()` (chỉ `parakeet_zs`), không đổi commit ghim. Đẩy lại v4 22:56.
- **23:04:** upload `gs2-cpu-*` B/C/D xong (D báo 403 lúc kiểm status ngay sau upload — Kaggle đang xử lý; sau đó `ready`, đủ file; `bundle.tar.gz` bị Kaggle tự giải nén, `decode_cpu.py` đã xử lý). Đẩy `push --stage tune`: B `gs2-cpu-tune-conformer-relpos`, C `gs2-cpu-tune-conextbimamba`, D `gs2-cpu-tune-mamba` (CPU, 0 GPU-h). Đang tải `gs2-vi-test` lên E (dự phòng).
- **23:10–23:26:** tune B, D v1 ERROR (C v1 cùng lỗi): cổng KenLM 200/200 qua, LM rộng dựng xong (GS2 train 4.996.366 câu, bỏ 156 trùng dev/test; 58.138 từ, ARPA 374 MB), gãy ở chấm điểm — `score_clean_test` import `correct_clean_test` (gradio/torch). Sửa `decode_cpu.py`: bước dựng WORK thay file đó bằng file chỉ mang hằng `OURS_VISIBLE_FROM` (không tải lại dataset); mô phỏng local chặn gradio/torch, WER trùng bản đầy đủ. Đẩy lại v2 cả 3 (23:26). E `gs2-vi-test` ready. Parakeet zs v4 QUEUED trên D (PhoWhisper D RUNNING 32′).
- **23:40 tune xong (B/C/D v2 COMPLETE, ~11′ mỗi kernel, CPU):** output `reports/results/gs2_eval_2026-10-19/40ms_tune/<enc>/tune.json` + log. Cổng 200/200 cả 3; 2.010 câu dev (seed 42). WER dev tb 3 encoder: tốt nhất LM rộng lmw 1,0 ws 2,0 = 14,21% (relPE 13,23 / ConExt 14,34 / B1 15,05); LM Vietcetera tốt nhất 15,88% (lmw 0,6 ws 1,0); cấu hình cố định cũ (vss 0,8/1,0) 15,04/15,95/16,69. **ws 2,0 là mép lưới** → hỏi người dùng: mở rộng ws (CPU ~11′) hay chốt (broad,1,0,2,0) cho final.
- **23:46:** người dùng chọn mở rộng ws. `launch_cpu.py push --grid` (mới; bước dựng WORK sửa 2 dòng `LMW`/`WS` của bản `decode_cpu.py` trong bundle) → đẩy tune v3 B/C/D, lưới LMW [0,8; 1,0; 1,3] × WS [2,5; 3,0; 4,0] × 2 LM. Chọn cấu hình chung trên hợp hai lưới (cùng 2.010 câu dev). Cron kiểm 5′ (job phiên). PhoWhisper D RUNNING ~55′; Parakeet zs D QUEUED.
- **23:57–00:05 (20/10):** tune lưới mở rộng xong (B/C/D v3, `40ms_tune_ext/`). Hợp 58 cấu hình: tốt nhất vẫn **LM rộng lmw 1,0 ws 2,0** (dev tb 14,21%; ws 2,5 → 14,23%, lmw 1,3/ws 3,0 → 14,29%) — nay nằm trong lưới → chốt theo quy tắc. Đẩy `push --stage final --final '{"lm":"broad","lmw":1.0,"ws":2.0}'` B/C/D lúc 00:00. PhoWhisper D COMPLETE (61′ GPU, không lỗi) → `gs2_eval_2026-10-19/phowhisper/`; clean-test kèm trùng `final_compare_2026-10-08` 203/203. `score_gs2`: PhoWhisper-small zero-shot GS2 test **12,63%** [12,31; 12,94] (Parakeet FT 12,15%). Parakeet zs D vẫn QUEUED.
- **00:05–00:15:** final B/C/D COMPLETE (5–10′ CPU, 4.422 câu × 3 seed, rc 0) → `40ms_tuned/<enc>/test_hyps.tsv`. `score_gs2` **chưa có** đọc `_lmtuned` (nhật ký 19/10 ghi nhầm) → thêm nạp `40ms_tuned/*/test_hyps.tsv` + chế độ `lmtuned` vào bootstrap hai tầng. Kết quả test GS2 (tb 3 seed): relPE 20,50% / ConExt 21,39% / B1 22,01% (LM cố định 22,46/23,25/23,81; greedy 27,77/27,90/28,27). Cặp hai tầng lmtuned: B1 > ConExt [+0,36; +0,88], B1 > relPE [+0,70; +2,18], ConExt > relPE [+0,05; +1,58] (sát 0). Mọi hệ train từ đầu vẫn kém PhoWhisper zs 12,63 / Parakeet FT 12,15 ~8–10 điểm. Còn: Parakeet zs D QUEUED; cập nhật note `so_sanh_hien_hanh` mục B.
- **Đóng phiên 20/10 ~00:35.** Chốt: cấu hình giải mã GS2 = LM rộng lmw 1,0 ws 2,0 (dev tb, hợp 58 cấu hình). Đã có đủ số test GS2 trừ Parakeet zero-shot. Code đổi (chưa commit): `scripts/kaggle/eval_gs2/eval_gs2.py` (`preinstall_nemo`), `scripts/kaggle/decode_gs2_cpu/decode_cpu.py` (file thay `correct_clean_test`, `GRID`), `launch_cpu.py` (`--grid`), `src/evaluation/score_gs2.py` (`_lmtuned`). Output mới: `gs2_eval_2026-10-19/{phowhisper,40ms_tune,40ms_tune_ext,40ms_tuned}/`, `gs2_scored_2026-10-19/` chấm lại.
- **Phiên sau bắt đầu từ:** (1) kiểm D `eval-gs2-parakeet-zs` v4 (QUEUED từ 22:56; chạy ~15′ khi có máy). Còn QUEUED → hỏi người dùng huỷ trên UI rồi đẩy `launch_gs2.py --mode parakeet_zs --account E --push` (E đã có `gs2-vi-test` ready). Xong → tải vào `gs2_eval_2026-10-19/parakeet_zs/` (qua `accounts.mask`), đối chiếu clean-test kèm với `final_compare_2026-10-08` cột `raw_parakeet_fp32`, `python -m src.evaluation.score_gs2`. (2) Cập nhật note `so_sanh_hien_hanh_2026-10-19.md` mục B (bảng greedy / LM cố định / LM dò dev, cặp hai tầng, PhoWhisper 12,63%; ghi ConExt vs relPE: lmtuned CI [+0,05; +1,58] sát 0, LM cố định "không phân biệt được" → nhạy cấu hình giải mã). (3) Hỏi người dùng: thêm 1–2 mô hình fine-tune? (4) Commit khi người dùng bảo. GPU A vẫn khoá.

### 2026-10-20 (sáng)
- **Parakeet zs D v4 ERROR** (chạy lúc nào đó trong đêm, vài phút): `preinstall_nemo` chạy được, nhưng wheel `nemo_toolkit` 2.6.2 cài kèm package `scripts/` (có `__init__.py`) vào site-packages → đè namespace `scripts/` của repo → `No module named 'scripts.kaggle'`. Sửa `eval_gs2.py`: gắn `sys.modules["scripts"]` vào `REPO_DIR/scripts` sau checkout (tái hiện + kiểm local với site-packages giả). Đẩy lên **E** 08:59 (`--account E`; D từng QUEUED 1,5 h).
- **Khung Ch3 (người dùng duyệt hướng 20/10: chỉ thêm thứ trả lời RQ, 0 GPU):** `docs/khoa_luan/ch3_khung.md` thêm Bảng 3.0 (cấu hình kiểu Bảng 1 Jiang 2024 — gộp L7), mục 3.3.6 + Bảng 3.5b (GS2: RQ1 thứ hạng ngoài kênh, Spearman 7 cấu hình 0,93; sd seed relPE 0,77 vs Mamba 0,10–0,16), đoạn GS2 ở 3.5.2 (RQ3, chờ Parakeet zs), 3.6 viết lại đối chiếu Jiang (**sửa lỗi bản 17/10**: nhận định độ phân giải của Jiang là về hiệu quả, không về WER; ngưỡng 1,5× lùi 60–90 s → ~20 s khi token 40 → 10 ms, ≈ tỉ lệ chuỗi), 3.7 thêm "một kênh nguồn" + nguồn dữ liệu pre-train. Bỏ: WER theo 30 nhóm GS2 (không thuộc RQ); WER theo độ dài câu GS2 chỉ làm nếu thừa thời gian. Bản note `so_sanh_hien_hanh` thêm mục B.3c.
- **09:25 Parakeet zs E COMPLETE** (22,9′ GPU E, không lỗi) → `gs2_eval_2026-10-19/parakeet_zs/`; clean-test kèm 203/203; GS2 12,66%. **B (GS2) hoàn tất.** FT − zs ngoài kênh −0,51 [−0,68; −0,33] (trong kênh −4,35); từ Việt FT **kém hơn** +0,53 [+0,35; +0,69], lợi chỉ ở từ ngoại → vào Ch3 3.5.2 + 3.6 (RQ3: nhãn giả một kênh dạy quy ước chép, không chuyển thích nghi âm học). Người dùng: chưa chuyển Word — tiếp tục viết sườn, dành riêng 1 tuần cho văn phong + Word sau.
- **Tổng hợp kết quả (người dùng 20/10: chốt trình bày gì trước khi viết sườn tiếp):** `docs/notes/ket_qua/tong_hop_ket_qua_2026-10-20.md` — câu trả lời 3 RQ một đoạn + bảng kiểm kê 40 kết quả (ID, nguồn, đề xuất vị trí Ch2/Ch3/PL/Ch4/Bỏ, cột Chốt để trống). Chờ người dùng duyệt; 6 dòng [CHỌN].
- **Kiểm trích dẫn Ch3 (20/10):** 8 khoá đều khớp (đọc PDF local + trang công bố): Zhang (B1 = ExtBiMamba dùng độc lập, Alg. 2 — gỡ [CẦN KIỂM], viết lại câu), Jiang (ICASSP 2025 theo DBLP), Miyazaki (Interspeech 2024), Zevallos ("≤ 10 s", sửa), Liu & Peng (Interspeech 2020 → khoá [LiuPeng2020]), Bouthillier (MLSys 2021 — chỉ nguyên tắc, không phải thủ tục hai tầng; note gap 9.6 cập nhật), Hägele (NeurIPS 2024), Yang GS2 (ACL 2025). Bảng trích dẫn cuối `ch3_khung.md`.
- **Phân tích đánh đổi (bản xem trước):** `docs/notes/ket_qua/danh_gia_danh_doi_2026-10-20.md` — bảng chi phí mọi hệ thống, 5 nhận xét (10 ms bị lấn át mọi trục; 40 ms chọn theo chi phí, ConExt lấn át B1; LM là đòn bẩy rẻ nhất; nhỏ vs pre-train: nhường 1,2 điểm trong kênh / 8,4 ngoài kênh; Mamba chỉ có giá trị ở audio dài), bảng khuyến nghị theo tình huống. Chờ người dùng duyệt.
- **Bộ slide bảo vệ (bản phác, người dùng muốn hình dung):** artifact https://claude.ai/artifact/XTrMQr4CvYq1WDVwDYUmcQ (riêng tư), 14 slide ~15′ theo Mẫu 3 (trình bày 15′ tối đa 20′, demo 5′, hỏi đáp ~10′; Rubric Mẫu 2: thuyết trình 10%, trả lời hội đồng 20%, phương pháp 40%). Số lấy từ lần chấm đã khoá; ghi chú lời nói + thời lượng mỗi slide. Chỗ trống: tên đề tài chính thức, họ tên/GVHD/ngày.
- **Slide (tiếp):** người dùng: slide thiếu cơ chế, không có điểm để hội đồng hỏi → thêm 3 slide (kiến trúc ba encoder kiểu Hình 3 Jiang; bên trong khối Mamba + đảo chuỗi theo độ dài thật; ví dụ thật train:944 "silicon valley…" qua 6 hệ thống + train:13537 "ready"→"rady" nhại lỗi nhãn) → 17 slide ~19–20′. Người dùng vẫn chưa ưng, chưa thấy slide đề tài tương tự → tìm tham khảo (đã kiểm link sống): Hung-yi Lee ML2025 Mamba (pdf/pptx), NeurIPS 2024 "Demystify Mamba in Vision" (slides), sigport ICASSP 2021 "A Better and Faster E2E Model for Streaming ASR" (Conformer), sigport ICASSP 2022 Conformer hybrid ASR Switchboard, Edinburgh ASR lecture CTC, UBC CPSC503 intro E2E ASR. Không tìm thấy slide công khai cho Mamba-ASR (ConMamba/Zhang/Miyazaki) hay bảo vệ luận văn ASR tiếng Việt.

### 2026-10-20 (đóng phiên)
- **Đã làm:** GS2 xong toàn bộ (Parakeet zs E 12,66% sau 2 lần sửa kernel: numpy, package `scripts` của NeMo); `score_gs2` thêm `_lmtuned`; khung Ch3 thêm Bảng 3.0, 3.3.6 + Bảng 3.5b, đoạn GS2 ở 3.5.2, viết lại đối chiếu Jiang (sửa lỗi gán nhận định hiệu quả cho độ chính xác); kiểm 8 trích dẫn Ch3 (bảng cuối `ch3_khung.md`); note tổng hợp kết quả (`tong_hop_ket_qua_2026-10-20.md`, cột Chốt trống) và phân tích đánh đổi (`danh_gia_danh_doi_2026-10-20.md`); bộ slide bản phác (artifact riêng tư https://claude.ai/artifact/XTrMQr4CvYq1WDVwDYUmcQ). Chốt: không khai "thứ tự quyết định sau v1"/"sự cố 13/10"; chưa chuyển Word (1 tuần riêng cho văn phong + Word sau).
- **Chưa commit** (chờ người dùng): `eval_gs2.py`, `decode_cpu.py`, `launch_cpu.py`, `score_gs2.py`, output `gs2_eval_2026-10-19/` + `gs2_scored_2026-10-19/`, các note mới/sửa, Plan, TODO (rà username/dữ liệu gated trước khi add).
- **Phiên sau bắt đầu từ:** (1) người dùng điền cột Chốt `tong_hop_ket_qua` (6 dòng [CHỌN]) + trả lời mục 5 `danh_gia_danh_doi`; (2) hỏi: giữ hay bỏ giới hạn số 6 "clean-test đã được nhìn trong giai đoạn thiết kế"; (3) slide: người dùng xem tham khảo rồi quyết hướng (bố cục, mức chi tiết cơ chế; 17 slide → chuyển 2 slide sang dự phòng?); (4) Hình 3.2 forest plot PNG (0 GPU); (5) thêm mô hình fine-tune — AI nghiêng không, người dùng chưa chốt; (6) commit khi được bảo. GPU A vẫn khoá; đóng băng GPU 25/10.

### 2026-10-23 — rà phương pháp; mục lục mới theo Jiang 2024; dựng văn phong khóa luận
- **Rà phương pháp** (chỉ đọc + script scratchpad đọc `per_sentence.tsv`, 0 GPU): lõi đúng — không rò rỉ (tokenizer,
  CMVN tính lại sau tách 29 video `f102ee9`; LM lọc trùng val/val_unseen/clean-test; Parakeet FT chỉ train trên manifest
  đã bỏ video giữ riêng); 0 chữ số trong output pre-train trên clean-test; code chấm tái lập đúng mọi CI hai tầng đã
  công bố. **Cần sửa chữ (chưa sửa, chờ người dùng):** (1) GS2 greedy B1 − ConExt +0,36 [+0,02; +0,72] → theo luật là
  "ConExt tốt hơn B1 ngoài kênh"; `tong_hop_ket_qua` mục 0, `danh_gia_danh_doi` N2, `ch3_khung` 3.3.6 đang ghi "giữ
  nguyên"/"không phân biệt được"; với LM ngoài kênh relPE tốt hơn ConExt và B1. (2) CI hai tầng là khoảng của hiệu giữa
  hai lần train đơn lẻ, không phải CI của trung bình 3 seed (độ nhạy trung bình-seed: chỉ cặp LM B1 − relPE đổi, [+0,06;
  +1,10]) — giữ luật, sửa câu mô tả. (3) "lợi thế B1 chỉ khi không lấy mẫu xuống" → "chỉ thấy ở cấu hình 10 ms"; tương
  tác (B1 − relPE) 10 ms − 40 ms = −2,64 [−3,82; −1,57] (mô tả, tính sau khi có số). (4) 10 ms − 40 ms cùng encoder:
  B1 +1,66 [+1,04; +2,31] · ConExt +4,14 [+3,26; +5,12] · relPE +4,30 [+3,36; +5,48]. (5) tỉ lệ sd seed relPE/Mamba: F
  p một phía 0,08/0,14; "nhất quán trên hai tập độc lập" sai (cùng 3 lần train). (6) "kiểm trực tiếp Jiang" → "phù
  hợp". (7) Khai: lmw/ws dò trên nửa val_unseen (cùng 29 video với clean-test). (8) So sánh A thiên lệch có lợi cho nhóm
  train từ đầu (cùng kênh, neo nhãn, 11 câu trùng, LM cùng kênh). (9) Giả định 30 nhóm GS2 = bản ghi chưa kiểm. (10)
  "GPU-giờ" = giờ tường kernel 2×T4. Nêu hệ quả, không đổi quyết định: 11/203 không khai; không khai thứ tự quyết định
  (mâu thuẫn note gap 9.4 điều 1); giới hạn #6.
- **Mục lục mới** `docs/khoa_luan/muc_luc_de_xuat.md` theo trình tự Jiang 2024 + Mẫu 5: ~59 tr (Mở đầu 5 · Ch1 lý thuyết
  11 · **Ch2 kiến trúc mô hình** 10, từ dưới lên · Ch3 thực nghiệm và kết quả 24 · Ch4 thảo luận và ứng dụng 7 · KL 2);
  bản 16/10 lưu `muc_luc_de_xuat_luu_2026-10-16.md`. Người dùng "được rồi" (duyệt hướng); 5 mục [CHỌN]/[GVHD] chưa trả
  lời. Mẫu 5: TLTK đánh số theo thứ tự chữ cái tác giả.
- **Văn phong:** skill `.claude/skills/van-phong-khoa-luan/SKILL.md` (gitignore). Chốt: đoạn kết quả "đặt câu hỏi rồi trả
  lời"; xưng "chúng tôi"; Việt kèm Anh lần đầu; người đọc là hội đồng CNTT; có công thức, không chứng minh; vài ví dụ đắt
  giá; giới hạn nói thẳng rồi chuyển hướng phát triển; chú thích đọc độc lập; tiểu kết ngắn; chỉ giải thích nguyên nhân
  khi có nguồn; tên mục danh từ; câu mời "Hãy xét…"; ký tự (– khoảng, − trừ, không —, _ chỉ trong mã); "điểm phần trăm"
  định nghĩa một lần rồi luân phiên cách nói. Đoạn giải thích chuẩn = đoạn S4/Δ người dùng gửi (từ quen "Cách mà",
  "song", "Lưu ý rằng"). **Chưa chốt:** nét riêng (câu đùa bị bác: cố gây cười, sai chỗ), cách mở chương, câu hỏi tu từ.
  Đoạn nghị luận seed 42: bản dài bị chê (dài + câu CI), bản ≤ 80 chữ người dùng vẫn "chưa ưng bụng".
- Mẫu thử `docs/khoa_luan/ch2_mau_thu.md` (2.2.1; ô [CẦN KIỂM]: Jiang dùng chung phép chiếu vào–ra?). Bài tập
  `docs/khoa_luan/bai_tap_van_phong.md`: người dùng chọn **Phần B — 15 đề ngoài dự án**, viết tay.
- Không chạy GPU, không sửa code, không commit.
- **Phiên sau bắt đầu từ:** (1) nhận bài viết tay → rút nét lặp → ghi skill mục 1/10 → viết lại 1–2 đề theo giọng khóa
  luận để so → viết 2.2.1 rồi trọn Ch2 (nhịp: một mục thử rồi cả chương); (2) hỏi 5 mục [CHỌN]/[GVHD] của mục lục, rồi sửa
  "Phần NỘI DUNG" đề cương mới cho khớp; (3) hỏi có áp các sửa chữ của bản rà phương pháp không; (4) việc cũ ở TODO.

### 2026-10-26 — GVHD hỏi "chỉ có vậy thôi à?": chấm lại theo Mẫu 2, đề xuất bổ sung (chưa làm)
- **Bối cảnh:** người dùng báo thầy hỏi "chỉ có vậy thôi à?"; hỏi khóa luận đủ chưa (thang 100), cần thêm gì.
- **Đã làm (0 GPU, chỉ đọc + script scratchpad):** đọc `docs/bieu_mau/Mau-2-Rubric-KLTN-Edit.docx` — 7 tiêu chí: thực tiễn +
  hiểu biết 10 · phương pháp/thiết kế/**giải pháp**/**sản phẩm**/khối lượng 40 · thuyết trình 10 · đọc tài liệu ngoại ngữ 5 ·
  tổng hợp + viết 10 · hình thức 5 · hỏi đáp 20; mỗi tiêu chí 5 mức 0/20/50/70/100%. Bảng điểm 15/10 chia theo cấu trúc tự
  đặt (không có tiêu chí thực tiễn, hỏi đáp 10). AI ước: nội dung hiện có + viết tốt ~78 (72–84); hụt ở tiêu chí 1 và phần
  giải pháp/sản phẩm của tiêu chí 2 (khớp ý thầy 17/10 "có giải pháp rõ ràng"); thêm (1) + (2) dưới đây ~86.
- **Kiểm dữ liệu:** 29 video giữ riêng (val_unseen + clean-test) = 11,7 h; theo số seg liền nhau: 81 chuỗi ≥ 1′, 26 ≥ 5′,
  22 ≥ 10′, dài nhất 38,5′ (câu train 10–15 s). Lý do 03/10 không đo WER audio dài ("validation gần như không có chuỗi seg
  liên tiếp dài") hết đúng từ khi tách video 04/10.
- **Nguồn đã kiểm:** Miyazaki 2024 mục 3.3, Hình 2 (PDF local: TEDLIUM2, ghép 3 đoạn liên tiếp; Conformer-Transformer AED suy
  giảm từ > 25 s, thêm Mamba giảm suy giảm); Alabi và cs. arXiv:2607.01502 (preprint 07/2026; CTC ký tự, ConMamba vs
  Conformer, ghép 60–240 s: cả hai suy giảm, ConMamba hơi ít hơn); DeciMamba arXiv:2406.14528 và Chiu 2019 arXiv:1911.02242
  (mức tóm tắt).
- **Đề xuất, chờ người dùng chốt (chưa chạy gì):** (1) WER audio dài trên 29 video giữ riêng: 9 checkpoint 40 ms ep37, nguyên
  chuỗi L ∈ {1, 5, 10′, cả chuỗi} vs đoạn gốc vs chia đoạn 15 s chồng lấn; greedy, bootstrap theo video, quy tắc chốt trước;
  CPU qua `mamba_ref.py` (0 GPU-h; kiểm khớp output GPU đã lưu trước) hoặc ~0,5–1 GPU-h nếu mở GPU sau mốc 25/10; ~2–3 ngày.
  (2) Công cụ phiên âm audio dài (mở rộng demo: mốc thời gian CTC, xuất SRT), ~1–2 ngày. (3) Tuỳ chọn: tốc độ CPU Conformer
  PE 40 ms (± int8), hình Pareto. Không đề xuất train/seed/LR mới; thêm mô hình fine-tune vẫn là ⛔ của người dùng (AI: không).
- **Người dùng: "để những người bạn cùng nghiên cứu"** → 4 tác tử chạy nền (chấm bịt mắt theo Mẫu 2; tổng quan tài liệu audio
  dài; fork đo khả thi CPU, không tạo WER; phản biện đề xuất). Kết quả tổng hợp ghi vào mục sau.
- **Tính mới (người dùng hỏi):** quét nhanh nguồn — không thấy công bố Mamba cho ASR tiếng Việt (Zevallos 2025 và MLMA không có
  tiếng Việt); bài VietSuperSpeech không báo WER, không định lượng nhãn. Ba điểm mới đứng được: so sánh có kiểm soát trên tiếng
  Việt hội thoại (theo phạm vi tra cứu); thứ hạng đổi theo độ phân giải token (tương tác −2,64 [−3,82; −1,57], mô tả, cấu
  hình 40 ms trộn 4 khác biệt); bộ đánh giá cho VietSuperSpeech (tách theo video, clean-test hiệu đính, đo nhãn giả, split
  gốc trùng video, bản phát hành khác bài báo).
- **Ý tưởng "hay ho" (người dùng hỏi; 0 GPU, không train, chờ chốt):** (A) tiêu đề tập podcast làm ngữ cảnh cho beam search
  (thêm từ vào từ điển + điểm thưởng; đo B-WER/U-WER theo Le 2021; Jung 2022 "Spell my name") — nhắm từ ngoại/tên riêng
  (3,1% số từ, ~21% số lỗi, WER 88–100%); (B) mở hộp đen Mamba: Δ theo thời gian (Gu & Dao 2023 mục 3.5.2, PDF local) +
  attention ẩn (Ali và cs., ACL 2025) cạnh attention Conformer; (C) độ tin cậy từ theo entropy (Laptev & Ginsburg
  arXiv:2212.08703) + transcript tô màu. Loại: nhầm hỏi/ngã (lỗi chỉ sai thanh 0,17–0,28% số từ — rq3 mục 4).
- **Tác tử A (chấm bịt mắt theo Mẫu 2) xong:** viết tốt 85 (8,5) · viết vội 59; GVHD chấm tiêu chí 2 mức Khá → trung bình
  ~8,1 → "Giỏi chưa chắc" (khớp AI). Đủ tốt nghiệp; rủi ro thật là thủ tục (Plan chỉ ghi 1 lần gặp trực tiếp 01/10; đề
  cương mới chưa gửi/duyệt). "Chỉ có vậy" có thể nhắm: độ rộng "loại" mô hình (thầy buổi 1: "ASR và fine-tune"), thiếu
  "giải pháp rõ ràng", thầy chưa thấy phần sau 17/10. **Báo cáo gửi thầy 17/10 ghi "kém mô hình 0,6B chỉ 1–3,5 điểm"**
  (đúng trên clean-test lúc đó; GS2 sau đó: ngoài kênh chênh 8–16 điểm) → cần cập nhật cho thầy. Câu hỏi mới phải chuẩn bị:
  FastConformer attention cục bộ chạy tới 11 h (Rekesh 2023) → "Mamba chạy được 1 giờ" phải nói rõ là so với attention toàn
  cục; câu hỏi "dùng AI tới đâu". A khuyên gặp thầy tuần này với 1–2 trang + thực đơn, trước khi làm bổ sung nào.
- **Chốt cuối phiên (người dùng):** làm **A** (tiêu đề tập làm ngữ cảnh) và **B** (mở hộp đen Mamba) — 0 GPU, không train;
  các đề xuất còn lại treo. Dừng tác tử B/C/D trước khi báo cáo (token ~90%). **Từ nay nương theo khối ▶ của `TODO.md`.**
- **Ghi đủ để không nhắc lại:** `docs/notes/review/danh_gia_bo_sung_2026-10-26.md` (đánh giá, điểm mới, đề xuất, các bước A/B).
- **Phiên sau bắt đầu từ:** `TODO.md` khối ▶ 26/10 — quy tắc phân tích A → người dùng chép 29 tiêu đề → code A → B.

### 2026-10-27 — triển khai A + B (dừng giữa chừng vì hết token)
- **Đã làm:** chốt quy tắc A/B trước khi chạy (`docs/notes/ket_qua/bo_sung_AB_quy_tac_2026-10-27.md`; A dùng danh sách tự
  động từ tên file video, chọn b trên `val_unseen_subset` seed 42; B dùng 30 câu val_unseen seed 42). Giao 2 tác tử song
  song (A: tiêu đề tập làm ngữ cảnh; B: hộp đen Mamba), CPU local, không GPU, không đẩy kernel (theo chỉ dẫn).
- **Khi đóng (token 94%):** hai tác tử còn chạy nền, chưa báo cáo; chưa kiểm kết quả, chưa commit.
- **Phiên sau bắt đầu từ:** `TODO.md` khối ▶ dòng "27/10" — kiểm đầu ra trên đĩa, đối chiếu quy tắc + mục "Sửa đổi", làm
  tiếp phần thiếu, tổng hợp số cho người dùng.
- **Phiên tiếp (27/10, sau khi mở lại):** hai tác tử đã ghi gần đủ trước khi phiên cũ đóng. AI kiểm và làm nốt: note kết quả A
  (`bo_sung_A_ket_qua_2026-10-27.md`), điền phần kiểm + kết quả của note B, vẽ hình (c) (`scripts/fig_mamba_m3.py`). Kiểm: (i)
  tái lập 203/203 câu × 9 run; hook B không đổi đầu ra; attention ẩn khớp bản tính thẳng. **A:** b = 4 (nửa tune 441 câu);
  (ii) trùng (i); (iii) B-WER −9 đến −12 điểm (8/9 run giảm) nhưng CI chứa 0, độ phủ 29 từ → không phân biệt được; WER,
  U-WER không đổi; b ≥ 8 hại (overbiasing). **B:** M2 < 1 ở 40 ms (mọi khối ConExt, nửa khối B1), ngược/lẫn ở 10 ms; M1 ≈ 1;
  M3 Conformer ~2–3 s mọi lớp, Mamba 2 s → 0,2–1 s theo độ sâu, 10 ms còn ~một nửa (theo giây). Chưa commit.
- **Phiên sau bắt đầu từ:** người dùng xem hình B + quyết chạy lại A với tiêu đề/mô tả có dấu; rồi viết (khối 23/10).

### 2026-10-27 (chiều) — A điều kiện riêng (iv): tiêu đề có dấu + mô tả tập, lấy tự động
- Người dùng: "đọc note 26–27 và tiếp tục theo hướng đó" → làm bước tiếp của A (note kết quả A mục 3.5), 0 GPU.
- **Chốt trước khi tải/tính:** quy tắc A8 (`bo_sung_AB_quy_tac_2026-10-27.md`) + "Sửa đổi" 27/10 chiều. Không cần người dùng
  chép tay: trang podcast vietcetera.com mang tiêu đề + mô tả YouTube của tập (`episode.video.snippet`); **không dùng
  `transcriptSSR`** (tưởng là bản ghi lời tập; kiểm sau: chỉ chứa id tập, bản ghi nếu có nằm sau API nội bộ — note
  `docs/notes/du_lieu/vietcetera_transcript_2026-10-27.md`). Sitemap không có trang video → 22 video ghép trang podcast theo Jaccard, 6 tìm
  tay, 1 (Yêu Lành SS5 EP4, tiêu đề trên trang đã đổi) giữ danh sách A2.
- Code: `bias_title.py` thêm `fetch-pages`, `lists-iv`, cờ `--variant iv` cho `grid`/`decode`/`score` (mặc định giữ hành vi
  cũ; demo không đổi). Nguyên văn trang ở `data/processed/bias_title/` (gitignore).
- **Kết quả (iv):** độ phủ 56 từ; b = 2; B-WER −4,8 đến −7,7 điểm, 9/9 run giảm, CI chứa 0 → không phân biệt được; WER/U-WER
  không đổi. Sửa `lexicon_with` (Windows: hai tiến trình cùng ghi một file từ điển tạm).
- Người dùng hỏi "giữ được mấy điểm mới" → giữ cả 3, không thêm (ghi mục 3 `danh_gia_bo_sung`); quét nguồn B: chưa thấy phân
  tích attention ẩn trên encoder ASR (2 lượt). Người dùng hỏi theo bảng Mẫu 2 làm gì thêm → AI xếp: tiêu chí 2 (+12) sản phẩm,
  tiêu chí 7 (+6) bảng trả lời 10 câu hỏi; gặp thầy trước.

### 2026-10-27 (tối) — SẢN PHẨM: phiên âm podcast dài → phụ đề
- **Người dùng chốt:** "tập trung hết mức cho phần Giải pháp/sản phẩm" → gộp việc treo (2) công cụ dài + SRT, (3) tốc độ
  CPU, (C) độ tin cậy; 0 GPU. Quy tắc chốt trước: `docs/notes/ket_qua/san_pham_podcast_quy_tac_2026-10-27.md`.
- **Đã làm:** `src/demo/longform.py` (cắt chỗ lặng 10–15 s → beam + LM đã khoá + ngữ cảnh A + mốc thời gian + độ tin cậy
  entropy → SRT/TXT/JSON/HTML); tab Gradio; chữ trùng `lm_hyp` 24/24; 17,8′ chạy 41 s CPU (RTF 0,038). Sửa lệch timestep
  flashlight (T + 2 vị trí).
- **Đo (val_unseen, nhãn giả, 38 chuỗi / 7 video / 3,01 h):** P1 tự cắt 8,22% vs đoạn gốc 8,02%, hiệu +0,20 [+0,07; +0,37] →
  tự cắt kém hơn, nhỏ (báo nguyên); P3 AUROC entropy 0,940 / max-prob 0,937, hiệu +0,003 [+0,002; +0,004]; P2 RTF CPU
  Conformer 0,044 · ConExt 0,067 · B1 0,109 (Mamba bản tham chiếu). Note `san_pham_podcast_ket_qua_2026-10-27.md`.
- **Sau đó:** sửa lọc ngữ cảnh của sản phẩm (hàm chung `context_words`; danh sách (iv) tạo lại trùng từng byte); hình Ch4
  `reports/figures/san_pham/` (ảnh HTML chụp bằng Chrome headless — ảnh người dùng chụp có thanh dấu trang/tab cá nhân, giữ
  local; sơ đồ quy trình đọc số từ file); hướng dẫn `docs/huong_dan_san_pham.md`; `requirements.txt` + `flashlight-text`,
  `scipy`. Người dùng hỏi gộp demo → AI nêu 3 mức; người dùng: **tạm để y, chưa quyết** (⛔ trong TODO).
- **Khuya:** người dùng hỏi còn làm gì + vì sao không 100 (Mẫu 2 chỉ 5 nấc, 100 = cả 3 người chấm Giỏi cả 7 tiêu chí; 83–88
  là kỳ vọng, không phải trần). Người dùng duyệt (2) + (4), 2 tác tử: `transcriptSSR` **chỉ là id tập** (AI ghi sai trước
  đó là "bản ghi lời" — đã đính chính ở mọi note); hình `fig_pareto_cpu.png`. (1) WER audio dài nguyên chuỗi: AI khuyên,
  chưa có lệnh.
- **27/10 18:17 — (1) WER audio dài nguyên chuỗi:** người dùng "chạy việc 1, chia nhỏ cho các bạn phối hợp". Quy tắc chốt
  trước (`audio_dai_nguyen_chuoi_quy_tac_2026-10-27.md`); 2 tác tử viết chạy/chấm (kiểm trên dữ liệu giả + chạy thử 2 chuỗi);
  Conformer chạy được tới 8.279 khung (11,3 GB), 8.918 khung chết → 10/38 cửa sổ "cả chuỗi" bỏ. Đang chạy 9 checkpoint
  tuần tự (~9–10 h).
- **19:16:** xong 2/9 (Conformer s42 18:45, 1.130 ok + 10 skip_mem; ConExt s42 19:13, 1.140 ok), B1 s42 đang chạy; ước xong
  cả 9 ~00:00–00:30 28/10. Người dùng hỏi đẩy Kaggle → AI khuyên không (lợi ~3–4 h rơi vào ban đêm; tốn công đóng gói ~800 MB,
  quota GPU, trộn thiết bị CPU/GPU); người dùng chưa yêu cầu lại. Đã bật theo dõi báo mỗi checkpoint xong (chỉ sống trong phiên).
- **Sau 19:16:** người dùng duyệt thông điệp mở đầu + sườn slide (mục 4) và "cập nhật mục lục và khung" (mục lục 5b; khung
  `docs/khoa_luan/ch3_khung_bo_sung_2026-10-27.md`); bảng hỏi đáp 24 câu `docs/khoa_luan/hoi_dap_bao_ve.md` (tác tử soạn, AI
  rà). **23:36 (1) xong:** cả ba encoder kém đi khi cửa sổ dài (1′ +0,3–0,4; 5′ +0,6–1,5; cả chuỗi B1 +0,96, ConExt +0,65,
  Conformer +1,27 trên 28 chuỗi); Mamba vs Conformer không phân biệt được ở mọi độ dài; B1 − ConExt cả chuỗi +0,31 [+0,01;
  +0,72]; Mamba RTF/RAM không đổi (17,8′ ≤ 7,1 GB), Conformer 9,6–12,9 GB ở 5,5′. Số sơ bộ seed 42 ("Mamba kém đi hơn ở 1′")
  không giữ. Đính chính: RAM B1 11,2 GB lúc kiểm là RSS cộng dồn (thật 5,9 GB).
- **Phiên sau bắt đầu từ (cập nhật 23:45):** chờ người dùng viết bài tập văn phong Phần B (~6 đề) + chọn 5 mục mục lục + Hình
  3.7 → viết văn xuôi theo khung; gộp demo (⛔); gặp thầy + bản 1–2 trang (⛔). Chưa commit gì từ 26/10.
- **(cũ, đã xong) Phiên sau bắt đầu từ:** (1) kiểm `reports/results/long_audio_2026-10-27/hyps/` đủ 9 file × 1.140 dòng (Conformer có
  10 `skip_mem`); thiếu → chạy lại vòng lệnh trong `TODO.md` (tự chạy tiếp). (2) `python -m src.evaluation.long_audio_score`
  → `python scripts/fig_audio_dai.py` → viết `docs/notes/ket_qua/audio_dai_nguyen_chuoi_ket_qua_2026-10-28.md` theo đúng mục 3
  quy tắc (Δ từng encoder × L, CI; so encoder; RAM/thời gian mô tả; kết quả bất lợi báo nguyên) → cập nhật bảng trạng thái
  `danh_gia_bo_sung` mục 4. (3) Còn treo: gộp demo (⛔), bảng trả lời 10 câu hỏi hội đồng, viết Ch4 phần sản phẩm, gặp thầy.
  Chưa commit gì từ 26/10 (danh sách file: `git status`).

### 2026-10-27 — TỔNG KẾT PHIÊN (26–27/10, đóng ~23:50)
- **Bối cảnh:** GVHD hỏi "chỉ có vậy thôi à?" → chấm lại theo Mẫu 2 (~78/100 nếu viết tốt; đòn bẩy: tiêu chí 2 "giải
  pháp/sản phẩm" +12, tiêu chí 7 hỏi đáp +6) → người dùng chọn A + B, rồi "dồn sức giải pháp/sản phẩm", rồi (1)(2)(4).
- **Đã làm + kết quả (mọi quy tắc chốt trước khi đo; sửa đổi ghi trong từng note quy tắc):**
  - A ngữ cảnh tiêu đề (iii) + (iv): B-WER giảm 8/9 và 9/9 run, CI chứa 0 → không phân biệt được; WER không đổi.
  - B hộp đen Mamba: Δ nhỏ ở khung lặng (40 ms); không reset ở ranh giới từ; Conformer 2–3 s mọi lớp, Mamba hẹp dần.
  - Sản phẩm `src/demo/longform.py` + tab Gradio + hướng dẫn: P1 +0,20 [+0,07; +0,37]; P3 AUROC 0,940/0,937; P2 RTF CPU
    0,044/0,067/0,109; hình quy trình, ảnh HTML (headless), Pareto.
  - Audio dài nguyên chuỗi (9 checkpoint, 18:17–23:36): ΔWER 1′ +0,34–0,38, 5′ +0,59–1,50, cả chuỗi +0,65–1,27 (mọi CI ≠ 0);
    Mamba vs Conformer không phân biệt được; B1 − ConExt cả chuỗi +0,31 [+0,01; +0,72]; Mamba RTF/RAM phẳng (17,8′ ≤ 7,1 GB),
    Conformer 9,6–12,9 GB ở 5,5′, dừng ở đó.
  - Bản ghi Vietcetera: không công bố. Bảng hỏi đáp 24 câu. Thông điệp mở đầu + sườn slide chốt; mục lục ~64,5 tr + khung.
- **Đính chính trong phiên (đã sửa mọi note):** (a) `transcriptSSR` không phải bản ghi lời (AI suy từ tên trường); (b) RAM
  B1 11,2 GB lúc kiểm là RSS cộng dồn (thật 5,9 GB); (c) số sơ bộ seed 42 "Mamba kém đi hơn ở 1′" không giữ với 3 seed;
  (d) giờ bắt đầu chạy ghi nhầm "28/10 00:30" (thật 27/10 18:17). Lỗi kỹ thuật đã sửa: lệch timestep flashlight (T + 2),
  tranh ghi file từ điển trên Windows, ngữ cảnh sản phẩm chưa lọc URL.
- **Đánh giá cuối phiên (người dùng hỏi "cần thêm gì"):** không cần thêm thí nghiệm. Còn: commit (⛔ người dùng duyệt), gặp
  thầy + bản 1–2 trang (⛔), CI cho chênh 1,2 điểm trong kênh, hình còn thiếu (3.1, 2.1–2.3, 1.1), Bảng 4.1/4.2, quét nguồn
  thêm cho 3.6, slide theo sườn mới, viết (sau bài tập văn phong). Tuỳ chọn: người hiệu đính thứ hai ~30 câu.
- **Phiên sau bắt đầu từ:** `TODO.md` khối ▶ (đóng phiên 27/10): hỏi người dùng duyệt commit + soạn bản gặp thầy; làm mục
  "AI làm được ngay" 1–3; chờ bài tập văn phong để viết.

### 2026-10-28 — viết nháp Ch1–Ch4 (0 GPU, không commit)
- **Đã làm:** nạp skill `but-phap` + `van-phong-khoa-luan`; viết nháp theo `muc_luc_de_xuat.md` 28/10, mỗi chương một thư mục
  `docs/khoa_luan/chN/` gồm `chN.md` (văn) + `nguon_chN.md` (nguồn từng số, chỗ treo). Ch2 ~2.950 chữ (rà bởi 2 tác tử:
  văn phong; công thức + code — số tham số đếm lại khớp từng tham số; sửa 2 chỗ sai: ablation khác Conformer PE ở **4** điểm,
  ConExt và Conformer PE không "chỉ khác khối trộn chuỗi"). Ch3 ~6.500 chữ (7 bảng; thảo luận/giới hạn cũ chuyển Ch4). Ch4
  ~2.800 chữ (Bảng 4.1 cập nhật hai ô "[chưa đo]"). Ch1 ~3.400 chữ (tác tử fork; công thức 1.1–1.23; trạng thái SSM đổi h → s,
  Ch3 (3.1)–(3.2) đổi chỉ số khung s → j cho khớp). Mở chương: Ch1 (c), Ch2 (d), Ch3 (a), Ch4 (d).
- **Phát hiện khi đối chiếu:** Jiang 2024 dùng chung phép chiếu vào–ra + lấy trung bình (giải ô [CẦN KIỂM] mẫu thử 23/10);
  câu "README mamba-ssm khuyên SSM fp32" trong mẫu thử sai nguồn → thay bằng sự cố NaN; BPE 1.000 **không có lý do ghi lúc
  chọn** (commit `7891303` 14/09 "bắt đầu thử 1000"; lý do giữ 03/10: CTC head không đổi).
- **AI đã áp vào văn Ch3/Ch4 (chưa sửa note gốc) các sửa chữ của bản rà phương pháp 23/10** — đáng chú ý: GS2 tham lam 40 ms
  ConExt tốt hơn B1 (+0,36 [+0,02; +0,72]) theo luật CI; mô tả CI hai tầng; khai LM dò trên nửa val_unseen. Danh sách:
  `docs/khoa_luan/ch3/nguon_ch3.md` mục 3. Người dùng chưa duyệt.
- **Phiên sau bắt đầu từ:** người dùng đọc Ch1–Ch4; trả lời các chỗ treo trong 4 tệp `nguon_chN.md` (BPE, sửa chữ 23/10,
  [CHỌN] Hình 3.4/3.7, [GVHD] UML, khoá trích dẫn mới); mỗi lần người dùng sửa → ghi `but-phap` mục 9. Sau đó: Mở đầu →
  Kết luận → Tóm tắt; hình còn thiếu (1.1, 2.1–2.3, 3.1).
- **Rà cuối phiên (người dùng hỏi "đã đúng đắn chưa"):** nội dung Ch1–Ch4 đúng về số/luật ở phần đã kiểm; chưa chắc: 10 sửa chữ
  23/10 (chờ duyệt), Ch1 chưa rà độc lập (ví dụ tách BPE *nghỉ ngơi* → ▁NGH, Ỉ, ▁NG, ƠI ở `ch1.md:110` chưa kiểm bằng tokenizer
  thật), 12 khoá trích dẫn chưa có PDF, CI Conformer PE + LM vs Parakeet FT. **Slide 20/10 lỗi thời** (7 chỗ, ghi ở TODO ▶) và
  không theo sườn chốt 27/10 (thiếu ngoại suy độ dài, bên trong encoder, ứng dụng, ba điểm mới; CI nên ở slide dự phòng).
- **Đóng phiên 28/10.** Phiên sau: người dùng duyệt chương (ưu tiên 10 sửa chữ 23/10, BPE) → AI dựng lại deck theo sườn chốt,
  giữ link cũ `https://claude.ai/artifact/XTrMQr4CvYq1WDVwDYUmcQ`.

### 2026-10-29 — ba việc 0 GPU không cần quyết định (không commit)
- **Đã làm:** (1) Kiểm ví dụ BPE `ch1.md:110` bằng `configs/tokenizer.model`: `NGHỈ NGƠI` → `▁NGH Ỉ ▁NG ƠI`, khớp (văn bản
  huấn luyện viết hoa). (2) CI chênh trong kênh Conformer PE 40 ms + LM − Parakeet tinh chỉnh: **+1,17 [+0,27; +1,99]**
  (hai tầng video × seed, 3 seed vs 1 run; từng seed +0,81/+1,20/+1,49); tham lam +3,21 [+2,06; +4,35]. Tính từ
  `clean_test_scored_2026-10-16_chung/per_sentence.tsv` + `bootstrap.two_level_bootstrap` (script một lần, không lưu repo).
  Văn Ch3 3.3.4, Ch4 4.2, `hoi_dap_bao_ve.md` sửa "chỉ cách ~1,2 điểm, chưa có CI" → "kém 1,2 điểm (CI…)", kèm giới hạn
  Parakeet một lần chạy. (3) Hình 3.1 forest plot: `scripts/fig_forest_cleantest.py` →
  `reports/figures/clean_test/fig_forest_cap_encoder.png` (7 hàng khớp Bảng 3.3; đơn sắc; điểm đặc = CI ≠ 0).
- **Chưa đổi:** thông điệp 3 ở mục 4 và slide 20/10 vẫn "~1,2 điểm" — sửa khi dựng lại deck.
- **Tiếp (người dùng "tiếp tục"):** Hình 1.1, 2.1–2.3 (`scripts/fig_kien_truc.py` → `reports/figures/kien_truc/`; Hình 2.1 đặt
  đổi tốc độ **trước** log-mel theo code, lệch mô tả nháp cũ); Bảng 4.2 đã có trong nháp, kiểm khớp note; quét nguồn 3.6
  lượt 3: không có công bố phân tích Δ/attention ẩn trên encoder ASR Mamba, thêm LongMamba (ICLR 2025) làm nguồn "phù hợp
  với" (`ch3/nguon_ch3.md` mục 5). Sau đó commit + push theo yêu cầu, **không đẩy `docs/`** (người dùng: "các file doc không đẩy").
- **Phiên sau bắt đầu từ:** người dùng duyệt chương + hình mới, BPE, 10 sửa chữ 23/10, gặp thầy; dựng lại deck theo sườn chốt.
