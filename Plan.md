# Plan.md — Kế hoạch, trạng thái & quyết định

Cập nhật lần cuối: **2026-10-15** (dọn lại mục 1–6 sau bản rà soát độc lập
`docs/notes/review_fable_2026-10-15.md`; nhật ký trước 2026-10-11 chuyển nguyên
văn sang `docs/notes/plan_nhat_ky_luu_tru_den_2026-10-08.md`). Đây là nơi **nối
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
bằng **đề cương mới** (nháp `docs/notes/de_cuong_moi_nhap.md`, chưa duyệt);
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

Bảng, hình, kết luận viết sẵn theo nhánh: `docs/notes/ket_qua_cuoi_khung_2026-10-15.md`.
Quy tắc phân tích chốt trước khi có số: `docs/notes/conextbimamba_gap_2026-10-12.md` mục 9.4.

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

## 3. Trạng thái hiện tại (2026-10-15 khuya)

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Dữ liệu, pipeline, Kaggle Dataset | ✅ | 3 tài khoản A/B/C đều train được |
| v1 30 epoch ×3 + Conformer dài 69 ep | ✅ | phụ lục; avg ep20–29 là số tham chiếu |
| 10 ms ×4 ep37 (B1, ConExt, relPE, không PE) | ✅ train + chấm clean-test 16/10 | B1 0,1432 · relPE 0,1694 · ConExt 0,1705 · không PE 0,2235 (`clean_test_results` bổ sung 16/10) |
| 40 ms ×3 seed 42 (ep37) | ✅ train + chấm clean-test | relPE 0,1211 · B1 0,1264 · ConExt 0,1309 (greedy); + LM 0,1024 / 0,1115 / 0,1081 |
| Seed 43/44 ở 40 ms | ▶ 6/6 run đẩy 16/10 08:36–08:49 (A ConExt, B relPE, C B1) | ~32 GPU-h, xong ~14:00–15:00 |
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
| 2026-09-14 | Hạ tầng thật là **Kaggle** (2x T4), không phải Colab; giữ nguyên `.docx` đã nộp | Lệch có chủ đích. `docs/notes/mamba_ssm_install_log.md` |
| 2026-09-14 | **Mamba-1/S6**, pin `mamba-ssm` `v2.3.1`; Mamba-3 chỉ nhắc như hướng phát triển | `docs/notes/mamba_versions.md` |
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
`docs/notes/plan_nhat_ky_luu_tru_den_2026-10-08.md` (2026-10-15, không sửa chữ nào); trước 2026-09-19 xem `git log`.

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
- **Phân tích ConExtBiMamba** (người dùng yêu cầu): `docs/notes/conextbimamba_gap_2026-10-12.md` — kém B1 3,2 điểm
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
     `kaggle kernels output <kernel> -p <dir> --file-pattern '.*\.(json|jsonl|log|csv)$'` (B: `KAGGLE_CONFIG_DIR=C:/Users/Dell/.kaggle_Duc`).
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
  kernel eval; công cụ + kernel RQ2 cuối; nháp đề cương mới `docs/notes/de_cuong_moi_nhap.md`.
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
  code tác giả dùng subsampling 4×; 4 lệch bài không phải lỗi. Nháp: đề cương (`docs/notes/de_cuong_moi_nhap.md`, chưa
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
  v2a_branch_2026-10-13}`. Vẫn không track: `docs/notes/de_cuong_moi_nhap.md`, `reports/results/error_analysis/` (chưa kiểm).
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
  `docs/notes/label_quality_direction.md`; 0 GPU-h, dựa trên bộ hiệu đính clean-test.
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
  < 0,3). Hướng dẫn mục 2a. Bảng tên chuẩn có nguồn: `docs/notes/clean_test_entities.md`.
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
  Toàn bộ kết quả + số: **`docs/notes/glued_words_2026-10-15.md`**; file agent: `D:/Model/kltn_decoding_2026-10-15/`.
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
- 21:30 Ghi nhận xét clean-test vào `docs/notes/clean_test_results_2026-10-15.md`. Người dùng: **có thể thêm tài khoản
  Kaggle, tối đa 10**; **ý nghĩa khóa luận phải thật**; sẽ nhờ Fable 5.1 rà soát toàn dự án và đề xuất phương án cuối —
  prompt `docs/notes/prompt_review_fable_2026-10-15.md` (đầu ra dự kiến `docs/notes/review_fable_2026-10-15.md`).
  **Phiên sau:** đọc bản rà soát của Fable (nếu có) cùng người dùng trước khi làm bất cứ việc GPU nào.

### 2026-10-15 (khuya) — rà soát độc lập toàn dự án (Fable 5.1, chỉ đọc)
- **Làm:** đọc CLAUDE/TODO/Plan/ARCHITECTURE, 9 note, `reports/results/` (clean-test đã chấm, RQ2 cuối, metrics mọi
  run), tra 10 nguồn → **`docs/notes/review_fable_2026-10-15.md`**. Không chạy GPU, không sửa code.
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
- **Làm:** `docs/notes/review_fable_scorecard_2026-10-15.md` (62/100 hôm nay → 84/100 nếu làm đúng phương án; việc 72 giờ
  tới); `docs/notes/ket_qua_cuoi_khung_2026-10-15.md` (9 bảng, 5 hình, 5 kết luận viết sẵn theo nhánh). **Dọn file hướng
  dẫn** (người dùng yêu cầu, đã ghi TODO từ 12/10): `CLAUDE.md` viết lại theo hiện trạng (hai cấu hình 10/40 ms, luật
  clean-test chỉ để báo cáo + ep37, không đề xuất GPU mới, tải output kernel ngay, cạm bẫy `python -I`); `TODO.md` viết
  lại gọn (bản cũ nguyên văn `docs/notes/todo_luu_tru_2026-10-15.md`); `Plan.md` mục 1–6 viết lại, nhật ký trước
  2026-10-11 chuyển nguyên văn sang `docs/notes/plan_nhat_ky_luu_tru_den_2026-10-08.md`.
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
