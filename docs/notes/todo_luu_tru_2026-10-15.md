# Lưu trữ TODO.md nguyên văn tới 2026-10-15 (trước khi viết lại)

File này là bản sao nguyên văn của `TODO.md` lúc dọn dẹp 2026-10-15 (sau rà soát Fable). Giữ để truy vết lịch sử
train full, checklist kiểm kernel, lệnh đẩy. `TODO.md` mới chỉ giữ việc còn mở.

---

# TODO tổng — Mamba vs Conformer ASR

Cập nhật lần cuối: **2026-10-13 15:10** (nhóm rẽ nhánh xong; relPE phiên 2; 40 ms ×3 chạy trên A + C); trước đó **2026-10-12** (fine-tune Parakeet chạy; công thức v2 code xong); trước đó **2026-10-05** (preflight B OK, kernel train thử thêm log); trước đó **2026-10-04** (ConExtBiMamba code + test CPU; bước 3 code dùng chung; front-end CMVN + SpecAugment, dropout Mamba, giữ CTC; test độc lập theo video). **Nguồn sự thật duy nhất cho việc cần làm /
đang chặn / đã xong.** Kế hoạch tuần + quyết định + nhật ký phiên ở
[`Plan.md`](Plan.md); kiến trúc ở [`ARCHITECTURE.md`](ARCHITECTURE.md).

Câu hỏi cần mang đi hỏi (người dùng/GVHD): [`QA.md`](QA.md).

## ▶▶▶▶▶▶▶ PHIÊN SAU BẮT ĐẦU Ở ĐÂY (ghi 2026-10-15 ~21:30)
- ✅ **Bản rà soát Fable 5.1 đã có: `docs/notes/review_fable_2026-10-15.md`** (prompt
  `docs/notes/prompt_review_fable_2026-10-15.md`) → đọc cùng người dùng trước mọi việc GPU; ⛔ 12 quyết định ở mục 4.
  Hai lỗ hổng 0 GPU-h phát hiện: **run Conformer relPE 10 ms (B s3) chưa tải/chưa chấm**; **nhóm 10 ms chấm clean-test
  bằng ep28 v1 thay vì ep37** theo quy tắc 9.4. Khuyến nghị GPU duy nhất: 2 seed × 3 mô hình 40 ms (~32 GPU-h).
  Tài khoản Kaggle có thể thêm tới **10** (theo note: không cần); yêu cầu: **ý nghĩa khóa luận phải thật**.
- Kết quả clean-test bản hiệu đính + nhận xét: `docs/notes/clean_test_results_2026-10-15.md`.

## ▶▶▶▶▶▶ CẬP NHẬT 2026-10-15 ~19:50
- ✅ Bảng giải mã 3 mô hình 40 ms trên nửa dev xong → `docs/notes/glued_words_2026-10-15.md` mục 3.1. Beam 10 + từ
  điển + LM: B1 0,0889 · ConExt 0,0859 · relPE 0,0812 (greedy 0,1029 / 0,1020 / 0,0992); từ dính → ~0.
- ⛔ Người dùng chọn: (a) chỉ bảng phụ giải mã (0 GPU) · (b) + BPE3000 rẽ nhánh (~4,5–6 GPU-h, cần code) · (c) CR-CTC
  (~18–36 GPU-h). Kèm: có áp giải mã cho cả 5 mô hình trên clean-test sau hiệu đính không.
- 🔄 Trang hiệu đính đang chạy http://127.0.0.1:7861. Người dùng báo **hiệu đính xong 203/203** (15/10 tối); nhiều câu
  quên ghi chú `tiếng Anh` — tự phát hiện: 33 câu có từ ngoài âm tiết Việt mà thiếu tag. **Người dùng duyệt cách làm
  (15/10):** bảng loại từ theo từng từ `data/processed/clean_test_word_tags.tsv` (`python -m src.data.word_tags`,
  230 từ + 8 lần từ mơ hồ, cột `guess` đoán sẵn) → ✅ người dùng duyệt (sửa ở cột `guess`: bar/new/street → tiếng
  anh, uba → viết tắt; AI chép guess → label). 20:23 AI sửa manifest theo lời người dùng: #11 upan → u penn, #86 med →
  mad + video adis và credit breatter → visual artist và creative director, #99 nthy → nft (sao lưu
  `clean_test_manifest.bak-20261015-before-fix86.json`; 3 câu có `corrected_at` 20:23; #86 thêm suboy → suboi). Bảng từ 239/241 (người dùng gán an, to → tiếng việt; oh my god, top → tiếng anh; on boarding → tiếng anh; `u@#190` → nhãn mới `từ đệm`) → ✅ **241/241**.
  File đang bị gitignore — hỏi có thêm
  ngoại lệ như manifest không (là đầu vào chấm WER).
- (c) CR-CTC — người dùng đang cân nhắc; số liệu bài + ước chi phí: note mục 3.2. Phương án chuẩn bị (chưa duyệt):
  3 mô hình 40 ms (B1, ConExt, relPE), cấu hình v2a sub4, chỉ đổi loss; theo bài: α 0,2, time mask ×2,5, nửa batch
  (8 × 2 bản) + nửa epoch (19) → ≈ chi phí run 40 ms hiện có (4,9–6,0 h/mô hình, ~17–19 h quota); mốc so = 3 run 40 ms
  đã có. ⛔ chốt: số epoch giảm LR khi tổng 19; cách nhân 2,5 time mask (ta 10 vùng ≤ 5%/vùng). Ước lợi (ngoại suy,
  chưa đo): −3% (GigaSpeech) đến −22% (Aishell-1) tương đối; Mamba chưa có bằng chứng.
- ✅ 15/10 ~21:00 **chấm lại clean-test bản hiệu đính xong** (`python -m src.evaluation.score_clean_test` → `reports/results/clean_test_scored_2026-10-15/`; số ở `Plan.md` nhật ký 21:00). Còn: Parakeet fine-tune chưa chấm clean-test (best.pt trên Kaggle, cần GPU — hỏi); cập nhật trang báo cáo.
- WER với tiếng Anh: vẫn dùng WER; báo thêm WER tách từ Việt / từ tiếng Anh; chốt danh sách cách viết thay thế trước khi chấm.

## ▶▶▶▶▶ PHIÊN SAU BẮT ĐẦU Ở ĐÂY (ghi 2026-10-15 ~17:00, dừng vì hết token)
1. Đọc `docs/notes/glued_words_2026-10-15.md` (từ dính: chẩn đoán xong, phương án train xong, giải mã dở).
2. Hoàn tất bảng giải mã 3 mô hình 40 ms trên nửa dev val_unseen (log-prob `D:/Model/kltn_checkpoints/logprobs_v2a_s4/`,
   code `D:/Model/kltn_decoding_2026-10-15/decoding/`) → ⛔ người dùng chọn: bảng phụ giải mã / BPE3000 / CR-CTC.
3. Bật lại trang hiệu đính (`python -m src.data.correct_clean_test`); người dùng hiệu đính tiếp (113/203). Chưa commit.

## ▶▶▶▶ PHIÊN SAU BẮT ĐẦU Ở ĐÂY (đóng phiên 2026-10-13 ~21:00) — chi tiết `Plan.md` nhật ký cùng ngày
1. **Hướng mới đã chốt: chất lượng nhãn VietSuperSpeech** → đọc `docs/notes/label_quality_direction.md` (người dùng đọc;
   mang đi hỏi thầy Dũng cùng đề cương). Không thêm run train cho hướng này.
2. B relPE s3: s2 hết trần ~22:10 → kiểm returncode + `latest.pt` → `launch.py --model v2a-conformer-relpos --session 3
   --account B --push` ✅ **đã đẩy 14/10 01:10** (s2 COMPLETE, returncode 0, dừng ep33 step 703; s3 resume đúng ep33
   step 703, VRAM 5,1 GiB, s/step đang giảm dần 1,64 → 1,33 lúc khởi động; ước xong ~04:00-06:00). Không in WER relPE.
3. ✅ Xong 13/10: 40 ms × 3 (38/38, 0 nonfinite; output `reports/results/train_full/v2a-s4-*/s1/`); RQ2 cuối (trọng số
   ngẫu nhiên, `reports/results/rq2_final_2026-10-13/`, chưa phân tích). WER 40 ms val_unseen (nhãn giả, khám phá):
   ConExt 0,1115 · B1 0,1134 (relPE 40 ms chưa xem — chờ quy tắc trục GPU-giờ).
4. ⛔ Chờ người dùng: số epoch "train dài 40 ms" (người dùng thiên về, AI ước ~80 epoch ~33 GPU-h — chưa có config);
   RQ chất lượng nhãn: RQ riêng hay gộp RQ3; gõ lại từ đầu ~20 câu để đo neo nhãn?; các mục ⛔ cũ ở khối dưới.
5. Không đề xuất thêm thí nghiệm GPU mới (người dùng 13/10: đổi hướng liên tục tốn thời gian). Độ sâu = nhận xét từ dữ liệu có sẵn.
6. Quota 13/10 21:00: A 18,2 h · B 22,0 h · C 12,2 h đã dùng (reset 17/10).

## ▶▶▶ CẬP NHẬT 2026-10-13 19:25
- ✅ A `train-full-v2a-s4-conextbimamba-s1` COMPLETE 38/38 (301′, 0 nonfinite, 0,149 s/step, 1,2 GiB): ep37 WER val
  0,1128 / val_unseen 0,1115 (nhãn giả, khám phá) → `reports/results/train_full/v2a-s4-conextbimamba/s1/`.
- 🔄 A `rq2-final-asr` đẩy 19:12 (`0090bd7`, trọng số ngẫu nhiên — encoder configs/v2 ≡ v2a khi tắt InterCTC; độ dài
  mặc định `DEFAULT_LENGTHS_S`), ước 60-80′. Quota A 17,4/30 h lúc đẩy. Kiểm: `fetch` output `rq2_final/`.
- 19:11: B relPE s2 ep28 (hết trần ~22:10, ~ep34) → đẩy `--session 3 --account B`; C s4-relPE ep32 (xong ~20:00),
  C s4-mamba ep28 (xong ~20:40) → tải output nhỏ. Sau đó C rảnh: việc kế tiếp là cổng seed 2 (⛔ hỏi người dùng).

## ▶▶ PHIÊN SAU BẮT ĐẦU Ở ĐÂY (ghi 2026-10-13 15:10) — chi tiết: `Plan.md` nhật ký "2026-10-13 (đóng phiên ~15:10)"

1. Kiểm 3 run 40 ms: A `<A>/train-full-v2a-s4-conextbimamba-s1` (~22:00), C `<C>/train-full-v2a-s4-conformer-relpos-s1`
   (~19:30), C `...-v2a-s4-mamba-s1` (~20:30) → tải output nhỏ → `reports/results/train_full/v2a-s4-*/s1/`.
2. B relPE phiên 2 hết trần ~22:00 (~ep34) → kiểm `latest.pt` → đẩy `--session 3 --account B`. **Không in WER relPE.**
3. Sau đó: chấm clean-test ep37 cả nhóm → RQ2 cuối.
4. ⛔ Chờ người dùng: quy tắc trục GPU-giờ (AI đã lỡ thấy 1 số WER relPE 13/10 12:58 — khai khi chốt); review
   `docs/khoa_luan/muc_luc_de_xuat.md`; xác nhận sửa Ch3 (ep37, không trung bình pha giảm); commit file mới (reports v2a,
   `scripts/plot_v2a_branch.py`); `<B>` trong `launch.py` public; các mục cũ (normalize_text, đề cương, lý do ConExt,
   eval ngoài miền).
5. ✅ 13/10: ConExt v2a ep37 0,1525/0,1506 · Conformer v2a 0,2094/0,2068 · B1 v2a 0,1271/0,1252 (val/val_unseen, nhãn giả);
   tài khoản C OK (2x T4, 8 dataset đọc được). Việc tay: hiệu đính (11/203), đề cương + hẹn GVHD trước ~17/10.

## ▶ PHIÊN SAU BẮT ĐẦU Ở ĐÂY (ghi 2026-10-12; bản 11/10 + cập nhật)

| Việc | Trạng thái | Làm gì |
|---|---|---|
| Thư gửi thầy Dũng (báo cáo lần 2, đổi tên đề tài, xin báo cáo sớm) | người dùng tự gửi | ⛔ hỏi: thầy trả lời chưa, ngày gặp → `Plan.md` mục 6 |
| Hiệu đính clean-test | 113/203 (14/10 tối) (`python -m src.data.correct_clean_test`) | xong → tính lại WER 5 mô hình từ `reports/results/final_compare_2026-10-08/predictions.tsv` + bootstrap → cập nhật trang báo cáo · **mở (người dùng):** quy tắc chấm từ đệm/lặp từ/cách viết thay thế; N-WER/CS-WER; câu sau mốc `OURS_VISIBLE_FROM` 14/10 23:17 đã thấy output mô hình → báo thêm WER trên 113 câu trước mốc · **15/10 (người dùng phát hiện):** ConExt/B1 ra *make up* ở #121 vì nhãn train tự mâu thuẫn (MAKE UP 80 / MAKEUP 11; tương tự START UP 26/194, CHECK IN 24/11, PLAY LIST 18/5, LIVE STREAM 5/52) → ứng viên danh sách cách viết thay thế lúc chấm, lấy từ nhãn train, chốt trước khi chấm · **15/10 RQ3 ứng viên:** lỗi *dính âm tiết* (nghỉơi, mìnhỉ — mất ranh giới từ + phụ âm đầu) chỉ gặp ở mô hình CTC: Conformer 387 · ConExt 159 · Parakeet 137 · B1 86 · PhoWhisper (AED) 5 từ trên 203 câu (predictions 08/10, đếm bằng regex âm tiết) → đưa vào phân tích lỗi, 0 GPU |
| Trang báo cáo | https://claude.ai/artifact/VFAfKywgz6UjTu3PGNXZF1 (công khai) | nguồn `docs/tong_ket/2026-10-08/mamba_asr_report.html` (gitignore); sửa → đăng lại cùng URL |
| Đề cương mới | 🔄 🔴 agent soạn nháp `docs/notes/de_cuong_moi_nhap.md` | người dùng duyệt → chuyển Word → gửi thầy trước ~15/10 |
| ~~Train thêm~~ | ✅ người dùng quyết 2026-10-12 không chờ GVHD → công thức v2 (dòng dưới) | QA Q11 coi như đã trả lời |
| **Fine-tune Parakeet lần chính** | ✅ COMPLETE 12/10 (9.379 step): WER val 12,1 → 8,2% · val_unseen 12,6 → 8,0% (nhãn giả) | best.pt 2,27 GB chỉ tải khi chấm clean-test sau hiệu đính; kết luận WER chỉ sau hiệu đính |
| **Phạm vi "A + 40 ms" (người dùng duyệt 2026-10-12 chiều, thay v2 mục 8)** | preflight v2 ✅ (12/10); agent đang làm: config công thức chung + rẽ nhánh v1, trung bình checkpoint + kernel eval, RQ2 cuối, nháp đề cương | (1) trung bình checkpoint v1 → (2) pha giảm rẽ nhánh v1 ×3 → (3) Conformer relPE từ đầu → (4) 40 ms ×3 → (5) RQ2. Cổng seed 2: xong trước ~22/10 + còn ≥ 60 GPU-h (áp đối xứng cả ba). Quy tắc phân tích chốt trước: note mục 9.4. **Người dùng giao quyền quyết định đẩy kernel cho AI (2026-10-12)** — vẫn báo sau mỗi lần đẩy. Đóng băng GPU 01/11. Mỗi kernel vẫn hỏi trước. Tài khoản C: chờ username + khoá. Chi tiết `docs/notes/conextbimamba_gap_2026-10-12.md` mục 9 |
| Hiệu đính: tên riêng / tiếng Anh | ✅ 14/10 cố vấn Whisper large-v3 (ngoài đội hình) trong công cụ + bảng tên chuẩn `docs/notes/clean_test_entities.md` | người dùng tự quyết bằng tai; Whisper bịa 6 câu (đã chặn), sai OPEX/Curious/Micah. Còn: sửa các garble còn sót ở câu đã lưu (#7,12,35,36,56,76,85,98,101,105,114…), P&G #59, upan→upenn #10 |
| Trung bình checkpoint · RQ2 cuối | ⬜ → thuộc v2 (P1, P5) | `src/evaluation/average_checkpoints.py`; RQ2 phải làm ấm từng độ dài; Conformer relpos VRAM bậc hai — audio dài OOM sớm hơn v1 |

Quota reset 2026-10-10. Train full đã xong — khối "TRAIN FULL" bên dưới là lịch sử.

## 🔴 TRAIN FULL ĐANG CHẠY — việc đầu tiên của phiên sau (ghi 2026-10-06)

**▶▶▶ 2026-10-08: TRAIN 30 EPOCH XONG CẢ BA** — Mamba B1 0,1562/0,1547 · ConExt 0,1875/0,1846 · Conformer 0,2519/0,2481
(WER val/val_unseen, best ep28 cả ba). Conformer dài (B) dừng 69/76, best ep66 0,1944/0,1924. Không kernel nào đang chạy.
Quota A 7,52 h · B 3,00 h (reset 10/10). Chi tiết + bảng cùng-thời-gian: `Plan.md` nhật ký 2026-10-08.
⛔ Chờ người dùng chọn: tải checkpoint (thư mục?) · N trung bình checkpoint · eval clean-test · pha decay? · Conformer dài s3
(7 epoch ≈ 1,5 h B)? · Parakeet fp32.
✅ Checkpoint đã lưu `D:\Model\kltn_checkpoints\` (6,6 GB) — người dùng đưa lên Drive. ✅ Benchmark subsampling (v3,
`reports/results/subsample_speed/v3/`): có subsampling Mamba vẫn 2,2× Conformer/step; ⚠️ RTF RQ2 phải làm ấm cuDNN từng shape.
Quota A còn 6,91 h.
✅ Bảng so sánh cuối 5 mô hình trên clean-test (nhãn giả): `reports/results/final_compare_2026-10-08/`. ⏳ Người dùng
hiệu đính 203 câu (`python -m src.data.correct_clean_test`) → tính lại WER từ `predictions.tsv`. Quota A còn ~6,6 h.

Lần 1 đã đẩy 2026-10-06 ~00:38 (giờ VN; lúc đóng phiên 00:41 cả hai RUNNING, quota bắt đầu tăng — **chưa xác nhận được dòng "lưu epoch00.pt"**, kiểm đầu tiên ở phiên sau), commit **`7660a6b`**, phiên 540 phút:
`<A>/train-full-conformer-s1` (A) · `<B>/train-full-mamba-s1` (B). Lịch: `training_plan_kaggle.md` mục 7.

**Kết quả kiểm lần 1 (2026-10-06 12:10):** Mamba s1 (B) COMPLETE, returncode 0, **15/30 epoch** (latest epoch 15
step 494), WER val 0,515 → 0,185 · val_unseen 0,181, giảm đều; nonfinite 0, VRAM 3,6 GiB, 0,669 s/step; quota B dùng
9,58 h. Conformer s1 (A) **ERROR log rỗng, quota A không tăng (5,44 h)** → lỗi khởi động, chưa train gì. Người dùng
duyệt → **đẩy 2026-10-06 ~12:17:** `conformer-s1` (A, version 3 — v1, v2 ERROR log rỗng không trừ quota; ghim commit
`bb8d0eb`, code y hệt `7660a6b`) và `mamba-s2` (B, version 2 — v1 ERROR log rỗng). Lúc đóng phiên (~13:00) cả hai
đang train: Conformer xong epoch 0 (WER 1,0, loss val 5,68 — bình thường), Mamba resume đúng epoch 15 step 494, xong
epoch 15 (WER val 0,192, nhích lên từ 0,185 — theo dõi epoch 16-17). **Phiên sau:** kiểm `conformer` s1 + `mamba` s2
theo checklist dưới (Conformer ~7 h → xong ~19:30; Mamba 540′ → hết ~21:30), hỏi người dùng trước khi đẩy lần 3
(`conextbimamba --session 1` A, `mamba --session 3 --account B`).

**Kiểm 2026-10-06 19:15:** Conformer s1 **COMPLETE, 30/30 epoch** (402′), best WER val 0,2519 (ep28), val_unseen
0,2481; không lỗi → Conformer xong train. Mamba s2 đang chạy (epoch 26, WER val 0,164). Quota còn A 17,85 h · B 13,50 h.
→ Người dùng duyệt, **đã đẩy `conextbimamba-s1` (A) ~19:45**, train từ 19:56, kết thúc chậm nhất ~04:45 07/10.
Phiên sau: kiểm Mamba s2 (hết ~21:30) + ConExt s1 → hỏi trước khi đẩy `mamba --session 3` / `conextbimamba --session 2`.

Demo: `python -m src.demo.app` (Conformer + Mamba B1 trên CPU, Mamba qua `mamba_ref.py`). Còn: ConExt (khi có best.pt); tab RQ2 audio dài + căn chỉnh CTC (theo bản phác) chưa làm.

**Cập nhật 21:25:** Mamba s2 **XONG 30/30** (best ep28 WER val 0,1562 · val_unseen 0,1547) → **không đẩy mamba s3**; còn tải checkpoint Mamba (`--ckpt_dest`, chờ người dùng chọn thư mục). ConExt s1 RUNNING epoch 4 (WER val ep3 0,324).

**▶▶ PHIÊN SAU (ghi 2026-10-07 04:30) — thay khối 20:17 bên dưới:**

| Việc | Trạng thái lúc đóng | Làm gì |
|---|---|---|
| Conformer s1 (A) | ✅ 30/30, best ep28 WER val 0,2519 · val_unseen 0,2481 | — |
| Mamba B1 s1+s2 (B) | ✅ 30/30, best ep28 WER val 0,1562 · val_unseen 0,1547 | tải checkpoint `--ckpt_dest` (chờ người dùng chọn thư mục) |
| ConExtBiMamba s1+s2 (A) | s1 ✅ 25 ep (best ep23 WER val 0,1960); **s2 v2 đẩy 04:31** (v1 lỗi khởi động, v2 chạy; ~5 epoch còn lại) | kiểm s2: resume đúng ep25, đủ 30/30, best; hết ~07:00 nếu không lỗi |
| #6 Conformer cùng GPU-giờ (B, 76 epoch) | RUNNING — v4 đẩy 04:06; **resume đúng epoch 30 step 0** (global_step 85.230); ep30 WER val 0,2545 · val_unseen 0,2503; 0,248 s/step; hết chậm nhất ~13:06 (ước dừng ~ep70) | **kiểm**: resume đúng epoch 30 từ `latest.pt` ep29, metrics liền mạch, WER mới; phiên 2 đợi quota reset 10/10 (hỏi trước) |
| Zero-shot (B) | PhoWhisper 0,155; Parakeet fp16 lỗi `⁇` 16/203 → fp32 hết lỗi | ⛔ người dùng: chạy lại Parakeet fp32 đủ 203 câu? |
| Demo | Conformer + Mamba B1 (CPU, `mamba_ref.py`) | thêm ConExt khi có best.pt |

Quota 04:29: A còn 9,34 h (ConExt s2 dùng ~2 h), B còn 11,53 h (phiên Conformer dài dùng tới 9 h → còn ~2,5); reset 2026-10-10.
Câu hỏi treo: trung bình checkpoint (N?); eval clean-test 3 mô hình; 30 hay 60 epoch cho cả nhóm; **lịch LR** — thứ tự Claude đề xuất (chưa chốt, Plan nhật ký 2026-10-07 04:15): ConExt s2 LR hằng tới 30 → trung bình checkpoint cả ba → mức lợi lớn mới làm pha decay riêng ~5,7 GPU-h; ablation Conformer cấu hình gốc; ghi vào `QA.md`?

**▶ PHIÊN SAU BẮT ĐẦU Ở ĐÂY (ghi 2026-10-06 20:17)** — tình trạng lúc đóng phiên:

| Kernel | Tài khoản | Đẩy lúc | Hết chậm nhất | Lúc đóng phiên |
|---|---|---|---|---|
| `<B>/train-full-mamba-s2` | B | 06/10 ~12:17 | ~21:30 06/10 | RUNNING, xong epoch 26 (WER val 0,164; tốt nhất 0,159 ep24) |
| `<A>/train-full-conextbimamba-s1` | A | 06/10 ~19:45 | ~04:45 07/10 | RUNNING, epoch 0 step 300 (cần ~2 phiên) |
| `<A>/train-full-conformer-s1` | A | — | — | **XONG 30/30**, best ep28 WER val 0,2519 · val_unseen 0,2481 |

Quota lúc đóng: A còn 17,33 h · B còn 12,51 h (reset 2026-10-10).
Việc: (1) checklist bên dưới cho `mamba` s2 (`--account B`) và `conextbimamba` s1 → `fetch.py` → báo người dùng;
(2) **hỏi** trước khi đẩy `mamba --session 3 --account B` (chỉ khi s2 chưa đủ 30 epoch — 34′/epoch, có thể vừa đủ)
và `conextbimamba --session 2` (A); sau khi đẩy chạy `watch_kernel.py` vài phút (lỗi khởi động log rỗng);
(3) Mamba xong 30 epoch → `fetch.py --ckpt_dest` phiên cuối → người dùng đưa lên Drive;
(4) còn treo từ báo cáo Conformer: trung bình checkpoint (N?) cho cả nhóm; chạy eval clean-test Conformer (CPU).

**Zero-shot (2026-10-06, B):** PhoWhisper 0,155 (203 câu). Parakeet fp16 lỗi `⁇` 16/203 câu → fp32 hết lỗi (chẩn đoán). ⛔ Chờ người dùng: chạy lại Parakeet fp32 đủ 203 câu (~0,1-0,15 GPU-h B). Còn mở: đối chiếu forward tự viết với `transcribe()` chính thức (hỏng do lhotse).

**LUẬT (người dùng): KHÔNG tự đẩy lần 2 / lần 3. Kiểm kết quả → trình người dùng → chỉ đẩy khi người dùng đồng ý.**
Lý do: không để mất trắng hàng giờ GPU vì phiên trước hỏng mà không ai thấy.

Kiểm mỗi phiên vừa xong (chỉ tải file nhỏ, không tải checkpoint):
1. `kaggle kernels status <kernel>` (B: `KAGGLE_CONFIG_DIR=<thư mục khoá B>`) — COMPLETE / ERROR?
2. `python scripts/kaggle/train_full/fetch.py --model <conformer|mamba|conextbimamba> --session <k> [--account B]`
   → in returncode, epoch xong, finished, commit. Đọc `reports/results/train_full/<model>/s<k>/`:
   - `session_summary.json`: `returncode` 0, `latest.epoch/step_in_epoch` tăng so với phiên trước, `run.commit` = 7660a6b
   - `metrics.jsonl`: `nonfinite_loss_steps` ≈ 0, loss giảm, WER val giảm (Conformer có thể WER 1,0 vài epoch đầu —
     bình thường), có `wall_seconds_total`
   - `logs/train_full_s<k>.log`: có dòng "lưu epochXX.pt" mỗi epoch, không có `!!!`/`LỖI`
3. Báo người dùng: tình trạng + số epoch + WER + quota còn (`kaggle quota`) → **hỏi** có đẩy phiên kế không.

Lệnh đẩy (chỉ sau khi người dùng đồng ý; launcher tự ghim commit của phiên 1 qua `_build/` — nếu `_build/` mất
thì thêm `--commit 7660a6b`):
- Lần 2: `python scripts/kaggle/train_full/launch.py --model conextbimamba --session 1 --push` (A, sau khi Conformer
  xong cả 30 epoch; nếu Conformer chưa xong thì `--model conformer --session 2`) và
  `python scripts/kaggle/train_full/launch.py --model mamba --session 2 --account B --push`
- Lần 3: `--model conextbimamba --session 2` (A) và `--model mamba --session 3 --account B`
- Phiên ERROR **log rỗng, quota không tăng** → đẩy lại cùng phiên. Phiên ERROR **sau khi đã train** → KHÔNG đẩy lại
  cùng phiên (version mới che output) — đẩy phiên kế (resume từ latest.pt của phiên lỗi).
- Hết train: `fetch.py ... --ckpt_dest <thư mục ngoài repo>` cho phiên cuối mỗi mô hình → người dùng đưa lên Drive.

## 🔴 Đang chặn — chờ quyết định từ GVHD

- [ ] **ĐỔI ĐỊNH HƯỚNG sau buổi GVHD lần 1 (ghi 2026-10-01) — ưu tiên số 1**:
      khảo sát Mamba (trọng tâm) vs Conformer pre-train + fine-tune vs 2-3 mô
      hình khác vs 1 baseline; không cần giống hệt cấu hình; nhận xét theo đánh
      đổi WER/tốc độ/tài nguyên. Cần làm rõ Q-a…Q-g trong
      `docs/notes/gvhd_buoi_1.md` mục 4 (danh sách mô hình, Mamba train từ đầu
      hay fine-tune, baseline, fine-tune hay zero-shot, đề cương/tên đề tài).
      **Chờ chốt trước bước 6 (train) và trước khi sửa phần riêng Conformer.**
      Bước 5 (tạo dataset) và hiệu đính clean-test vẫn làm được.
      Đã biết thêm: thầy không gợi ý mô hình, dặn **"đừng ôm đồm"**; **sẽ sửa
      đề cương** (làm bản mới, không sửa docx cũ); buổi sau dự kiến Tuần 9.
      A1 (nhãn tiếng Anh) thầy giao **người dùng tự quyết** sau khi tìm hiểu
      ngôn ngữ pre-train của từng mô hình. A2-A4: chưa được trả lời.
      **AI đã tra ứng viên + phân tích A1:** `docs/notes/survey_model_candidates.md`
      (đề xuất 4 mô hình: Mamba + Conformer nhỏ train từ đầu, Parakeet-CTC-0.6B-vi,
      PhoWhisper-small; A1 nghiêng lọc khỏi train + báo clean-test 2 mức).
      **Chốt 2026-10-02: đủ 5 mô hình** (Mamba hai chiều + Conformer nhỏ train
      từ đầu, Parakeet-CTC-0.6B-vi, PhoWhisper-small, wav2vec2-base-vi); **duyệt
      mọi thay đổi phép đo** (RQ2 = so hiệu quả giữa mô hình + ghép đoạn đo RTF
      theo độ dài; chuẩn hóa văn bản chung trước WER; A3 giữ + nêu hạn chế,
      tách nhóm zero-shot/fine-tune khi bàn). **A1 chốt 2026-10-03** (mục dưới). Còn: viết đề cương điều chỉnh
      (bản mới) trước buổi Tuần 9.

- [x] **RQ2 — CHỐT 2026-10-03 (người dùng): phương án 2 + đổi cách hỏi.**
      RQ2 thành so hiệu quả giữa các mô hình (RTF, độ trễ, VRAM, tham số,
      GPU-giờ); dải độ dài lấy từ **audio ghép nhân tạo** (đoạn liên tiếp cùng
      video, chỉ để đo, không train); WER chỉ trên đoạn gốc 10-15 s. Việc code
      ở 🟡 "RQ2". Ghi chú gốc bên dưới.
      Sai lệch số liệu: số liệu thật
      (67.405 mẫu/245,42h, audio gần như đồng nhất 10-15s) khác đề cương đã
      đăng ký (52.023/267,39h, dải 3-30s) — RQ2 (RTF theo độ dài audio) không
      còn đủ dải để chứng minh ưu thế O(n) của Mamba. 4 phương án trong
      `docs/notes/dataset_discrepancy.md`.
      *Không chặn train (rà 2026-09-30):* cả 4 phương án chỉ đổi phần đo RTF
      sau train (RTF không phụ thuộc trọng số → đo được song song lúc train).
      Chỉ ảnh hưởng train nếu GVHD muốn WER trên audio dài hoặc thêm dữ liệu
      train — nên hỏi thêm: "RQ2 chỉ cần RTF hay cần cả WER audio dài?"
      *AI rà 2026-10-03:* các đoạn số seg liền nhau **không liền về thời gian**
      (87% đoạn < 15 s kết thúc ở chỗ lặng đã bị cắt bỏ; 13% cắt cứng ở 15,0 s;
      5,9% số seg khuyết). Ghép vẫn hợp lệ để đo RTF/độ trễ/VRAM theo độ dài
      (chi phí chỉ phụ thuộc số khung), gọi là "audio ghép nhân tạo". **Không**
      đo được WER audio dài: validation chỉ là ~10% đoạn rải từ chính video
      train → chuỗi seg liên tiếp ≥ 60 s chỉ có 1. Đề xuất: RQ2 chỉ đo hiệu quả
      theo độ dài, ghép từ cả train + val; WER chỉ trên đoạn 10-15 s.

- [x] **A1 — nhãn sai do audio không phải tiếng Việt — CHỐT 2026-10-03
      (người dùng): lọc C ∩ L.** Người dùng nghe kiểm: 149 video phỏng vấn khách
      nước ngoài (tiếng Anh) + video Alive Kicking Ep4 (tiếng Nhật, nhãn trông
      như tiếng Việt → lọc theo nhãn không đủ). LID Whisper-small trên audio
      (kernel `lid-asr`, ~65 phút T4) khớp các nhóm đã nghe. Loại nếu LID ≠ `vi`
      HOẶC thuộc 149 video HOẶC nhãn < 20% dấu → `data/splits/excluded.tsv`
      (`python -m src.data.filter_language`). Train còn **48.340 câu/175,77 h**
      (từ 220,84 h), val 5.140/18,65 h, **clean-test 192 câu**. Khóa luận: coi
      dữ liệu là tiếng Việt, một mức WER trên 192 câu, không bàn tiếng Anh.
      Chi tiết: `docs/notes/survey_model_candidates.md` mục 6.

- [x] **Mamba hai chiều — CHỐT 2026-10-02 (người dùng), phương án B1**; việc
      sửa code chuyển xuống 🟡. Ghi chú gốc: Mamba hiện chỉ nhìn quá khứ, Conformer nhìn cả câu →
      biến gây nhiễu cho RQ1. Phương án B1 (hai chiều, 14 lớp × 2 khối, cộng) =
      12.285.696 tham số (+0,67%), vẫn 28 khối như cũ; cần đảo chuỗi theo độ dài
      thật, benchmark lại. Chi tiết: `docs/notes/mamba_bidirectional.md`. **Chốt
      trước khi train; hỏi GVHD (A4).** Code B1 xong (`df72f11`), chưa chạy CUDA thật.

## 🟡 Sẵn sàng làm ngay — kế hoạch Kaggle đã duyệt 2026-09-28

Chi tiết + lý do: `docs/notes/training_plan_kaggle.md` mục 5 (D1-D11 đã
duyệt, D3 → **4 shard**, train full + đánh giá theo epoch).

- [x] **1. Kernel CPU (D9) — xong 2026-09-28** (`scripts/kaggle/check_env/`):
      `/tmp` trống ~1.200 GB, tải HF ẩn danh ~2,4 h cho 67.405 file (0 lỗi),
      output giữ đủ 600 file, tar 590 MB/s. Chi tiết: `training_plan_kaggle.md` mục 5.
- [x] **2. Kernel GPU benchmark (D9) — xong 2026-09-30** (~16 phút quota):
      bảng đủ 8 cấu hình ở `training_plan_kaggle.md` mục 5. Tóm tắt: workers
      không giúp; AMP Conformer 0,56 s/step (17,7 h/30 epoch), Mamba 1,02
      (32,1 h); DDP+AMP Conformer **chậm đi** 0,77 (24,4 h), Mamba 0,48 (15,0 h).
- [x] **D8 chốt 2026-09-30: cả hai encoder DDP 2 GPU + SyncBN + AMP**, batch
      toàn cục 16 (8/GPU). Ước 30 epoch: Conformer 23,2 h, Mamba 16,0 h (chưa
      tính eval). Số đo: `training_plan_kaggle.md` mục 5.
- [x] **`PACK_TAR = True` chốt 2026-09-30**: 5 dataset đều là tar; đầu mỗi phiên
      train giải nén vào `/tmp/audio_cache` (**không** `/kaggle/working`).
- [x] **3. Sửa code dùng chung (D5, D7, D8, D10) — code xong 2026-10-04,
      chưa commit.** `manifest_indices` (nay là `manifest_rows`) đọc `data/splits/*.tsv` + bỏ
      `excluded.tsv` (train 48.340, `train_shard0` 12.086, val 5.140,
      clean-test 192); `HF_REVISION`; `AUDIO_CACHE_DIR` từ biến môi trường;
      `train.py`: `torchrun` DDP + SyncBN + AMP (front-end + CTC loss fp32),
      sampler tất định theo (seed, epoch) bỏ được step đã học, `--max_minutes`,
      checkpoint mỗi 20 phút ghi nguyên tử, `--resume_from`, `--train_manifests`,
      `--epochs`, eval chia GPU; `src/data/extract_audio.py` giải nén tar. Lịch LR
      giữ warmup + hằng số.
      **Đã test CPU local (Conformer):** resume giữa epoch ra trọng số **giống hệt**
      chạy liền (max |dW| = 0, tắt dropout/augment); `eval_clean_test` 192 câu;
      giải nén tar thật. **Chưa test được:** DDP (PyTorch Windows không có gloo
      device), NCCL/SyncBN/AMP, Mamba → bước đầu của kernel train thử.
      Khởi động: `load_dataset` ~45 s/lần kể cả đã cache → làm ấm bằng
      `python -m src.data.vietsuperspeech_dataset` rồi chạy `HF_HUB_OFFLINE=1`
      (0,2 s).
- [x] **Test độc lập theo video — chốt + code 2026-10-04** (người dùng chọn B,
      theo video): `src/data/make_heldout.py` giữ riêng 29 video (~6%, seed 42);
      train 45.442 / val 4.824 / val_unseen 3.011 / clean-test 203, rời nhau.
      Tokenizer + CMVN tính lại; `train.py` log thêm WER val_unseen. Không phải
      tạo lại 5 Kaggle Dataset (lọc lúc đọc). Chi tiết `Plan.md` mục 4.
- [x] **Front-end + dropout — chốt và code 2026-10-04** (người dùng duyệt, căn
      cứ `docs/notes/frontend_decoder_survey.md`): giữ CTC; CMVN toàn cục
      (`configs/cmvn_stats.json`, 2000 câu train seed 42); SpecAugment 2 freq
      mask F=27 + 10 time mask ≤ 5%, không time warp, chỉ lúc train; dropout 0,1
      cho Mamba (đầu ra mỗi khối); giữ log-mel 80/25/10 ms. Mục `features` trong
      2 yaml, dựng model qua `train.build_model`. Test: CMVN đúng công thức, mask
      không lọt padding (~22% khung, ~33% kênh), dropout tắt khi eval, bất biến
      padding Mamba giữ (khối giả, 9,5e-7), train 3 epoch CPU loss 113 → 41.
- [x] **Theo khuyến nghị tác giả Mamba — sửa 2026-09-30** (người dùng duyệt):
      `build_optimizer` miễn weight decay cho `A_log`/`D`, `weight_decay: 0.01`
      ghi rõ trong 2 yaml; `MambaEncoder` thêm `norm_f`, chia `out_proj` cho
      √n_layers, residual fp32. Test local: Conformer giống hệt từng bit so với
      trước; Mamba kiểm bằng khối giả (không CUDA). **Còn: chạy khối Mamba thật
      trên Kaggle** (gộp vào lần chạy GPU kế tiếp) + đo lại `param_count`
      (dự kiến 12.292.864, +0,73%).
- [x] **4. Manifest 4 shard + val — xong 2026-09-30** (`src/data/make_shards.py`
      → `data/splits/`, ~9 MB TSV): kiểm rời nhau, hợp = 60.656, không lẫn
      validation, val 6.499 + clean-test 250 (đối chiếu nội dung theo index).
      Mỗi shard 15.164 câu / 55,21 h / 6,36 GB, lệch phân bố ≤ 0,1 điểm %, phủ
      996/998 token BPE. Đã commit + push 2026-09-30.
- [ ] **5. Kernel CPU tạo 5 dataset** — *2026-10-04: đã tạo `<A>/vss-asr-train-shard0..3`
      (private, đã chia sẻ `<B>` — B liệt kê được file). **Kaggle tự giải nén tar
      khi tạo Dataset** → dataset chứa `<part>/audio/...wav` + `<part>.tsv`, không còn tar;
      `extract_audio` đã sửa nhận cả hai dạng (test local wav thật, chưa commit).
      **Xong 2026-10-04:** thêm `vss-asr-val` + `mamba-wheels`, chia sẻ B; kernel CPU
      `<B>/data-check-asr` gắn 6 dataset → DATA CHECK OK (thiếu audio 0 ở cả 4 tập,
      chép 27 GB/317 s).* — *2026-10-02: 5 kernel
      `make-dataset-asr-{shard0..3,val}` **xong cả 5, qua kiểm tra**
      (15.164×4 + 6.499 file, 0 lỗi, 16 kHz mono, duration ≤ 0,1 s; ~6 GiB/shard,
      val 2,6 GiB). **Còn (việc tay trên UI):** tab Output của từng kernel →
      New Dataset, private, tên đề xuất `vss-asr-train-shard{0..3}`,
      `vss-asr-val`; **không push version mới** các kernel này trước khi tạo
      xong; chia sẻ sang tài khoản B; ghi slug dataset vào đây.*
      (4 train + 1 val), private —
      `scripts/kaggle/make_dataset/make_dataset.py` (`PACK_TAR = True`), test
      local 20 file thật. Còn: chạy 5 lần (sửa `PART`), mỗi lần ~40 phút CPU
      (không tốn GPU); tạo Dataset từ output; chia sẻ sang tài khoản B (D6).
- [ ] **6. Train full** — **sẵn sàng 2026-10-06** (preflight full đạt, thử nối phiên đạt): lịch ở
      `training_plan_kaggle.md` mục 7 — A: Conformer (1 phiên) → ConExtBiMamba (2 phiên); B: B1 (3 phiên).
      Chạy bằng `scripts/kaggle/train_full/launch.py`. **Chờ người dùng duyệt bắt đầu.**
- [x] **Ghi nhận cho báo cáo trong lúc train (2026-10-06, người dùng yêu cầu):** checkpoint trọng số mỗi epoch
      (`epochs/epochXX.pt`), `metrics.jsonl` thêm lr / `resumed_mid_epoch` / `wall_seconds_total` (cộng dồn qua
      phiên) / `epoch_end_utc`; `session_summary.json` thêm commit + phiên bản + GPU + `setup_minutes`, chụp config;
      `scripts/kaggle/train_full/fetch.py` tải output sau mỗi phiên. Test CPU local (resume cộng dồn đúng).
- [x] **Early stopping — CHỐT 2026-10-06 (người dùng): KHÔNG dùng.** Train cố định 30 epoch, chọn best.pt theo
      WER val; trung bình checkpoint (icefall/ESPnet) để ngỏ, làm sau train từ `epochs/*.pt`.

## 🟡 Mới 2026-10-02 — chờ duyệt chạy

- [x] **Commit + push** `df72f11` (Mamba B1, chuẩn hóa văn bản, kernel
      zero_shot, notes) — push 2026-10-03.
- [x] **Commit + push** thay đổi 2026-10-03 + 2026-10-04 (`9d137a7`, `a4fe49e`,
      `21452f5` code bước 3 + front-end, `7fb8492` Plan) — push 2026-10-04.
- [x] **Train lại tokenizer BPE trên 48.340 nhãn đã lọc — xong 2026-10-03**
      (người dùng duyệt; `python -m src.data.filter_language --train-tokenizer`).
      Cùng tham số, vocab vẫn 1001 (gồm blank) → số tham số hai mô hình không
      đổi. 121/1000 piece đổi: bỏ mảnh giả tiếng Anh (`AND`, `ATION`, `KING`…),
      thêm âm tiết Việt; piece có dấu 663 → 771. Val sạch: 59,7 token/câu (cũ
      60,7), 0 unk, decode khớp 5.332/5.332. Conformer: 2 bước train thật CPU,
      loss 136,7 → 80,1. Mamba chưa test (cần CUDA; tokenizer không chạm
      encoder). `data/splits/summary.json` (`bpe_tokens`) còn tính theo
      tokenizer cũ — chạy lại thống kê khi sửa `make_shards`.
- [ ] Kernel zero_shot: đọc clean-test bản 2 (203 câu) — **đã sửa 2026-10-04**; còn **một mức WER** (bỏ
      `wer_vi_label` hoặc đọc `excluded.tsv`) — sửa trước khi chạy.
- [x] **Hàm bootstrap theo khối video** — `src/evaluation/bootstrap.py`
      (2026-10-04; 1 mô hình hoặc so cặp; Liu & Peng arXiv:1912.09508). Test CPU:
      WER khớp `train.py`. Còn: dùng cho clean-test/zero-shot khi có kết quả.
- [x] **Commit + push** phần giữ riêng video — `f102ee9`, push 2026-10-04.
- [x] ~~Kernel GPU benchmark lần 3~~ — gộp vào kernel train thử (CHECK_CODE
      kiểm B1 + ConExtBiMamba với CUDA thật, s/step đo trong `metrics.jsonl`).
- [x] **Môi trường GPU — XONG 2026-10-05 (phương án A):** `env-check-asr` ENV CHECK OK (Py 3.13 /
      torch 2.11, wheel cp313 build sm_75, CHECK_CODE + DDP/AMP 3 mô hình). **Còn (việc tay):**
      thêm version `mamba-wheels` từ output `env-check-asr` + chia sẻ B.
- [x] ~~⛔ Môi trường GPU sau sự cố 2026-10-04/05~~ (`training_plan_kaggle.md` mục 6):
      ghim image cũ thất bại trên máy GPU (log 0 dòng, Cancel). Người dùng chọn **A** build
      mamba-ssm v2.3.1 trên image mặc định (Py 3.13, torch 2.11?) hay **B** cài torch 2.10
      + wheel v2.3.2/causal-conv1d ≥1.6.2 dựng sẵn → kernel kiểm tra ngắn `-t 1200` → train thử.
- [x] **Preflight tài khoản B** — **PREFLIGHT OK 2026-10-05** (18 phút, 0,29 GPU-h của B; Plan nhật ký).
- [x] **Train thử A — XONG 2026-10-05** (88′, ~1,5 GPU-h). WER val_unseen: ConExt 0,439 · Conformer 0,564 ·
      B1 0,747 (B1 trượt G1/G2: loss NaN dưới fp16, 41,6% câu rỗng; trên câu không rỗng B1 0,560 ≈ Conformer).
      Chi tiết: `Plan.md` nhật ký 2026-10-05 (chiều).
- [x] **Chẩn đoán NaN của B1** — 2026-10-05: tràn fp16 bên trong khối Mamba; sửa `mamba_fp32` (khối Mamba
      fp32 mọi encoder, theo U-Mamba/README mamba-ssm). Kernel `diag-b1-asr`; Plan nhật ký 2026-10-05 (tối).
- [x] **Train thử lại B1** — XONG 2026-10-05: 0 NaN, qua G0-G3, WER val_unseen 0,465 (Conformer 0,564, ConExt
      0,439; B1 − ConExt +0,026 [+0,019; +0,034]); 0,733 s/step. Chi tiết Plan nhật ký 2026-10-05 (tối).
- [ ] **Commit** `mamba_fp32` (mamba_encoder + conextbimamba_encoder), ARCHITECTURE, note tiền xử lý, kernel
      `diag_b1`/`retrain_b1` — chờ người dùng yêu cầu. Train full phải clone code đã có `mamba_fp32`.
- [x] Đo lại s/step ConExtBiMamba với khối Mamba fp32 — **0,432 s/step** (preflight full 2026-10-05). (số 0,326 cũ là fp16) — chọn cách đo (b1/b2/b3) ở
      `docs/notes/trial_results_2026-10-05.md` mục 7.
- [x] **Tiền xử lý audio — CHỐT 2026-10-05: assert + SP on-the-fly, không volume, giữ CMVN toàn cục.** Đã tra thêm 2026-10-05 (`audio_preprocessing_survey.md` mục 5). Đề xuất: assert định
      dạng; speed perturbation on-the-fly 0,9/1,0/1,1 cả 3 mô hình (nếu preflight không nghẽn DataLoader); không volume;
      giữ CMVN toàn cục. **Chờ người dùng chốt.**
- [x] **Chính sách khối Mamba fp32 — CHỐT 2026-10-05: giữ `mamba_fp32`.** Đã tra 2026-10-05 (`docs/notes/mamba_precision_survey.md`). Đề xuất giữ
      `mamba_fp32` (tiền lệ U-Mamba; bf16 không có trên T4; RMSNorm trong khối là đổi kiến trúc). **Chờ người dùng chốt.**
- [x] **Giữ 2 hay 3 mô hình — CHỐT 2026-10-05: giữ cả 3** (dự phòng đề xuất: thiếu quota thì bỏ B1 trước).
- [ ] **Tăng tốc train: bucketing theo độ dài — CHỐT 2026-10-05 (phương án B, cả 3 mô hình; không làm A)** — **code +
      test CPU xong 2026-10-05** (padding 17–19% → 0,5%), chưa commit. Còn: đo s/step 3 mô hình trong preflight.
- [ ] **Speed perturbation + kiểm định dạng audio — CHỐT 2026-10-05, code + test CPU xong**, chưa commit. Còn: preflight
      kiểm DataLoader không nghẽn (resample CPU) + B1/ConExt chạy được.
- [x] (sửa `1727ec7`) Sửa phụ: `trial_report` lấy s/step từ epoch không toàn step bỏ (B1 ghi sai 0,114 s/step); `train.py` nhánh
      bỏ step `del loss, log_probs` (giữ đồ thị → VRAM 2,1 → 3,9 GiB) — sửa code dùng chung, test cả 3 encoder.
- [ ] **Kernel train thử** — lần 1 **ERROR** (image Py 3.13 không nhận wheel cp312, chưa
      train gì); đã ghim `docker_image` + `dataset_sources`, thêm log tiến độ + watchdog
      treo 20 phút (test xong); **chờ duyệt push + chạy lại** (quota A còn 28,87 h).
      `scripts/kaggle/train_trial/` — **duyệt 2026-10-04**
      (tiêu chí G0-G3/K1-K5/C giữ ngưỡng đề xuất, tập chính val_unseen, 4 lệch
      bài ConExt xác nhận, quota tài khoản A, `-t 19800`). 3 mô hình ×
      `train_shard0` × 5 epoch, 1 seed, ~2,5-3,5 GPU-h. Còn: theo dõi kernel,
      tải output (chỉ `trial_report.*`, `metrics.jsonl`, eval jsonl — không tải
      checkpoint), người dùng chốt giữ 2 hay 3.
- [ ] **Kernel GPU zero_shot** Parakeet + PhoWhisper trên 203 câu clean-test (~0,5 GPU-h)
      — xin duyệt quota. Kiểm Parakeet chạy được trên T4.
- [x] **Pilot RQ2 2026-10-05** (trọng số ngẫu nhiên, 3 mô hình × {1×, 4×} + B1 fp16, 5-3.600 s): Mamba dốc ≈ 1,
      Conformer ≈ 2; điểm giao 60 s (ConExt) / 120 s (B1); không OOM tới 3.600 s. `docs/notes/rq2_pilot_2026-10-05.md`.
- [x] **Subsampling — CHỐT 2026-10-05: giữ không subsampling.** Pilot: 4× làm mọi mô hình nhanh hơn nhưng đẩy
      điểm giao ra ~4× (240/480 s).
- [ ] **RQ2 (chốt 2026-10-03, chưa code):** script ghép đoạn liên tiếp cùng
      video (từ train + val, đã bỏ `excluded.tsv`) thành audio dài theo các mức
      độ dài; sửa `rtf.py` bỏ bucket 3-30 s, đo RTF/độ trễ/VRAM cho cả 5 mô
      hình. Không phụ thuộc trọng số → chạy được với trọng số ngẫu nhiên
      trước khi train xong. Mức độ dài cụ thể: AI đề xuất, người dùng duyệt.
- [ ] **Đội hình mới 2026-10-03 + train thử BiMamba vs ConExtBiMamba** (người
      dùng chốt hướng; Plan mục 4). Việc: viết `ConExtBiMambaEncoder` (khối
      Conformer tùy biến, ½FFN → ExtBiMamba → conv → ½FFN → LN; dùng lại
      `reverse_padded`/khối B1), khớp ~12M; train thử trên tập nhỏ vài giờ
      GPU. **Chốt 2026-10-03:** baseline = **Conformer-12M** train từ đầu;
      pre-train = **Parakeet-CTC-0.6B-vi**, **dự bị PhoWhisper-small** (bỏ
      wav2vec2 → kernel zero_shot rút về 2 mô hình; vẫn phải kiểm NeMo/T4); train thử = **1 shard,
      ~5 epoch, 3 mô hình** (BiMamba, ConExtBiMamba, Conformer làm mốc), ước
      ~2-3 GPU-h, so WER val + đường loss + s/step + VRAM. Thứ hạng sớm chỉ để
      loại phương án tệ rõ. **Đổi 2026-10-04 (người dùng): không chọn 1 trong
      #1/#5 nữa — sau train thử chốt giữ 2 hay 3 mô hình, mong cả 3 đủ tốt;
      1 seed + biện pháp ổn định.** Tiêu chí "đủ tốt" AI đã đề xuất, chờ duyệt. **`ConExtBiMambaEncoder` code xong 2026-10-04** (6 lớp, ffn 928,
      12.241.536 tham số +0,31%, test CPU khối giả; 4 lệch bài cần xác nhận —
      `lineup_preparation.md` mục 5). Thứ tự: bước 3 → `ConExtBiMamba` + test CPU khối giả → kernel
      train thử (gộp benchmark lần 3).
      **Danh sách chuẩn bị từng mô hình: `docs/notes/lineup_preparation.md`.**
- [x] **Mamba chồng thuần hay thêm FFN — CHỐT 2026-10-04 (người dùng): giữ B1,
      không làm biến thể B1 + FFN**; ConExtBiMamba (đã trong train thử) là
      phương án "có FFN". Căn cứ 2405.12609 Bảng XVI (+FFN chỉ ~1,6 điểm) và XII
      (khung Conformer mới cải thiện lớn) — `docs/notes/frontend_decoder_survey.md` mục 5.
- [ ] Xác nhận các thiết lập tác nhân tự chọn: ~~`out_proj` chia √28~~ (chốt giữ 2026-10-05); greedy
      không LM; chữ số giữ nguyên; đo batch 1 fp16.
- [x] Viết lại `Plan.md` mục 1-5, `CLAUDE.md`, `README.md` theo định hướng
      khảo sát — xong 2026-10-04.
- [ ] Khung **đề cương mới** (dựa `Plan.md` mục 1-2) trước buổi GVHD Tuần 9.

## 🟢 Việc tay — song song bất cứ lúc nào, không chặn code

- [x] ~~Khi hiệu đính clean-test: thêm cột nhãn ngôn ngữ audio~~ — thay bằng
      LID trên audio + người dùng nghe mẫu (A1 chốt 2026-10-03).

- [ ] Nghe & hiệu đính **203 câu clean-test mới** (29 video giữ riêng, 7 câu/video —
      thay bản 192 câu ngày 2026-10-04) trong `data/processed/clean_test_manifest.json`
      (cột `corrected_text` còn trống) — dùng mục 5 của
      `notebooks/02_dataset_eda.ipynb`.

## 📘 Khóa luận (bản viết) — kế hoạch, mốc và quy cách ở `Plan.md` mục 6

**Làm ngay / sớm**
- [ ] **Gặp GVHD (tối đa 2 tuần/lần, GVHD ký logbook)**: cho AI biết ngày gặp
      gần nhất + ngày hẹn kế tiếp để ghi vào bảng ở `Plan.md` mục 6. Nên hỏi:
      🔴 quyết định RQ2 · hình thức nộp hiện hành (biểu mẫu 2018) · dạng "thiết
      kế" ở Chương 2. *(việc tay của người dùng)*
- [ ] **Sao lưu file Word khóa luận lên Drive** `specialized-thesis/khoa_luan/`
      (`docs/khoa_luan/KhoaLuan_Mamba_vs_Conformer_ASR.docx`, gitignore). Đây là
      **bản làm việc duy nhất**; script sinh file đã bỏ — đừng dựng lại kẻo mất
      bản sửa tay.
- [ ] **Điền chỗ trống trên bìa** (tô vàng): Bộ môn, Khóa, logo khoa. Xác nhận
      tên đề tài trên bìa: bìa ghi `Conformer`, đề cương đã nộp ghi
      `Con-Former` — hỏi GVHD nên theo bản nào.
- [ ] **Rà bản nháp Chương 1** (~9 trang, đoạn tô vàng = ghi chú cần xử lý):
      đối chiếu 12 trích dẫn với nguồn gốc (điền từ trí nhớ AI, chưa kiểm),
      đọc lại 3 bài Mamba-ASR (số liệu AISHELL-1 lấy từ đề cương), chuyển công
      thức sang Word Equation, vẽ Hình 1.1-1.3. Mục tiêu xong trước Tuần 7.
- [ ] **Bắt đầu ghi Tài liệu tham khảo** ngay khi trích dẫn (chỉ tài liệu thực
      sự trích dẫn; quy ước ghi ở Mẫu 5).

**Theo tuần (chưa đến hạn)**
- [ ] Tuần 6-7: Ch2 Phân tích & thiết kế (từ `ARCHITECTURE.md`).
- [ ] Sau Tuần 8/12/13: Ch3 Thực nghiệm — chèn kết quả từ `reports/`; phần mô
      tả dữ liệu chờ 🔴.
- [ ] Tuần 14: Ch4 Demo & tổng kết. Tuần 15: Mở đầu, Kết luận, Tóm tắt, Phụ lục.
- [ ] **Trước hết đợt Kaggle:** sao lưu `best.pt` của cả hai thí nghiệm lên
      Drive (cần cho đĩa CD, mục `SETUP`).
- [ ] Khi nộp: danh sách kiểm tra ở `Plan.md` mục 6 (2 cuốn + 2 CD, đề cương có
      chữ ký, GVHD duyệt trước khi in).

## ⚠️ Cần kiểm tra / rủi ro đã biết (không phải việc làm ngay)

**Repo public**
- [ ] `notebooks/02_dataset_eda.ipynb` nhúng audio YouTube (20 output
      `<audio>`, ~5MB) — trong repo public và lịch sử git. Người dùng chọn bỏ
      qua (2026-09-19). Chưa kiểm license VietSuperSpeech. Khi xử lý: xoá
      output (`nbstripout`); muốn gỡ khỏi lịch sử phải viết lại lịch sử + force
      push (hỏi trước).
- [ ] Lịch sử git vẫn chứa các file đã gỡ index: `.docx` (đề cương, biểu mẫu
      khoa) và file TLCN có tên+MSSV người thứ ba trong tên file. Chưa quyết
      định có viết lại lịch sử không.
- [ ] `docs/de_cuong/De_Cuong_Chi_Tiet_Mamba_ASR.docx` trên đĩa **khác bản đã
      commit** (trước đó `git status` báo modified; nghi Word tự lưu). Giờ file
      không còn track nên git không hiện diff nữa — mở kiểm tra nội dung trước
      khi upload Drive, đừng để mất bản edit tay. Không sửa nội dung file.

**Code / thiết kế** (chi tiết ở `ARCHITECTURE.md` mục 8)
- [ ] `train.py` viết lại 2026-10-04 **chưa chạy trên Kaggle** (DDP/NCCL,
      SyncBN, AMP, Mamba, `extract_audio`) — kernel train thử phải chạy một
      vòng ngắn (vài step + cắt `--max_minutes` + resume) trước khi train thật.
- [ ] **Ước lượng thời gian train vượt hạn mức**: ~1,3 s/step (batch 16, T4,
      chưa tính tải dữ liệu) × 3.791 step/epoch ≈ 80 phút/epoch → 30 epoch ≈
      40 GPU-giờ **mỗi mô hình**, trong khi hạn mức Kaggle 30 GPU-giờ/tuần.
      Liên quan mục `train.py` không AMP/1 GPU bên dưới — chưa quyết.
      *2026-09-30: đã đo (fp32 1 GPU: Conformer 42,5 h, Mamba 48,7 h); lựa
      chọn tăng tốc là mục ⏸ D8 ở trên.*
- [ ] Mamba đơn hướng (đang cân nhắc hai chiều — 🔴 ở trên), chưa mask padding; hai
      encoder không subsampling (T'=T ≈ 1000-1500 khung). Cần nêu trong phần
      thảo luận; kiểm tra đề cương đã đề cập chưa.
- [x] ~~Chênh ngoài lõi encoder: Mamba không dropout, front-end không CMVN/SpecAugment~~
      — xử lý 2026-10-04 (dropout 0,1 cho Mamba, CMVN toàn cục, SpecAugment).
      Còn nêu khi viết: vị trí dropout trong khối Mamba là lựa chọn của dự án.
- [ ] `best.pt` chọn theo WER val (= `validation` trừ clean-test, D5 — đã
      tách tập từ 2026-10-04); val và clean-test vẫn rải từ cùng video với train
      → lưu ý khi diễn giải.
- [ ] `rtf.py`/`survey.py` chia bucket độ dài theo đề cương (3-30s), không
      khớp dữ liệu thật — RQ2 đã chốt 2026-10-03, việc sửa chuyển lên 🟡 "RQ2".

## ✅ Đã hoàn thành

- [x] **Xác nhận Mamba trên Kaggle T4 (2026-09-24)**, chạy từ local qua Kaggle
      CLI (`scripts/kaggle/verify_mamba.py`): `param_count` in Mamba
      12.292.352 vs Conformer 12.204.288 → **0,72% ĐẠT**. 5 step train batch 16
      (audio 15 s) + eval 16 câu, cả hai encoder chạy qua: Conformer 1,35 s/step,
      VRAM đỉnh 7,76 GiB; Mamba 1,29 s/step, 5,95 GiB. Wheel `causal-conv1d`
      1.5.4 + `mamba-ssm` 2.3.1 (cp312, torch 2.10+cu128) nằm trong output của
      kernel.

- [x] **Push lên GitHub (2026-09-24)**: 6 commit (`f101acb`..`f8d5af4`), gồm
      gỡ 12 file `.docx` khỏi index. File cũ vẫn nằm trong lịch sử git (xem ⚠️).

- [x] ~~Upload tài liệu `.docx` (đề cương, biểu mẫu, notes) lên Google Drive~~
      — **người dùng bỏ qua (2026-09-24)**. Các file chỉ còn trên đĩa local
      (gitignore) + bản cũ trong lịch sử git.

- [x] **Script đo WER clean-test (2026-09-21)**:
      `python -m src.evaluation.eval_clean_test --config configs/model_<x>.yaml`.
      Test thật với 250 câu thật (Conformer, CPU, trọng số ngẫu nhiên: chạy hết
      pipeline, WER vô nghĩa). Còn: chạy với checkpoint thật khi có; Mamba chưa
      test (Kaggle); reference hiện là `pseudo_label` cho cả 250 câu vì
      `corrected_text` chưa hiệu đính (🟢) — script in rõ số câu mỗi loại.

- [x] **`param_count.py` đọc yaml (2026-09-21)**: dùng `build_encoder` của
      `train.py`; test local: Conformer 12.204.288 khớp số đã đo, đổi
      `n_layers` trong yaml thì số đổi theo. Nhánh Mamba chưa chạy (không CUDA).

- [x] **Chuẩn hoá tài liệu gốc (2026-09-19)**: tách `Plan.md` (kế hoạch/trạng
      thái/quyết định/nhật ký), `ARCHITECTURE.md` (sơ đồ, luồng dữ liệu, bản đồ
      file), `docs/CONVENTIONS.md` (quy ước), rút gọn `README.md`, cập nhật
      `CLAUDE.md`. Rà repo public, bổ sung `.gitignore`, gỡ 12 file nhị phân
      khỏi index. *(Đã commit local, chưa push — xem 🟡.)*
- [x] **Training loop thật** (`139f018`): checkpoint/resume tự động
      (`checkpoints/<experiment_name>/{latest,best}.pt`), eval WER định kỳ,
      tensorboard. Test thật với dữ liệu thật trên ConformerEncoder (CPU) — resume
      đúng epoch/global_step. Sửa bug: `lr: 3e-4` bị PyYAML đọc thành string →
      `3.0e-4`. Nhánh Mamba chưa test (xem ⚠️).
- [x] Đề cương chi tiết + tóm lược, đăng ký với GVHD (Tuần 1-2).
- [x] Cấu trúc project + skeleton code toàn bộ pipeline.
- [x] **Rủi ro kỹ thuật lớn nhất**: cài `mamba-ssm` CUDA kernel trên Kaggle T4
      (pin tag v2.3.1), forward pass thật chạy được (Tuần 1).
- [x] Khảo sát thống kê VietSuperSpeech (`src/data/survey.py`), phát hiện sai
      lệch số liệu so với đề cương (Tuần 3).
- [x] Tokenizer BPE tiếng Việt (vocab_size=1000, 60.656 transcript).
- [x] Clean-test manifest 250 câu (seed=42).
- [x] Notebook EDA `notebooks/02_dataset_eda.ipynb` (đã test chạy thật).
- [x] Sửa bug decode audio: cột `audio` là string path, không tự giải mã.
- [x] `src/data/prefetch_audio.py`: tải song song đúng 67.405 file cần dùng,
      resume qua session, lỗi từng file không sập cả batch (test logic xong,
      chưa chạy full — xem ⏸).
- [x] Push code lên GitHub: https://github.com/VuKhah/specialized-thesis

- [ ] **Sau khi phạm vi "A + 40 ms" chạy: dọn lại file hướng dẫn** (CLAUDE.md, Plan.md, TODO.md — khối lịch sử train full, notes lỗi thời) — người dùng yêu cầu 2026-10-12.
