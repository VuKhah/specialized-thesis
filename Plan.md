# Plan.md — Kế hoạch, trạng thái & quyết định

Cập nhật lần cuối: **2026-10-04** (viết lại mục 1-5 theo định hướng khảo sát sau
buổi GVHD lần 1). Đây là nơi để **nối tiếp giữa các phiên làm việc**: kế hoạch
theo tuần, trạng thái mức tuần, quyết định đã chốt/đang treo, nhật ký phiên.
Việc chi tiết ở [`TODO.md`](TODO.md); kiến trúc ở
[`ARCHITECTURE.md`](ARCHITECTURE.md); luật cho AI ở [`CLAUDE.md`](CLAUDE.md);
quy ước ở [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md).

> Đọc nhanh khi mở phiên mới: mục 3 (đang ở đâu) → mục 5 (quyết định đang mở)
> → mục 6 (khóa luận, lịch gặp GVHD) → **mục cuối của nhật ký (mục 7)**.

## 1. Mục tiêu

**Khảo sát Mamba (trọng tâm) làm encoder ASR tiếng Việt hội thoại** trên
VietSuperSpeech (sau lọc A1 và giữ riêng video test: train 45.442 câu / 165,18 h), nhận xét theo đánh
đổi **độ chính xác – tốc độ – tài nguyên** (GVHD lần 1, 2026-10-01). Đề cương
đã nộp (so sánh có kiểm soát Mamba vs Conformer) sẽ được thay bằng **đề cương
mới** trước buổi Tuần 9; `docs/de_cuong/*.docx` cũ giữ nguyên.

**Đội hình (chốt 2026-10-03, `docs/notes/lineup_preparation.md`):**

| # | Mô hình | Vai trò | Cách dùng |
|---|---|---|---|
| 1 | Mamba hai chiều B1 (ExtBiMamba chồng) | Trọng tâm | Train từ đầu ~12M |
| 5 | ConExtBiMamba (khung Conformer, MHSA → ExtBiMamba) | Ứng viên trọng tâm, chọn #1 hay #5 sau **train thử** | Train từ đầu ~12M |
| 2 | Conformer-12M | Baseline có kiểm soát | Train từ đầu, cùng pipeline |
| 4 | Parakeet-CTC-0.6B-vi (dự bị PhoWhisper-small) | Tham chiếu có pre-train | Zero-shot → fine-tune nếu còn quota |
| 3 | Mamba một chiều | Chỉ trích dẫn (arXiv:2405.12609) | Không train |

**Hai mức so sánh — không trộn khi kết luận:**
- **Nhóm train từ đầu (#1/#5, #2): có kiểm soát.** Cùng front-end (log-mel +
  CMVN toàn cục + SpecAugment), tokenizer BPE 1000, CTC head, `train.py`, eval;
  ~12M tham số, không subsampling. **Chỉ encoder khác nhau** — luật trong
  `CLAUDE.md` vẫn áp cho nhóm này.
- **Nhóm pre-train (#4): tham chiếu.** Pipeline, front-end, dữ liệu, kích
  thước riêng của mô hình; chỉ dùng chung chuẩn hóa văn bản trước WER và tập
  đánh giá.

**Câu hỏi nghiên cứu (bản làm việc, chờ đề cương mới):**
- **RQ1 — độ chính xác:** WER (greedy, không LM, sau `normalize_text`) trên
  clean-test 203 câu (video giữ riêng, hiệu đính tay), val_unseen 3.011 câu
  (video giữ riêng) và val 4.824 câu (video đã gặp), theo epoch cho nhóm train
  từ đầu; khoảng tin cậy bằng bootstrap theo khối video.
- **RQ2 — hiệu quả:** RTF, độ trễ, VRAM, số tham số, GPU-giờ train; theo độ dài
  trên **audio ghép nhân tạo** (chốt 2026-10-03, chỉ để đo, không đo WER audio dài).
- **RQ3 — phân tích lỗi** định tính trên clean-test.

## 2. Lộ trình (viết lại 2026-10-04, thay lộ trình 15 tuần của đề cương cũ)

Mốc ngày suy từ "Tuần 6 ≈ 2026-10-01" (nhật ký GVHD); tuần tính từ thứ Hai.
Quota: 30 GPU-h/tuần × 2 tài khoản Kaggle (A, B). Ước train full 30 epoch sau
A1 (×0,80 số D8): Conformer ~18,5 h, Mamba B1 ~12,8 h — **kiểm lại bằng số đo
thật ở train thử**.

| Tuần | Ngày (≈) | Nội dung |
|---|---|---|
| 6 | 28/09–04/10 | ✅ A1 lọc dữ liệu, tokenizer mới, RQ2 chốt, đội hình chốt; bước 3 code dùng chung; front-end CMVN + SpecAugment, dropout Mamba |
| 7 | 05/10–11/10 | `ConExtBiMambaEncoder` (~12M, test CPU khối giả); 5 Kaggle Dataset (UI); tiêu chí chọn #1/#5 (người dùng duyệt); **kernel train thử** 1 shard × ~5 epoch × 3 mô hình (~2-3 GPU-h, kèm kiểm DDP/AMP/Mamba thật); kernel zero-shot Parakeet + PhoWhisper (192 câu, kiểm NeMo/T4) |
| 8 | 12/10–18/10 | Chọn #1 hay #5; **bắt đầu train full song song** (A: Conformer, B: Mamba đã chọn); code RQ2 (ghép audio, sửa `rtf.py`), đo RTF/VRAM bằng trọng số ngẫu nhiên; khung đề cương mới |
| 9 | 19/10–25/10 | Train full tiếp; **gặp GVHD lần 2**: đề cương mới, kết quả train thử, zero-shot, các quyết định front-end/CTC |
| 10 | 26/10–01/11 | Hết train full; WER theo epoch trên val, clean-test; ước VRAM/thời gian fine-tune Parakeet (vài step) |
| 11 | 02/11–08/11 | Fine-tune Parakeet (hoặc PhoWhisper) nếu quota cho phép; đo RQ2 mọi mô hình **trên cùng một máy** |
| 12 | 09/11–15/11 | Tổng hợp RQ1 + RQ2 (bảng, biểu đồ đánh đổi); clean-test đã hiệu đính tay xong (việc tay, làm dần từ bây giờ) |
| 13 | 16/11–22/11 | RQ3 phân tích lỗi; Ch3 Thực nghiệm |
| 14 | 23/11–29/11 | Demo Gradio; hoàn thiện mã; Ch2, Ch4 |
| 15 | 30/11–06/12 | Hoàn thiện khóa luận, slide, báo cáo thử, nộp GVHD |

Dự phòng: train full có thể kéo sang Tuần 11 (Kaggle cắt phiên, train lại khi
lỗi); fine-tune pre-train là phần **bỏ được đầu tiên** nếu thiếu quota/thời gian
("đừng ôm đồm") — khi đó #4 chỉ có zero-shot.

## 3. Trạng thái hiện tại (mức tuần — chi tiết ở `TODO.md`)

Đang ở **cuối Tuần 6** (2026-10-04).

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Dữ liệu | ✅ | Manifest 4 shard + val; A1 lọc (`excluded.tsv`); giữ riêng 29 video cho test (`heldout_videos.tsv`); tokenizer BPE + CMVN tính lại; 5 kernel tạo dataset xong — **còn tạo Dataset trên UI + chia sẻ tài khoản B** (việc tay) |
| Pipeline chung (#1, #2, #5) | ✅ code · 🟡 test | DDP/AMP/resume giữa epoch, CMVN + SpecAugment, dropout Mamba (2026-10-04). Test CPU Conformer đạt; **DDP/NCCL/AMP/Mamba thật chưa chạy** (Kaggle) |
| #1 Mamba B1 | 🟡 | Code xong, test bằng khối giả; chưa chạy CUDA thật |
| #5 ConExtBiMamba | ⬜ | Chưa code |
| #2 Conformer-12M | ✅ code | Train thử CPU đạt |
| #4 Parakeet / PhoWhisper | 🟡 | Kernel zero-shot có sẵn (test CPU 4 câu); đã đọc clean-test bản 2 (203 câu); còn rút về 2 mô hình; chưa chạy Kaggle |
| RQ2 | ⬜ | Đã chốt cách đo, chưa code |
| clean-test hiệu đính tay | ⬜ | 203 câu từ 29 video giữ riêng (thay bản 192 câu ngày 10-04), việc tay |
| Đề cương mới | ⬜ | Trước Tuần 9 |
| Khóa luận | 🔄 | File khung + nháp Ch1 (mục 6) |

## 4. Quyết định đã chốt

Theo thứ tự thời gian. Quyết định bị thay thế giữ lại để truy vết, đánh dấu
**(thay bởi …)**.

| Ngày | Quyết định | Lý do / nơi ghi |
|---|---|---|
| 2026-09-14 | Hạ tầng thật là **Kaggle** (2x T4), không phải Colab; chỉ sửa tài liệu nội bộ, **giữ nguyên** `.docx` đã nộp | Lệch có chủ đích. `docs/notes/mamba_ssm_install_log.md` |
| 2026-09-14 | Dùng **Mamba-1/S6**, pin `mamba-ssm` tag `v2.3.1`; **Mamba-3 tạm gác** (chỉ nhắc như hướng phát triển) | Khớp trích dẫn đề cương; tránh lỗi tilelang/tvm. `docs/notes/mamba_versions.md` |
| 2026-09-14 | Tokenizer BPE `vocab_size=1000`; clean-test 250 mẫu, seed 42, lấy từ `validation` | Tuần 3. Sau A1: tokenizer train lại trên nhãn đã lọc, clean-test còn 192 |
| 2026-09-16 | Prefetch **chỉ đúng 67.405 file** cần dùng, không `snapshot_download` cả thư mục | Repo HF có 118.259 file, thừa ~43% |
| — | ~~Train Conformer baseline trước, Mamba sau~~ **(thay bởi 09-28: train song song 2 tài khoản)** | Thứ tự đề cương cũ |
| 2026-09-19 | **Tài liệu nhị phân (`.docx`, biểu mẫu) lưu trên Google Drive, không đưa lên GitHub** (repo public); gỡ 12 file khỏi index | `docs/CONVENTIONS.md` mục 7 |
| 2026-09-19 | **Khóa luận viết bằng Word (`.docx` + PDF) theo Mẫu 5**, lưu `docs/khoa_luan/` (gitignore) + sao lưu Drive; **viết song song theo chương từ sớm** | Mẫu 3 "vừa làm vừa viết", Mẫu 5 yêu cầu 50-100 trang. Mục 6 |
| 2026-09-19 | Chuẩn tài liệu gốc: `CLAUDE.md` (luật AI) · `Plan.md` (kế hoạch/trạng thái) · `ARCHITECTURE.md` (sơ đồ) · `TODO.md` · `README.md` (người ngoài) · quy ước → `docs/CONVENTIONS.md` | Chống lệch trạng thái giữa nhiều file |
| 2026-09-24 | ~~Mamba một chiều `n_layers: 28`~~ khớp tham số bằng **tăng độ sâu**, `expand: 2`, `d_model: 256`, `d_state: 16` **(thay bởi 10-02: B1 14 lớp × 2 khối, vẫn 28 khối)** | Giữ siêu tham số mặc định bài Mamba gốc |
| 2026-09-26 | **Giữ không subsampling** (`T' = T`) | Subsampling rút ngắn chuỗi → làm yếu RQ2. Không đề xuất lại làm cách tăng tốc. `docs/notes/training_plan_kaggle.md` |
| 2026-09-26 | **2 tài khoản Kaggle của 2 người khác nhau**; train **chế độ nền** (≤ 9 h/phiên). ~~3 shard~~ **(thay bởi 09-28: 4 shard)** | `docs/notes/training_plan_kaggle.md` |
| 2026-09-28 | **Duyệt D1-D11** (`training_plan_kaggle.md` mục 5): A train Conformer / B train Mamba; WAV; **4 shard** vòng tròn theo video seed 42 + 1 dataset val; gắn cả shard, xáo chung; val = `validation` trừ clean-test (**ghi rõ ở Chương 3**); notebook CPU tạo dataset; `--max_minutes` + checkpoint theo step; pin HF `cbf624ae9b`; giữ 30 epoch | Đĩa đỉnh lúc tạo ~12,7 GB / 20 GB |
| 2026-09-28 | **Train full dữ liệu, lấy kết quả theo epoch**; so sánh ở cùng số epoch; không train tập con làm cách train chính | Dừng lúc nào cũng có kết quả hợp lệ. Ràng buộc: lịch LR warmup + hằng số. (Train thử 1 shard ngày 10-03 là để **chọn thiết kế**, không phải cách train chính) |
| 2026-09-30 | **D8: DDP 2 GPU + SyncBatchNorm + AMP** (fp16 autocast, tham số fp32), batch toàn cục 16 | Benchmark: Conformer 23,2 h, Mamba 16,0 h / 30 epoch (trước A1). `training_plan_kaggle.md` mục 5 |
| 2026-09-30 | **Dataset dạng tar** (`PACK_TAR = True`), giải nén vào `/tmp` đầu mỗi phiên | Số đo tốc độ chỉ đúng khi đọc đĩa local |
| 2026-09-30 | Theo khuyến nghị tác giả Mamba: miễn weight decay `A_log`/`D`, `norm_f`, chia `out_proj` cho √(số khối), residual fp32; **không** LR riêng cho Δ, **không** document packing | Mã nguồn mamba-ssm v2.3.1 |
| 2026-10-02 | ~~Đội hình 5 mô hình (thêm wav2vec2-base-vi)~~ **(thay bởi 10-03)**. Định hướng **khảo sát**, Mamba trọng tâm; loại `kyle/vi-asr-fastconformer-114m` (đã train trên VietSuperSpeech), Zipformer-30M (mô hình sinh nhãn, license ND) | Sau buổi GVHD lần 1. `docs/notes/gvhd_buoi_1.md`, `survey_model_candidates.md` |
| 2026-10-02 | **Mamba hai chiều B1** (14 lớp × 2 khối xuôi/ngược, chung LN, cộng; 12.285.696 tham số) | `docs/notes/mamba_bidirectional.md`. Code `df72f11` |
| 2026-10-02 | **Duyệt thay đổi phép đo**: chuẩn hóa văn bản chung trước WER; A3 giữ + nêu hạn chế; tách nhóm zero-shot/fine-tune khi bàn | `survey_model_candidates.md` |
| 2026-10-03 | **A1: lọc C ∩ L** — loại LID Whisper-small ≠ `vi`, 149 video phỏng vấn nước ngoài, nhãn < 20% dấu. Train 175,77 h (48.340 câu), val 18,65 h (5.140), clean-test 192. Khóa luận coi dữ liệu là tiếng Việt, một mức WER | `survey_model_candidates.md` mục 6, `data/splits/excluded.tsv` |
| 2026-10-03 | **RQ2: phương án 2 + đổi cách hỏi** — so hiệu quả giữa mô hình (RTF, độ trễ, VRAM, tham số, GPU-giờ); dải độ dài từ **audio ghép nhân tạo** (đoạn liên tiếp cùng video, train + val); WER chỉ trên đoạn gốc 10-15 s | `docs/notes/dataset_discrepancy.md` |
| 2026-10-03 | **Không train Mamba một chiều** — chỉ trích dẫn arXiv:2405.12609 (Bảng XII) | "Đừng ôm đồm"; cờ `bidirectional: false` giữ trong code |
| 2026-10-03 | **Giữ kích thước ~12M** cho nhóm train từ đầu | Cỡ Conformer-S (Gulati 2020); vừa quota; đầu "siêu nhẹ" trên trục đánh đổi |
| 2026-10-03 | **Đội hình hiện hành** (mục 1): B1 + ConExtBiMamba (chọn một sau train thử 1 shard × ~5 epoch × 3 mô hình) + Conformer-12M + Parakeet-CTC-0.6B-vi (dự bị PhoWhisper-small) + Mamba một chiều chỉ trích dẫn | `docs/notes/lineup_preparation.md` |
| 2026-10-04 | **Giữ CTC** (không decoder attention/Mamba, không transducer) | Speech Slytherin Bảng 3; transducer không khả thi khi không subsampling. `docs/notes/frontend_decoder_survey.md` |
| 2026-10-04 | **Front-end: CMVN toàn cục + SpecAugment** (2 freq mask F=27, 10 time mask ≤ 5%, không time warp, chỉ lúc train); giữ log-mel 80 / 25 / 10 ms | SpecAugment Bảng 3/4/6; mọi hệ đối chứng đều chuẩn hoá. Cùng note |
| 2026-10-04 | **Dropout 0,1 cho Mamba** (đầu ra mỗi khối, trước residual) | 2405.12609 Bảng I-III, ConMamba yaml. Cùng note |
| 2026-10-04 | **Giữ B1, không làm biến thể B1 + FFN** | 2405.12609 Bảng XVI/XII. Cùng note |
| 2026-10-04 | **Test độc lập theo video** (thay clean-test 09-14 và một phần D5): giữ riêng ngẫu nhiên 29 video (~6%, seed 42, video ≥ 20 câu) khỏi train/val; clean-test mới 203 câu (7/video) để hiệu đính; phần còn lại = val_unseen 3.011 câu; val chọn checkpoint = 4.824 câu. Train 45.442 câu / 165,18 h. Tokenizer + CMVN tính lại | 561/562 video `validation` trùng train → test cũ chỉ đo "đã gặp", lệch khi so với pre-train. Chuẩn: LibriSpeech, VIVOS tách theo người nói. Chọn theo video (không theo chương trình) để test đại diện phân bố; hạn chế: MC chương trình lớn có thể đã gặp. `src/data/make_heldout.py`, `docs/notes/dataset_discrepancy.md` |

## 5. Quyết định đang mở / treo / rủi ro đã biết

| Trạng thái | Vấn đề | Chi tiết |
|---|---|---|
| 🔴 Chờ viết + GVHD duyệt | **Đề cương mới** (mục tiêu, đội hình, RQ, lộ trình theo mục 1-2) — trước buổi Tuần 9 | `docs/notes/gvhd_buoi_1.md`; mục 1-2 ở trên là bản làm việc |
| 🟡 Chờ AI đề xuất → người dùng duyệt | **Tiêu chí chọn #1 (B1) hay #5 (ConExtBiMamba)** sau train thử — phải duyệt **trước** khi chạy kernel | `TODO.md` mục đội hình |
| 🟡 Chờ AI đề xuất → người dùng duyệt | Các mức độ dài audio ghép cho RQ2 | `TODO.md` RQ2 |
| 🟡 Chờ người dùng xác nhận | Thiết lập AI tự chọn: `out_proj` chia √28; greedy không LM; chữ số giữ nguyên; đo RTF batch 1 fp16; vị trí dropout trong khối Mamba; 10 time mask (NeMo gợi ý ít hơn cho mô hình nhỏ — núm chỉnh nếu hội tụ chậm) | `docs/notes/frontend_decoder_survey.md` |
| ⚠️ Rủi ro | Parakeet chưa chắc chạy được trên T4/NeMo; dữ liệu pre-train có thể trùng VietSuperSpeech (rò rỉ) | `lineup_preparation.md` mục 4; dự bị PhoWhisper |
| ⚠️ Rủi ro | `train.py` mới chưa chạy trên GPU (DDP/NCCL/SyncBN/AMP), Mamba chưa chạy CUDA thật sau B1 | Kiểm ở bước đầu kernel train thử |
| ⚠️ Việc tay | 5 Kaggle Dataset trên UI + chia sẻ tài khoản B; hiệu đính 203 câu clean-test | `TODO.md` 🟡 bước 5, 🟢 |
| ⚠️ Biết, chưa xử lý | `notebooks/02_dataset_eda.ipynb` nhúng audio YouTube trong repo **public**, có cả trong lịch sử git | Người dùng chọn bỏ qua (2026-09-19). Chưa kiểm license VietSuperSpeech |
| ⚠️ Chưa quyết | Tên người thứ ba + MSSV trong tên file TLCN (còn trong lịch sử git); xoá cần viết lại lịch sử + force push | Chỉ làm khi người dùng yêu cầu |

## 6. Khóa luận (bản viết) và làm việc với GVHD

Bản khóa luận viết theo các biểu mẫu của khoa (Mẫu 0/3/5, ban hành 2018, lưu ở
`docs/bieu_mau/` trên máy + Drive). Đề cương chỉ ghi Tuần 14
(Chương 1-4) và Tuần 15 (Mở đầu, Kết luận, TLTK, nộp GVHD); Mẫu 3 dặn *"vừa
làm vừa viết"* và Mẫu 5 yêu cầu **50-100 trang** nội dung → viết song song theo
chương từ sớm (đề xuất được người dùng đồng ý 2026-09-19; không đổi kế hoạch đã
đăng ký). **Đã dựng file khung + bản nháp Chương 1 (2026-09-21)** — xem nhật ký.

**Lưu ở đâu:** Word `.docx` (+ xuất PDF) trong `docs/khoa_luan/` trên máy
(gitignore, không lên GitHub), sao lưu Drive `specialized-thesis/khoa_luan/`.
Chỉ giữ **một bản làm việc** để khỏi lệch. Hình/bảng số liệu vẫn sinh từ code
vào `reports/`, rồi mới chèn vào Word.

### Cấu trúc và mốc viết (theo đề cương; cột "khi nào" là đề xuất)

| Phần | Lấy từ | Khi nào | Trạng thái |
|---|---|---|---|
| Mở đầu | Phần Mở đầu trong đề cương | Dựng khung sớm, hoàn thiện Tuần 15 | 🔄 khung 6 mục, chưa có nội dung |
| Ch1 Cơ sở lý thuyết | `docs/notes/mamba_versions.md`, tài liệu đã đọc Tuần 1-2 | Từ nay đến Tuần 7 (không phụ thuộc kết quả) | 🔄 bản nháp đầu, chờ rà |
| Ch2 Phân tích và thiết kế hệ thống | `ARCHITECTURE.md`, `docs/CONVENTIONS.md` | Tuần 6-7 (lúc chờ máy train) | ⬜ (đã có khung mục) |
| Ch3 Thực nghiệm và đánh giá | `reports/`, `docs/notes/dataset_discrepancy.md`, `survey_model_candidates.md` mục 6 (A1), `frontend_decoder_survey.md` | Mô tả dữ liệu viết được ngay (A1, RQ2 đã chốt); chèn kết quả sau Tuần 10, 12, 13 (mục 2) | ⬜ |
| Ch4 Ứng dụng minh họa và tổng kết | `src/demo/` | Tuần 14 | ⬜ |
| Kết luận, Tóm tắt, Phụ lục | — | Tuần 15 | ⬜ |
| Tài liệu tham khảo | — | **Ghi dần từ bây giờ** (chỉ liệt kê tài liệu thực sự được trích dẫn) | 🔄 12 mục, chưa đối chiếu nguồn |

### Hình thức (Mẫu 5, tóm tắt)

- A4 dọc, Times New Roman, dãn dòng 1,5; lề trên/dưới/phải 2 cm, trái 3 cm; số
  trang giữa bên dưới. Chương/mục đánh số Ả-rập (`2.1`, `2.1.1`), không dùng La Mã.
- Cỡ chữ: tên chương 14 (HOA, đậm, giữa) · mục 1: 13 (HOA, đậm, trái) · mục 2:
  13 (thường, đậm) · mục 3: 13 (thường, nghiêng) · nội dung 13 (đều 2 bên) ·
  bảng 12 · tên bảng 11 đậm (giữa, **trên** bảng) · tên hình 11 đậm (giữa,
  **dưới** hình) · chú thích bảng 10 nghiêng · TLTK 11.
- Thứ tự trong quyển: bìa chính → bìa phụ → nhận xét GVHD → nhận xét GVPB →
  lời cám ơn → **đề cương chi tiết có chữ ký GVHD** → mục lục → danh mục hình/
  bảng/ký hiệu → tóm tắt → Mở đầu / Nội dung / Kết luận → TLTK → Phụ lục.
- Nên dựng ngay file Word với style Heading 1/2/3 đúng cỡ chữ trên, để viết
  đến đâu đúng hình thức đến đó.

### Danh sách kiểm tra khi nộp (Mẫu 3, 5)

- [ ] GVHD duyệt lần cuối **trước** khi in đóng bìa.
- [ ] 2 cuốn bìa mềm bao phim + 2 CD/DVD dán bìa sau, nhãn đĩa đúng mẫu.
- [ ] CD có: `<Tên phần mềm>/{SETUP,SOURCE}` · `THESIS/{DOC,PDF}` · `REF` ·
      `SOFT` · file hướng dẫn + thông tin liên lạc. `SETUP` cần dữ liệu thử
      khớp kết quả báo cáo → **sao lưu `best.pt` của cả hai thí nghiệm ra khỏi
      Kaggle lên Drive** trước khi hết đợt.
- [ ] Đề cương chi tiết có chữ ký đóng vào quyển.
- [ ] Bảo vệ: trình bày ~15 phút (tối đa 20) + chạy thử 5 phút + hỏi đáp ~10
      phút; hội đồng ≥3 người chấm theo rubric (Mẫu 2), lệch >1,5 điểm thì
      thương thảo lại.
- [ ] Sau bảo vệ: sửa theo hội đồng, in lại, có trang nhận xét GVHD/GVPB (GVPB ký
      xác nhận đã sửa), nộp lại quyển + CD.

### Nhật ký gặp GVHD (Mẫu 0/3: gặp **tối đa 2 tuần/lần**, GVHD ký logbook)

Không báo cáo định kỳ có thể bị coi như không làm luận văn (Mẫu 3). Sổ logbook
giấy do GVHD giữ; bảng này chỉ để không quên lịch và để phiên sau biết GVHD
đã dặn gì. Người dùng cung cấp thông tin, AI không tự điền.

| Ngày gặp | Đã báo cáo | GVHD dặn / quyết định | Hẹn buổi sau (≤ 2 tuần) |
|---|---|---|---|
| Tuần 6 (≈ 2026-10-01) — buổi báo cáo lần 1 | Tiến độ Tuần 4-5, câu hỏi A1-A4/B1/C1-C4 (`docs/tong_ket/2026-09-30/`) | **Đổi định hướng:** khảo sát so sánh nhiều mô hình, Mamba là trọng tâm; đối chứng không cần giống hệt, Conformer nên dùng bản pre-train + fine-tune; thêm 2-3 mô hình khác + 1 baseline; nhận xét theo đánh đổi độ chính xác/tốc độ/tài nguyên. A1 nhãn tiếng Anh: "tự đọc rồi làm" (xét theo ngôn ngữ pre-train). "Đừng ôm đồm". **Sẽ sửa đề cương** (bản mới). Chi tiết `docs/notes/gvhd_buoi_1.md` | Dự kiến **Tuần 9** (> 2 tuần — lưu ý Mẫu 3) |

**Câu hỏi nên hỏi ở buổi gặp tới:** (1) báo cáo RQ2 đã chốt phương án 2 + đổi cách
hỏi (audio ghép nhân tạo, không đo WER audio dài) để thầy biết khi duyệt đề
cương mới (`docs/notes/dataset_discrepancy.md`); (2) hình thức nộp hiện hành (biểu mẫu
ban hành 2018: bìa mềm + CD còn áp dụng không); (3) phần "thiết kế" ở Ch2 cần
dạng nào (Mẫu 5 nhắc UML cho đề tài ứng dụng, đề tài này thiên về nghiên cứu). (4) duyệt đề cương mới theo mục 1-2 (đội hình, hai
mức so sánh, RQ); (5) báo cáo các lựa chọn kỹ thuật có căn cứ tài liệu: giữ
CTC, CMVN + SpecAugment, dropout Mamba (`docs/notes/frontend_decoder_survey.md`).

## 7. Nhật ký phiên (chỉ thêm vào cuối, không sửa mục cũ)

Mỗi mục: làm gì · chốt gì · phiên sau bắt đầu từ đâu.

### Trước 2026-09-19
Xem `git log` (Tuần 1 → Tuần 4). Mốc chính: cài `mamba-ssm` (4 lần thử),
khảo sát dataset, phát hiện sai lệch số liệu, prefetch script, train loop thật
(`139f018`).

### 2026-09-19
- **Làm:** rà lại toàn bộ tài liệu gốc và tách thành bộ nguồn sự thật mới
  (`CLAUDE.md`, `Plan.md`, `ARCHITECTURE.md`, `docs/CONVENTIONS.md`, `TODO.md`,
  `README.md`). Đọc code thật để viết `ARCHITECTURE.md` (shape, interface,
  bản đồ file). Rà repo public: phát hiện notebook EDA nhúng audio, tên người
  thứ ba trong tên file; thêm rule `.gitignore`; gỡ 12 file nhị phân khỏi index.
- **Chốt:** tài liệu nhị phân → Google Drive; bỏ qua notebook EDA; treo quyết
  định prefetch (mục 5); khóa luận viết bằng Word ở `docs/khoa_luan/` + Drive,
  viết song song theo chương (mục 6). Người dùng đã tự xoá file TLCN khỏi đĩa.
- **Đã đọc biểu mẫu khoa (Mẫu 0/3/5)** để dựng mục 6: hình thức, cấu trúc quyển,
  danh sách nộp, quy định gặp GVHD ≤ 2 tuần/lần.
- **Phát hiện khi đọc code (chưa sửa, ghi ở `ARCHITECTURE.md` mục 8):**
  `param_count.py` không đọc yaml; hai encoder không subsampling (T'=T, 100
  khung/s); Mamba đơn hướng và chưa mask padding; `train.py` 1 GPU, không AMP;
  clean-test nằm trong `validation` dùng chọn `best.pt`.
- **Kết phiên:** đã commit local toàn bộ thay đổi tài liệu/`.gitignore`;
  **chưa push** (GitHub vẫn hiển thị `.docx` cũ cho tới khi push).
- **Phiên sau bắt đầu từ:** (1) upload `.docx` lên Drive, cho link để sửa link
  trong README, rồi mới quyết định push (xem `TODO.md` 🟡); (2) tune tham số Mamba (cần sửa `param_count.py`
  đọc yaml trước); (3) khi có quyết định: prefetch trên Kaggle; (4) **khóa
  luận:** người dùng cho biết ngày gặp GVHD gần nhất để điền bảng ở mục 6, rồi
  dựng file Word đúng Mẫu 5 và bắt đầu Ch1.

### 2026-09-21
- **Làm:** sửa `src/models/param_count.py` để đọc `configs/model_*.yaml` và dựng
  encoder qua `build_encoder` của `train.py` (trước đó gọi constructor bằng
  tham số mặc định nên chỉnh yaml không đổi kết quả). Cập nhật `CLAUDE.md`,
  `ARCHITECTURE.md` (bảng file + mục 8-f), `TODO.md`.
- **Test:** local (không CUDA) — Conformer theo yaml = 12.204.288, khớp số đã đo;
  đổi `n_layers` 8→4 trong yaml tạm thì ra 6.112.512 (script thật sự đọc yaml).
  **Nhánh Mamba chưa chạy** (cần `mamba-ssm`, chỉ có trên Kaggle).
- **Phiên sau bắt đầu từ:** chạy `python -m src.models.param_count` trên Kaggle
  để có số Mamba, chỉnh `configs/model_mamba.yaml` tới chênh < 5%; các mục còn
  lại của phiên 2026-09-19 (Drive/push, prefetch, khóa luận, bảng GVHD) vẫn mở.
- **Làm (tiếp):** viết `src/evaluation/eval_clean_test.py` — đo WER 1 checkpoint
  trên clean-test, dùng chung cho hai encoder (`--config`), ghi
  `reports/results/<exp>_clean_test.json` gồm ref/hyp từng câu cho RQ3. Sửa hai
  docstring lỗi thời (`wer.py`, `vietsuperspeech_dataset.py`).
- **Test:** chạy thật 250 câu clean-test (tải audio thật), Conformer, CPU, trọng
  số ngẫu nhiên (`--allow_random_init`) → chạy hết pipeline, WER 4,34 (vô nghĩa,
  chỉ chứng minh script chạy). Chưa test với checkpoint đã train (chưa có) và
  chưa test Mamba (cần CUDA). Suy luận CPU batch 1 mất ~13 phút cho 250 câu.
- **Chốt trong code:** reference = `corrected_text` nếu có, không thì
  `pseudo_label` (hiện 0/250 câu đã hiệu đính); `batch_size` mặc định 1 vì
  Mamba chưa mask padding.
- **Ghi nhận (cuối phiên):** người dùng xác nhận hạn mức `/kaggle/working` = 20 GB,
  nhỏ hơn ~27 GB audio → prefetch nguyên bản không vừa. Quyết định ⏸ vẫn treo,
  chưa chọn phương án; xem `TODO.md` ⏸.

### 2026-09-21 (khóa luận)
- **Làm:** dựng file Word `docs/khoa_luan/KhoaLuan_Mamba_vs_Conformer_ASR.docx`
  (gitignore) theo Mẫu 5: A4, Times New Roman, dãn 1,5, lề 2/2/3/2 cm, số trang
  giữa dưới, style chương/mục đúng cỡ chữ, đánh số tự động `Chương n:` / `n.m` /
  `n.m.k`, mục lục tự sinh, bảng viết tắt. Trang trước nội dung: bìa chính + bìa
  phụ, chỗ đóng phiếu GVHD/GVPB, lời cám ơn, đề cương có chữ ký, mục lục, danh
  mục, tóm tắt. Khung Mở đầu, Ch2-4, Kết luận, TLTK, Phụ lục. **Viết bản nháp
  Chương 1** (~9 trang: ASR/log-mel/CTC, Transformer/Conformer, SSM/S4/Mamba,
  Bảng 1.1 so sánh độ phức tạp, 12 mục TLTK).
- **Kiểm tra:** validate schema OK; mở bằng Word, cập nhật mục lục, xuất PDF (26
  trang) và xem lại bìa, mục lục, trang công thức, trang bảng.
- **Chốt / lưu ý:** file Word là **bản làm việc duy nhất** — script sinh đã bỏ,
  đừng dựng lại. Chưa điền: Bộ môn, Khóa, logo (tô vàng). Bìa ghi `Conformer`,
  đề cương đã nộp ghi `Con-Former` — chờ người dùng/GVHD quyết định. 12 trích
  dẫn điền từ trí nhớ AI, **chưa đối chiếu nguồn**. Ba hình (1.1-1.3) chỉ là ô
  giữ chỗ.
- **Phát hiện khi viết Ch1 (ghi vào bản nháp, chưa vào `ARCHITECTURE.md`):**
  `torchaudio.models.Conformer` (2.8.0) dùng `MultiheadAttention` thường, **không
  có mã hóa vị trí tương đối** như bài báo Conformer gốc, và encoder của dự án
  cũng không cộng mã hóa vị trí nào → vị trí chỉ đến từ tích chập. Cùng với
  việc không subsampling, cần thảo luận ở Ch2/Hạn chế.
- **Phát hiện khác:** `docs/de_cuong/De_Cuong_Chi_Tiet_Mamba_ASR.docx` trên đĩa
  ghi 67.405 mẫu/245,42 h, **khác** con số 52.023/267,39 h mà `TODO.md` 🔴 nói
  là đề cương đã đăng ký (phạm vi vẫn ghi Colab, `dev-test`). Nhiều khả năng
  bản trên đĩa đã được sửa tay sau khi đăng ký (khớp cảnh báo "khác bản đã
  commit" ở `TODO.md`). Chưa sửa/kiểm chứng — người dùng cần xác nhận bản nào là
  bản đã nộp GVHD trước khi dựa vào con số nào.
- **Phiên sau bắt đầu từ:** sao lưu file Word lên Drive; điền bìa; rà Ch1 theo
  đoạn tô vàng; Mở đầu chờ ý kiến GVHD; các mục treo khác (🔴 RQ2, ⏸ prefetch —
  hạn mức Kaggle 20 GB < 27 GB) vẫn mở.

### 2026-09-24
- **Làm:** tune số tham số Mamba khớp Conformer. Không có CUDA local nên đếm
  bằng bản sao shape của `mamba_ssm.Mamba` (đọc `mamba_simple.py` tag v2.3.1
  trên GitHub): mỗi block 437.760 + LayerNorm 512, `input_proj` 20.736. Cấu
  hình cũ (10 layer) chỉ 4,40 M (−64 %).
- **Chốt (người dùng chọn):** `n_layers: 28`, `expand: 2` → 12.292.352 tham số
  (+0,72 %). Đã sửa `configs/model_mamba.yaml`, `ARCHITECTURE.md`.
- **Chưa kiểm chứng:** chưa chạy `param_count.py` với mamba-ssm thật; chưa chạy
  forward/train với 28 layer (VRAM, tốc độ trên T4).
- **Phiên sau bắt đầu từ:** trên Kaggle chạy `param_count` + vài step
  `train.py` cho Mamba; các mục 🔴/⏸ vẫn mở.
- **Chốt thêm (người dùng):** bỏ qua việc upload tài liệu `.docx` lên Google
  Drive; push không còn phụ thuộc việc này. Sửa `README.md` (bỏ câu "lưu trên
  Drive"), `TODO.md`. Mục sao lưu file Word khóa luận vẫn giữ.
- **Push:** đã push 6 commit lên `origin/master` (`98406d9..f8d5af4`) theo yêu cầu
  người dùng. Phiên sau: lên Kaggle `git pull` rồi chạy `param_count` + vài step
  `train.py` cho Mamba.
- **Kaggle từ máy local:** cài Kaggle CLI (khoá `~/.kaggle/kaggle.json` có sẵn),
  viết `scripts/kaggle/verify_mamba.py` (build wheel `causal-conv1d` v1.5.4 +
  `mamba-ssm` v2.3.1, `param_count`, 5 step train + eval cho cả hai encoder) và
  gửi chạy bằng `kaggle kernels push`. `kernel-metadata.json` gitignore vì chứa
  username. **Kết quả (kernel ~18 phút):** `param_count` 0,72% ĐẠT; smoke test 5
  step batch 16 cả hai encoder chạy qua — Conformer 1,35 s/step, 7,76 GiB;
  Mamba 1,29 s/step, 5,95 GiB. Build `causal-conv1d` mất ~13 phút, `mamba-ssm`
  lấy wheel dựng sẵn. Ghi vào `mamba_ssm_install_log.md` (Lần 5).
- **Phát hiện:** ước lượng ~80 phút/epoch → 30 epoch ≈ 40 GPU-giờ/mô hình, vượt
  hạn mức 30 GPU-giờ/tuần. Ghi ⚠️ `TODO.md`, chưa quyết hướng xử lý (AMP / 2 GPU
  / giảm epoch).
- **Phiên sau bắt đầu từ:** quyết định chi phí train (mục trên) và ⏸ prefetch —
  hai việc này cùng chặn train Conformer thật.
- **Kết phiên:** tạo `QA.md` (9 câu hỏi mở: 3 chặn tiến độ — RQ2, prefetch, chi
  phí train; 6 cần xác nhận), đăng ký vai trò trong `docs/CONVENTIONS.md` mục 1.
- **Phiên sau bắt đầu từ:** đọc `QA.md` xem câu nào đã có trả lời. Nếu chưa có:
  đề xuất đo AMP / 2 GPU / thời gian tải dữ liệu trên Kaggle (Q3, cần người
  dùng đồng ý), hoặc làm trích dẫn + hình Chương 1. Wheel `mamba-ssm` +
  `causal-conv1d` nằm trong output kernel `verify-mamba-asr` — tái dùng thay vì
  build lại.

### 2026-09-26
- **Làm:** review dataset cho Q2 (người dùng yêu cầu, có tra web) →
  `docs/notes/prefetch_storage_review.md`. Đo bằng HF API: cần đúng 67.405
  file = **28,27 GB (26,33 GiB)**; FLAC lossless ~60,5 % (~17,1 GB, đo 300
  file); HF rate limit 3.000 (ẩn danh)/5.000 (token) resolver/5 phút ⇒ tải
  lại mỗi phiên tốn ≥ 67-112 phút. So sánh 5 phương án A-E, **chưa chọn**.
- **Phát hiện (liên quan Q1/Q4):** train+dev chỉ gồm kênh **vietcetera**
  (manifest liệt kê 4 nguồn); README card HF đã cũ (32.267 mẫu/103 h). Repo HF
  đứng yên từ 2026-02-22 (`cbf624ae9b`), code chưa pin revision.
- **Thảo luận tiếp (cùng phiên):** chia dữ liệu nhiều tài khoản / gộp trọng
  số kiểu LLM (DiLoCo) → với 2 tài khoản không lợi, đề xuất mỗi tài khoản 1 mô
  hình; nguyên nhân ~80 phút/epoch; chạy nền khi tắt máy; chia 3 shard phân
  tầng (mô phỏng thật: lệch ≤ 0,1 điểm %); phát hiện val gần như toàn bộ từ
  video đã có trong train (Q10); độ liền mạch các đoạn cho phương án ghép RQ2.
- **Chốt (người dùng):** giữ không subsampling; chia train thành 3 Kaggle
  Dataset; dùng 2 tài khoản; train chế độ nền. Ghi ở mục 4.
- **Tổng hợp + đề xuất D1-D14 (chờ duyệt):** `docs/notes/training_plan_kaggle.md`.
  Chưa sửa code, chưa tạo dataset.
- **Xác nhận (người dùng):** 2 tài khoản Kaggle thuộc 2 người khác nhau → D14
  đã giải quyết (Q11 đóng).
- **Phiên sau bắt đầu từ:** người dùng duyệt D1-D13 (nhất là D9: chạy kernel
  CPU + GPU benchmark). D12/D13 mang đi hỏi GVHD.

### 2026-09-27
- **Làm:** tổng hợp các câu hỏi cần thầy Dũng giải đáp vào `Report.md` (gốc
  repo): A — cần quyết định (RQ2/Q1 → đề xuất phương án 2; val trùng video/Q10
  → giữ split, nêu hạn chế; hạn chế kiến trúc/Q9; số epoch/Q3 → giữ 30),
  B — hình thức (Q5, Q6), C — lệch đề cương đã xử lý. `QA.md` trỏ tới file này.
- **Phiên sau bắt đầu từ:** ghi câu trả lời của GVHD vào `Plan.md` mục 4/6,
  xoá mục tương ứng khỏi `QA.md`/`Report.md`; người dùng duyệt D1-D13.

### 2026-09-28
- **Làm:** giải thích D1-D11 cho người dùng; so sánh 3 vs 4 shard, các loại
  lộ dữ liệu (giữa shard: không; clean-test trong tập chọn `best.pt`: có, D5
  sửa; val cùng video train: có, do split gốc, D13 chờ GVHD); so sánh train
  tập con / shard lần lượt / cộng dồn / full theo epoch.
- **Chốt (người dùng):** D1-D11 đồng ý, D3 đổi sang **4 shard**; D4/D5 đồng ý
  với điều kiện minh bạch (ghi rõ val trừ clean-test ở Chương 3); **train full,
  đánh giá theo epoch**. Ghi ở mục 4 và `docs/notes/training_plan_kaggle.md` mục 5.
- **Làm (tiếp):** kernel CPU `check-env-asr` chạy xong trên Kaggle (kết quả ở
  `docs/notes/training_plan_kaggle.md` mục 5); viết kernel GPU benchmark
  `scripts/kaggle/benchmark/benchmark.py` (AMP giữ front-end log-mel fp32),
  dry-run CPU nhánh không-AMP qua (265 s/step trên CPU — chỉ để kiểm luồng
  code). `.gitignore` đổi thành `scripts/kaggle/**/kernel-metadata.json`.
- **Phiên sau bắt đầu từ:** người dùng đồng ý → chạy benchmark GPU (~30-40
  phút quota), rồi chọn tăng tốc theo D8 và sửa code dùng chung (bước 3).

### 2026-09-30
- **Làm:** chạy kernel GPU `benchmark-asr` (người dùng đồng ý; ~16 phút
  quota, lần đầu lỗi vì đường dẫn wheel `kernel_sources` đổi thành
  `/kaggle/input/notebooks/<user>/<kernel>/` → sửa tìm đệ quy). Kết quả đủ 8
  cấu hình: `docs/notes/training_plan_kaggle.md` mục 5. Viết + chạy full
  `src/data/make_shards.py` → `data/splits/` (4 shard × 15.164 câu + val
  6.499; mọi kiểm tra đạt, `--check` đọc lại đạt). Thêm hằng số `HF_REVISION`
  vào `vietsuperspeech_dataset.py` (chưa dùng trong Dataset). Viết kernel CPU
  `scripts/kaggle/make_dataset/make_dataset.py`, test local 20 file thật.
- **Chờ người dùng (D8):** cả hai encoder phải cùng cấu hình train.
  (a) AMP, 1 GPU: Conformer 17,7 h, Mamba 32,1 h (> 30 h/tuần → Mamba sang
  tuần 2); huấn luyện giống hệt fp32 1 GPU về batch/BN. (b) AMP + DDP 2 GPU:
  Conformer 24,4 h, Mamba 15,0 h, cả hai vừa quota; nhưng BN của Conformer
  tính trên 8 câu/GPU (cần SyncBatchNorm để tương đương) và Conformer chậm đi
  chưa rõ lý do. (c) đo thêm DDP (SyncBN, ít worker/GPU) ~10 phút quota rồi
  mới chọn. **AI đề xuất (c)** rồi nghiêng về (b) + SyncBN nếu Conformer DDP
  không chậm hơn 1 GPU; mọi số giờ chưa tính eval + đọc `/kaggle/input`.
- **Cũng chờ:** `PACK_TAR` cho kernel tạo dataset; commit + push manifest.
- **Phiên sau bắt đầu từ:** quyết định D8 → sửa code dùng chung (bước 3);
  push manifest → chạy `make_dataset` 5 lần.
- **Làm (tiếp):** rà khuyến nghị tác giả Mamba (bài báo + README + mã nguồn
  mamba-ssm v2.3.1) so với code. Đã đúng: tính lại trạng thái ở backward
  (kernel tự làm), tham số fp32 + autocast, khởi tạo Δ/A. **Sửa (người dùng
  duyệt):** miễn weight decay `A_log`/`D` (`build_optimizer`, code dùng chung —
  Conformer không đổi, kiểm từng bit), `norm_f`, chia `out_proj` cho
  √n_layers, residual fp32. Tham số Mamba 12.292.864 (+0,73%). Xác nhận không
  mask padding là đúng với Mamba đơn hướng. Xem xét tài liệu người dùng dán về
  "khuyến nghị của tác giả": LR riêng cho Δ là thực hành của S4, không phải
  mamba-ssm; document packing không áp dụng cho ASR theo câu — không đổi code.
- **Phiên sau (bổ sung):** chạy khối Mamba thật trên Kaggle cùng lần GPU kế tiếp.
- **Làm (tiếp):** D8 phương án (c) — người dùng chọn. Benchmark lần 2 xong
  (kernel v4, ~13 phút; v3 hỏng do AI quên nhúng yaml, mất ~9 phút). Bảng ở
  `training_plan_kaggle.md` mục 5. Khối Mamba thật với các sửa hôm nay chạy
  được, tham số 12.292.864. Máy GPU: 4 lõi CPU, `/tmp` trống 1,1 TB.
- **Đề xuất chờ duyệt:** D8 = cả hai encoder DDP 2 GPU + SyncBN + AMP
  (Conformer 23,2 h, Mamba 16,0 h); `PACK_TAR = True`, giải nén vào `/tmp`.
- **Phiên sau bắt đầu từ:** người dùng duyệt D8 + `PACK_TAR` + commit/push →
  sửa code dùng chung (bước 3), chạy `make_dataset` 5 lần.
- **Chốt (người dùng):** D8 = 2 GPU + SyncBN + AMP; `PACK_TAR = True`; commit
  + push (mục 4).
- **Commit + push (người dùng yêu cầu):** `64dc53a` (sửa theo khuyến nghị
  Mamba), `256396c` (manifest, kernel Kaggle, chốt D8/tar). Lệnh push đã chạy
  xong trước khi người dùng kịp huỷ để yêu cầu kiểm tra khoá; quét *sau* push
  toàn bộ 61 file trên GitHub: không có khoá Kaggle/Claude/HF/GitHub, không có
  username Kaggle (metadata kernel vẫn gitignore). **Từ nay quét nội dung
  trước mỗi lần push.** Commit cập nhật tiến độ này quét trước rồi mới push.
- **Phiên sau bắt đầu từ:** chạy `make_dataset` (CPU, sửa `PART`) lần lượt 5
  phần → tạo Dataset từ output, chia sẻ sang tài khoản B; song song sửa code
  dùng chung bước 3 (D5, D7, D8, D10) cho cả hai encoder.
- **Chốt cuối phiên (người dùng):** RQ2 **không chặn train** — cả 4 phương án
  chỉ đổi phần đo RTF sau train (RTF không phụ thuộc trọng số, đo được song
  song lúc train). Cứ tạo dataset + train; buổi gặp GVHD tới hỏi thêm "RQ2 chỉ
  cần RTF hay cần cả WER trên audio dài?" (ghi ở 🔴 `TODO.md`).
- **Đóng phiên 2026-09-30.** Quota GPU đã dùng hôm nay ~38 phút (benchmark v1
  lỗi ~1, v2 ~15, v3 lỗi ~9, v4 ~13). GitHub ở `93c8132`; `Plan.md`/`TODO.md`
  có sửa sau commit đó (mục này + ghi chú RQ2) — chưa commit.
- **Phiên sau bắt đầu từ:** (1) push kernel `make_dataset` với
  `PART = "train_shard0"` (CPU, ~40 phút), kiểm output tar → tạo Kaggle
  Dataset private, lặp cho shard1-3 + val; hỏi cách chia sẻ sang tài khoản B
  (D6). (2) Song song: sửa code dùng chung bước 3 (val trừ clean-test,
  `--max_minutes` + checkpoint theo step, DDP + SyncBN + AMP, giải nén tar vào
  `/tmp/audio_cache`, pin `HF_REVISION`) cho cả hai encoder. (3) Chạy thử
  `train.py` `main()` trên Kaggle (tốn GPU → hỏi trước).

### 2026-09-30 (tổng kết)
- **Làm:** rà lại toàn bộ dự án (TODO, Plan, ARCHITECTURE, notes, git log) và
  viết bản tổng kết tiến độ + dashboard HTML mô phỏng kết quả cuối (RQ1/RQ2/RQ3)
  vào `docs/tong_ket/2026-09-30/` (ban đầu tạo ngoài repo, người dùng yêu cầu chuyển vào; chưa commit). Số trên
  dashboard có nhãn "Mô phỏng" là số giả lập để hình dung bố cục Chương 3,
  **không** dùng trong khóa luận. Không đổi code.
- **Phiên sau bắt đầu từ:** như mục 2026-09-30 ở trên (kernel `make_dataset`
  5 lần; sửa code dùng chung bước 3).
- **Làm (tiếp):** theo yêu cầu người dùng (buổi gặp GVHD đầu tiên sau 6 tuần),
  hỏi đáp rồi dựng `docs/tong_ket/2026-09-30/trinh_bay_gvhd.html` (trang cuộn:
  đề tài, sản phẩm cuối, bản phác demo Gradio đầy đủ, luồng, kiến trúc, dữ liệu,
  huấn luyện, mục lục khóa luận, tiến độ, hạn chế) và `cau_hoi_gvhd.html` (9
  câu: A1-A4 thiết kế thí nghiệm, B1 lệch đề cương, C1-C4 hình thức; có ô ghi
  quyết định). `docs/tong_ket/` đưa vào `.gitignore` (người dùng chọn).
- **Phát hiện:** ~19,4% transcript train và 58/250 câu clean-test là tiếng Anh
  phiên sai → 🔴 mới trong `TODO.md`, có thể chặn train nếu GVHD chọn lọc.
- **Phiên sau bắt đầu từ:** ghi câu trả lời của GVHD (nút "Chép tất cả câu trả
  lời" trên trang câu hỏi) vào `Plan.md` mục 4/5, `TODO.md`, `QA.md`; rồi tiếp
  kernel `make_dataset` + sửa code dùng chung.

### 2026-10-01
- **Làm:** phiên hỏi đáp để người dùng hiểu kỹ dự án (chặng 1 bức tranh lớn,
  chặng 2 CTC, tiền xử lý, kiến trúc; so sánh ASR vs TTS). In cấu trúc Conformer
  thật từ code (tham số từng khối). Vẽ 2 hình cho báo cáo/khóa luận:
  `reports/figures/kien_truc_tong_the.{svg,png}`,
  `reports/figures/khoi_conformer_vs_mamba.{svg,png}` (PNG ×2 để chèn Word).
  Thêm `mamba_tong_the.{svg,png}` (Mamba-CTC phóng to 3 tầng: mô hình → khối
  Mamba → quét S6 kèm công thức) và `kien_truc_tung_lop.{svg,png}` (AI hiểu
  nhầm yêu cầu "từng layer"; người dùng chưa quyết giữ hay xoá).
- **Ghi nhận mới (chưa quyết):** Mamba không dropout vs Conformer 0,1; không
  CMVN — `TODO.md` ⚠️.
- **Phiên sau bắt đầu từ:** tiếp hỏi đáp (chặng 5 khớp tham số → 8 đánh giá)
  nếu người dùng muốn; các mục của 2026-09-30 vẫn mở.
- **Đánh giá đề cương** (người dùng yêu cầu, chỉ đọc 2 file `.docx`, không sửa):
  điểm mạnh là thiết kế ablation; vấn đề chính: số liệu dữ liệu sai và hai bản
  mâu thuẫn, Mamba một chiều là biến gây nhiễu chưa nêu, TLTK chỉ 3 mục, tuyên
  bố mới lạ quá mạnh, taxonomy RQ3 chưa vận hành được ("nhiễu nền" là nguyên
  nhân, thiếu "dấu thanh"), không có kiểm định thống kê (gợi ý bootstrap CI).
  Chưa ghi thành file — chỉ mục Mamba một chiều được ghi (dưới).
- **Chốt (người dùng):** ghi nhận nghi vấn Mamba một chiều, **cân nhắc hai
  chiều** → `docs/notes/mamba_bidirectional.md`, 🔴 `TODO.md`, mục 5. Chưa đổi
  code.


### 2026-10-01 → 2026-10-02 (sau buổi GVHD lần 1)
- **Ghi ý kiến GVHD** (Tuần 6, buổi sau dự kiến Tuần 9): đổi định hướng sang
  **khảo sát**, Mamba là trọng tâm, đối chứng không cần giống hệt, Conformer nên
  pre-train; "đừng ôm đồm"; A1 người dùng tự quyết; **sẽ sửa đề cương** (bản
  mới). `docs/notes/gvhd_buoi_1.md`.
- **Tra ứng viên** → `docs/notes/survey_model_candidates.md`. Loại
  `kyle/vi-asr-fastconformer-114m` (đã train trên VietSuperSpeech) và Zipformer-30M
  (sinh nhãn, license ND). Đính chính: nhãn hỏng là chuỗi giả tiếng Anh (không
  phải âm tiết Việt); Parakeet là Việt–Anh CS (không phải chỉ tiếng Việt).
- **Chốt (người dùng 2026-10-02):** đội hình 5 mô hình; Mamba hai chiều B1;
  duyệt thay đổi phép đo (mục 4).
- **A1:** người dùng hỏi căn cứ ngưỡng 20% → **không có căn cứ tài liệu**, AI tự
  đặt; số liệu: phân bố hai đỉnh, ngưỡng 10-50% chỉ đổi ~1.100 câu; chưa chứng
  minh audio là tiếng Anh. Kiểm bằng cột ngôn ngữ audio khi hiệu đính
  clean-test. Chưa chốt lọc.
- **3 tác nhân song song:**
  1. Mamba B1 xong (`mamba_encoder.py`, yaml `n_layers: 14`, `bidirectional:
     true`): 12.285.696 tham số, 6 test CPU khối giả đạt; `out_proj` chia √28
     (tác nhân tự quyết, AI đồng ý — người dùng chưa xác nhận). Benchmark lần 3
     (`CHECK_CODE` + đo Mamba B1) sửa sẵn, **chưa chạy**.
  2. Kernel `scripts/kaggle/zero_shot/` (Parakeet, PhoWhisper-small,
     wav2vec2-base-vi; greedy không LM; WER 2 mức + RTF + VRAM) chạy thật CPU
     local 4 câu cả 3 mô hình, **chưa chạy Kaggle** (~0,5 GPU-h). Code dùng
     chung mới: `src/evaluation/text_normalize.py`, `wer.py` tự chuẩn hóa (WER
     cũ không đổi). Card Parakeet không liệt kê T4.
  3. Tạo 5 dataset: 5 kernel CPU riêng `make-dataset-asr-{shard0..3,val}`; lúc
     đóng phiên 4/5 COMPLETE, shard2 RUNNING; **chưa kiểm log, chưa tạo Dataset**
     (tác nhân dừng giữa chừng khi đóng phiên).
- **Chưa commit/push** (người dùng chưa duyệt). Benchmark + zero-shot cần push
  code lên GitHub trước (kernel clone repo).
- **Phiên sau bắt đầu từ:** (1) xin duyệt commit + push + chạy 2 kernel GPU;
  (2) kiểm log 5 kernel `make-dataset-asr-*` (đủ số file, 16 kHz, duration),
  tạo 5 Kaggle Dataset từ output (có thể phải qua UI); (3) cập nhật mục 1 +
  lộ trình mục 2 + `CLAUDE.md` theo định hướng khảo sát; (4) khung đề cương mới
  trước Tuần 9.
- **Bổ sung sau khi đóng phiên (2026-10-02):** đã commit `df72f11` (chưa
  push). 5 kernel `make-dataset-asr-*` đều COMPLETE, kiểm đạt (0 lỗi tải dù
  32-40 lỗi HTTP 429/shard được retry tự xử lý). CLI/API không tạo được Dataset
  từ output kernel → người dùng tạo trên UI (hướng dẫn ở `TODO.md` bước 5).

### 2026-10-03
- **Đầu phiên:** rà trạng thái; chỉ ra `TODO.md` lệch thực tế (commit `df72f11`
  đã có, code Mamba B1 đã xong). Người dùng xóa 8 hình trong `reports/figures/`
  (chưa từng commit; không dùng được). Còn trỏ tới hình: `docs/tong_ket/2026-09-30/bao_cao_gvhd.html`
  (báo cáo đã gửi, giữ nguyên), `docs/notes/mamba_bidirectional.md` dòng 51.
- **A1 — nguyên nhân nhãn sai:** nhãn do Zipformer chỉ-tiếng-Việt sinh; gặp
  audio nước ngoài thì ghép mảnh từ Latin trong vocab thành chuỗi giả tiếng
  Anh. Nhãn hỏng dồn theo video (97% trong 150 video, toàn phỏng vấn khách nước
  ngoài). Người dùng nghe mẫu đầu/giữa/cuối 149 video (`data/raw/en_video_samples/`,
  gitignore) → đúng tiếng Anh; nghe 30 câu hỏng còn sót → train#15087 là
  **tiếng Nhật** (nhãn cả video sai, kể cả câu trông như tiếng Việt).
- **LID trên audio:** kernel `scripts/kaggle/lid/lid.py` (`tieunhi/lid-asr`, gắn
  output 5 kernel `make-dataset-asr-*` làm `kernel_sources`, đọc tar tuần tự),
  Whisper-small, 67.405 đoạn, ~65 phút T4 (gấp đôi ước). Khớp mọi nhóm đã nghe.
  **Chốt C ∩ L** (mục 4) → `src/data/filter_language.py` → `data/splits/excluded.tsv`
  (13.733 đoạn). `lid.csv` ở `data/processed/` (không track).
- **RQ2 (vẫn chờ GVHD, AI chỉ phân tích):** đoạn liền số seg không liền thời
  gian; ghép vẫn hợp lệ cho RTF/VRAM theo độ dài, không đo được WER audio dài
  vì validation rải từ chính video train (chi tiết `TODO.md` 🔴 RQ2).
- **Đã push `df72f11`; commit `9d137a7`** (A1, chưa push).
- **Train lại tokenizer BPE** trên 48.340 nhãn đã lọc (người dùng duyệt):
  121/1000 piece đổi (bỏ mảnh giả tiếng Anh, thêm âm tiết Việt), vocab và số
  tham số không đổi; Conformer train thật 2 bước CPU đạt; Mamba chưa test
  (CUDA). Chưa commit.
- **Lịch phiên sau (theo thứ tự; ⛔ = cần người dùng duyệt trước):**
  1. ⛔ **Push** `9d137a7` + commit tokenizer — mọi kernel Kaggle clone repo,
     không push thì kernel chạy code/tokenizer cũ.
  2. **Bước 3 — sửa code dùng chung** (việc lớn nhất, cả hai encoder):
     `VietSuperSpeechDataset` dùng `HF_REVISION` + đọc `data/splits/*.tsv` + bỏ
     `excluded.tsv`; `eval_clean_test` chỉ 192 câu; DDP `torchrun` + SyncBN +
     AMP fp16 (front-end fp32); `DistributedSampler.set_epoch`; `--max_minutes`
     + checkpoint theo step, resume giữa epoch; giải nén tar vào
     `/tmp/audio_cache`. Kiểm: Conformer vài step thật trên CPU local.
  3. **Kernel zero_shot:** đổi sang một mức WER trên 192 câu (đọc `excluded.tsv`).
  4. ⛔ **Kernel GPU benchmark lần 3** (~15 phút): Mamba B1 chạy CUDA thật +
     code bước 3 + tokenizer mới; đo lại `param_count` Mamba.
  5. ⛔ **Kernel GPU zero_shot** (~0,5 GPU-h): Parakeet/PhoWhisper/wav2vec2 trên
     192 câu; xác nhận Parakeet chạy trên T4.
  6. **Tài khoản B:** thử gắn output kernel của A (`kernel_sources`); không được
     thì người dùng tạo 5 Dataset trên UI rồi chia sẻ.
  7. **Viết:** `Plan.md` mục 1-2, `CLAUDE.md` (luật "chỉ encoder khác nhau"),
     `README.md` theo định hướng khảo sát; khung đề cương mới (trước Tuần 9).
  - Chờ người dùng/GVHD, không tự làm: RQ2 (phân tích đã có ở `TODO.md`);
    xác nhận `out_proj` chia √28, greedy không LM, chữ số giữ nguyên, batch 1
    fp16; dropout Mamba / CMVN (phải quyết trước train).

### 2026-10-04
- **Đầu phiên:** `master` đi trước origin 2 commit (`9d137a7`, `a4fe49e`) chưa
  push; tài liệu 10-03 chưa commit. Theo lịch, làm bước 3 (code dùng chung) —
  push để cuối phiên hỏi duyệt.
- **Bước 3 xong (code, chưa commit):** dataset đọc manifest + bỏ `excluded.tsv`
  + `HF_REVISION` + `AUDIO_CACHE_DIR` từ env; `train.py` viết lại (DDP/SyncBN/AMP,
  sampler resume giữa epoch, `--max_minutes`, checkpoint 20 phút, eval chia GPU);
  `extract_audio.py`; `eval_clean_test` 192 câu. Chi tiết + kết quả test ở
  `TODO.md` bước 3.
- **Người dùng hỏi decoder / log-mel → AI tra tài liệu, người dùng duyệt cả 5
  đề xuất** (mục 4, `docs/notes/frontend_decoder_survey.md`): giữ CTC, CMVN
  toàn cục, SpecAugment, dropout Mamba 0,1, giữ B1. Đã code + test CPU.
- **Phát hiện:** `load_dataset` ~45 s/lần kể cả cache (lần đầu ~2 phút) →
  làm ấm cache rồi `HF_HUB_OFFLINE=1` (0,2 s). DDP không test được trên Windows
  (PyTorch không có gloo device/libuv) → kiểm ở Kaggle.
- **Lịch phiên sau (⛔ = cần người dùng duyệt):**
  1. ⛔ **Commit + push** (2 commit cũ + tài liệu 10-03 + code hôm nay) — mọi
     kernel clone repo.
  2. **`ConExtBiMambaEncoder`** + yaml khớp ~12M + test CPU khối giả.
  3. AI đề xuất **tiêu chí chọn** BiMamba vs ConExtBiMamba → người dùng duyệt.
  4. **Kernel train thử** (1 shard, ~5 epoch, 3 mô hình): bước đầu chạy ngắn
     DDP + cắt `--max_minutes` + resume, `param_count` Mamba thật; ⛔ quota ~2-3 GPU-h.
     Cần 5 Dataset trên UI (việc tay, bước 5) hoặc gắn output kernel.
  5. Kernel zero_shot rút về Parakeet + PhoWhisper, một mức WER 192 câu.
  6. Viết `Plan.md` mục 1-2, `CLAUDE.md`, `README.md`, khung đề cương mới.
- **Viết lại `Plan.md` mục 1-5** (người dùng yêu cầu): mục tiêu theo định hướng
  khảo sát + đội hình + hai mức so sánh (có kiểm soát / tham chiếu) + RQ bản
  làm việc; lộ trình Tuần 6-15 có mốc ngày (suy từ Tuần 6 ≈ 2026-10-01); trạng
  thái theo hạng mục; bảng quyết định theo thời gian, đánh dấu mục bị thay thế;
  mục 5 chỉ còn việc đang mở. Còn viết lại `CLAUDE.md`/`README.md`.
- **Viết lại `CLAUDE.md` + `README.md`** theo hướng khảo sát (người dùng yêu
  cầu): đề tài + đội hình + hai mức so sánh; luật "chỉ encoder khác nhau" giờ
  áp cho nhóm train từ đầu, nhóm pre-train giữ pipeline riêng; thêm luật đề
  xuất phải có nguồn, quota GPU phải xin duyệt; cạm bẫy cập nhật (manifest/
  excluded, `HF_HUB_OFFLINE`, tar, DDP không test được local, B1 không cần
  mask); README thêm lệnh chạy Kaggle và số liệu sau A1. Tên đề tài trên
  README ghi là "tên làm việc" — tên chính thức theo đề cương mới.
- **Tỷ lệ train/val/test (người dùng hỏi):** kích thước hợp lý (clean-test ≈
  VIVOS test), nhưng 100% clean-test và 99,98% val thuộc video đã train (do
  `validation` gốc rải từ video train) → đề xuất 3 phương án, **người dùng chọn
  B: giữ riêng theo video** (mục 4). `make_heldout.py` → 29 video; train 45.442 /
  val 4.824 / val_unseen 3.011 / clean-test 203, rời nhau từng đôi (đã kiểm).
  Tokenizer train lại (4/1000 piece đổi; val_unseen + clean-test giải mã khớp
  100%, 0 unk), CMVN tính lại (|Δ| ≤ 0,08). `train.py` chấm thêm val_unseen mỗi
  epoch (`extra_eval_manifests`, chỉ log); dataset nhận `(split, index)` trộn hai
  split; `eval_clean_test`, kernel zero-shot, notebook 02 đọc manifest bản 2. Test
  CPU: eval_clean_test 203 câu, train 2 epoch có 2 tập eval, zero-shot chuẩn bị
  mẫu đạt. Train thử 1 shard: `train_shard0` còn 11.364 câu.
- **Đóng phiên 2026-10-04 — tóm tắt thay đổi trong ngày** (đã push tới `ffecb6f`;
  phần giữ riêng video **chưa commit**):
  - Code dùng chung (bước 3): dataset theo manifest + lọc lúc đọc, `train.py`
    DDP/AMP/resume giữa epoch, `extract_audio.py`; `HF_HUB_OFFLINE` sau khi làm ấm.
  - Front-end CMVN toàn cục + SpecAugment, dropout Mamba 0,1, giữ CTC, giữ B1
    (căn cứ `docs/notes/frontend_decoder_survey.md`).
  - Viết lại `Plan.md` mục 1-5, `CLAUDE.md`, `README.md` theo hướng khảo sát.
  - Test độc lập theo video: 29 video giữ riêng; train 45.442 / val 4.824 /
    val_unseen 3.011 / clean-test 203; tokenizer + CMVN tính lại.
  - Hướng dẫn người dùng tạo 5 Kaggle Dataset trên UI (slug đề xuất
    `vss-asr-train-shard{0..3}`, `vss-asr-val`, private, chia sẻ tài khoản B
    quyền xem) — **lọc lúc đọc nên 5 dataset không phải tạo lại** sau thay đổi
    giữ riêng video.
- **Lịch phiên sau (thay lịch đầu ngày; ⛔ = cần người dùng duyệt):**
  1. ⛔ **Commit + push** phần giữ riêng video (`make_heldout.py`,
     `heldout_videos.tsv`, `val_unseen.tsv`, clean-test bản 2, tokenizer, CMVN,
     code + tài liệu) — kernel Kaggle clone repo.
  2. **Kiểm 5 Kaggle Dataset** nếu người dùng đã tạo: `kaggle datasets files
     tieunhi/<slug>` (đủ `.tar` + `.tsv`, đúng kích thước); ghi slug + username
     tài khoản B vào `kernel-metadata.json` (gitignore).
  3. **`ConExtBiMambaEncoder`** (½FFN → ExtBiMamba → conv → ½FFN → LN, dùng lại
     `reverse_padded` + cặp khối B1, dropout 0,1), khớp ~12M, `configs/model_conextbimamba.yaml`,
     đăng ký `build_encoder`; đối chiếu chi tiết khối với 2405.12609 Bảng XIV;
     test CPU khối giả (shape, padding, tham số); cập nhật `ARCHITECTURE.md`.
  4. **Đề xuất tiêu chí chọn B1 hay ConExtBiMamba** (có căn cứ tài liệu) →
     người dùng duyệt **trước** khi chạy train thử.
  5. **Hàm bootstrap theo khối video** cho WER (khoảng tin cậy; dùng cho
     train thử, clean-test, zero-shot) — Liu và cs. 1912.09508.
  6. **Kernel train thử** (`train_shard0` = 11.364 câu, ~5 epoch, 3 mô hình):
     bước đầu chạy ngắn DDP + cắt `--max_minutes` + resume, `param_count` Mamba
     thật; ⛔ quota ~2-3 GPU-h. Cần bước 2-4.
  7. **Kernel zero-shot** rút về Parakeet + PhoWhisper, một mức WER trên 203
     câu, kiểm Parakeet chạy được trên T4; ⛔ quota ~0,5 GPU-h.
  8. Khung **đề cương mới** (dựa `Plan.md` mục 1-2) trước buổi GVHD Tuần 9.
  - **Việc tay của người dùng (song song):** tạo 5 Kaggle Dataset + chia sẻ B;
    hiệu đính 203 câu clean-test (notebook 02 mục 5).
  - **Chờ người dùng xác nhận:** `out_proj` chia √28; greedy không LM; chữ số
    giữ nguyên; đo RTF batch 1 fp16; vị trí dropout trong khối Mamba; 10 time mask.
