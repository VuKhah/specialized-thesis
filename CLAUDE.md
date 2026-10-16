# CLAUDE.md

Luật cho Claude Code khi làm việc trong repo này. Ngắn có chủ đích: kế hoạch/
trạng thái ở `Plan.md`, việc cần làm ở `TODO.md`, kiến trúc ở
`ARCHITECTURE.md`, quy ước chi tiết ở `docs/CONVENTIONS.md`. **Không lặp lại
nội dung các file đó ở đây** — chỉ ghi thứ cần biết *trước khi chạm vào code*.
Dọn lại 2026-10-15 sau bản rà soát độc lập `docs/notes/review_fable_2026-10-15.md`.

## Đề tài (3 dòng)

KLTN **khảo sát Mamba làm encoder ASR tiếng Việt hội thoại** trên VietSuperSpeech
(nhãn giả Zipformer; clean-test 203 câu/29 video giữ riêng đã hiệu đính tay).
Nhóm train từ đầu ~12M, cùng pipeline CTC, **chỉ encoder khác**: Mamba hai chiều
B1, ConExtBiMamba, Conformer PE tương đối (bản không PE là ablation), ở **hai cấu
hình 10 ms và 40 ms**. Nhóm pre-train (Parakeet-CTC-0.6B-vi, PhoWhisper-small)
chỉ là mốc tham chiếu. Ba RQ: độ chính xác · hiệu quả theo độ dài audio và
GPU-giờ · đặc điểm lỗi + chất lượng nhãn. Chi tiết, số liệu, kết luận viết sẵn
theo nhánh: `Plan.md` mục 1 và `docs/notes/ket_qua_cuoi_khung_2026-10-15.md`.
Đề cương mới chưa duyệt; thực tế lệch đề cương cũ (Kaggle thay Colab, số liệu
dataset, RQ2 audio ghép) là có chủ đích, đã ghi.

## Đầu mỗi phiên (theo thứ tự)

1. `TODO.md` khối ▶ trên cùng và mục ⛔ — việc đang chặn, quyết định đang chờ.
2. `Plan.md` mục 3 (đang ở đâu), mục 5 (quyết định mở), **mục cuối nhật ký
   (mục 7)**. Mục 6 là khóa luận + lịch gặp GVHD (≤ 2 tuần/lần).
3. `git log --oneline -10` + `git status` — đừng giả định trạng thái phiên trước.
4. Bàn về phạm vi, thí nghiệm, cách kết luận → đọc **bản phản biện chốt**
   `docs/notes/review_fable_2026-10-15.md` (mục 2 phương án, mục 3 những gì đã
   loại, mục 4 quyết định mở) trước khi đề xuất bất cứ gì.
5. Đụng code → `ARCHITECTURE.md` (mục 8 cạm bẫy, 8-v2a công thức chung).
   Quy tắc phân tích chốt trước: `docs/notes/conextbimamba_gap_2026-10-12.md`
   mục 9.4. Quy tắc chấm clean-test: `docs/notes/clean_test_results_2026-10-15.md`
   mục 1.

## Cuối mỗi phiên

Thêm 1 mục **ở cuối** nhật ký `Plan.md` (đã làm gì · chốt gì · phiên sau bắt đầu
từ đâu), cập nhật `TODO.md`. Đổi luồng dữ liệu/shape/interface → `ARCHITECTURE.md`.
Output kernel Kaggle (file nhỏ: json/jsonl/log/csv) **tải về `reports/results/`
ngay trong phiên kernel xong** — bài học run Conformer relPE 10 ms (B) bị bỏ quên.
Không tự commit khi người dùng chưa yêu cầu.

## Luật cứng — không vi phạm dù không ai nhắc lại

- **Không tự chọn phương án cho quyết định đang mở** (⛔ trong `TODO.md`, mục 5
  `Plan.md`). Những quyết định này thuộc người dùng/GVHD — hỏi trước, không
  "chọn phương án hợp lý nhất" rồi làm. Đề xuất kỹ thuật phải có nguồn công bố
  đã kiểm; chỗ không có bằng chứng ghi rõ là suy luận.
- **Không đề xuất thí nghiệm GPU mới** ngoài danh sách đã duyệt (người dùng
  2026-10-13: đổi hướng liên tục tốn thời gian; review 15/10 mục 3 liệt kê
  những gì đã loại và vì sao). Hỏi "độ sâu" → trả lời bằng nhận xét từ dữ liệu
  sẵn có.
- **Clean-test đã hiệu đính chỉ để báo cáo.** Không chọn checkpoint, seed, tham
  số LM, quy tắc chấm theo nó. Checkpoint bảng chính = **ep37** (cuối pha giảm)
  cho mọi run; chọn gì cũng trên `val`. Kết luận "A tốt hơn B" chỉ khi CI 95%
  bootstrap cặp theo video không chứa 0; chứa 0 → viết "không phân biệt được",
  không "tương đương"/"nhỉnh hơn". Kết quả bất lợi cho Mamba vẫn báo nguyên.
- **Không sửa nội dung `docs/de_cuong/*.docx`** — đã nộp GVHD. Đề cương mới là
  file mới.
- **Hai mức so sánh, không trộn:**
  - *Nhóm train từ đầu*: chỉ encoder khác; front-end (log-mel + CMVN +
    SpecAugment), tokenizer BPE1000, CTC head, `train.py`, eval, công thức
    (`configs/v2a/`, 38 epoch WSD) dùng chung. Điểm hoán đổi duy nhất:
    `build_encoder` trong `src/training/train.py`; dựng model qua `build_model`.
    Mọi thay đổi công thức áp cho **cả ba** encoder, cùng thứ tự, không ưu tiên
    mô hình nào.
  - *Nhóm pre-train*: giữ pipeline riêng; chỉ dùng chung `normalize_text` và
    tập đánh giá. Khi kết luận phải tách nhóm, nêu điều kiện khác.
- **Repo GitHub là PUBLIC.** Không commit: `data/raw/`, `data/processed/*` (trừ
  `clean_test_manifest.json` và các ngoại lệ đã ghi trong `.gitignore`),
  `checkpoints/*`, audio/model, `.docx`/biểu mẫu/bản nháp `docs/khoa_luan/`
  (Google Drive), khoá/token, **username Kaggle**, họ tên-MSSV người thứ ba. Không
  `git add -f`. Notebook track phải xoá output. Chi tiết: `docs/CONVENTIONS.md` mục 7.
- **Không viết lại lịch sử git / force push** khi người dùng chưa yêu cầu rõ.
- **Quota GPU Kaggle là tài nguyên của người dùng** (3 tài khoản A/B/C, 30 h/tuần
  mỗi tài khoản, reset thứ Sáu). Chỉ đẩy kernel trong phạm vi đã duyệt, báo ngay
  sau mỗi lần đẩy kèm ước thời gian; kernel ngoài phạm vi phải hỏi trước.

## Cạm bẫy đã biết (đừng khám phá lại)

- Console Windows cp1252 → Python local phải `sys.stdout.reconfigure(encoding="utf-8")`
  hoặc `PYTHONIOENCODING=utf-8`; **`python -I` bỏ qua biến môi trường** → dùng
  `reconfigure` trong script.
- Chạy mọi `python -m src...` **từ gốc repo** (đường dẫn cache tương đối theo cwd).
- Cột `audio` của VietSuperSpeech là string path tương đối, không tự giải mã —
  cần file local rồi `soundfile` (đã xử lý trong `src/data/vietsuperspeech_dataset.py`).
- Split thật: `train` / `validation` (không phải `dev-test`). Chọn mẫu qua
  `manifest_rows` (bỏ `excluded.tsv` A1 và video giữ riêng `heldout_videos.tsv`),
  index theo `HF_REVISION`; câu val_unseen/clean-test đến từ cả hai split → luôn
  mang `split`.
- `load_dataset` ~45 s/lần kể cả đã cache → làm ấm rồi `HF_HUB_OFFLINE=1`.
- `mamba-ssm` pin tag `v2.3.1`, build `--no-build-isolation`; wheel cp313 có trong
  dataset `mamba-wheels`. Log: `docs/notes/mamba_ssm_install_log.md`.
- Kaggle: audio là 5 Dataset tar → `python -m src.data.extract_audio` vào
  `/tmp/audio_cache`, đặt `AUDIO_CACHE_DIR`. Kernel kẹt cấp máy (RUNNING, 0 log
  > 15′, quota tăng) đã xảy ra 3 lần → huỷ, quota hoàn, đẩy lại. API log stream
  trả 500 → chỉ canh status + quota, đọc log sau khi xong. Một tài khoản chạy
  được 2 kernel GPU song song.
- **Local không test được** Mamba (CUDA) và DDP (Windows không có gloo) — khối
  Mamba giả cho test CPU; Conformer (kể cả relPE) chạy CPU được, dùng để chấm
  cục bộ khi cần.
- **Trong code:** cấu hình 10 ms không subsampling (`T' = T`); cấu hình 40 ms
  dùng `Conv2dSubsampling4` qua mục `subsampling` trong `configs/v2a/*_sub4.yaml`
  (+1,84M tham số) — đổi độ phân giải là đổi **cấu hình thí nghiệm**, không phải
  cách tăng tốc. B1 đảo theo độ dài thật (`reverse_padded`, không `torch.flip`);
  khối Mamba luôn fp32 (`mamba_fp32`); front-end và CTC loss fp32 dưới AMP.
  Đo RTF/độ trễ phải làm ấm từng độ dài (cuDNN dựng kế hoạch conv theo shape).
  Đầy đủ: `ARCHITECTURE.md` mục 8.
- LM/từ điển cho giải mã chỉ train từ `train_transcripts_filtered.txt` (bản
  `train_transcripts.txt` cũ chứa 179/203 câu clean-test).

## Ngôn ngữ & phong cách

- Code (tên file/hàm/biến): tiếng Anh, `snake_case`.
- Docstring, comment, commit message, tài liệu: tiếng Việt.
- Comment/docstring chỉ giải thích *why* không hiển nhiên, không mô tả lại *what*.

## Trước khi báo 1 việc là xong

Chạy thử thật với dữ liệu thật; sửa code dùng chung thì xác nhận mọi encoder
nhóm train từ đầu chạy được, hoặc nói rõ bên nào chưa test được (Mamba cần
CUDA, DDP cần Linux — chỉ trên Kaggle). Checklist: `docs/CONVENTIONS.md` mục 10.
