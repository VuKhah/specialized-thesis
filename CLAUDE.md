# CLAUDE.md

Luật cho Claude Code khi làm việc trong repo này. Ngắn có chủ đích: kế hoạch/
trạng thái ở `Plan.md`, việc cần làm ở `TODO.md`, kiến trúc ở
`ARCHITECTURE.md`, quy ước chi tiết ở `docs/CONVENTIONS.md`. **Không lặp lại
nội dung các file đó ở đây** — chỉ ghi thứ cần biết *trước khi chạm vào code*.

## Đề tài (1 dòng)

KLTN **khảo sát Mamba (trọng tâm) làm encoder ASR tiếng Việt hội thoại** trên
VietSuperSpeech, nhận xét theo đánh đổi WER – tốc độ – tài nguyên, so với
Conformer-12M train từ đầu và một mô hình pre-train (Parakeet-CTC-0.6B-vi).
Đội hình, hai mức so sánh, RQ: `Plan.md` mục 1. Định hướng đổi sau buổi GVHD
lần 1 (2026-10-01); đề cương mới chưa duyệt.

## Đầu mỗi phiên (theo thứ tự)

1. `TODO.md` mục 🔴 và ⏸ — có quyết định đang chờ/đang treo không.
2. `Plan.md` mục 3 (đang ở đâu), mục 5 (quyết định mở), và **mục cuối của
   nhật ký (mục 7)** — phiên trước dừng ở đâu, phiên này bắt đầu từ đâu. Mục 6
   là khóa luận (bản viết) + lịch gặp GVHD (≤ 2 tuần/lần) — xem qua khi việc
   liên quan.
3. `git log --oneline -10` + `git status` — code tiến triển nhanh hơn trí nhớ,
   đừng giả định trạng thái từ phiên trước còn đúng.
4. Nếu sắp đụng code: `ARCHITECTURE.md` (luồng dữ liệu, shape, mục 8 cạm bẫy
   trong code). Việc theo từng mô hình: `docs/notes/lineup_preparation.md`.
   Dataset/RQ2: `docs/notes/dataset_discrepancy.md`. Lý do front-end/CTC/
   dropout: `docs/notes/frontend_decoder_survey.md`.

## Cuối mỗi phiên

Thêm 1 mục **ở cuối** nhật ký trong `Plan.md` (đã làm gì · chốt gì · phiên sau
bắt đầu từ đâu) và cập nhật `TODO.md`. Đổi luồng dữ liệu/shape/interface thì
cập nhật `ARCHITECTURE.md`. Bảng "sự kiện → file nào" ở
`docs/CONVENTIONS.md` mục 1. Không tự commit khi người dùng chưa yêu cầu.

## Luật cứng — không vi phạm dù không ai nhắc lại

- **Không tự chọn phương án cho quyết định đang mở** (🔴 hoặc ⏸ trong
  `TODO.md`, mục 5 `Plan.md`, vd. tiêu chí chọn B1 hay ConExtBiMamba, mức độ
  dài RQ2, nội dung đề cương mới). Những quyết định này thuộc về người
  dùng/GVHD — hỏi trước, không "chọn phương án hợp lý nhất" rồi làm luôn.
  Đề xuất kỹ thuật phải dựa trên tài liệu công bố đã kiểm, ghi nguồn; chỗ
  không có bằng chứng thì nói rõ, không đoán.
- **Không sửa nội dung `docs/de_cuong/*.docx`** — đã nộp GVHD. Đề cương mới là
  file mới; thực tế lệch đề cương thì ghi vào `Plan.md`/`TODO.md`/`docs/notes`.
- **Hai mức so sánh, không trộn:**
  - *Nhóm train từ đầu* (Mamba B1, ConExtBiMamba, Conformer-12M): **chỉ encoder
    khác nhau.** Front-end (log-mel + CMVN + SpecAugment), tokenizer, CTC head,
    `train.py`, eval dùng chung; điểm hoán đổi duy nhất là khởi tạo encoder
    (`ASREncoder`, `build_encoder` trong `src/training/train.py`). Sửa code dùng
    chung thì nghĩ tới **mọi** encoder của nhóm. Dựng model qua `build_model`
    (đọc mục `features` của yaml), đừng tự gọi `CTCASRModel(...)`.
  - *Nhóm pre-train* (Parakeet, dự bị PhoWhisper): giữ pipeline/front-end riêng
    của mô hình; chỉ dùng chung `normalize_text` trước WER và tập đánh giá
    (clean-test 203 câu từ video giữ riêng). Khi kết luận phải tách nhóm, nêu điều kiện khác nhau.
- **Repo GitHub là PUBLIC.** Không commit: `data/raw/`, `data/processed/*` (trừ
  `clean_test_manifest.json`), `checkpoints/*`, audio/model, **file `.docx` /
  biểu mẫu / bản nháp khóa luận `docs/khoa_luan/` (lưu Google Drive)**, khoá/token, họ tên-MSSV người thứ ba. Không
  `git add -f` để vượt `.gitignore`. Notebook track trên git phải xoá output
  (đặc biệt `IPython.display.Audio` nhúng audio). Chi tiết:
  `docs/CONVENTIONS.md` mục 7.
- **Không viết lại lịch sử git / force push** khi người dùng chưa yêu cầu rõ.
- **Quota GPU Kaggle (30 h/tuần/tài khoản) là tài nguyên của người dùng** —
  chạy kernel GPU phải xin duyệt, kèm ước thời gian.

## Cạm bẫy đã biết (đừng khám phá lại)

- Console Windows mặc định cp1252 → script Python chạy local phải
  `PYTHONIOENCODING=utf-8` hoặc `sys.stdout.reconfigure(encoding="utf-8")`
  (kể cả lệnh `python -` một dòng dùng trong phiên), nếu không crash khi print
  tiếng Việt.
- Chạy mọi `python -m src...` **từ gốc repo** (đường dẫn cache là tương đối
  theo cwd).
- Cột `audio` của VietSuperSpeech là **string path tương đối**, không phải
  `datasets.Audio` tự giải mã — phải có file local rồi đọc bằng `soundfile`
  (đã xử lý trong `src/data/vietsuperspeech_dataset.py`).
- Tên split thật trên HF Hub: `train` / `validation` — **không phải**
  `dev-test` như đề cương ghi nhầm. Chọn mẫu qua `manifest_rows` (đọc
  `data/splits/*.tsv`, **đã bỏ `excluded.tsv`** — A1 — và với manifest shard bỏ
  thêm **video giữ riêng** `heldout_videos.tsv`), index theo `HF_REVISION`. Câu
  của val_unseen/clean-test đến từ cả hai split → luôn mang theo `split`.
- `load_dataset` mất ~45 s/lần kể cả đã cache → làm ấm
  (`python -m src.data.vietsuperspeech_dataset`) rồi chạy `HF_HUB_OFFLINE=1`.
- Cài `mamba-ssm`: pin tag `v2.3.1`, build `--no-build-isolation`. Nhánh `main`
  kéo Mamba-3/tilelang/tvm, lỗi Python 3.12. Log: `docs/notes/mamba_ssm_install_log.md`.
- Môi trường thật là **Kaggle** (2x T4), không phải Colab như đề cương ghi —
  lệch có chủ đích, đã xác nhận, không phải lỗi.
- Audio trên Kaggle: 5 Kaggle Dataset dạng tar (4 shard train + 1 val) →
  `python -m src.data.extract_audio` vào `/tmp/audio_cache`, đặt
  `AUDIO_CACHE_DIR`. Đừng để `__getitem__` tự tải lẻ khi train (~1-4 s/file).
- **Local không test được:** Mamba (cần CUDA) và DDP (PyTorch Windows không có
  gloo device) — dùng khối Mamba giả cho test CPU, DDP/AMP kiểm trên Kaggle.
- **Trong code:** không subsampling (`T' = T`, chốt vì RQ2); Mamba hai chiều B1
  đảo theo độ dài thật (`reverse_padded`, **không** `torch.flip`) nên không
  cần mask; `train.py` DDP + AMP, front-end và CTC loss luôn fp32 — đầy đủ ở
  `ARCHITECTURE.md` mục 8.

## Ngôn ngữ & phong cách

- Code (tên file/hàm/biến): tiếng Anh, `snake_case`.
- Docstring, comment, commit message, tài liệu: tiếng Việt.
- Comment/docstring chỉ giải thích *why* không hiển nhiên (lịch sử quyết định,
  cạm bẫy), không mô tả lại *what*. Ví dụ: `docs/CONVENTIONS.md` mục 4.

## Trước khi báo 1 việc là xong

Chạy thử thật với dữ liệu thật (không chỉ đọc code thấy hợp lý); sửa code dùng
chung thì xác nhận mọi encoder của nhóm train từ đầu chạy được, hoặc nói rõ
bên nào chưa test được (Mamba/ConExtBiMamba cần CUDA, DDP cần Linux — chỉ có
trên Kaggle). Checklist: `docs/CONVENTIONS.md` mục 10.
