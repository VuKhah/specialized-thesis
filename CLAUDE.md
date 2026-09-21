# CLAUDE.md

Luật cho Claude Code khi làm việc trong repo này. Ngắn có chủ đích: kế hoạch/
trạng thái ở `Plan.md`, việc cần làm ở `TODO.md`, kiến trúc ở
`ARCHITECTURE.md`, quy ước chi tiết ở `docs/CONVENTIONS.md`. **Không lặp lại
nội dung các file đó ở đây** — chỉ ghi thứ cần biết *trước khi chạm vào code*.

## Đề tài (1 dòng)

KLTN so sánh matched-parameter Mamba (SSM) vs Conformer làm encoder trong cùng
1 pipeline CTC ASR tiếng Việt hội thoại, train từ đầu trên VietSuperSpeech.
Chi tiết: `README.md`.

## Đầu mỗi phiên (theo thứ tự)

1. `TODO.md` mục 🔴 và ⏸ — có quyết định đang chờ/đang treo không.
2. `Plan.md` mục 3 (đang ở tuần nào), mục 5 (quyết định mở), và **mục cuối của
   nhật ký (mục 7)** — phiên trước dừng ở đâu, phiên này bắt đầu từ đâu. Mục 6
   là khóa luận (bản viết) + lịch gặp GVHD (≤ 2 tuần/lần) — xem qua khi việc
   liên quan.
3. `git log --oneline -10` + `git status` — code tiến triển nhanh hơn trí nhớ,
   đừng giả định trạng thái từ phiên trước còn đúng.
4. Nếu sắp đụng code: `ARCHITECTURE.md` (luồng dữ liệu, shape, mục 8 cạm bẫy
   trong code). Nếu liên quan dataset/RQ2: `docs/notes/dataset_discrepancy.md`.

## Cuối mỗi phiên

Thêm 1 mục **ở cuối** nhật ký trong `Plan.md` (đã làm gì · chốt gì · phiên sau
bắt đầu từ đâu) và cập nhật `TODO.md`. Đổi luồng dữ liệu/shape/interface thì
cập nhật `ARCHITECTURE.md`. Bảng "sự kiện → file nào" ở
`docs/CONVENTIONS.md` mục 1. Không tự commit khi người dùng chưa yêu cầu.

## Luật cứng — không vi phạm dù không ai nhắc lại

- **Không tự chọn phương án cho quyết định đang mở** (🔴 hoặc ⏸ trong
  `TODO.md`, vd. hướng xử lý sai lệch dataset ảnh hưởng RQ2, nơi/cách chạy
  prefetch). Những quyết định này thuộc về người dùng/GVHD — hỏi trước, không
  "chọn phương án hợp lý nhất" rồi làm luôn.
- **Không sửa nội dung `docs/de_cuong/*.docx`** — đã nộp GVHD. Thực tế lệch đề
  cương thì ghi vào `Plan.md`/`TODO.md`/`docs/notes`, không sửa ngược docx.
- **Không để code dùng chung (front-end log-mel, tokenizer, CTC head, training
  loop, eval) lệch giữa hai thí nghiệm.** Điểm hoán đổi duy nhất là chỗ khởi
  tạo encoder (`ASREncoder`, `src/models/encoder_base.py`). Sửa code dùng chung
  thì nghĩ tới cả hai kiến trúc.
- **Repo GitHub là PUBLIC.** Không commit: `data/raw/`, `data/processed/*` (trừ
  `clean_test_manifest.json`), `checkpoints/*`, audio/model, **file `.docx` /
  biểu mẫu / bản nháp khóa luận `docs/khoa_luan/` (lưu Google Drive)**, khoá/token, họ tên-MSSV người thứ ba. Không
  `git add -f` để vượt `.gitignore`. Notebook track trên git phải xoá output
  (đặc biệt `IPython.display.Audio` nhúng audio). Chi tiết:
  `docs/CONVENTIONS.md` mục 7.
- **Không viết lại lịch sử git / force push** khi người dùng chưa yêu cầu rõ.

## Cạm bẫy đã biết (đừng khám phá lại)

- Console Windows mặc định cp1252 → script Python chạy local phải
  `PYTHONIOENCODING=utf-8` hoặc `sys.stdout.reconfigure(encoding="utf-8")`
  (kể cả lệnh `python -` một dòng dùng trong phiên), nếu không crash khi print
  tiếng Việt.
- Chạy mọi `python -m src...` **từ gốc repo** (đường dẫn cache là tương đối
  theo cwd).
- Cột `audio` của VietSuperSpeech là **string path tương đối**, không phải
  `datasets.Audio` tự giải mã — phải tải qua `hf_hub_download`/prefetch rồi đọc
  bằng `soundfile` (đã xử lý trong `src/data/vietsuperspeech_dataset.py`).
- Tên split thật trên HF Hub: `train` / `validation` — **không phải**
  `dev-test` như đề cương ghi nhầm.
- Cài `mamba-ssm`: pin tag `v2.3.1`, build `--no-build-isolation`. Nhánh `main`
  kéo Mamba-3/tilelang/tvm, lỗi Python 3.12. Log: `docs/notes/mamba_ssm_install_log.md`.
- Môi trường thật là **Kaggle** (2x T4), không phải Colab như đề cương ghi —
  lệch có chủ đích, đã xác nhận, không phải lỗi.
- Trước khi train thật: `python -m src.data.prefetch_audio` (không
  `snapshot_download` cả repo — thừa ~43%); đừng để `__getitem__` tự tải lẻ khi
  train (~1-4 s/file × 60k+ mẫu). *Chạy ở đâu và lưu cache thế nào: đang treo,
  xem `Plan.md` mục 5.*
- **Trong code:** hai encoder không subsampling; Mamba chưa mask padding;
  `train.py` 1 GPU không AMP; docstring vài file còn cũ — đầy đủ ở
  `ARCHITECTURE.md` mục 8.

## Ngôn ngữ & phong cách

- Code (tên file/hàm/biến): tiếng Anh, `snake_case`.
- Docstring, comment, commit message, tài liệu: tiếng Việt.
- Comment/docstring chỉ giải thích *why* không hiển nhiên (lịch sử quyết định,
  cạm bẫy), không mô tả lại *what*. Ví dụ: `docs/CONVENTIONS.md` mục 4.

## Trước khi báo 1 việc là xong

Chạy thử thật với dữ liệu thật (không chỉ đọc code thấy hợp lý); sửa code dùng
chung thì xác nhận cả hai encoder chạy được, hoặc nói rõ bên nào chưa test được
(Mamba cần CUDA — chỉ có trên Kaggle). Checklist: `docs/CONVENTIONS.md` mục 10.
