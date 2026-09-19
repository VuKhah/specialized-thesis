# CONVENTIONS.md — Quy ước làm việc cho dự án

Tra cứu khi cần, không phải đọc mỗi phiên (luật cứng nằm ở `CLAUDE.md`, kế
hoạch/trạng thái ở `Plan.md`, việc cần làm ở `TODO.md`, kiến trúc ở
`ARCHITECTURE.md`). Đây là các quy ước đang áp dụng trong repo, gom lại để
dự án lớn lên mà không vỡ cấu trúc và để hai thí nghiệm Mamba/Conformer không
bị lệch nhau — điều kiện sống còn của đề tài (matched-parameter, chỉ hoán
đổi encoder).

Cập nhật lần cuối: 2026-09-19 (tách ra từ `Plan.md` bản cũ).

## 1. Vai trò từng file — một nguồn sự thật cho mỗi loại thông tin

| File | Là nguồn sự thật cho | KHÔNG dùng để |
|---|---|---|
| `CLAUDE.md` | Luật cứng + cạm bẫy cho AI, thứ tự đọc đầu phiên. Ngắn. | Kế hoạch, trạng thái, mô tả kiến trúc. |
| `Plan.md` | Lộ trình theo tuần, trạng thái **mức tuần**, quyết định đã chốt/đang treo, nhật ký phiên. | Danh sách việc chi tiết (→ `TODO.md`). |
| `ARCHITECTURE.md` | Luồng dữ liệu, luồng tensor, interface, bản đồ file, hai môi trường. | Tiến độ, việc cần làm. |
| `TODO.md` | Việc đang chặn / sẵn sàng / đã xong, nhãn 🔴🟡🟢⏸. | Giải thích kiến trúc/lịch sử quyết định. |
| `README.md` | Giới thiệu, RQ, cài đặt, cách chạy — cho người ngoài (GVHD, hội đồng). | Trạng thái chi tiết (chỉ trỏ sang `Plan.md`/`TODO.md`). |
| `docs/notes/*.md` | Điều tra/quyết định có chiều sâu, mỗi vấn đề 1 file, viết một lần, không xoá. | Việc cần làm ngắn hạn. |
| `docs/CONVENTIONS.md` | Quy ước (file này). Thay đổi hiếm. | Tiến độ. |

**Quy tắc chống lệch:** không chép trạng thái sang nhiều nơi. Nơi nào không
phải nguồn sự thật thì chỉ trỏ link. Khi có sự kiện, sửa đúng file theo bảng:

| Sự kiện | Sửa |
|---|---|
| Xong / bị chặn / phát sinh 1 việc | `TODO.md` |
| Chốt hoặc treo 1 quyết định | `Plan.md` (bảng quyết định) + `docs/notes/` nếu cần phân tích dài |
| Đổi luồng dữ liệu, shape tensor, interface, thêm/bỏ module | `ARCHITECTURE.md` |
| Xong/đổi 1 mốc tuần | `Plan.md` (bảng trạng thái) |
| Kết thúc phiên làm việc | `Plan.md` (**thêm** 1 mục ở cuối nhật ký, không sửa mục cũ) |
| Phát hiện cạm bẫy mà session sau dễ dẫm lại | `CLAUDE.md` (1-2 dòng) hoặc `ARCHITECTURE.md` mục 8 |

## 2. Cấu trúc thư mục — quy tắc thêm file mới

```
ARCHITECTURE.md, Plan.md, TODO.md,   tài liệu gốc — xem mục 1
CLAUDE.md, README.md
docs/CONVENTIONS.md                  file này
docs/notes/*.md                      điều tra/quyết định (track trên git)
docs/de_cuong/, docs/bieu_mau/       file .docx — CHỈ nằm trên máy + Google Drive, gitignore
docs/khoa_luan/                      bản viết khóa luận (Word + PDF) — cùng quy tắc, gitignore
notebooks/                           chỉ notebook CHẠY ĐƯỢC (đã test), không để nháp
notebooks/temp/                      nháp/output chạy thử — gitignore
configs/                             1 file yaml = 1 thí nghiệm (mục 5)
src/{data,features,tokenizer,models,training,evaluation,demo}/
reports/                             số liệu, biểu đồ — output tái tạo được từ code
checkpoints/, data/, runs/           KHÔNG commit (mục 6)
```

- Code dùng chung cho cả Mamba lẫn Conformer (front-end, tokenizer, CTC head,
  training loop, eval) → không đặt riêng theo tên kiến trúc.
- Code riêng 1 kiến trúc → tên file có tiền tố kiến trúc (`mamba_encoder.py`,
  `conformer_encoder.py`), cùng implement `ASREncoder`
  (`src/models/encoder_base.py`). Không rải `if arch == "mamba"` trong code
  dùng chung; điểm hoán đổi duy nhất là chỗ khởi tạo encoder (`build_encoder`
  trong `train.py`).
- Notebook mới: `NN_ten-muc-dich.ipynb`, số theo trình tự chạy. Phải chạy thử
  thật trước khi giao. **Không để output nhúng audio** (mục 7).
- Ghi chú điều tra/quyết định dài → file mới trong `docs/notes/`, không nhét
  vào README.

## 3. Đặt tên

- File/thư mục/biến Python: `snake_case`, tiếng Anh.
- Docstring/comment/commit message/tài liệu: tiếng Việt.
- `experiment_name` (trong yaml): `<kiến_trúc>_<mô tả ngắn>`, vd.
  `conformer_ctc_baseline`. Dùng đặt tên checkpoint/log; **không đổi giữa
  chừng** (gãy resume/so sánh).
- Split dataset: đúng tên thật trên HF Hub — `train`, `validation` (không dùng
  `dev-test`, đề cương ghi nhầm — xem `docs/notes/dataset_discrepancy.md`).

## 4. Quy ước code

- Type hint cho tham số hàm public.
- Docstring đầu file **chỉ khi cần ghi lý do/lịch sử/cạm bẫy không hiển nhiên
  từ code**. Không mô tả lại điều tên hàm/biến đã nói.
- Comment inline giải thích **why**, không phải **what**.
- Không thêm try/except/fallback cho tình huống không thể xảy ra trong
  pipeline nội bộ. Validate ở biên: dữ liệu từ HF Hub, input người dùng (demo).
- Code dùng chung sửa 1 chỗ áp dụng cho cả hai thí nghiệm. Nếu thay đổi chỉ có
  nghĩa cho 1 kiến trúc, đó là dấu hiệu code đặt sai chỗ.
- Windows/tiếng Việt: script chạy local đặt `PYTHONIOENCODING=utf-8` hoặc
  `sys.stdout.reconfigure(encoding="utf-8")` ở đầu (cp1252 sẽ crash khi print).
- Chạy mọi lệnh `python -m src...` **từ gốc repo**: các đường dẫn như
  `data/raw/audio_cache` là tương đối theo thư mục hiện hành (đã có lần chạy
  notebook từ `notebooks/` đẻ ra `notebooks/data/raw/...`).

## 5. Config & thí nghiệm (`configs/*.yaml`)

- Mỗi thí nghiệm 1 file yaml độc lập, đầy đủ tham số (không kế thừa/override
  ngầm) — dễ diff `model_conformer.yaml` với `model_mamba.yaml`.
- Số tham số hai encoder khớp trong < 5%. Dùng `python -m src.models.param_count`,
  không ước lượng tay. **Lưu ý:** script này hiện dựng encoder bằng tham số
  mặc định trong code chứ không đọc yaml (xem `ARCHITECTURE.md` mục 8).
- Comment trong yaml ghi rõ giá trị nào phụ thuộc môi trường (vd. `batch_size`
  theo VRAM Kaggle).
- Số thực dạng khoa học phải có dấu chấm thập phân: `3.0e-4`, không phải
  `3e-4` (PyYAML đọc thành string).
- Seed cố định cho mọi bước ngẫu nhiên ảnh hưởng khả năng tái lập (chia tập,
  sample clean-test, init model, shuffle) — đã dùng `seed=42` cho clean-test.

## 6. Dữ liệu & checkpoint

- Không commit raw audio, checkpoint, cache, tensorboard log — `.gitignore`
  chặn. Ngoại lệ duy nhất: `data/processed/clean_test_manifest.json` (công sức
  tay). Thêm file "công sức tay" khác thì thêm ngoại lệ tường minh, không đổi
  rule chung. Không dùng `git add -f` để vượt `.gitignore`.
- Cache audio (`data/raw/audio_cache/`) không tự đồng bộ giữa local và Kaggle;
  mỗi môi trường tự chạy prefetch riêng. Đừng giả định cache "đã có".
- Checkpoint đặt theo `checkpoints/<experiment_name>/{latest,best}.pt`.

## 7. Repo public — cái gì được lên GitHub

Repo `VuKhah/specialized-thesis` **đang public**. Nguyên tắc: chỉ code, config,
ghi chú `.md`, số liệu tổng hợp nhỏ.

- **Không** đưa lên: file `.docx`/biểu mẫu/đề cương (lưu Google Drive), audio
  (kể cả nhúng base64 trong output notebook), checkpoint, token/khoá
  (`.env`, `kaggle.json`, `hf_token*`), họ tên/MSSV người thứ ba (kể cả trong
  tên file).
- Notebook track trên git: xoá output trước khi commit (Clear All Outputs hoặc
  `nbstripout`), nhất là output `IPython.display.Audio` — nhúng nguyên audio
  YouTube vào git. *Ngoại lệ hiện có: `02_dataset_eda.ipynb` còn output —
  đã biết, chưa xử lý, xem `TODO.md`.*
- `.gitignore` chỉ có tác dụng với file **chưa track**. File đã lỡ commit vẫn
  nằm trong lịch sử; gỡ hẳn cần viết lại lịch sử + force push (thao tác ra
  ngoài, không hoàn tác — chỉ làm khi người dùng yêu cầu rõ).

## 8. Git

- Commit message tiếng Việt, tiền tố tuần nếu gắn tuần cụ thể
  (`"Tuần N: <mô tả>"`), không thì mô tả thẳng nội dung.
- Commit thẳng `master` (dự án cá nhân). Thử nghiệm có rủi ro làm hỏng baseline
  dùng chung (đổi `ASREncoder`, đổi front-end) thì dùng branch riêng rồi merge.
- Trước khi commit: `git status` kiểm tra không có file bị Word/notebook tự sửa
  ngoài ý muốn; không có file nằm ngoài dự định (đặc biệt `.docx`, audio).
- Chỉ commit khi người dùng yêu cầu.

## 9. Khi gặp vấn đề/quyết định mở

1. Ghi đầy đủ trong `docs/notes/<ten-van-de>.md`: dữ kiện, vì sao là vấn đề,
   các phương án + đánh đổi.
2. Thêm 1 dòng vào `TODO.md` (🔴 nếu đang chặn, ⏸ nếu người dùng chủ động
   treo) và 1 dòng vào bảng quyết định mở trong `Plan.md`. Trỏ sang note, không
   lặp chi tiết.
3. **Không tự chọn phương án thay người dùng/GVHD** nếu vấn đề ảnh hưởng thiết
   kế nghiên cứu — kể cả khi có phương án "hợp lý nhất".
4. Khi có quyết định: thêm mục "Quyết định cuối" vào note (không xoá lịch sử),
   chuyển dòng sang bảng "đã chốt" trong `Plan.md`, chuyển TODO sang ✅.

## 10. Definition of Done — trước khi coi 1 việc là xong

- [ ] Code/script **chạy thật** ít nhất 1 lần với dữ liệu thật (không chỉ đọc
      code thấy hợp lý). Mẫu tốt: `prefetch_audio.py` (5 file thật + 1 file
      lỗi cố ý), `train.py` resume test.
- [ ] Sửa code dùng chung (front-end/tokenizer/CTC head/training loop/eval):
      xác nhận **cả hai** encoder vẫn chạy, hoặc ghi rõ bên nào chưa test được
      và vì sao (vd. Mamba cần CUDA — chỉ có trên Kaggle) vào `TODO.md`.
- [ ] Nếu đổi luồng dữ liệu/shape/interface → `ARCHITECTURE.md` đã cập nhật.
- [ ] `TODO.md`/`Plan.md` cập nhật đúng vai trò (mục 1), không để lệch.
- [ ] Nếu phát sinh quyết định mở mới → đã theo mục 9.

## 11. Môi trường & khả năng tái lập

- Môi trường thật: **Kaggle** (2x T4, 30 GPU-giờ/tuần, session ~9-12h) — đề
  cương đã nộp ghi Colab, đây là lệch có chủ đích đã xác nhận, không sửa
  ngược lại đề cương.
- `requirements.txt` là danh sách tối thiểu, không phải lockfile. Thêm thư viện
  mới thì thêm vào kèm 1 dòng comment lý do nếu không hiển nhiên.
- Cài `mamba-ssm`: luôn pin tag `v2.3.1`, không dùng `main` (kéo theo
  Mamba-3/tilelang/tvm, lỗi Python 3.12 — xem
  `docs/notes/mamba_ssm_install_log.md`). Mamba-3 đang tạm gác
  (`docs/notes/mamba_versions.md`).
- Training có checkpoint/resume (đã xong, commit `139f018`) vì Kaggle giới
  hạn session liên tục.
