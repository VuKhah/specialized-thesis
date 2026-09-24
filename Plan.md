# Plan.md — Kế hoạch, trạng thái & quyết định

Cập nhật lần cuối: **2026-09-21**. Đây là nơi để **nối tiếp giữa các phiên làm
việc**: kế hoạch theo tuần, trạng thái mức tuần, quyết định đã chốt/đang treo,
nhật ký phiên. Việc chi tiết ở [`TODO.md`](TODO.md); kiến trúc ở
[`ARCHITECTURE.md`](ARCHITECTURE.md); luật cho AI ở [`CLAUDE.md`](CLAUDE.md);
quy ước ở [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md).

> Đọc nhanh khi mở phiên mới: mục 3 (đang ở đâu) → mục 5 (quyết định đang mở)
> → mục 6 (khóa luận, lịch gặp GVHD) → **mục cuối của nhật ký (mục 7)**.

## 1. Mục tiêu

So sánh có kiểm soát (matched-parameter, cùng pipeline CTC, train từ đầu trên
VietSuperSpeech) **Mamba** và **Conformer** làm encoder ASR tiếng Việt hội
thoại. Ba câu hỏi: **RQ1** WER · **RQ2** RTF/latency theo độ dài audio ·
**RQ3** phân loại lỗi. Điều kiện sống còn: chỉ encoder khác nhau, mọi thứ còn
lại dùng chung (`ARCHITECTURE.md` mục 1).

## 2. Lộ trình 15 tuần (theo đề cương đã đăng ký)

| Tuần | Nội dung theo đề cương |
|---|---|
| 1-2 | Đọc lý thuyết; cài môi trường; thử cài `mamba-ssm` (rủi ro cao nhất); hoàn thiện đề cương |
| 3 | Chuẩn bị dữ liệu: khảo sát, log-mel, tokenizer BPE, clean-test 200-300 câu hiệu đính tay (seed cố định) |
| 4-5 | Pipeline CTC dùng chung (feature, CTC head, train/eval loop, WER); cài baseline Conformer-CTC |
| 6-7 | Huấn luyện Conformer-CTC; tinh chỉnh siêu tham số; theo dõi checkpoint |
| 8 | Đánh giá Conformer: WER trên dev-test (= `validation`) + clean-test; đo RTF/latency theo độ dài audio |
| 9-10 | Cài Mamba-CTC (khớp tham số với Conformer); huấn luyện từ đầu |
| 11 | Tiếp tục/tinh chỉnh Mamba-CTC; xử lý hội tụ chậm nếu có |
| 12 | Đánh giá Mamba: WER, RTF/latency; so sánh trực tiếp |
| 13 | Phân tích lỗi định tính (5-6 loại lỗi); bảng số liệu, biểu đồ |
| 14 | Demo Gradio; hoàn thiện mã; viết Chương 1-4 |
| 15 | Hoàn thiện luận văn, slide, báo cáo thử, nộp GVHD |

Mốc theo tuần chỉ là khung; tiến độ thật ở mục 3. Nội dung tuần 8 và 12 (RTF
theo độ dài audio) đang bị chặn bởi quyết định 🔴 ở mục 5.

## 3. Trạng thái hiện tại (mức tuần — chi tiết ở `TODO.md`)

Đang ở **Tuần 4-5**.

| Tuần | Trạng thái | Ghi chú |
|---|---|---|
| 1-2 | ✅ Xong | Đề cương đã đăng ký. `mamba-ssm` v2.3.1 chạy được kernel CUDA trên Kaggle T4 |
| 3 | ✅ Xong, trừ việc tay | Khảo sát, tokenizer (vocab 1000), clean-test 250 mẫu. **Còn:** hiệu đính tay cột `corrected_text` |
| 4-5 | 🔄 Đang làm | Xong: dataset, front-end, CTC head, train loop (checkpoint/resume, eval WER, tensorboard, test thật với Conformer), script prefetch. **Còn:** chạy prefetch full (⏸ treo), xác nhận `MambaEncoder` + `train.py` trên Kaggle, khớp tham số hai encoder |
| 6-7 | ⬜ Chưa | Train Conformer baseline |
| 8 | ⬜ Chưa | Bị 🔴 (RQ2) một phần |
| 9-15 | ⬜ Chưa | — |

Song song các tuần: **viết khóa luận** (🔄 đã có file khung + nháp Ch1 — mục 6) và **gặp GVHD
tối đa 2 tuần/lần** (mục 6, chưa có lịch ghi).

## 4. Quyết định đã chốt

| Ngày | Quyết định | Lý do / nơi ghi |
|---|---|---|
| 2026-09-14 | Hạ tầng thật là **Kaggle** (2x T4), không phải Colab; chỉ sửa tài liệu nội bộ, **giữ nguyên** `.docx` đã nộp | Lệch có chủ đích. `docs/notes/mamba_ssm_install_log.md` |
| 2026-09-14 | Dùng **Mamba-1/S6**, pin `mamba-ssm` tag `v2.3.1`; **Mamba-3 tạm gác** (chỉ nhắc như hướng phát triển) | Khớp trích dẫn đề cương; tránh lỗi tilelang/tvm. `docs/notes/mamba_versions.md` |
| 2026-09-14 | Tokenizer BPE `vocab_size=1000`; clean-test 250 mẫu, seed 42, lấy từ `validation` | Tuần 3 |
| 2026-09-16 | Prefetch **chỉ đúng 67.405 file** cần dùng, không `snapshot_download` cả thư mục | Repo HF có 118.259 file, thừa ~43% |
| — | Train **Conformer baseline trước**, Mamba sau | Đúng thứ tự đề cương |
| 2026-09-19 | **Tài liệu nhị phân (`.docx`, biểu mẫu) lưu trên Google Drive, không đưa lên GitHub** (repo public). Đã gỡ 12 file khỏi index, thêm rule `.gitignore` | Đã commit local, **chưa push**. `docs/CONVENTIONS.md` mục 7 |
| 2026-09-19 | **Khóa luận viết bằng Word (`.docx` + PDF) theo Mẫu 5**, lưu `docs/khoa_luan/` (gitignore) + sao lưu Drive, không lên GitHub; **viết song song theo chương từ sớm** thay vì dồn Tuần 14-15 | Mẫu 3 "vừa làm vừa viết", Mẫu 5 yêu cầu 50-100 trang. Mục 6 |
| 2026-09-24 | Mamba khớp tham số bằng **tăng độ sâu**: `n_layers: 28`, `expand: 2`, `d_model: 256`, `d_state: 16` → 12.292.352 vs Conformer 12.204.288 (+0,72%). Phương án khác đã cân nhắc: 14 layer × `expand: 4` (+0,66%) | Giữ siêu tham số mặc định bài Mamba gốc, dễ biện luận. Chưa xác nhận trên Kaggle (`TODO.md`) |
| 2026-09-19 | Chuẩn tài liệu gốc: `CLAUDE.md` (luật AI) · `Plan.md` (kế hoạch/trạng thái) · `ARCHITECTURE.md` (sơ đồ) · `TODO.md` · `README.md` (người ngoài) · quy ước → `docs/CONVENTIONS.md` | Chống lệch trạng thái giữa nhiều file |

## 5. Quyết định đang mở / treo / rủi ro đã biết

| Trạng thái | Vấn đề | Chi tiết |
|---|---|---|
| 🔴 **Chờ GVHD** | Hướng xử lý sai lệch số liệu dataset ảnh hưởng RQ2 (67.405 mẫu/245,42h, audio 10-15 s vs đề cương 3-30 s) | 4 phương án trong `docs/notes/dataset_discrepancy.md`. **Không tự chọn.** |
| ⏸ **Treo (2026-09-19)** | Prefetch full: hướng nêu là chạy trên Kaggle, nhưng cách lưu cache chưa chốt | ~27 GB; hạn mức `/kaggle/working` = 20 GB (người dùng xác nhận 2026-09-21) → 27 GB không vừa, các vị trí đĩa khác chưa kiểm chứng và việc mỗi phiên bắt đầu trống, cache local không chuyển sang được. Lưu ý: nếu phải tải lại mỗi phiên sẽ tốn GPU-giờ |
| ⚠️ Biết, chưa xử lý | `notebooks/02_dataset_eda.ipynb` nhúng audio YouTube (20 output `<audio>`, ~5 MB) trong repo **public**, có cả trong lịch sử git | Người dùng chọn bỏ qua (2026-09-19). Chưa kiểm license VietSuperSpeech |
| ⚠️ Chưa quyết | Tên người thứ ba + MSSV trong tên file TLCN (đã gỡ index, còn trong lịch sử git); xoá khỏi lịch sử cần viết lại lịch sử + force push | Chỉ làm khi người dùng yêu cầu |

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
| Ch3 Thực nghiệm và đánh giá | `reports/`, `docs/notes/dataset_discrepancy.md` | Mô tả dữ liệu khi 🔴 có quyết định; chèn kết quả sau Tuần 8, 12, 13 | ⬜ |
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
| *(chưa ghi — cần người dùng cung cấp buổi gần nhất)* | | | |

**Câu hỏi nên hỏi ở buổi gặp tới:** (1) 🔴 hướng xử lý sai lệch dataset/RQ2
(`docs/notes/dataset_discrepancy.md`); (2) hình thức nộp hiện hành (biểu mẫu
ban hành 2018: bìa mềm + CD còn áp dụng không); (3) phần "thiết kế" ở Ch2 cần
dạng nào (Mẫu 5 nhắc UML cho đề tài ứng dụng, đề tài này thiên về nghiên cứu).

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
