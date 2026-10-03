# Plan.md — Kế hoạch, trạng thái & quyết định

Cập nhật lần cuối: **2026-09-26**. Đây là nơi để **nối tiếp giữa các phiên làm
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

Đang ở **Tuần 6** (người dùng đính chính 2026-10-01; việc kỹ thuật vẫn ở mức Tuần 4-5 của lộ trình cũ, lộ trình sẽ viết lại theo định hướng mới).

| Tuần | Trạng thái | Ghi chú |
|---|---|---|
| 1-2 | ✅ Xong | Đề cương đã đăng ký. `mamba-ssm` v2.3.1 chạy được kernel CUDA trên Kaggle T4 |
| 3 | ✅ Xong, trừ việc tay | Khảo sát, tokenizer (vocab 1000), clean-test 250 mẫu. **Còn:** hiệu đính tay cột `corrected_text` |
| 4-5 | 🔄 Đang làm (cập nhật 2026-09-30) | Xong: dataset, front-end, CTC head, train loop (checkpoint/resume, eval WER, tensorboard), khớp tham số (Mamba 12.292.864 vs 12.204.288, +0,73%), `MambaEncoder` chạy thật trên Kaggle T4, sửa theo khuyến nghị tác giả Mamba, benchmark GPU (D8 chốt: 2 GPU + SyncBN + AMP → ước Conformer 23,2 h, Mamba 16,0 h), manifest 4 shard + val, kernel tạo dataset (tar). **Còn:** chạy kernel tạo 5 dataset (CPU); sửa code dùng chung theo D5/D7/D8/D10 (bước 3 `TODO.md`); chạy thử `train.py` `main()` trên Kaggle |
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
| 2026-09-26 | **Giữ không subsampling** (`T' = T`) | Subsampling rút ngắn chuỗi → làm yếu RQ2. Không đề xuất lại làm cách tăng tốc. `docs/notes/training_plan_kaggle.md` |
| 2026-09-26 | Dữ liệu train chia **3 Kaggle Dataset** (shard); dùng **2 tài khoản Kaggle của 2 người khác nhau** (người dùng xác nhận); train ở **chế độ nền** (batch, ≤ 9 h/phiên), không chia phiên 2 h | Chi tiết + đề xuất D1-D14 chờ duyệt: `docs/notes/training_plan_kaggle.md` |
| 2026-09-28 | **Duyệt D1-D11** (`docs/notes/training_plan_kaggle.md` mục 5): A train Conformer / B train Mamba; WAV; **4 shard** (không phải 3) vòng tròn theo video seed 42 + 1 dataset val; gắn cả shard, xáo chung; val = `validation` trừ 250 câu clean-test (**ghi rõ ở Chương 3**); notebook CPU tạo dataset; `--max_minutes` + checkpoint theo step; `num_workers` → AMP → DDP theo benchmark; pin HF `cbf624ae9b`; giữ 30 epoch | 4 shard để đĩa đỉnh lúc tạo (nếu phải tar) ~12,7 GB thay vì ~17 GB / 20 GB |
| 2026-09-28 | **Train full dữ liệu, lấy kết quả theo epoch**; so sánh hai mô hình ở cùng số epoch. Không train tập con, không train shard lần lượt; bỏ pilot 1 shard | Dừng lúc nào cũng có kết quả hợp lệ. Ràng buộc: lịch LR giữ warmup + hằng số (có decay thì phải xét lại) |
| 2026-09-30 | **D8: cả hai encoder DDP 2 GPU + SyncBatchNorm + AMP** (fp16 autocast, tham số fp32), batch toàn cục 16 | Benchmark 2 lần: Conformer 23,2 h, Mamba 16,0 h cho 30 epoch, đều < 30 h/tuần/tài khoản; cùng một cách chạy cho cả hai, SyncBN giữ Conformer tương đương 1 GPU batch 16. `docs/notes/training_plan_kaggle.md` mục 5 |
| 2026-09-30 | **Dataset dạng tar** (`PACK_TAR = True`), giải nén vào `/tmp` đầu mỗi phiên | Số đo tốc độ chỉ đúng khi đọc đĩa local; `/tmp` máy GPU trống 1,1 TB; 4 lõi CPU không bù được I/O mạng |
| 2026-09-30 | Theo khuyến nghị tác giả Mamba: miễn weight decay `A_log`/`D`, `norm_f`, chia `out_proj` cho √n_layers, residual fp32; **không** LR riêng cho Δ, **không** document packing | Mã nguồn mamba-ssm v2.3.1; nhật ký 2026-09-30 |
| 2026-10-02 | **Định hướng khảo sát, đội hình 5 mô hình**: Mamba-CTC (trọng tâm) + Conformer-CTC ~12M (baseline) train từ đầu; Parakeet-CTC-0.6B-vi; PhoWhisper-small; wav2vec2-base-vi (zero-shot trước, fine-tune sau). Loại `kyle/vi-asr-fastconformer-114m` (đã train trên VietSuperSpeech) và Zipformer-30M (mô hình sinh nhãn, license ND) | Sau buổi GVHD lần 1. `docs/notes/gvhd_buoi_1.md`, `survey_model_candidates.md`. Mục 1, `CLAUDE.md` (luật "chỉ encoder khác nhau") cần viết lại |
| 2026-10-02 | **Mamba hai chiều** (B1: 14 lớp × 2 khối xuôi/ngược, cộng; 12.285.696 tham số) | `docs/notes/mamba_bidirectional.md`. Chưa sửa code |
| 2026-10-02 | **Duyệt thay đổi phép đo**: RQ2 thành so hiệu quả giữa mô hình (RTF, VRAM, tham số, GPU-giờ) + ghép đoạn liên tiếp chỉ để đo RTF theo độ dài; chuẩn hóa văn bản chung trước WER; A3 giữ nguyên + nêu hạn chế, tách nhóm zero-shot/fine-tune khi bàn | `survey_model_candidates.md` |
| 2026-10-03 | **A1: lọc C ∩ L** — loại đoạn có LID Whisper-small ≠ `vi`, 149 video phỏng vấn nước ngoài (≥ 80% nhãn < 20% dấu), hoặc nhãn < 20% dấu. Train 175,77 h (48.340 câu), val 18,65 h, clean-test 192 câu. Khóa luận coi dữ liệu là tiếng Việt, một mức WER, không bàn tiếng Anh | Người dùng nghe kiểm (tiếng Anh + tiếng Nhật). `survey_model_candidates.md` mục 6, `data/splits/excluded.tsv` |
| 2026-09-19 | Chuẩn tài liệu gốc: `CLAUDE.md` (luật AI) · `Plan.md` (kế hoạch/trạng thái) · `ARCHITECTURE.md` (sơ đồ) · `TODO.md` · `README.md` (người ngoài) · quy ước → `docs/CONVENTIONS.md` | Chống lệch trạng thái giữa nhiều file |

## 5. Quyết định đang mở / treo / rủi ro đã biết

| Trạng thái | Vấn đề | Chi tiết |
|---|---|---|
| 🔴 **Đã chốt (2026-10-02/03, xem mục 4); còn đề cương mới** | **Đổi định hướng sau buổi GVHD lần 1**: từ so sánh có kiểm soát Mamba vs Conformer (train từ đầu, khớp tham số) sang **khảo sát** Mamba vs nhiều mô hình (Conformer pre-train, 2-3 mô hình khác, baseline). Ảnh hưởng mục 1 (mục tiêu), ràng buộc "chỉ encoder khác nhau", RQ, lộ trình mục 2 | `docs/notes/gvhd_buoi_1.md` mục 4 (Q-a…Q-g). Chưa sửa mục 1/`CLAUDE.md` cho tới khi chốt |
| 🔴 **Chờ GVHD** | Hướng xử lý sai lệch số liệu dataset ảnh hưởng RQ2 (67.405 mẫu/245,42h, audio 10-15 s vs đề cương 3-30 s) | 4 phương án trong `docs/notes/dataset_discrepancy.md`. **Không tự chọn.** |
| ✅ **Chốt hai chiều (2026-10-02)** | Mamba một chiều vs hai chiều (biến gây nhiễu RQ1); người dùng nghiêng về cân nhắc hai chiều | `docs/notes/mamba_bidirectional.md`; hỏi GVHD (A4); chốt trước khi train |
| ✅ **Đã duyệt (2026-09-28)** | D1-D11 (D3 → 4 shard) + train full, đánh giá theo epoch | Mục 4; `docs/notes/training_plan_kaggle.md` mục 5. D12/D13 vẫn chờ GVHD |
| ✅ **Đã chốt (2026-09-28)** | Prefetch full: không tải về máy/`/kaggle/working` để train, mà notebook CPU tải từ HF rồi đóng thành 5 Kaggle Dataset (4 shard train + 1 val), train gắn qua `/kaggle/input` | Theo D6 đã duyệt; `docs/notes/training_plan_kaggle.md` mục 5 |
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
| Tuần 6 (≈ 2026-10-01) — buổi báo cáo lần 1 | Tiến độ Tuần 4-5, câu hỏi A1-A4/B1/C1-C4 (`docs/tong_ket/2026-09-30/`) | **Đổi định hướng:** khảo sát so sánh nhiều mô hình, Mamba là trọng tâm; đối chứng không cần giống hệt, Conformer nên dùng bản pre-train + fine-tune; thêm 2-3 mô hình khác + 1 baseline; nhận xét theo đánh đổi độ chính xác/tốc độ/tài nguyên. A1 nhãn tiếng Anh: "tự đọc rồi làm" (xét theo ngôn ngữ pre-train). "Đừng ôm đồm". **Sẽ sửa đề cương** (bản mới). Chi tiết `docs/notes/gvhd_buoi_1.md` | Dự kiến **Tuần 9** (> 2 tuần — lưu ý Mẫu 3) |

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
