# Kế hoạch dữ liệu + train trên Kaggle (tổng hợp phiên 2026-09-26)

Tổng hợp thảo luận phiên 2026-09-26: vấn đề đã nêu, quyết định người dùng
đã chốt, và **đề xuất của AI** cho các quyết định còn bỏ dở. Phần đề xuất
**chưa được duyệt** — chưa sửa code, chưa tạo dataset. Số liệu gốc và nguồn
web ở [`prefetch_storage_review.md`](prefetch_storage_review.md).

## 0. Đề xuất cho các quyết định còn bỏ dở — tóm tắt (chờ duyệt)

Bản rút gọn của mục 3 (đầy đủ lý do + người duyệt ở mục 3).

| Việc | Đề xuất |
|---|---|
| Chia 2 tài khoản (D1) | Tài khoản A train Conformer, B train Mamba — không gộp trọng số |
| Lưu dữ liệu (D2-D4, D6) | WAV (không FLAC), 3 shard ~8,5 GB chia vòng tròn theo video (seed 42), tạo bằng notebook CPU; train gắn cả 3, xáo trộn chung |
| Val / test (D5) | Val = validation trừ 250 câu clean-test (6.499 câu, dataset thứ 4); test = clean-test sau hiệu đính |
| Phiên train (D7) | Chạy nền ~9 h + `--max_minutes` tự lưu và thoát; checkpoint theo step, resume giữa epoch |
| Tăng tốc (D8, D9, D11) | Đo benchmark trước; `num_workers` → AMP → 2 GPU; giữ 30 epoch; không subsampling (đã chốt) |
| Tái lập (D10) | Pin revision HF `cbf624ae9b`; cùng commit + seed ở hai tài khoản; đo RTF cả hai mô hình trên cùng máy |
| RQ2 — Q1 (D12, **GVHD**) | Phương án 2: ghép đoạn liên tiếp chỉ để đo RTF |
| Val trùng video train — Q10 (D13, **GVHD**) | Giữ split của tác giả, nêu là hạn chế |
| Tài khoản (D14) | ✅ Đã xác nhận: 2 người khác nhau, mỗi người tự chạy bằng khoá API của mình |

Thứ tự khi được duyệt: kernel kiểm chứng (CPU + GPU benchmark) → sửa code
dùng chung → manifest shard → notebook CPU tạo 4 dataset → train song song
hai tài khoản.

## 1. Vấn đề đã nêu trong phiên

| # | Vấn đề | Số liệu / bằng chứng |
|---|---|---|
| V1 | Dữ liệu train+val không vừa đĩa: máy cá nhân hạn chế, `/kaggle/working` 20 GB | Cần 67.405 WAV = **28,27 GB** (đo qua HF API) |
| V2 | Tải lại từ HF mỗi phiên rất tốn | Rate limit HF 3.000 (ẩn danh) / 5.000 (token) lượt/5 phút ⇒ ≥ 67-112 phút GPU mỗi lần tải đủ |
| V3 | Train quá lâu so với quota | ~1,3 s/step × 3.791 step ≈ 80 phút/epoch (**chưa tính đọc dữ liệu**) → 30 epoch ≈ 40 GPU-giờ/mô hình > 30 GPU-giờ/tuần |
| V4 | Nguyên nhân chậm | Không subsampling (T ≈ 1.500 khung, attention O(T²)); fp32 không AMP; 1/2 GPU; `DataLoader` không `num_workers` |
| V5 | Không thể ngồi canh máy 40 h | `train.py` chỉ lưu checkpoint cuối epoch → ngắt giữa epoch mất tới ~80 phút |
| V6 | Chia dữ liệu nhỏ có làm lệch/trùng/lọt không | Mô phỏng chia 3 shard phân tầng: phân bố gần như giống hệt (mục 7 review) |
| V7 | Validation không độc lập với train | 6.748/6.749 câu val (và toàn bộ clean-test) đến từ video đã có trong train → WER lạc quan |
| V8 | Clean-test nằm trong tập chọn `best.pt` | Đã ghi ⚠️ từ trước |
| V9 | Ghép đoạn liên tiếp (phương án 2 RQ2) có khả thi | Chuỗi ≥ 3 đoạn liên tiếp: chỉ dev 57 · chỉ train 17.585 · lẫn 21.795; 6,3 % chỉ số đoạn bị thiếu, không có timestamp |
| V10 | Dataset lệch mô tả | Train+val chỉ 1 kênh (vietcetera), không phải 4; README HF cũ; repo HF đứng yên từ 2026-02-22 (`cbf624ae9b`) |
| V11 | Gộp trọng số kiểu LLM | DiLoCo/FedAvg khả thi về lý thuyết nhưng với 2 máy không lợi GPU-giờ, thêm biến phương pháp |
| V12 | Nhiều tài khoản Kaggle | Điều khoản Kaggle: 1 người 1 tài khoản; có trường hợp bị khoá |

## 2. Quyết định đã chốt trong phiên (người dùng)

| Quyết định | Ghi chú |
|---|---|
| **Giữ không subsampling** (`T' = T`) | Subsampling rút ngắn chuỗi → làm yếu RQ2 (ưu thế O(n) của Mamba trên chuỗi dài). Không đưa lại vào danh sách tăng tốc |
| **Chia dữ liệu train thành 3 Kaggle Dataset** | Để mỗi notebook tạo dataset vừa 20 GB; mỗi shard phải đại diện đủ thành phần |
| **Dùng 2 tài khoản Kaggle, thuộc 2 người khác nhau** | Người dùng xác nhận → không vướng điều khoản "1 người 1 tài khoản" (D14 đóng) |
| **Chạy train ở chế độ nền (batch)**, không chia phiên 2 h | Kernel chạy khi tắt máy (đã chứng minh với `verify-mamba-asr`); giới hạn 9 h/phiên GPU |

## 3. Đề xuất tối ưu cho các quyết định còn bỏ dở (chờ duyệt)

Tiêu chí: (1) không lệch phép so sánh hai encoder, (2) không đổi thiết kế đã
đăng ký nếu chưa có GVHD, (3) ít GPU-giờ, (4) ít sửa code dùng chung.

| # | Quyết định | Đề xuất | Lý do | Ai duyệt |
|---|---|---|---|---|
| D1 | Chia việc cho 2 tài khoản | **Mỗi tài khoản train 1 mô hình** (A: Conformer, B: Mamba), không gộp trọng số | Cùng tổng GPU-giờ như DiLoCo nhưng không thêm biến phương pháp, code gần như không đổi | Người dùng |
| D2 | Định dạng lưu | **WAV**, không FLAC | Mỗi shard ~8,5 GB đã vừa 20 GB; không phải đổi đuôi file trong code; đọc nhanh hơn (0,56 vs 4,55 ms/file) | Người dùng |
| D3 | Cách chia shard | **Vòng tròn trong từng video**, seed 42 (đã mô phỏng: lệch ≤ 0,1 điểm % theo độ dài) | Cân bằng tốt nhất; chuỗi liên tiếp vẫn dựng được từ manifest đầy đủ khi gắn đủ 3 shard | Người dùng |
| D4 | Dùng shard khi train | **Gắn cả 3 shard cùng lúc, xáo trộn toàn cục** | `/kaggle/input` không tính 20 GB → shard chỉ là cách lưu, train giữ nguyên | Người dùng |
| D5 | Val / test | Val = validation **trừ 250 câu clean-test** (6.499 câu, dataset thứ 4 ~2,8 GB); test = clean-test sau hiệu đính | Tách tập chọn mô hình khỏi tập báo cáo (V8) | Người dùng |
| D6 | Tạo dataset | Notebook **CPU** (không tốn GPU), có `HF_TOKEN` qua Kaggle Secrets, đóng gói tar nếu xác nhận giới hạn 500 file; dataset **private**; chia sẻ sang tài khoản kia (nếu Kaggle cho phép, không thì mỗi bên tự tạo) | Không dùng mạng nhà, không tốn quota GPU | Người dùng |
| D7 | Phiên train | Batch ~9 h + cờ **`--max_minutes ~510`** tự lưu và thoát; **checkpoint theo step (~20 phút) + resume giữa epoch** (thứ tự dữ liệu cố định theo seed/epoch); phiên sau gắn output phiên trước làm input | Không mất tiến độ nếu Kaggle cắt ở 9 h (chưa rõ output có được giữ khi bị cắt) | Người dùng |
| D8 | Tăng tốc | **`num_workers` + `pin_memory`** (không ảnh hưởng kết quả) → **AMP fp16** nếu benchmark cho thấy Mamba ổn định → **2 GPU (DDP)** chỉ khi vẫn thiếu quota | Theo thứ tự rủi ro tăng dần; không subsampling | Người dùng |
| D9 | Đo trước khi sửa | 1 kernel **GPU benchmark** (~30-40 phút quota: hiện tại / `num_workers` / AMP / 2 GPU × 2 encoder) + 1 kernel **CPU** (`df -h`, tốc độ tải HF có/không token, tar > 500 file, chia sẻ dataset) | Mọi con số tốc độ hiện chỉ là ước | Người dùng |
| D10 | Tái lập | **Pin revision HF `cbf624ae9b`**; `AUDIO_CACHE_DIR` đọc từ config/env (nhiều gốc); hai tài khoản cùng commit code, cùng seed; **đo RTF cả hai mô hình trên cùng 1 máy** sau khi train | Loại biến do hạ tầng | Người dùng |
| D11 | Số epoch (Q3) | Giữ 30 epoch; chỉ bàn giảm nếu sau D8 vẫn vượt quota | Giảm epoch cần GVHD | Người dùng → GVHD |
| D12 | RQ2 (Q1, 🔴) | Đề xuất trình GVHD: **phương án 2 — ghép đoạn liên tiếp chỉ để đo RTF** (không cần nhãn → dùng được 17-21 nghìn chuỗi, dải tới 30-60 s+); WER trên audio dài (nếu cần) chỉ dùng 57 chuỗi thuần dev | Không đổi dữ liệu train, giữ được câu hỏi O(n) vs O(n²) | **GVHD** |
| D13 | Val phụ thuộc video train (Q10) | Đề xuất trình GVHD: **giữ nguyên split của tác giả, nêu rõ là hạn chế** | So sánh hai mô hình vẫn công bằng; rút video khỏi train làm lệch đề cương + thêm việc | **GVHD** |
| D14 | Tài khoản Kaggle | ✅ **Đã xác nhận 2026-09-26:** 2 người khác nhau. Mỗi người tự chạy kernel bằng tài khoản/khoá API của mình — **không chia sẻ khoá `kaggle.json`** (dùng tài khoản của người khác cũng trái điều khoản) | — | — |

## 4. Thứ tự thực hiện đề xuất (sau khi duyệt)

1. Chạy kernel CPU + kernel GPU benchmark (D9).
2. Sửa code dùng chung (D5, D7, D8, D10): cho cả hai encoder, test local
   Conformer + Mamba trên Kaggle.
3. Script tạo manifest shard + val có kiểm tra: 3 shard rời nhau, hợp lại
   đúng 60.656 file, không chứa file dev/clean-test; commit manifest.
4. Notebook CPU tạo 4 dataset (3 train + 1 val).
5. Train song song: tài khoản A Conformer, tài khoản B Mamba, ~5 phiên mỗi
   bên (ít hơn nếu AMP hiệu quả).

Việc song song không phụ thuộc: hiệu đính 250 câu clean-test; mang D12, D13,
Q9 đi hỏi GVHD.
