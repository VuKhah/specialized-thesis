# Kế hoạch dữ liệu + train trên Kaggle (tổng hợp phiên 2026-09-26)

Tổng hợp thảo luận phiên 2026-09-26: vấn đề đã nêu, quyết định người dùng
đã chốt, và **đề xuất của AI** cho các quyết định còn bỏ dở. Phần đề xuất
**chưa được duyệt** — chưa sửa code, chưa tạo dataset. Số liệu gốc và nguồn
web ở [`prefetch_storage_review.md`](prefetch_storage_review.md).

> **2026-09-28: người dùng đã duyệt D1-D11** (D3 đổi sang **4 shard**) và
> chọn cách train "full, đánh giá theo epoch" — xem [mục 5](#5-duyệt-2026-09-28).
> Các mục 0-4 bên dưới giữ nguyên như bản đề xuất gốc; chỗ nào lệch thì
> mục 5 thắng.

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

## 5. Duyệt (2026-09-28)

| # | Kết quả | Ghi chú |
|---|---|---|
| D1 | ✅ Đồng ý | A: Conformer, B: Mamba |
| D2 | ✅ Đồng ý | WAV |
| D3 | ✅ **Đổi: 4 shard** (không phải 3), vẫn vòng tròn theo video, seed 42 | Mỗi shard ~15.164 câu, ~6,4 GB. Lý do: nếu phải đóng gói tar (giới hạn 500 file, chờ D9 kiểm), đĩa đỉnh lúc tạo ~12,7 GB thay vì ~17 GB / 20 GB. Số shard không ảnh hưởng train (D4). Cần chạy lại mô phỏng phân bố cho 4 shard khi viết script. Tổng dataset: 4 train + 1 val = 5 |
| D4 | ✅ Đồng ý (điều kiện: không giảm minh bạch) | Chỉ là cách lưu, không đổi dữ liệu/cách train |
| D5 | ✅ Đồng ý (điều kiện: không giảm minh bạch) | **Phải ghi rõ trong Chương 3**: val = `validation` trừ 250 câu clean-test (6.499 câu) và lý do (tách tập chọn mô hình khỏi tập báo cáo) |
| D6-D11 | ✅ Đồng ý | — |
| D12, D13 | Chờ GVHD | Không đổi |

### Cách train: full dữ liệu, đánh giá theo epoch (người dùng chọn)

Người dùng hỏi có nên chỉ train tập con vài GB, hoặc tách shard rồi train
lần lượt, để có kết quả sớm và bớt lo deadline. Đã so sánh:

- **Tập con thay cho full**: lệch đề cương (cần GVHD); ~12M tham số train từ
  đầu, CTC không LM với vài chục giờ dữ liệu dễ cho WER rất cao (ước lượng,
  chưa đo) → chênh lệch hai mô hình dễ là nhiễu; ít dữ liệu có thể nghiêng về
  kiến trúc có định kiến quy nạp mạnh hơn (conv của Conformer) → đổi câu hỏi
  nghiên cứu.
- **Train shard lần lượt, thay thế** (shard 1 → chỉ shard 2 → …): quên dần,
  lệch theo thứ tự shard → loại.
- **Cộng dồn shard** (1 → 1+2 → …): không quên, có đường cong theo lượng dữ
  liệu, nhưng quy trình không chuẩn, lẫn với thời gian train, tốn GPU-giờ hơn
  → không làm (có thể là phần mở rộng nếu GVHD muốn).
- **Chọn: train full, mỗi epoch thấy toàn bộ dữ liệu, lấy kết quả theo
  epoch.** Checkpoint epoch *k* là kết quả hợp lệ "full dữ liệu, *k* epoch" →
  dừng lúc nào cũng có kết quả, kết hợp D7 (`--max_minutes`, checkpoint theo
  step). **So sánh hai mô hình luôn ở cùng số epoch.** Bỏ đề xuất pilot 1
  shard — epoch đầu của lần train full đã bắt lỗi `main()`/checkpoint.

**Ràng buộc cần giữ:** lịch LR hiện tại là warmup rồi **hằng số**
(`src/training/train.py`, `warmup_lr_lambda`), nên dừng ở epoch bất kỳ không
bị "dở dang lịch LR". Nếu sau này thêm cosine/linear decay theo tổng epoch thì
tính chất "dừng lúc nào cũng được" mất — phải cân nhắc lại.

### Thứ tự thực hiện (thay mục 4 ở các chỗ khác)

1. Kernel CPU + kernel GPU benchmark (D9).
2. Sửa code dùng chung (D5, D7, D8, D10) cho cả hai encoder.
3. Script manifest **4 shard** + val, có kiểm tra (4 shard rời nhau, hợp lại
   đúng 60.656 file, không chứa file dev/clean-test); commit manifest.
4. Notebook CPU tạo **5 dataset** (4 train + 1 val).
5. Train full song song hai tài khoản, đánh giá theo epoch.

### Kết quả kernel CPU `check-env-asr` (D9 bước 1, 2026-09-28)

Script: `scripts/kaggle/check_env/check_env.py` (4 lõi CPU, không GPU).

| Đo | Kết quả | Hệ quả |
|---|---|---|
| Đĩa `/kaggle/working` | 21,0 GB (trống 20,9) | Khớp giới hạn 20 GB đã biết |
| Đĩa `/tmp` (= `/`) | Trống **~1.200 GB** (đĩa chung của máy chủ, không phải hạn mức riêng) | Ghi tạm WAV + tar khi tạo shard **không** bị giới hạn 20 GB → lý do chọn 4 shard vì đĩa đỉnh không còn quyết định; 4 shard vẫn giữ (đã chốt, mỗi shard ~6,4 GB vừa thoải mái output 20 GB) |
| `/dev/shm` | 15 GB | — |
| Tải HF ẩn danh, 8 luồng, 150 file | 19 s, 3,3 MB/s, **0 lỗi, 0 lỗi 429** | Ngoại suy 67.405 file ≈ **2,4 h** (~36 phút/shard). Tốc độ ~7,9 file/s, sát giới hạn 3.000 lượt/5 phút (10/s) → tăng luồng không giúp nếu không có token |
| Tải HF có token | Chưa đo — secret `HF_TOKEN` chưa gắn vào kernel | Không bắt buộc: ẩn danh đã đủ nhanh cho notebook CPU |
| Output kernel > 500 file | Giữ đủ **600/600** file | Chưa chứng minh giới hạn file khi *tạo Dataset* từ output (15k file/shard) — kiểm ở bước tạo dataset thật; tar vẫn là phương án dự phòng |
| Tốc độ tar (không nén) | 590 MB/s | Đóng tar 1 shard ~6,4 GB mất ~11 s → không đáng kể |

Ghi chú: tải output kernel về máy local đi từng file, 600 file nhỏ mất ~10
phút — không ảnh hưởng train (dataset gắn qua `/kaggle/input`), nhưng đừng
tải output shard về local.

### Kết quả kernel GPU `benchmark-asr` (D9 bước 2, 2026-09-30)

Script: `scripts/kaggle/benchmark/benchmark.py`, 2x T4, commit code `ed043f5`,
batch toàn cục 16 (DDP: 8/GPU), 3 step khởi động + 20 step đo, audio thật
(368 file đầu split `train`). Kernel chạy 917 s (~15 phút quota; lần đẩy đầu
lỗi sau ~1 phút vì wheel gắn ở `/kaggle/input/notebooks/<user>/<kernel>/…`).
Số giờ = s/step × 3.791 step × 30 epoch, **chưa tính eval và đọc dữ liệu từ
`/kaggle/input`**.

| Encoder | workers | AMP | GPU | s/step | 30 epoch (h) | VRAM đỉnh (GiB) | loss đầu → cuối | AMP scale |
|---|---|---|---|---|---|---|---|---|
| Conformer | 0 | – | 1 | 1,345 | 42,5 | 7,76 | 133,4 → 6,45 | – |
| Conformer | 4 | – | 1 | 1,387 | 43,8 | 7,76 | 133,4 → 6,45 | – |
| Conformer | 4 | ✓ | 1 | **0,561** | **17,7** | 4,89 | 133,4 → 6,68 | 1024 |
| Conformer | 4 | ✓ | 2 | 0,774 | 24,4 | 2,62 | 127,8 → 6,64 | 1024 |
| Mamba | 0 | – | 1 | 1,540 | 48,7 | 6,04 | 121,0 → 8,49 | – |
| Mamba | 4 | – | 1 | 1,573 | 49,7 | 6,04 | 121,0 → 8,49 | – |
| Mamba | 4 | ✓ | 1 | 1,016 | 32,1 | 3,58 | 121,0 → 10,12 | 512 |
| Mamba | 4 | ✓ | 2 | **0,476** | **15,0** | 1,95 | 118,0 → 10,32 | 512 |

Đọc kết quả:
- **`num_workers` không giúp** khi audio nằm đĩa local (GPU là nút cổ chai);
  với `/kaggle/input` chưa đo — giữ vài worker không hại.
- **AMP**: Conformer ×2,4; Mamba ×1,5. Loss hữu hạn ở mọi cấu hình. Mamba
  AMP có 1 lần tràn (scale 1024 → 512, bình thường với GradScaler) và loss
  cuối cao hơn fp32 (10,1 vs 8,5) — 23 step quá ít để kết luận; cần theo dõi
  loss/`amp_scale` ở epoch đầu train thật.
- **DDP ngược chiều giữa hai encoder**: Mamba nhanh ×2,1 (trên tuyến tính —
  nghi nhiễu phép đo 20 step), Conformer **chậm đi** (0,56 → 0,77). Chưa rõ
  nguyên nhân; nghi: 2 × 4 worker trên máy ít lõi CPU, overhead all-reduce
  qua PCIe, hoặc kernel attention kém hiệu quả ở batch 8.
- **Conformer có BatchNorm** (module conv của `torchaudio.models.Conformer`),
  Mamba chỉ LayerNorm → DDP không SyncBN làm thống kê BN tính trên 8 câu/GPU
  thay vì 16 — khác biệt thật về huấn luyện giữa 1 GPU và 2 GPU (với Mamba thì
  DDP tương đương toán học với 1 GPU).

### Benchmark lần 2 — D8 phương án (c) (2026-09-30)

Người dùng chọn (c): đo thêm trước khi chọn. Kernel `benchmark-asr` v4, 754 s
(v3 hỏng vì quên nhúng yaml có `weight_decay`, mất ~9 phút quota → thêm dừng
sớm khi cấu hình đầu lỗi). Code = commit `ed043f5` + **nhúng 6 file local chưa
push** (`OVERLAY`: 2 yaml, `train.py`, `mamba_encoder.py`,
`vietsuperspeech_dataset.py`, `make_shards.py`) — tức đã gồm các sửa theo
khuyến nghị tác giả Mamba. AMP ở mọi cấu hình, 3 + **40** step, optimizer =
`build_optimizer` thật. Máy GPU: **4 lõi CPU**, `/tmp` trống **1,1 TB**.

| Encoder | GPU | workers/tiến trình | SyncBN | s/step | 30 epoch (h) | VRAM (GiB) | loss đầu → cuối | AMP scale |
|---|---|---|---|---|---|---|---|---|
| Conformer | 1 | 4 | – | **0,563** | **17,8** | 4,89 | 133,4 → 5,98 | 1024 |
| Conformer | 2 | 2 | – | 0,695 | 22,0 | 2,62 | 127,8 → 5,92 | 1024 |
| Conformer | 2 | 2 | ✓ | 0,735 | 23,2 | 2,62 | 128,0 → 5,92 | 1024 |
| Mamba | 1 | 4 | – | 1,113 | 35,2 | 3,61 | 134,4 → 6,62 | 1024 |
| Mamba | 2 | 2 | – | **0,505** | **16,0** | 1,97 | 130,6 → 6,16 | 1024 |

- **Lặp lại được lần 1**: Conformer 1 GPU 0,563 (lần 1: 0,561); Mamba 2 GPU
  nhanh ×2,2 so với 1 GPU (lần 1: ×2,1) → hiện tượng thật, không phải nhiễu
  (nguyên nhân chưa rõ).
- **Conformer DDP chậm vì chính DDP**, không chỉ vì worker: giảm 4 → 2
  worker/tiến trình còn 0,695 (lần 1: 0,774), vẫn chậm hơn 1 GPU; SyncBN thêm
  ~6%.
- **Khối Mamba thật sau khi sửa** (norm_f, √n_layers, residual fp32,
  miễn weight decay): chạy được cả 1 và 2 GPU; tham số **12.292.864** (khớp dự
  kiến, +0,73%); đúng **56** tham số miễn weight decay (`A_log` + `D` × 28);
  AMP scale giữ 1024 (lần 1 tụt 512); loss cuối 6,6 (lần 1 trước khi sửa:
  10,1). 43 step chỉ là dấu hiệu, chưa phải bằng chứng. Mamba 1 GPU chậm hơn
  lần 1 ~10% (1,113 vs 1,016) — giá của residual fp32 + norm_f.

## 6. Sự cố train thử 2026-10-04 và quy trình chạy kernel GPU

**Diễn biến.** Lần 1 (đẩy 12:44 UTC): CLI báo RUNNING ~3 h nhưng script chỉ bắt
đầu 15:45 (chờ cấp máy, không tính quota) rồi lỗi sau 16 s — image GPU mặc định đã
lên **Python 3.13**, pip từ chối wheel mamba cp312. Lần 2 (17:11): ghim
`docker_image` của `verify-mamba-asr` (Python 3.12) → máy được cấp, quota tạm tính
~50 phút, **log 0 dòng** (kể cả `nvidia-smi`), người dùng Cancel 18:02; quota quay
về 1,13 h (không bị trừ). Mất ~5 h thời gian thực, gần như không mất quota.

**Nguyên nhân.**
1. *Gốc:* môi trường Kaggle tự đổi. Mọi kernel mới (và version mới không ghim) nhận
   image mặc định mới nhất. `verify-mamba-asr` build wheel 24/09 trên Python 3.12;
   `lid-asr` 03/10 đã chạy Python 3.13. Kernel train thử viết sau đó không ghim
   môi trường và không kiểm phiên bản trước khi cài wheel.
2. *Lần 2:* image cũ ghim được trên máy CPU (kernel thử chạy, cài được wheel) nhưng
   trên máy GPU không khởi động được script. **Chưa rõ vì sao** (giả thuyết: tải image,
   hoặc image cũ không hợp driver 580/CUDA 13 của máy GPU hiện tại) — không kiểm được
   vì log trống. → Không dựa vào ghim image cũ nữa.
3. *Không thấy được tình trạng:* trạng thái RUNNING của CLI không phân biệt chờ máy /
   máy đã cấp nhưng script chưa chạy / đang chạy; trước đó không có log tiến độ.

**Quy trình từ nay (mọi kernel GPU):**
1. **Kernel kiểm tra ngắn trước** mỗi khi đổi môi trường (kernel mới, image mới,
   wheel mới): `-t 1200` (20 phút trần), chỉ cài + `check_env` + CHECK_CODE + vài step.
   Qua rồi mới chạy dài.
2. **`check_env` là dòng đầu tiên**: in python/torch/CUDA/số GPU, lệch `EXPECT_*` thì
   dừng ngay.
3. **Theo dõi bằng `scripts/kaggle/watch_kernel.py`** (local): status + log + quota mỗi
   phút; quota tăng mà 15 phút không có log mới → báo động để người dùng Stop
   (kaggle.com/code → View Active Events; CLI không có lệnh dừng).
4. Trong kernel: log tiến độ mỗi 50 step + eval, watchdog 20 phút im lặng (đã có).
5. Mọi kernel GPU của dự án dùng **cùng một môi trường đã kiểm**, ghi trong `EXPECT_*`
   — điều kiện để số đo tốc độ giữa các mô hình so được.

**Kết quả phương án A (kernel `env-check-asr`, 2026-10-04 18:20-18:45 UTC, ~25 phút GPU):**
image GPU mặc định = Python 3.13.15, torch 2.11.0+cu128, torchaudio 2.11.0+cu128, nvcc 12.8,
driver 580 / CUDA 13.0, 2 × T4. Build chỉ sm_75: `causal-conv1d` 1.5.4 **3,1 phút**,
`mamba-ssm` 2.3.1 **6,1 phút** (v2.3.1 biên dịch được với torch 2.11). CHECK_CODE OK (B1 +
ConExt, fp32 + AMP). DDP 2 GPU + AMP, `train_shard0`, ~3 phút mỗi mô hình, đều thoát đúng
hạn: B1 0,465 s/step (VRAM đỉnh 2,1 GiB), ConExt 0,416 (2,4), Conformer 0,369 (2,6) — số đo
trong vài trăm step đầu, **nhanh hơn** benchmark 30/09 (Conformer 0,77, Mamba 0,48); chưa rõ
vì torch mới hay vì batch ngẫu nhiên ngắn hơn — train thử đo lại theo epoch. Wheel cp313 nằm
trong output `env-check-asr`. Tài khoản B cần chúng qua Dataset (thêm version mới cho
`mamba-wheels`).

**Chuẩn bị train thử lần 3 (2026-10-05) — khoá những phần chưa từng chạy trên GPU:**
- Wheel cp313 trong dataset **`tieunhi/mamba-wheels-v2`** (người dùng tạo từ output
  `env-check-asr`, chia sẻ B; cả hai tài khoản liệt kê được, kích thước khớp bản build). Mọi
  kernel lấy wheel từ dataset này — một nguồn cho cả hai tài khoản.
- `train.py --limit_train N --limit_eval N` (chỉ để kiểm tra). Kernel train thử chạy
  **pipeline tí hon** trước khi train: mỗi mô hình 1 epoch 64 câu + eval 64 câu/tập qua đúng
  đường thật (train → eval DDP gom 2 GPU → `best.pt` → `metrics.jsonl` → `trial_report`),
  `check=True` → hỏng là dừng sau ~1 phút/mô hình. Test CPU local với manifest thật: đủ file,
  `trial_report` ra bảng (bắt được lỗi không tạo thư mục output — đã sửa).
- `PREFLIGHT_ONLY = True` → chỉ chạy phần kiểm tra (môi trường, wheel, CHECK_CODE, chép 27 GB,
  pipeline tí hon, resume) rồi dừng: kernel kiểm tra cho tài khoản B (`vuvanduc1/preflight-asr`,
  bản sao `train_trial.py` dựng ở scratchpad lúc đẩy; metadata riêng `id` + `dataset_sources`).
- Thứ tự đề xuất: preflight B (~15 phút GPU của B) → train thử A (~15 phút preflight + ~2,3 h train).

