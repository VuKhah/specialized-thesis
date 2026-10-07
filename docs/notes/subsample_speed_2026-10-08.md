# Tốc độ train: subsampling và độ chính xác khối Mamba (2026-10-08)

Người dùng yêu cầu ghi lại để hỏi GVHD. Kernel `tieunhi/subsample-speed-asr`
(`scripts/kaggle/subsample_speed/subsample_speed.py`), quota tài khoản A tổng
~0,6 GPU-h (v1 ~1′ lỗi cài gói, v2 24′, v3 12′). Số liệu gốc:
`reports/results/subsample_speed/v2/`, `.../v3/` (`subsample_speed_result.json`
+ log kernel).

## 1. Câu hỏi

Train full (30 epoch, 2×T4, DDP, AMP) cho thấy Mamba B1 **chậm hơn Conformer
~2,6× mỗi step** (0,683 vs 0,26 s/step) và ConExtBiMamba chậm hơn ~1,55×
(0,40 s/step). Hai bài có đo tốc độ **train** của Mamba cho ASR lại báo ngược lại:

| Nguồn | Điều kiện | Kết quả |
|---|---|---|
| Zhang và cs., arXiv:2405.12609, Bảng XIII | LibriSpeech-100, V100, Mamba thay MHSA trong Conformer (~120M), có subsampling | phút/epoch: Conformer 21,4 · ConInnBiMamba 18,3 · ConExtBiMamba 19,8 |
| Zevallos và cs., Interspeech 2025, mục 5.3 | 9 ngôn ngữ ít tài nguyên, 2× GPU Hopper, token 40 ms, có cả audio ghép 10-60 s | ConMamba ít hơn Conformer **40-45%** giờ train (110 epoch) |
| Jiang và cs., Speech Slytherin, arXiv:2407.09732 | L40, chỉ đo **suy luận** | ConMamba nhanh hơn Conformer chỉ khi audio > 60 s; ASR (độ phân giải thấp) "ít hoặc không có lợi thế" |
| Miyazaki và cs., arXiv:2406.16808 | A100, chỉ đo **suy luận** | RTF: Mamba hai chiều 0,152 · Conformer 0,193 · Transformer 0,118 |

(Bảng V/VII/IX của 2405.12609 có s/step nhưng là bài **tăng cường tiếng nói**,
mô hình 3-13M — không phải ASR.)

Dự án khác các bài trên ở hai điểm đo được: **không subsampling** (khung 10 ms,
chuỗi dài gấp 4 so với token 40 ms) và **khối Mamba ép fp32** (`mamba_fp32`, vì
T4 không có bf16 và fp16 làm tràn số — `mamba_precision_survey.md`). Phép đo
này tách hai yếu tố đó.

## 2. Điều kiện đo

- **Độc lập với code train:** kernel clone repo ở commit `bb8d0eb` (commit của
  train full), không sửa file nào trong repo; subsampling và bản fp16 chỉ gắn
  vào model trong tiến trình con của kernel.
- 1× Tesla T4, batch 8 (bằng số câu mỗi GPU lúc train DDP 2×8), AMP fp16 như
  train, AdamW + clip grad + CTC fp32 (cùng hàm `build_model`/`build_optimizer`/
  `ctc_loss` của train).
- Audio giả: nhiễu Gauss, độ dài 10-15 s chia cho speed perturb {0,9; 1; 1,1},
  xếp theo độ dài rồi cắt batch (≈ bucketing của train full), nạp sẵn trên GPU.
  Trọng số khởi tạo ngẫu nhiên (seed 0).
- **Ba biến thể** cho mỗi encoder (yaml y hệt train full):
  - **a**: nguyên bản (T' = T, khối Mamba fp32);
  - **b**: Conv2d subsampling 4× trước encoder (2 conv 3×3 stride 2, 256 kênh,
    kiểu Conv2dSubsampling của ESPnet), thay `input_proj`; T' ≈ T/4 (khung
    40 ms); mỗi encoder thêm ~1,8M tham số (12,2M → 14,0-14,1M);
  - **c**: T' = T, khối Mamba chạy dưới autocast fp16 như phần còn lại
    (Conformer không có khối Mamba → không có biến thể c).
- **v3 (số dùng):** chạy 2 lượt qua cùng các batch, chỉ đo lượt 2 (45 step
  train, 20 batch eval). Lý do ở mục 4.

## 3. Kết quả (v3, trung vị s/step)

| Encoder | a: như train full | b: subsampling 4× | c: khối Mamba fp16 | Log train full (2×T4 DDP) |
|---|---|---|---|---|
| Conformer-12M | **0,195** | **0,075** | — | 0,26 |
| Mamba B1 | 0,663 (3,4× Conformer) | 0,165 (**2,2×**) | 0,445 | 0,683 |
| ConExtBiMamba | 0,385 (2,0× Conformer) | 0,110 (**1,5×**) | 0,282 | 0,40 |

| Tăng tốc | Conformer | Mamba B1 | ConExtBiMamba |
|---|---|---|---|
| nhờ subsampling 4× (a → b) | 2,6× | **4,0×** | 3,5× |
| nhờ khối Mamba fp16 (a → c) | — | 1,5× | 1,4× |

| VRAM đỉnh khi train (GiB) | a | b | c |
|---|---|---|---|
| Conformer | 3,14 | 1,48 | — |
| Mamba B1 | 3,89 | 1,63 | 2,61 |
| ConExtBiMamba | 3,46 | 1,56 | 2,97 |

Suy luận (forward, batch 8, ~13 s/câu), giây/batch: a: Conformer 0,059 · B1
0,176 · ConExt 0,104; b: 0,026 · 0,056 · 0,038. Mọi cấu hình loss hữu hạn, AMP
bỏ step ≤ 1.

**Độ tin cậy:** biến thể a của cả ba khớp log train full (Mamba 0,663 vs 0,683;
ConExt 0,385 vs 0,40). Conformer đo nhanh hơn log (0,195 vs 0,26) — phần chênh
khả dĩ là overhead DDP/đọc dữ liệu thật mà phép đo 1 GPU, dữ liệu nạp sẵn không
có (chưa tách được). v2 chạy mỗi cấu hình 2 lần (lần 2 ngược thứ tự): lệch < 3%.

## 4. Sự cố đo ở v2 — cạm bẫy cho RQ2

v2 đo ngay lượt đầu: Conformer và ConExt tốn **~1,1 s/step cố định**, không đổi
theo độ dài chuỗi (a ≈ b, Conformer 1,38 vs 1,32 s/step), kể cả khi chỉ forward
(eval ~0,35 s/batch); Mamba B1 không bị và khớp log. Lượt 2 trên cùng batch
(v3) về đúng log train. Giải thích khả dĩ (**giả thuyết, chưa kiểm bằng
profiler**): cuDNN dựng kế hoạch tính conv cho mỗi shape mới — conv module của
Conformer/ConExt (depthwise Conv1d k=31) và Conv2d subsampling đi qua cuDNN,
còn khối Mamba dùng kernel `causal_conv1d` riêng. Lúc train 2.841 step/epoch,
shape lặp lại nên đã có trong cache.

→ **Khi đo RTF/độ trễ RQ2 trên audio có độ dài thay đổi, phải làm ấm ở đúng
từng độ dài trước khi đo**, nếu không Conformer/ConExt bị đo chậm gấp nhiều
lần (pilot RQ2 2026-10-05 đã làm ấm trước khi đo — giữ nguyên cách đó).

## 5. Nhận xét

1. **Không subsampling là nguyên nhân chính của khoảng cách tốc độ, fp32 là
   nguyên nhân phụ:** subsampling tăng tốc Mamba B1 4,0× (Conformer chỉ 2,6×),
   fp16 tăng tốc 1,5×.
2. **Kể cả có subsampling, Mamba B1 vẫn chậm hơn Conformer ~2,2×/step, ConExt
   ~1,5×** trong điều kiện T4 + 12M tham số + câu ~13 s. Không tái hiện được
   "Mamba train nhanh hơn" của 2405.12609 / Zevallos 2025; phù hợp nhận xét của
   Speech Slytherin rằng Mamba ít lợi thế với ASR câu ngắn. Các bài kia còn khác
   GPU (V100 / Hopper), bf16, kích thước mô hình và (Zevallos) audio dài tới 60 s
   — **chưa đo** các yếu tố này, nên không kết luận yếu tố nào quyết định.
3. Thống nhất với pilot RQ2 (`rq2_pilot_2026-10-05.md`): ở 10-15 s Conformer
   nhanh nhất; ưu thế của Mamba chỉ xuất hiện ở audio dài (điểm giao 60 s ConExt /
   120 s B1 khi không subsampling).

## 6. Hạn chế của phép đo

- 1 GPU, không DDP, dữ liệu nạp sẵn → chỉ so **tỉ lệ** giữa cấu hình; số giây
  tuyệt đối không thay được log train.
- Trọng số ngẫu nhiên: biến thể c không nói được fp16 có tràn số với trọng số
  đã train hay không (lỗi 2026-10-05 xuất hiện sau vài epoch train).
- Chưa đo tổ hợp b + c (subsampling + Mamba fp16).
- Subsampling thêm ~1,8M tham số (như nhau cho cả ba); không đo WER — muốn biết
  WER khi có subsampling phải train lại.

## 7. Câu hỏi cho GVHD

1. Có đưa phép đo này vào khóa luận (Thảo luận / Hạn chế, giải thích đánh đổi
   tốc độ) không?
2. Có cần train thêm **một nhóm có subsampling 4×** cho cả ba mô hình để có WER
   + tốc độ ở cấu hình gần tài liệu không? (Mở lại quyết định 2026-09-26 "không
   subsampling", vốn chốt vì RQ2 — nhóm chính giữ nguyên, đây là nhóm bổ sung.)
3. Conformer-12M đang lệch cấu hình chuẩn (Gulati 2020) ở 3 điểm: không
   subsampling, không mã hóa vị trí, LR hằng thay Noam/warmup. Có cần một bản
   Conformer gần chuẩn để kiểm tính công bằng của so sánh không?
4. Nhóm pre-train: chỉ zero-shot hay cần fine-tune (Q-d, `gvhd_buoi_1.md`)?

## 8. Đề xuất dùng quota nếu còn ~90 GPU-h (AI đề xuất, CHƯA CHỐT)

Ước thời gian dựa trên log train full (Conformer 6,6 h · ConExt 10,0 h · Mamba
16,8 h cho 30 epoch, cộng ~6′ setup/phiên). Số cho nhóm subsampling là **ước
lượng** = s/step biến thể b × (log train / biến thể a), chưa tính khả năng khâu
đọc dữ liệu thành nút cổ chai khi step nhanh hơn ~3× — vì vậy ghi khoảng.

| Mức | Việc | GPU-h ước | Trả lời được gì | Điều kiện |
|---|---|---|---|---|
| 0 — bắt buộc | Đo RQ2 cuối (RTF/độ trễ/VRAM 5 mô hình theo độ dài), eval clean-test 3 mô hình, trung bình checkpoint, Parakeet fp32 đủ 203 câu, dự phòng phiên lỗi | 6-10 | Số liệu chính của khóa luận | Không cần train |
| 1 — đề xuất | **Nhóm subsampling 4×** cho cả ba, 30 epoch, mọi thứ khác giữ nguyên; trước đó 1 kernel thử ngắn đo s/step thật | 12-20 (+1 thử) | WER + tốc độ ở cấu hình gần tài liệu; Mamba còn hơn WER khi chuỗi ngắn lại không; giảm lệch cấu hình của Conformer | GVHD đồng ý mở lại chốt 2026-09-26 (câu 2) |
| 2 — có điều kiện | Pha decay LR ~5 epoch cho nhóm chính (+ nhóm subsampling nếu có) | 6-9 | LR hằng "để lại" bao nhiêu WER | Chỉ khi trung bình checkpoint (mức 0) cho thấy lợi lớn — thứ tự đề xuất 2026-10-07 |
| 3 — chọn một | (A) seed thứ hai cho nhóm chính **hoặc** (B) fine-tune mô hình pre-train | (A) ~34 · (B) chưa ước được — cần kernel thử ~2 h | (A) độ vững của thứ hạng WER; (B) yêu cầu "pre-train + fine-tune" của GVHD | (B) chỉ khi GVHD yêu cầu (câu 4); (A) chỉ khi bootstrap trên clean-test (`src/evaluation/bootstrap.py`, không tốn GPU) chưa đủ |

Tổng mức 0-3: ~60-75 GPU-h, còn ~15-30 h dự phòng (phiên lỗi khởi động, chạy
lại). Không đề xuất: chạy nốt Conformer dài tới 76 epoch (ít thông tin mới);
60 epoch cho cả ba (~33 h, pha decay rẻ hơn — chưa có bằng chứng nào tốt hơn);
train Mamba fp16 (tràn số đã gặp 2026-10-05); thêm mã hóa vị trí cho Conformer
(câu 3) — đúng hướng chuẩn hóa nhưng **chưa tra mức ảnh hưởng**, cần GVHD quyết
trước khi tra/làm.
