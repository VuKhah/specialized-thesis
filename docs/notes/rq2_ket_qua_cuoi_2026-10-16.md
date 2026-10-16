# RQ2 — kết quả cuối: bảng, hình, kết luận (2026-10-16)

Dựng bằng `scripts/rq2_final_tables.py` từ số đã đo (`reports/results/rq2_final_2026-10-13/`, T4, batch 1, trọng số ngẫu
nhiên, làm ấm từng độ dài, chia đôi tìm OOM) và `metrics.jsonl` của các run; **không chạy GPU mới**. Output:
`reports/results/rq2_tables_2026-10-16/` (`table_3_6_train_cost.csv`, `table_3_7_inference.csv`, `crossover.csv`,
`numerics.md`), `reports/figures/rq2/fig_3_4_latency_vram.png`, `fig_3_5_wer_rtf.png`. Chạy lại khi 6 run seed về
(Bảng 3.6 thêm GPU-h; Hình 3.5 đổi WER 40 ms sang trung bình 3 seed — hiện là seed 42).

## Bảng 3.6 — chi phí huấn luyện (2×T4, 38 epoch tới ep37)

| Cấu hình | Encoder (M) | + subsampling (M) | Tổng (M) | s/step | GPU-h | VRAM train (GiB/GPU) | Step không hữu hạn |
|---|---|---|---|---|---|---|---|
| B1 40 ms | 12,33 | 1,84 | 14,43 | 0,187 | 6,00 | 1,31 | 0 |
| ConExt 40 ms | 12,29 | 1,84 | 14,38 | 0,150 | 4,86 | 1,22 | 0 |
| Conformer PE 40 ms | 12,78 | 1,84 | 14,87 | 0,164 | 5,28 | 1,30 | 0 |
| B1 10 ms | 12,29 | — | 12,54 | 0,688 | 21,33* | 3,61 | 0 |
| ConExt 10 ms | 12,24 | — | 12,50 | 0,399 | 12,75* | 3,18 | 0 |
| Conformer PE 10 ms | 12,73 | — | 12,99 | 0,585 | 19,20 | 5,14 | 0 |
| Conformer không PE 10 ms | 12,20 | — | 12,46 | 0,260 | 8,40* | 2,88 | 0 |

\* rẽ nhánh từ v1 ep29: GPU-h gồm 30 epoch v1 (commit khác); s/step, VRAM chỉ của pha giảm 8 epoch.

## Bảng 3.7 — chi phí suy luận theo độ dài (độ trễ trung vị, giây; RTF trong `table_3_7_inference.csv`)

| Cấu hình | 15 s | 60 s | 240 s | 3.600 s | VRAM 3.600 s (GiB) | OOM (chạy được / hỏng) |
|---|---|---|---|---|---|---|
| B1 40 ms | 0,029 | 0,029 | 0,107 | 1,65 | 8,7 | không |
| ConExt 40 ms | 0,020 | 0,022 | 0,075 | 1,18 | 8,7 | không |
| Conformer PE 40 ms | 0,022 | 0,035 | 0,420 | OOM | — | 638 / 656 s |
| B1 10 ms | 0,026 | 0,075 | 0,317 | 5,21 | 5,7 | không |
| ConExt 10 ms | 0,020 | 0,051 | 0,203 | 3,40 | 6,0 | không |
| Conformer PE 10 ms | 0,030 | 0,393 | OOM | OOM | — | 163 / 165 s |
| Conformer không PE 10 ms | 0,015 | 0,062 | 0,644 | 156,7 | 2,8 | không |
| Parakeet 0,6B fp32 (80 ms) | 0,095 | 0,414 | 2,63 | OOM | — | 578 / 585 s |

## Điểm giao (Mamba nhanh hơn Conformer từ độ dài nào) — cùng cấu hình là chính

| Cặp | Nhanh hơn từ (nội suy) | Nhanh hơn ≥ 1,5× từ |
|---|---|---|
| ConExt 40 ms vs Conformer PE 40 ms | mọi độ dài (chênh 5–10% ≈ 1–2 ms tới 45 s) | 60 s |
| B1 40 ms vs Conformer PE 40 ms | ~53 s | 90 s |
| ConExt 10 ms vs Conformer PE 10 ms | ~6 s | 20 s |
| B1 10 ms vs Conformer PE 10 ms | ~13 s | 20 s |
| ConExt 10 ms vs Conformer không PE 10 ms | ~45 s | 105 s |
| B1 10 ms vs Conformer không PE 10 ms | ~90 s | 180 s |

Cột "≥ 1,5×" là **mô tả**, đặt sau khi thấy số (16/10) vì ở câu ngắn độ trễ mọi mô hình bị chi phí cố định mỗi lần chạy
(~15–30 ms) chi phối — "nhanh hơn" ở đó chỉ vài ms. Các cặp khác cấu hình ở `crossover.csv`.

## Kết luận RQ2 — bản sau phản biện 16/10 (thay KL3 của `ket_qua_cuoi_khung_2026-10-15.md`)

Phản biện 16/10 (3 agent: thiết kế hình, phương pháp, kiểm VRAM). Đổi so với bản sáng: kết luận về câu ngắn là về
**cài đặt** chứ không phải kiến trúc; lợi thế audio dài chỉ là **chi phí**, chưa phải khả năng phiên âm; ngưỡng
"Mamba nhanh hơn" chỉ đúng **so với Conformer PE cài bias tường minh**; VRAM 40 ms đã giải thích được.

**Đoạn viết sẵn cho Ch3 (3.4):**
> Trên GPU T4 với batch 1, ở độ dài câu hội thoại (10–15 s) độ trễ của cả ba encoder ~12M đều trong khoảng 15–30 ms
> và gần như không đổi theo độ dài tới ~45 s: chi phí cố định của mỗi lần chạy chi phối vùng này nên phép đo không
> phân biệt được các kiến trúc. Khi suy luận một lần trên toàn chuỗi, chi phí của hai encoder Mamba tăng gần tuyến
> tính và chạy hết 3.600 s audio (1,2–5,2 s, ≤ 8,7 GiB), trong khi Conformer PE tương đối với bias vị trí tường minh
> hết bộ nhớ từ ~165 s (10 ms) và ~656 s (40 ms) — thuộc tính của cách cài đặt, vì bản dùng SDPA vẫn chạy hết nhưng
> chậm hơn khoảng hai bậc. Các ngưỡng mà Mamba nhanh hơn được đo so với Conformer PE tường minh, trong điều kiện số
> học không đồng nhất (khối Mamba fp32), và chưa có bản Conformer SDPA ở 40 ms làm mốc dưới. Nghiên cứu không đánh
> giá WER trên audio dài, nên kết quả chỉ nói về chi phí tính toán, không nói về khả năng phiên âm audio dài không
> chia đoạn.

**Các điểm cụ thể:**
1. **Câu ngắn:** ở 40 ms ConExt 20 ms ≈ Conformer PE 22 ms, B1 29 ms. Chênh lệch có thể do cài đặt (số lần gọi kernel,
   ép kiểu fp32 của khối Mamba, `cudnn_benchmark=false`) — **suy luận**; CUDA graph / compile / batch lớn có thể đổi
   thứ hạng.
2. **Ngưỡng so với Conformer PE tường minh** (cùng cấu hình): 40 ms — ConExt ≥ 1,5× từ 60 s, B1 từ 90 s; 10 ms — cả
   hai từ ~20 s. **So với attention SDPA, ngưỡng ở 40 ms có thể ~3–4 phút** (ước thô từ Conformer không PE 10 ms ở
   độ dài L ≈ 40 ms ở 4L, bỏ qua conv subsampling — **suy luận**; không có số đo).
3. **OOM của Conformer PE** là thuộc tính cài bias vị trí tường minh [B, H, T, 2T−1].
4. **VRAM 40 ms > 10 ms ở audio rất dài — đã giải thích (agent kiểm 16/10, đếm numel thật trên CPU ở T = 3,6k và 36k
   rồi ngoại suy tuyến tính):** Conv2d #1 của subsampling sinh [1, 256, 180k, 39] ≈ 1,8·10⁹ phần tử ≈ 3,35 GiB fp16;
   ReLU không inplace tạo tensor mới cùng cỡ → ~6,7 GiB cùng lúc; cộng front-end + trọng số ≈ 7,1 GiB; phần dư ~1,6 GiB
   (đoán là workspace cuDNN — chưa đo). Bằng chứng phụ: B1 và ConExt 40 ms có **cùng** đỉnh 8,713 GiB dù encoder khác.
   Một khối Mamba fp32 ở 10 ms (T = 360k) cần ~4–5,9 GiB tạm thời, khớp 5,65 GiB đo được (dưới `no_grad` đỉnh = một
   khối). Câu cho khóa luận: *"Ở cấu hình 40 ms, VRAM đỉnh khi suy luận audio rất dài do khối Conv2d subsampling quyết
   định chứ không phải encoder: lớp tích chập đầu tiên sinh bản đồ đặc trưng 256 kênh trên toàn trục thời gian mới
   giảm 2×, lớn hơn activation của một khối Mamba ở 10 ms; vì vậy rút ngắn chuỗi vào encoder 4× không giảm bộ nhớ
   đỉnh với audio rất dài."* Script kiểm: scratchpad phiên 16/10 (`sub_mem.py`).
5. **Chi phí huấn luyện:** 40 ms ba mô hình lệch ≤ 24% (4,86–6,00 GPU-h), Conformer PE rẻ hơn B1. Cột GPU-h 10 ms
   **không so trực tiếp được** giữa run rẽ nhánh v1 (gồm 30 epoch v1, commit khác) và Conformer PE 10 ms train từ đầu.

## Nhỏ từ đầu vs lớn có pre-train (Parakeet) — được phép viết gì

So cặp trên clean-test (bootstrap theo video, 16/10): Parakeet FT − Conformer PE 40 ms greedy −2,68 [−3,54; −1,86];
− B1 10 ms −4,89 [−6,13; −3,87]; Parakeet FT (greedy) − Conformer PE 40 ms + LM −0,81 [−1,56; −0,09], riêng từ ngoại
+7,86 [+2,97; +13,79] (mô hình nhỏ + LM tốt hơn; phép so lệch vì Parakeet chưa gắn LM). `best.pt` của Parakeet chọn
trên `val` (`finetune_parakeet.py`: tập đầu của EVAL_MANIFESTS).
- **Được viết:** "Trong điều kiện đo này, mô hình lớn có pre-train và fine-tune đạt WER thấp nhất (9,4% [7,7; 11,3]);
  các mô hình ~12M rẻ hơn rõ về độ trễ và VRAM. Do khác đồng thời kích thước, dữ liệu, độ phân giải khung và số học,
  so sánh này không tách được đóng góp của pre-train." Khuyến nghị thực dụng ở Ch4: có T4 và chấp nhận ~600M →
  fine-tune cho WER thấp hơn với GPU-h người dùng bỏ ra tương đương (~7 h so với 5–6 h); giới hạn bộ nhớ/thiết bị suy
  luận → mô hình ~12M.
- **Không được viết:** "pre-train đáng/không đáng chi phí", "mô hình nhỏ đủ dùng", "mô hình nhỏ phù hợp audio dài"
  (chưa đo WER audio dài), hay bất kỳ phát biểu nhân quả nào.
- Parakeet: đo tốc độ ở fp32 (fp16 tràn) → "chậm 3–5×" trộn kích thước với số học; OOM ~580 s là cấu hình attention
  toàn cục mặc định của NeMo (có chế độ attention cục bộ — **chưa kiểm**); không tính chi phí pre-train; nghi trùng
  dữ liệu; fine-tune trên nhãn giả.

## Hình (bản 16/10 trưa, sau phản biện thiết kế)

> **Người dùng 16/10 ~10:50: KHÔNG chấp nhận thiết kế 2×2 cho Hình 3.4** — chờ người dùng nêu thiết kế muốn
> dùng (PNG hiện tại trong `reports/figures/rq2/` là bản 2×2, chưa dùng). **Hình 3.5: người dùng sẽ hỏi lại** — chưa
> chốt. Mô tả dưới đây là phương án đã bị bác (3.4) / đang chờ (3.5), giữ để truy vết.

**Hình 3.4** — lưới 2×2 (cột 40 ms / 10 ms; hàng độ trễ / VRAM), cùng trục y theo hàng; chỉ nhóm ~12M; nhãn trực tiếp;
nét + marker riêng từng encoder (đọc được đen trắng); "×" điểm đo cuối trước OOM; đường ngang giới hạn T4.
> *Hình 3.4. Độ trễ suy luận trung vị (a, b) và VRAM đỉnh (c, d) theo độ dài audio của các encoder train từ đầu ~12M
> ở khung 40 ms (trái) và 10 ms (phải). Đo trên T4, batch 1, audio ghép từ cùng một video, có làm ấm từng độ dài;
> khối Mamba chạy fp32, phần còn lại autocast fp16. Dải xám: câu 10–15 s, nơi mọi mô hình có độ trễ 15–30 ms và chênh
> nhau 1–2 ms. "×": điểm đo cuối trước khi hết bộ nhớ; OOM của Conformer PE do cách cài bias vị trí tường minh O(T²),
> bản không PE (SDPA) chạy hết 3.600 s. Ở (c), B1 và ConExt trùng nhau vì đỉnh do khối Conv2d subsampling quyết định.
> Parakeet-CTC-0,6B (khác điều kiện) có số liệu ở Bảng 3.7.*

**Hình 3.5** — trục x độ trễ (ms) ở câu 15 s, thang tuyến tính từ 0 (thang log phóng đại chênh 1–2 ms); marker đặc
40 ms / rỗng 10 ms; Parakeet chỉ là hai đường ngang tham chiếu WER, không đặt trên trục độ trễ.
> *Hình 3.5. WER clean-test (greedy, CI 95% bootstrap theo video) theo độ trễ suy luận trung vị trên câu 15 s (T4,
> batch 1) của các encoder train từ đầu ~12M. Marker đặc: khung 40 ms [sau L3: WER trung bình 3 seed]; rỗng: khung
> 10 ms, 1 seed. Điều kiện số học không đồng nhất (khối Mamba fp32, phần còn lại fp16). Đường xám ngang: WER của
> Parakeet-CTC-0,6B (pre-train, 608M) — mốc tham chiếu khác điều kiện, không đặt trên trục độ trễ (Bảng 3.7). CI chồng
> nhau không cho phép xếp hạng; so sánh cặp ở Bảng 3.3.*

## Hạn chế RQ2 phải khai
- Một loại GPU (T4), batch 1; chưa đo throughput theo batch; không CUDA graph / compile; `cudnn_benchmark=false`.
- Số học không đồng nhất: khối Mamba fp32, Conformer fp16, Parakeet fp32 toàn bộ.
- Thiếu Conformer không PE (SDPA) ở 40 ms; mốc relPE là cài đặt O(T²) tường minh.
- Không có WER trên audio dài; mô hình train trên câu 10–15 s; không đo kịch bản chia đoạn bằng VAD.
- Trọng số ngẫu nhiên; audio ghép từ một video; độ trễ chỉ tính giải mã greedy (chưa tính beam + LM).
- Điểm giao nội suy trên lưới thưa; ngưỡng ≥ 1,5× đặt sau khi thấy số; bảng chỉ báo trung vị (p25/p75 có trong csv).
- GPU-h 10 ms trộn run rẽ nhánh v1 và run train từ đầu; 40 ms hiện 1 seed (đang chạy thêm 2).
- Phần dư ~1,6 GiB trong VRAM 40 ms chưa phân rã.
- Parakeet: khác framework, khung 80 ms, attention mặc định, không tính chi phí pre-train, nghi trùng dữ liệu, fine-tune
  trên nhãn giả.
