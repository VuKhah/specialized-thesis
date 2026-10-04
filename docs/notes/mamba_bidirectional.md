# Nghi vấn: Mamba một chiều so với Conformer hai chiều

Ghi 2026-10-01. **Trạng thái: ĐÃ CHỐT hai chiều (người dùng, 2026-10-02) — phương án B1, đã sửa code 2026-10-02 (mục cuối), chưa chạy trên GPU.** ~~đang cân nhắc chuyển sang hai chiều — chưa
quyết, chưa đổi code.** Người dùng muốn cân nhắc; mang câu A4 đi hỏi GVHD
(`docs/tong_ket/2026-09-30/cau_hoi_gvhd.html`, local). Phải chốt **trước khi
train** vì đổi kiến trúc Mamba.

## Vấn đề

`MambaEncoder` hiện là 28 lớp `x + Mamba(LN(x))`, mỗi `mamba_ssm.Mamba` quét
**trái → phải** (conv1d nhân quả + selective scan): khung t chỉ thấy khung ≤ t.
`ConformerEncoder` dùng self-attention không mask → mỗi khung thấy **cả câu**.

Hệ quả cho RQ1: nếu Conformer cho WER thấp hơn, không tách được do "SSM kém
attention" hay do "chỉ nhìn một phía". Đây là biến gây nhiễu mà đề cương không
nêu (đề cương chỉ nói "cùng pipeline, khớp tham số").

Thêm hai điểm liên quan:
- Bài AISHELL-1 đề cương dẫn số liệu (arXiv:2410.00070, CER 5,55% vs 7,49%) là
  **streaming**, không đối chiếu thẳng với thiết lập hiện tại.
- Các công trình Mamba cho ASR không streaming (ConMamba; nhóm bài so sánh
  InnBiMamba/ExtBiMamba) thường dùng **hai chiều**. *Tên bài và chi tiết cần
  đối chiếu nguồn trước khi trích — đang ghi theo trí nhớ AI.*

## Phương án

| | Mô tả | Tham số encoder | Chênh vs Conformer |
|---|---|---|---|
| **U** — giữ nguyên | 28 lớp một chiều | 12.292.864 | +0,73% |
| **B1** — hai chiều "ngoài" (ExtBiMamba) | 14 lớp; mỗi lớp 2 khối `Mamba` riêng: một khối quét chuỗi xuôi, một khối quét chuỗi **đảo ngược theo độ dài thật**, đảo kết quả lại rồi **cộng**; chung 1 LayerNorm | 12.285.696 | **+0,67%** |
| **B2** — hai chiều "trong" (kiểu Vim/BiMamba v2) | Chung `in_proj`/`out_proj`, hai nhánh conv+scan riêng | cần tính lại | — |
| **U + B1** | Train cả hai, Mamba một chiều thành ablation | — | — |

Tính B1 (2026-10-01): mỗi khối 437.760 + LN 512 cho mỗi lớp, `input_proj`
20.736, `norm_f` 512 → `20.736 + 14 × (2 × 437.760 + 512) + 512`. B1 vẫn là
**28 khối Mamba** như hiện nay → số tham số và FLOPs gần như không đổi.

## Nếu chọn hai chiều (B1) phải làm gì

- **Mask/đảo theo độ dài thật**: nhánh ngược mà đảo cả tensor đã pad thì phần
  pad (ở cuối) thành đầu chuỗi và lọt vào trạng thái. Phải đảo từng câu theo
  `feat_lengths` (gather chỉ số), không `torch.flip` cả tensor. Lý do "không cần
  mask" trong `mamba_encoder.py` và `ARCHITECTURE.md` mục 8-e **hết đúng**.
- Sửa `MambaEncoder` + `configs/model_mamba.yaml` (`n_layers: 14`, thêm cờ hai
  chiều); đo lại `param_count` trên Kaggle.
- **Benchmark lại** Mamba (DDP 2 GPU + AMP): 28 lần quét như cũ nhưng thêm
  gather đảo chiều; ước 16,0 h/30 epoch không còn chắc.
- Giữ các sửa theo khuyến nghị tác giả: miễn weight decay `A_log`/`D`,
  `norm_f`, chia `out_proj` — chia cho √(số khối cộng vào residual); cần xét lại
  √14 hay √28.
- Cập nhật hình `reports/figures/mamba_tong_the.*`, `khoi_conformer_vs_mamba.*`,
  `kien_truc_*`.
- Ước lệch lịch khoảng 1 tuần (sửa + test + benchmark), chưa tính train.

## Được và mất

| | U (một chiều) | B1 (hai chiều) |
|---|---|---|
| RQ1 công bằng về ngữ cảnh | Không | Có |
| RQ2 chi phí theo độ dài | O(T) | Vẫn O(T) (hai lần quét) |
| Dùng được cho streaming | Có — một luận điểm riêng | Không |
| Đúng bài Mamba gốc | Có | Biến thể phổ biến trong ASR |
| Không cần mask padding | Có | Không — phải xử lý |
| Việc phải làm thêm | Không | Code + test + benchmark |

## Câu hỏi còn mở

1. GVHD chấp nhận đổi kiến trúc so với đề cương ("Mamba" không nói rõ một hay
   hai chiều) không, hay muốn giữ và nêu hạn chế?
2. Nếu B1: gộp hai hướng bằng cộng (không thêm tham số) hay nối + Linear
   (+131k tham số/lớp, phải khớp lại)?
3. Có đủ quota để train thêm U làm ablation không (tài khoản B, ~16 h)?

## Đã triển khai (2026-10-02)

- `src/models/mamba_encoder.py`: cờ `bidirectional` (mặc định code `False` để
  yaml cũ/checkpoint cũ vẫn dựng đúng bản một chiều); `configs/model_mamba.yaml`
  `n_layers: 14`, `bidirectional: true`. Khối xuôi giữ tên `layers`, khối ngược
  `layers_bwd`, chung `norms`. Gộp hai hướng bằng **cộng** (câu hỏi mở 2 — theo
  B1 đã chốt, không thêm tham số).
- Đảo theo độ dài: `reverse_padded` (gather chỉ số `L-1-t` cho `t < L`, giữ
  nguyên padding ở cuối; tự nghịch đảo).
- Chia `out_proj`: **√28** (= √(2 khối/lớp × 14 lớp)) cho cả hai nhánh. Lý do:
  `_init_weights` chia cho √(`n_residuals_per_layer` × `n_layer`), tức số đầu ra
  khối độc lập cộng vào residual; B1 cộng 2 `out_proj` mỗi lớp. Trùng hệ số bản
  một chiều 28 lớp.
- **Đã test (CPU, khối Mamba-1 giả thuần PyTorch cùng shape tham số, scan tham
  chiếu tuần tự):** 1 khối 437.760; encoder B1 **12.285.696** (khớp mục trên);
  bản một chiều 28 lớp 12.292.864; 56 tham số `A_log`/`D` được miễn weight decay;
  std `out_proj` giảm đúng 1/√28 cả hai nhánh; shape vào/ra; bất biến padding
  (mẫu chạy riêng vs trong batch với padding rác biên độ 1e3: lệch 6e-7, float32);
  đối chứng `torch.flip` cả tensor lệch 8e-2 (test có độ nhạy); nhánh ngược
  nhìn được tương lai, bản một chiều thì không (lệch đúng 0); gradient tới khối
  ngược; Conformer qua `build_encoder` không đổi (12.204.288).
- **Chưa test:** kernel CUDA thật (`mamba_ssm` + `causal_conv1d`), AMP fp16, DDP,
  tốc độ. Đã sửa sẵn `scripts/kaggle/benchmark/benchmark.py` (lần 3: chỉ Mamba,
  AMP 1 GPU và 2 GPU, kèm `CHECK_CODE` kiểm tham số/padding/tương lai bằng
  kernel thật trước khi đo) — chưa push/chạy.

## Đối chiếu nguồn (2026-10-03)

Đối chiếu `src/models/mamba_encoder.py` với mã mamba-ssm **tag v2.3.1**
(`modules/mamba_simple.py`, `modules/block.py`, `models/mixer_seq_simple.py`,
`models/config_mamba.py`) và hai bài báo.

**Nguồn ExtBiMamba đã xác minh:** Zhang, Zhang, Liu, Xiao và cs., *Mamba in
Speech: Towards an Alternative to Self-Attention*, arXiv:2405.12609 (v6,
04/2025), Algorithm 2: `H' = Norm(H)`; mỗi chiều có in_proj/conv/SSM/out_proj
riêng; `H_l = y_fwd + y_bwd + H` — **B1 khớp đúng** (chung 1 norm, cộng, một
residual). InnBiMamba (chung in/out_proj) là kiểu Vision Mamba, chính là B2.

**Khớp mã gốc / bài Mamba (Gu & Dao, arXiv:2312.00752):** khối `Mamba` là của
thư viện (S4D-Real cho A, D = 1, dt khởi tạo [0,001; 0,1], dt_rank = 16,
d_state 16, d_conv 4, expand 2 = mặc định bài báo); pre-norm residual tương
đương `Block` (Add → Norm → Mixer); `norm_f`; chia `out_proj` theo
`_init_weights` (kaiming a=√5 rồi chia √n); residual fp32; miễn weight decay
`A_log`/`D`; không dropout. `reverse_padded` chạy thử: đảo đúng trong độ dài
thật, padding giữ nguyên, tự nghịch đảo. Số tham số tính tay theo shape v2.3.1:
437.760/khối × 28 + 14 LN + norm_f + input_proj = **12.285.696**, khớp note.

**Lệch nhỏ (có chủ đích hoặc không đáng kể):**
- LayerNorm thay RMSNorm: `MambaConfig` mặc định `rms_norm=True` (công thức
  "Transformer++" của bài), nhưng `MixerModel` hỗ trợ cả hai và bài gọi là
  "standard normalization". LayerNorm giống Conformer torchaudio.
- Hệ số √28 cho B1 là suy luận của dự án (mã gốc không có hai chiều).
- Công thức train (AdamW 0,01, β mặc định, clip 5, warmup + hằng số) là của
  pipeline chung, khác công thức LM của bài (wd 0,1, β2 0,95, clip 1, cosine);
  LayerNorm/bias vẫn bị weight decay (các công thức LM thường miễn). Áp dụng
  như nhau cho Conformer.
- A thực (real): bài chỉ chuyển sang phức cho audio dạng sóng thô; mamba-ssm
  `Mamba` chỉ hỗ trợ A thực. Đầu vào ở đây là log-mel 100 khung/s.

**Rủi ro lớn (chưa quyết):** bài 2405.12609 mục V-B/V-D báo **Mamba/BiMamba
chồng độc lập** (đúng thiết kế hiện tại) cho ASR **kém rõ rệt** Transformer/
Conformer, kể cả khi tăng số lớp để khớp tham số, và **khó train ổn định**;
lý do nêu: khối Mamba ít phi tuyến, ASR cần thêm FFN. Cấu hình tốt của họ là
thay MHSA trong Conformer bằng ExtBiMamba (ConExtBiMamba). Bài dùng ESPnet có
subsampling, khác thiết lập ở đây — không suy thẳng, nhưng phải nêu khi bàn
kết quả và cân nhắc trước khi train.

**Bổ sung 2026-10-03 — FFN không phải lời giải chắc chắn.** Bảng XVI của
2405.12609 (LibriSpeech100, WER dev/test): ExtBiMamba chồng độc lập 38,5/37,7;
**+ FFN 34,9/36,1** (chỉ ~1,6 điểm), + residual 42,1/41,7 (tệ hơn — tức mô hình
độc lập của họ vốn *không có* residual, khác thiết kế ở đây vốn có pre-norm
residual + `norm_f`); Transformer 8,0/8,4. Mức cải thiện lớn chỉ đến khi đặt
ExtBiMamba vào khung Conformer (thay MHSA, giữ macaron FFN + conv module):
ConExtBiMamba 5,9/6,0 vs Conformer 6,3/6,5 (Bảng XII), ổn định hơn qua seed trên
AN4 (Bảng XV). Mọi mô hình của họ đều đã có SpecAugment (cấu hình ESPnet) → SpecAugment
không cứu được Mamba độc lập trong bài. Hệ quả: bằng chứng "Mamba chồng thuần
kém" yếu hơn tưởng (baseline của họ thiếu residual); bằng chứng "thêm FFN là
đủ" cũng yếu.
