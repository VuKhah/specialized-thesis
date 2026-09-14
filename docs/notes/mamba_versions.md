# Mamba, S6, Mamba-2, Mamba-3 — khác nhau ra sao, và đề tài dùng bản nào

Ghi chú tra cứu cho Chương 1 (Cơ sở lý thuyết) — "Cơ sở toán học của State
Space Model (S4, S6/Mamba)". Không phải nội dung để copy nguyên vào báo
cáo, nhưng đủ chính xác để bạn viết lại theo văn phong của mình.

## Dòng thời gian & quan hệ

```
S4 (2021, Gu et al.)
  → SSM cấu trúc, tuyến tính bất biến theo thời gian (LTI): A, B, C CỐ ĐỊNH
    cho mọi token. Tính được bằng convolution (FFT) → nhanh nhưng không
    "chọn lọc" nội dung như attention.

S6 = "Selective SSM" (12/2023, Gu & Dao, arXiv:2312.00752)
  → Cải tiến cốt lõi: Δ, B, C trở thành HÀM CỦA INPUT (data-dependent) →
    mất tính LTI, không convolution-hóa được nữa → cần thuật toán "selective
    scan" (quét song song tối ưu phần cứng, kỹ thuật recomputation giống
    FlashAttention) để vẫn chạy nhanh trên GPU.
  → "Mamba" = KHỐI (block) kiến trúc đầy đủ dùng S6 làm thành phần trộn
    chuỗi (sequence-mixing), kèm causal depthwise conv1d + gating SiLU —
    giống cấu trúc một gated-MLP block. Nói cách khác:
    S6 = lớp/cơ chế lõi bên trong.  Mamba = toàn bộ khối kiến trúc dùng S6.
  → Trong `mamba_ssm` (Python package): class `Mamba`.

Mamba-2 (2024, Dao & Gu, "Transformers are SSMs", ICML 2024)
  → Phát hiện: nếu ràng buộc ma trận trạng thái A thành "vô hướng nhân ma
    trận đơn vị" (scalar-times-identity) theo từng head — đơn giản hơn A
    dạng đường chéo theo từng kênh của Mamba-1 — thì SSM tương đương về
    toán học với một dạng linear/masked attention có cấu trúc ("State
    Space Duality" — SSD).
  → Nhờ tương đương này, tính bằng thuật toán dựa trên nhân ma trận theo
    khối (chunked matmul) thay vì scan tuần tự thuần túy → tận dụng
    tensor core GPU tốt hơn nhiều → train nhanh hơn Mamba-1 khoảng 2-8 lần,
    cho phép d_state (kích thước trạng thái) lớn hơn nhiều mà vẫn rẻ.
  → Đánh đổi: A bị ràng buộc về vô hướng thực (real scalar) làm mất khả
    năng biểu diễn động lực học "xoay" (rotational dynamics) → Mamba-2 yếu
    hơn hẳn trên các tác vụ state-tracking đơn giản (vd. parity check,
    modular arithmetic).
  → Trong `mamba_ssm`: class `Mamba2`.

Mamba-3 (~3/2026, Dao et al., arXiv:2603.15569)
  → Mục tiêu thiết kế khác hẳn: Mamba-2 tối ưu cho TỐC ĐỘ TRAIN, Mamba-3
    tối ưu cho HIỆU QUẢ INFERENCE.
  → 3 cải tiến kỹ thuật chính:
    1. Rời rạc hóa kiểu hình thang (trapezoidal discretization) thay vì
       Euler (dùng trong Mamba-1/2) — xấp xỉ phương trình vi phân liên tục
       chính xác hơn.
    2. Công thức MIMO (Multi-Input Multi-Output) — tăng arithmetic
       intensity, quan trọng khi decode (bị giới hạn bởi băng thông bộ
       nhớ, không phải compute).
    3. Ma trận trạng thái A dạng SỐ PHỨC (không chỉ vô hướng thực như
       Mamba-2) — khôi phục lại khả năng biểu diễn "xoay" đã mất ở
       Mamba-2, liên hệ lý thuyết với Data-Dependent RoPE. Sửa được các
       lỗi state-tracking mà Mamba-2 mắc phải.
    + Thêm QK-Norm/"BCNorm" giúp ổn định huấn luyện.
  → Đạt perplexity ngang Mamba-2 nhưng chỉ cần một nửa kích thước trạng
    thái; ở quy mô 1.5B tham số, vượt các baseline (kể cả Gated DeltaNet)
    trên downstream accuracy.
  → Trong `mamba_ssm` (nhánh `main`, từ v2.3.2): `mamba_ssm.modules.mamba3`.
    **Đây chính là phần gây lỗi cài đặt** (kéo theo `tilelang`/`tvm`, xem
    `mamba_ssm_install_log.md`) — không liên quan gì đến việc dùng Mamba-1.

## Đề tài dùng bản nào?

**Mamba/S6 cổ điển (Mamba-1)** — đúng như trích dẫn trong đề cương (Gu &
Dao, arXiv:2312.00752, mục "Cơ sở toán học của SSM (S4, S6/Mamba)" và tài
liệu tham khảo [6]). Lý do:

- Đề cương đã đăng ký với GVHD chỉ nói đến "Mamba (State Space Model có
  chọn lọc)" — không đề cập Mamba-2/3. Đổi sang bản khác là mở rộng phạm vi
  ngoài những gì đã đăng ký.
- Mamba-2/Mamba-3 là các kiến trúc *khác*, không phải bản "vá lỗi" của
  Mamba-1 — đổi bản sẽ đổi luôn câu chuyện nghiên cứu (Mamba-2 đánh đổi lấy
  tốc độ train, Mamba-3 tối ưu cho inference production ở quy mô lớn — cả
  hai không phải trọng tâm so sánh matched-parameter WER/RTF/error taxonomy
  mà đề tài đặt ra).
- `mamba_ssm.Mamba` (pin tag `v2.3.1`, không dùng bản `main`) chính là
  Mamba-1/S6 cổ điển — khớp 100% với những gì đã đăng ký.

Nếu về sau muốn bàn thêm trong phần "Hướng phát triển" ở Kết luận, có thể
nhắc Mamba-2/3 như hướng mở rộng tương lai (đã có sẵn trong cùng thư viện,
`mamba_ssm.Mamba2`) — nhưng không đổi phạm vi thực nghiệm chính.

## Nguồn

- Gu, A., Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective
  State Spaces*. arXiv:2312.00752, 2023.
- Dao, T., Gu, A. *Transformers are SSMs: Generalized Models and Efficient
  Algorithms Through Structured State Space Duality*. ICML 2024.
- Dao, T. et al. *Mamba-3: Improved Sequence Modeling using State Space
  Principles*. arXiv:2603.15569, 2026. Blog: tridao.me/blog/2026/mamba3-part1/
