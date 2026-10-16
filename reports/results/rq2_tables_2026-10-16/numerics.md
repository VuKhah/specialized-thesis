# Điều kiện đo RQ2 (Bảng điều kiện số học — đưa vào Ch2 2.7.3 / Ch3 3.4)

| Mục | Nhóm train từ đầu (7 cấu hình) | Parakeet-CTC-0.6B-vi |
|---|---|---|
| Phần cứng | Tesla T4 (14.6 GiB), 1 GPU | như bên trái |
| Phần mềm | Python 3.13.15, torch 2.11.0+cu128, cuDNN 91900, commit `0090bd7` | NeMo (tiến trình riêng) |
| Số học | autocast fp16; front-end, log_softmax fp32; **khối Mamba fp32** | **fp32 toàn bộ** (fp16 tràn số ở 16/203 câu) |
| Attention | Conformer không PE: SDPA (flash/mem-efficient bật); Conformer PE tương đối: bias vị trí **tường minh** [B, H, T, 2T−1] → bộ nhớ O(T²) | FastConformer (subsampling 8×) |
| Trọng số | ngẫu nhiên seed 0 — chi phí không phụ thuộc giá trị trọng số | đã pre-train |
| Đầu vào | audio ghép các đoạn liên tiếp cùng một video (274 đoạn, 3.603 s), batch 1 | như bên trái |
| Đo | 3–30 lần/độ dài sau làm ấm (cuDNN dựng kế hoạch theo shape), báo trung vị; OOM tìm bằng chia đôi 4 bước | như bên trái |

Kết luận phải gọi đúng tên: Conformer PE tương đối hết bộ nhớ ở audio dài là **thuộc tính cách cài bias vị trí
tường minh**, không phải của Conformer nói chung — bản không PE (SDPA) chạy hết 3.600 s.
