# Log cài đặt mamba-ssm trên Colab T4

Ghi lại kết quả từng lần chạy `notebooks/00_setup_environment.ipynb`, theo
đúng nhắc nhở trong chính notebook — để không phải dò lại từ đầu mỗi lần mở
máy Colab mới.

## Lần 1 — 2026-09-14

**Môi trường:** Colab free-tier, GPU Tesla T4 (compute capability 7.5),
driver CUDA 13.0, `torch 2.10.0+cu128`.

**Cách 1 — `pip install mamba-ssm causal-conv1d>=1.4.0`: THẤT BẠI.**

```
error: subprocess-exited-with-error
× Getting requirements to build wheel did not run successfully.
```

Lỗi xảy ra ngay ở bước **lấy metadata** (trước cả bước biên dịch CUDA
kernel thật sự). Nguyên nhân khả dĩ nhất: `setup.py` của `causal-conv1d`/
`mamba-ssm` cần `import torch` để xác định đúng CUDA/ABI tag khi build,
nhưng pip mặc định build trong **môi trường cô lập (PEP 517 build
isolation)** không có torch — torch cài sẵn trên Colab (2.10.0+cu128) không
nhìn thấy được từ trong đó. Ngoài ra `mamba-ssm`/`causal-conv1d` thường
chỉ publish prebuilt wheel cho một số tổ hợp torch/CUDA cụ thể, torch
2.10+cu128 (rất mới) nhiều khả năng chưa có wheel sẵn → phải build từ
source.

Vì cài đặt thất bại hoàn toàn, bước kiểm tra sau đó (cả CUDA kernel thật
lẫn fallback `selective_scan_ref` thuần PyTorch) đều báo
`ModuleNotFoundError('mamba_ssm')` — **đây không phải là fallback thật sự
thất bại, mà vì package chưa được cài được gì cả.** Không thể kết luận gì
về tính khả dụng của fallback thuần PyTorch từ lần chạy này.

**Kết luận lần 1:** chưa xác nhận được rủi ro cao nhất — cần thử Cách 2
(`--no-build-isolation`, build từ source, dùng đúng torch đã cài sẵn).
Notebook đã được cập nhật để thử cách này làm ưu tiên tiếp theo.

**Việc cần làm lần 2:** chạy lại `notebooks/00_setup_environment.ipynb`
(đã sửa), ghi kết quả tiếp vào file này (Lần 2 — ...).
