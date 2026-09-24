# Log cài đặt mamba-ssm trên Kaggle T4

Ghi lại kết quả từng lần chạy `notebooks/00_setup_environment.ipynb`, theo
đúng nhắc nhở trong chính notebook — để không phải dò lại từ đầu mỗi lần mở
máy Kaggle mới.

> **Đính chính (2026-09-14):** môi trường thực tế của cả 3 lần dưới đây là
> **Kaggle** (2x T4), không phải Google Colab như ghi nhầm ban đầu — dấu
> hiệu là `nvidia-smi` luôn trả về 2 GPU (Colab free-tier chỉ cấp 1x T4).
> Không ảnh hưởng đến các chẩn đoán kỹ thuật (build isolation, Mamba-3/
> tilelang) vì chúng độc lập với nền tảng Colab hay Kaggle.

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

## Lần 2 — 2026-09-14

**Môi trường:** Colab T4 mới (session khác), `torch 2.10.0+cu128`, driver
CUDA 13.0, Python 3.12.13.

**Cách 2 — build từ source với `--no-build-isolation`: CÀI THÀNH CÔNG.**

```
pip install packaging ninja                                            -> return code 0
pip install --no-build-isolation git+.../Dao-AILab/causal-conv1d       -> return code 0
pip install --no-build-isolation git+.../state-spaces/mamba            -> return code 0
```

Xác nhận đúng giả thuyết ở Lần 1: build isolation là nguyên nhân, dùng
`--no-build-isolation` để setup.py thấy torch cài sẵn đã giải quyết được.

**Nhưng test forward pass với CUDA kernel thật vẫn THẤT BẠI, lỗi khác:**

```
AttributeError("attribute '__dict__' of 'type' objects is not writable")
```

Chưa có traceback đầy đủ ở lần chạy này (cell cũ chỉ in `repr(e)`) nên chưa
xác định được lỗi xảy ra ở đâu (import `mamba_ssm`, khởi tạo `Mamba(...)`,
hay lúc gọi forward). Notebook đã được cập nhật 2 việc để chẩn đoán tiếp ở
lần chạy sau:
1. In full traceback (`traceback.print_exc()`) thay vì chỉ `repr(e)`.
2. Thêm bước **restart runtime sau khi cài, trước khi import** — nghi vấn
   hàng đầu: torch đã được import trước khi cài package build từ source
   trong cùng session, custom CUDA op đăng ký vào torch dispatcher có thể
   cần runtime mới để nhận đúng trạng thái.

**Tin tốt — phương án dự phòng đã xác nhận hoạt động:**
`from mamba_ssm.ops.selective_scan_interface import selective_scan_ref` →
**import thành công.** Đề tài không bị chặn ngay cả khi CUDA kernel thật
không chạy được — có thể tiếp tục với `selective_scan_ref` (chậm hơn, cần
ghi rõ trong Chương 2 và ước lượng lại thời gian train).

**Bonus — xác nhận schema dataset VietSuperSpeech:**
`dict_keys(['audio', 'text', 'duration', 'source'])`. `text` là transcript
(đúng như giả định trong `src/data/vietsuperspeech_dataset.py`), `source`
là tên file video nguồn — có thể dùng để nhóm/loại nhiễu theo kênh khi cần.

## Lần 3 — 2026-09-14

User gửi full traceback của lỗi `__dict__ of 'type' objects is not
writable` (trước đó chỉ có `repr(e)`). **Nguyên nhân thật, không liên quan
đến torch dispatcher hay build isolation:**

```
from mamba_ssm import Mamba
  -> mamba_ssm/__init__.py: from mamba_ssm.modules.mamba3 import Mamba3
    -> mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py: import tilelang
      -> tilelang/__init__.py: import tvm
        -> tvm/ir/attrs.py: @tvm_ffi.register_object("ir.DictAttrs")
          -> tvm_ffi/registry.py: setattr(type_cls, name, field.as_property(type_cls))
             AttributeError: attribute '__dict__' of 'type' objects is not writable
```

Nhánh `main` của `state-spaces/mamba`, từ tag **v2.3.2** (5/2025), gộp thêm
kiến trúc **Mamba-3** — và `mamba_ssm/__init__.py` **luôn luôn** import
`Mamba3` (không có try/except), kéo theo dependency nặng `tilelang` → `tvm`.
Bản thân `tvm_ffi` có bug đăng ký class attribute không tương thích Python
3.12.13 (không liên quan gì đến CUDA/GPU hay code của đề tài). Đề tài chỉ
cần kiến trúc Mamba/S6 cổ điển (Mamba-1/2), không dùng Mamba-3.

**Fix:** pin cài đặt về tag **`v2.3.1`** (release cuối cùng trước khi gộp
Mamba-3, 10/03/2025):

```
pip install --no-build-isolation git+https://github.com/state-spaces/mamba@v2.3.1
```

Đã cập nhật `notebooks/00_setup_environment.ipynb` theo fix này.

## Lần 4 — 2026-09-14 — ĐÃ GIẢI QUYẾT

**Môi trường:** Kaggle T4 x2, torch 2.10.0+cu128, Python 3.12.13 (session mới).

Chạy lại notebook đã pin `v2.3.1`:

```
pip install packaging ninja                                              -> return code 0
pip install --no-build-isolation git+.../Dao-AILab/causal-conv1d         -> return code 0
pip install --no-build-isolation git+.../state-spaces/mamba@v2.3.1       -> return code 0
```

Test forward pass CUDA kernel thật:

```
from mamba_ssm import Mamba
m = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).to("cuda")
y = m(torch.randn(2, 32, 64, device="cuda"))
-> OK — mamba-ssm (CUDA kernel) hoạt động. Output shape: torch.Size([2, 32, 64])
```

`MAMBA_CUDA_OK = True`, `FALLBACK_OK = True` (selective_scan_ref cũng vẫn
import được). Kết luận notebook: **dùng CUDA kernel thật cho tốc độ
train/inference tốt nhất.**

## Kết luận cuối cùng — rủi ro cao nhất của đề tài đã được xác nhận an toàn

Cài đặt đúng cho `mamba-ssm` (Mamba/S6 cổ điển, không dính Mamba-3) trên
Kaggle T4:

```bash
pip install packaging ninja
pip install --no-build-isolation git+https://github.com/Dao-AILab/causal-conv1d
pip install --no-build-isolation git+https://github.com/state-spaces/mamba@v2.3.1
```

Dùng `from mamba_ssm import Mamba` bình thường trong `src/models/mamba_encoder.py`
— CUDA kernel thật hoạt động, không cần fallback `selective_scan_ref` (vẫn
giữ như phương án dự phòng nếu môi trường Kaggle thay đổi version sau này).
Có thể tiếp tục sang Tuần 3 (khảo sát VietSuperSpeech, xây tokenizer) theo
đúng kế hoạch trong đề cương.

## Lần 5 — 2026-09-24 (kernel Kaggle chạy từ local, `scripts/kaggle/verify_mamba.py`)

**Môi trường:** Kaggle 2x T4, Python 3.12, `torch 2.10.0+cu128`.

Pin thêm `causal-conv1d` về tag **v1.5.4** (trước đây dùng `main`). Dùng
`pip wheel --no-build-isolation --no-deps` để giữ lại wheel:
- `causal-conv1d` 1.5.4: build từ source **~13 phút** (`MAX_JOBS=4`), wheel 135 MB.
- `mamba-ssm` 2.3.1: **~15 giây** — `setup.py` tìm được wheel dựng sẵn trên
  GitHub release khớp torch/CUDA nên không phải build; wheel 534 MB.
- `from mamba_ssm import Mamba` OK; forward + backward thật chạy được
  (smoke test 5 step train).

Hai wheel nằm trong output của kernel `verify-mamba-asr` — lần sau có thể gắn
output đó làm input và `pip install --no-deps` thẳng, bỏ qua 13 phút build.
