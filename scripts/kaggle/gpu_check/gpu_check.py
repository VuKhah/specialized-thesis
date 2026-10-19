"""Kernel Kaggle **GPU** ngắn — kiểm tài khoản mới có được cấp 2x T4 không (19/10: D xếp hàng > 1 h không được cấp;
thêm E). In môi trường + một phép nhân ma trận trên mỗi GPU; ~1 phút GPU nếu được cấp máy."""
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
t0 = time.time()
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, capture_output=True,
                     text=True).stdout, flush=True)
import torch  # noqa: E402

print("python", sys.version.split()[0], "| torch", torch.__version__, "| GPU", torch.cuda.device_count(), flush=True)
for i in range(torch.cuda.device_count()):
    x = torch.randn(4096, 4096, device=f"cuda:{i}")
    torch.cuda.synchronize(i)
    t = time.time()
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize(i)
    print(f"cuda:{i} {torch.cuda.get_device_name(i)} matmul 10x {time.time() - t:.2f}s", flush=True)
print(f"GPU OK {time.time() - t0:.0f}s", flush=True)
