"""Theo dõi một kernel Kaggle từ máy local: trạng thái, số dòng log, quota GPU.

Lý do (sự cố train thử 2026-10-04): trạng thái RUNNING của CLI gồm cả lúc chờ cấp
máy (lần 1 chờ ~3 h) lẫn lúc máy đã cấp mà script chưa chạy (lần 2: quota tăng
~50 phút, log 0 dòng). Chỉ nhìn status thì không phân biệt được. Script này báo
động khi quota tăng mà log không có dòng mới quá `--no_log_minutes`, để người dùng
bấm Stop sớm (CLI không có lệnh dừng kernel).

    python scripts/kaggle/watch_kernel.py tieunhi/train-trial-asr
    KAGGLE_CONFIG_DIR=C:/Users/Dell/.kaggle_Duc python scripts/kaggle/watch_kernel.py vuvanduc1/<kernel>

Thoát 0 khi kernel kết thúc, 2 khi báo động (kernel vẫn chạy — người dùng quyết Stop).
"""

import argparse
import re
import subprocess
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding != "utf-8":
        _stream.reconfigure(encoding="utf-8")


def kaggle(*args: str, timeout: int = 90) -> str:
    try:
        r = subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, env={**__import__("os").environ, "PYTHONUTF8": "1"})
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:  # `logs -f` không tự thoát: lấy phần đã nhận trước khi cắt
        out = e.stdout or b""
        return out.decode("utf-8", errors="replace") if isinstance(out, bytes) else out


def gpu_used_hours() -> float | None:
    m = re.search(r"^GPU\s+([\d.]+)h", kaggle("quota"), re.M)
    return float(m.group(1)) if m else None


def log_lines(kernel: str, running: bool) -> list[str]:
    # Khi kernel đang chạy, `kernels logs` (không -f) trả rỗng; chỉ `-f` trả log — đọc ~20 s rồi cắt.
    out = kaggle("kernels", "logs", "-f", kernel, timeout=20) if running else kaggle("kernels", "logs", kernel)
    if "Server Error" in out:
        return []
    if out.lstrip().startswith("["):  # kernel đã kết thúc: CLI trả mảng JSON thay vì văn bản
        try:
            import json
            out = "".join(e.get("data", "") for e in json.loads(out))
        except ValueError:
            pass
    # Log stream gửi UTF-8 nhưng CLI giải mã latin-1 → đảo lại để đọc được tiếng Việt.
    try:
        out = out.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    # Bỏ dòng thống kê của ptxas khi build mamba (`--ptxas-options=-v`, hàng nghìn dòng).
    return [l for l in out.splitlines() if l.strip() and "Warning" not in l and "re.sub" not in l
            and "bytes stack frame" not in l and "ptxas info" not in l]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("kernel")
    p.add_argument("--interval", type=float, default=60, help="Giây giữa 2 lần hỏi")
    p.add_argument("--no_log_minutes", type=float, default=15,
                   help="Quota đã tính chừng này phút mà chưa có dòng log nào → báo động")
    args = p.parse_args()

    # `logs -f` luôn stream từ đầu log và bị cắt sau ~20 s → log dài thì không đọc tới cuối, số dòng
    # không tăng được dù kernel vẫn in. Vì vậy chỉ báo động khi quota tính mà **chưa có dòng log nào**
    # (đúng sự cố lần 2); treo giữa chừng do watchdog trong kernel lo (im lặng 20 phút → kill).
    q0, shown, quota_rising_since = gpu_used_hours(), 0, None
    print(f"[{time.strftime('%H:%M:%S')}] theo dõi {args.kernel}; GPU đã dùng {q0} h", flush=True)
    while True:
        status = kaggle("kernels", "status", args.kernel).strip().split('"')[-2:-1] or ["?"]
        status = status[0].replace("KernelWorkerStatus.", "")
        lines, q = log_lines(args.kernel, running=status in ("RUNNING", "QUEUED")), gpu_used_hours()
        if len(lines) > shown:
            for l in lines[shown:]:
                print("  │ " + l[:220], flush=True)
            shown = len(lines)
        if q is not None and q0 is not None and q > q0 and quota_rising_since is None:
            quota_rising_since = time.time()
        charged = (time.time() - quota_rising_since) / 60 if quota_rising_since else 0
        last = lines[-1][:120] if lines else "—"
        print(f"[{time.strftime('%H:%M:%S')}] {status} | GPU đã dùng {q} h "
              f"({f'tính {charged:.0f}′' if quota_rising_since else 'chưa tính'}) | dòng đọc được cuối: {last}",
              flush=True)
        if status not in ("RUNNING", "QUEUED"):
            print(f"Kernel kết thúc: {status}", flush=True)
            sys.exit(0)
        if quota_rising_since and not lines and charged > args.no_log_minutes:
            print(f"!!! BÁO ĐỘNG: quota đã tính {charged:.0f} phút mà chưa có dòng log nào — xem UI, cân nhắc "
                  f"Stop (kaggle.com/code → View Active Events)", flush=True)
            sys.exit(2)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
