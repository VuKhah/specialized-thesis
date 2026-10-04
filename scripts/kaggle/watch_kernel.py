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
    except subprocess.TimeoutExpired:
        return ""


def gpu_used_hours() -> float | None:
    m = re.search(r"^GPU\s+([\d.]+)h", kaggle("quota"), re.M)
    return float(m.group(1)) if m else None


def log_lines(kernel: str) -> list[str]:
    out = kaggle("kernels", "logs", kernel)
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
    return [l for l in out.splitlines() if l.strip() and "Warning" not in l and "re.sub" not in l]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("kernel")
    p.add_argument("--interval", type=float, default=60, help="Giây giữa 2 lần hỏi")
    p.add_argument("--no_log_minutes", type=float, default=15,
                   help="Quota tăng mà log không có dòng mới quá chừng này phút → báo động")
    args = p.parse_args()

    q0, n_seen, last_new, quota_rising_since = gpu_used_hours(), 0, time.time(), None
    print(f"[{time.strftime('%H:%M:%S')}] theo dõi {args.kernel}; GPU đã dùng {q0} h", flush=True)
    while True:
        status = kaggle("kernels", "status", args.kernel).strip().split('"')[-2:-1] or ["?"]
        status = status[0].replace("KernelWorkerStatus.", "")
        lines, q = log_lines(args.kernel), gpu_used_hours()
        if len(lines) > n_seen:
            for l in lines[n_seen:]:
                print("  │ " + l[:220], flush=True)
            n_seen, last_new = len(lines), time.time()
        if q is not None and q0 is not None and q > q0 and quota_rising_since is None:
            quota_rising_since = time.time()
        silent = (time.time() - max(last_new, quota_rising_since or time.time())) / 60
        print(f"[{time.strftime('%H:%M:%S')}] {status} | log {n_seen} dòng | GPU đã dùng {q} h "
              f"({'đang tính' if quota_rising_since else 'chưa tính'}) | im lặng {silent:.0f}′", flush=True)
        if status not in ("RUNNING", "QUEUED"):
            print(f"Kernel kết thúc: {status}", flush=True)
            sys.exit(0)
        if quota_rising_since and silent > args.no_log_minutes:
            print(f"!!! BÁO ĐỘNG: quota đang tính mà {silent:.0f} phút không có log mới — xem UI, cân nhắc Stop "
                  f"(kaggle.com/code → View Active Events)", flush=True)
            sys.exit(2)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
