"""Sinh + push kernel fine-tune Parakeet (`finetune_parakeet_kernel.py`).

    python scripts/kaggle/finetune_parakeet/launch.py --mode trial                 # chỉ sinh, xem lại
    python scripts/kaggle/finetune_parakeet/launch.py --mode trial --push
    python scripts/kaggle/finetune_parakeet/launch.py --mode main --account B --push --extra --total_steps 6000

Bản sinh ở `_build/<slug>/` (gitignore như train_full — metadata chứa username). Ghim `COMMIT` = HEAD local;
từ chối nếu HEAD chưa có trên origin hoặc src/scripts có thay đổi chưa commit (kernel clone từ GitHub).
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding != "utf-8":
        _stream.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BUILD = HERE / "_build"
sys.path.insert(0, str(HERE.parent))
import accounts  # noqa: E402 — username/thư mục khoá chỉ ở máy local (repo public)

# Không cần wheel mamba; dataset do A sở hữu, đã chia sẻ B.
DATASETS = ["{A}/vss-asr-train-shard0", "{A}/vss-asr-train-shard1", "{A}/vss-asr-train-shard2",
            "{A}/vss-asr-train-shard3", "{A}/vss-asr-val"]
KERNEL = "finetune_parakeet_kernel.py"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["trial", "main"])
    ap.add_argument("--account", default="A", choices=accounts.names())
    ap.add_argument("--session_minutes", type=int, default=420, help="Trần lần chính (chốt 2026-10-11: ~7 GPU-h)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Tham số cho finetune_parakeet (để cuối)")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    user = accounts.username(args.account)
    name = f"finetune-parakeet-{args.mode}"
    commit = git("rev-parse", "HEAD")
    if git("branch", "-r", "--contains", commit) == "":
        sys.exit(f"LỖI: commit {commit[:8]} chưa có trên origin — push trước (kernel clone từ GitHub)")
    if git("status", "--porcelain", "--", "src", "scripts"):
        sys.exit("LỖI: src/scripts có thay đổi chưa commit — kernel sẽ không chạy đúng code đang thấy")

    code = (HERE / KERNEL).read_text(encoding="utf-8")
    subs = {"MODE": json.dumps(args.mode), "COMMIT": json.dumps(commit),
            "SESSION_MINUTES": str(args.session_minutes), "MAIN_ARGS": json.dumps(args.extra)}
    for key, value in subs.items():
        code, n = re.subn(rf"^{key}(: [^=]+)? = .*$", lambda m: f"{key}{m.group(1) or ''} = {value}", code,
                          count=1, flags=re.M)
        if n != 1:
            sys.exit(f"LỖI: không tìm thấy hằng {key} trong {KERNEL}")

    out = BUILD / name
    out.mkdir(parents=True, exist_ok=True)
    (out / KERNEL).write_text(code, encoding="utf-8")
    meta = {"id": f"{user}/{name}", "title": name, "code_file": KERNEL, "language": "python", "kernel_type": "script",
            "is_private": "true", "enable_gpu": "true", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [accounts.ref(d) for d in DATASETS], "competition_sources": [], "kernel_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "build.json").write_text(json.dumps({"commit": commit, **subs}, indent=2), encoding="utf-8")
    print(f"{out}: {accounts.mask(meta['id'])} | commit {commit[:8]} | {args.session_minutes} phút | extra {args.extra}")

    if args.push:
        env = accounts.env(args.account)
        timeout = (args.session_minutes + 30) * 60 if args.mode == "main" else 150 * 60
        subprocess.run(["kaggle", "kernels", "push", "-p", str(out), "-t", str(timeout)], env=env, check=True)
        print(f"Theo dõi: python scripts/kaggle/watch_kernel.py {args.account}:{name}")


if __name__ == "__main__":
    main()
