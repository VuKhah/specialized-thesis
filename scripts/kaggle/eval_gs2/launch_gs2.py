"""Đẩy kernel `eval_gs2.py` lên đúng tài khoản cho từng chế độ (phương án B, 2026-10-19).

Chia theo dữ liệu mỗi tài khoản đọc được: A có dataset checkpoint 40 ms + output fine-tune Parakeet; 10 ms cũng chạy
trên A (dataset `kltn-v2a-10ms-ep37`, kernel thứ hai song song) vì B tạo dataset bị 403 ngày 19/10; C (PhoWhisper) và
D (Parakeet zero-shot) chỉ cần audio và mô hình công khai trên HF. Username đọc từ `kaggle.json` lúc chạy, bản sinh ra ở `_build/` (gitignore) — repo public.

    python scripts/kaggle/eval_gs2/launch_gs2.py --mode 40ms            # chỉ sinh _build/, in lệnh
    python scripts/kaggle/eval_gs2/launch_gs2.py --mode 40ms --push
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
BUILD = HERE / "_build"
sys.path.insert(0, str(HERE.parent))
import accounts  # noqa: E402 — username/thư mục khoá chỉ ở máy local (repo public)
PLAN = {  # chế độ → (tài khoản, dataset của chính tài khoản đó, kernel_sources)
    "40ms": ("A", ["mamba-wheels-v2", "kltn-v2a-s4-ep37", "kltn-v2a-s4-seeds-ep37", "gs2-vi-test"],
             ["finetune-parakeet-main"]),
    "40ms_dev": ("A", ["mamba-wheels-v2", "kltn-v2a-s4-ep37", "kltn-v2a-s4-seeds-ep37", "gs2-vi-dev"], []),
    "10ms": ("A", ["mamba-wheels-v2", "kltn-v2a-10ms-ep37", "gs2-vi-test"], []),  # B: tạo dataset 403 (19/10)
    "phowhisper": ("C", ["gs2-vi-test"], []),
    "parakeet_zs": ("D", ["gs2-vi-test"], []),
}
WHEELS_OWNER = "A"  # mamba-wheels-v2 của A, đã chia sẻ cho B và C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=PLAN)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--account", choices=accounts.names(), default=None,
                    help="Ghi đè tài khoản (19/10: C, D QUEUED > 1 h không được cấp GPU → chạy pre-train trên A)")
    args = ap.parse_args()
    account, datasets, kernels = PLAN[args.mode]
    account = args.account or account
    user = accounts.username(account)
    wheels_user = accounts.username(WHEELS_OWNER)
    sources = [f"{wheels_user if d == 'mamba-wheels-v2' else user}/{d}" for d in datasets]

    out = BUILD / args.mode
    out.mkdir(parents=True, exist_ok=True)
    code = (HERE / "eval_gs2.py").read_text(encoding="utf-8").replace('MODE = "__MODE__"', f'MODE = "{args.mode}"')
    (out / "eval_gs2.py").write_text(code, encoding="utf-8")
    slug = f"eval-gs2-{args.mode}"
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}", "title": slug, "code_file": "eval_gs2.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": sources, "competition_sources": [],
        "kernel_sources": [f"{user}/{k}" for k in kernels]}, indent=2), encoding="utf-8")
    print(f"{args.mode}: tài khoản {account}, {len(sources)} dataset, {len(kernels)} kernel nguồn → {out}")
    if args.push:
        env = accounts.env(account)
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(out), "-t", "10800"], env=env, text=True,
                           capture_output=True, encoding="utf-8", errors="replace")
        print((r.stdout + r.stderr).replace(user, f"<{account}>").strip())
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
