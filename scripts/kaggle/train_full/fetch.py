"""Tải output một phiên train full về local ngay khi phiên xong (output kernel Kaggle có thể mất khi đẩy
version mới / xoá kernel — số liệu cho báo cáo không được chỉ nằm trên Kaggle).

    python scripts/kaggle/train_full/fetch.py --model conformer --session 1
    python scripts/kaggle/train_full/fetch.py --model mamba --session 2 --account B --ckpt_dest D:/kltn_checkpoints

Vào `reports/results/train_full/<[tag-]model>/s<k>/` (giữ cấu trúc output kernel):
  - file nhỏ, commit được: session_summary.json, config_s<k>.yaml, checkpoints/<exp>/metrics.jsonl, logs/
  - lớn, gitignore: checkpoints/<exp>/eval_*.jsonl (ref/hyp từng câu), runs/ (tensorboard)
Output phiên k đã gồm metrics/eval/epochs của mọi phiên trước (train_full chép nối) → bộ đủ là của phiên
cuối; logs/ và gpu_util.csv thì riêng từng phiên.
`--ckpt_dest`: tải thêm best.pt + các `epochs/epochXX.pt` chưa có ở đó (ngoài repo — người dùng đưa lên Drive).
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding != "utf-8":
        _stream.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import ACCOUNTS, REPO, slug  # noqa: E402

SMALL = r"(session_summary\.json|config_s\d+\.yaml|checkpoints/[^/]+/metrics\.jsonl|logs/.*)"
LARGE = r"(checkpoints/[^/]+/eval_.*\.jsonl|runs/.*)"


def kaggle_output(kernel: str, dest: Path, pattern: str, env: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["kaggle", "kernels", "output", kernel, "-p", str(dest), "--file-pattern", pattern],
                       env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        sys.exit(f"LỖI tải {kernel} ({pattern}): {r.stdout[-500:]}{r.stderr[-500:]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--account", default="A", choices=ACCOUNTS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ckpt_dest", default=None, help="Thư mục ngoài repo để tải best.pt + epochs/*.pt")
    args = ap.parse_args()

    user, config_dir = ACCOUNTS[args.account]
    kernel = f"{user}/{slug(args.model, args.session, args.tag)}"
    env = {**os.environ, "PYTHONUTF8": "1", **({"KAGGLE_CONFIG_DIR": config_dir} if config_dir else {})}
    run_name = f"{args.tag}-{args.model}" if args.tag else args.model
    out = REPO / "reports" / "results" / "train_full" / run_name / f"s{args.session}"
    kaggle_output(kernel, out, SMALL, env)
    kaggle_output(kernel, out, LARGE, env)

    summary = json.loads((out / "session_summary.json").read_text(encoding="utf-8"))
    metrics = list(out.glob("checkpoints/*/metrics.jsonl"))
    n_eval = len(list(out.glob("checkpoints/*/eval_*.jsonl")))
    print(f"{kernel} → {out}\n  returncode {summary['returncode']}, epoch xong {summary['epochs_done']}/"
          f"{summary['epochs_target']}, finished {summary['finished']}, kernel {summary['kernel_minutes']:.0f}′ "
          f"(setup {summary.get('setup_minutes', 0):.0f}′), commit {summary.get('run', {}).get('commit', '?')[:8]}"
          f"\n  metrics: {[str(m.relative_to(out)) for m in metrics]}, eval jsonl: {n_eval}")

    if args.ckpt_dest:
        dest = Path(args.ckpt_dest) / args.model
        have = {p.name for p in (dest / "epochs").glob("epoch*.pt")} if (dest / "epochs").exists() else set()
        last = summary["epochs_done"]
        missing = [f"epoch{e:02d}\\.pt" for e in range(last) if f"epoch{e:02d}.pt" not in have]
        pattern = r"checkpoints/[^/]+/(best\.pt" + (r"|epochs/(" + "|".join(missing) + r")" if missing else "") + r")"
        tmp = dest / "_dl"
        kaggle_output(kernel, tmp, pattern, env)
        got = []
        for f in tmp.rglob("*.pt"):
            target = dest / ("epochs" if f.parent.name == "epochs" else "") / f.name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(f, target)
            got.append(f.name)
        n_new = sum(name.startswith("epoch") for name in got)
        print(f"  checkpoint → {dest}: {sorted(got)} ({n_new}/{len(missing)} epoch cần tải) — đưa lên Drive")
        if n_new < len(missing):
            print("  !!! thiếu checkpoint epoch trong output kernel — kiểm tra trước khi xoá kernel", flush=True)


if __name__ == "__main__":
    main()
