"""Sinh + push kernel một phiên train full (`train_full.py`), nối phiên qua `kernel_sources`.

    python scripts/kaggle/train_full/launch.py --model conformer --session 1            # chỉ sinh, xem lại
    python scripts/kaggle/train_full/launch.py --model conformer --session 1 --push
    python scripts/kaggle/train_full/launch.py --model mamba --session 2 --account B --push

Bản sinh ra nằm ở `scripts/kaggle/train_full/_build/<slug>/` (gitignore — metadata chứa username).
Ghim `COMMIT` = HEAD local và từ chối nếu HEAD chưa có trên origin (kernel clone từ GitHub) hoặc cây
làm việc bẩn ở file code. Phiên k > 1 ghim **cùng commit với phiên 1** (đọc từ `_build/` phiên 1) trừ
khi `--commit` chỉ định khác.
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
REPO = HERE.parents[2]  # git chạy ở gốc repo: pathspec `src`/`configs` tương đối theo cwd
BUILD = HERE / "_build"
ACCOUNTS = {  # tài khoản → (username Kaggle, KAGGLE_CONFIG_DIR) — xem memory/kaggle CLI
    "A": ("tieunhi", None),
    "B": ("vuvanduc1", "C:/Users/Dell/.kaggle_Duc"),
}
MODELS = {"conformer": "configs/model_conformer.yaml", "mamba": "configs/model_mamba.yaml",
          "conextbimamba": "configs/model_conextbimamba.yaml"}
# Dataset do A sở hữu, đã chia sẻ B (TODO bước 5).
DATASETS = ["tieunhi/vss-asr-train-shard0", "tieunhi/vss-asr-train-shard1", "tieunhi/vss-asr-train-shard2",
            "tieunhi/vss-asr-train-shard3", "tieunhi/vss-asr-val", "tieunhi/mamba-wheels-v2"]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def slug(model: str, session: int, tag: str) -> str:
    return f"train-full{('-' + tag) if tag else ''}-{model}-s{session}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--account", default="A", choices=ACCOUNTS)
    ap.add_argument("--commit", default=None, help="Mặc định: phiên 1 = HEAD; phiên sau = commit của phiên 1")
    ap.add_argument("--session_minutes", type=int, default=540)
    ap.add_argument("--tag", default="", help="Tiền tố slug cho lần thử (vd. 'chain') — tách khỏi train thật")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Tham số thêm cho train.py (để cuối)")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    user, config_dir = ACCOUNTS[args.account]
    name = slug(args.model, args.session, args.tag)
    first = BUILD / slug(args.model, 1, args.tag) / "build.json"
    if args.commit:
        commit = args.commit
    elif args.session > 1 and first.exists():
        commit = json.loads(first.read_text(encoding="utf-8"))["commit"]
    else:
        commit = git("rev-parse", "HEAD")
    if git("branch", "-r", "--contains", commit) == "":
        sys.exit(f"LỖI: commit {commit[:8]} chưa có trên origin — push trước (kernel clone từ GitHub)")
    if args.session == 1 and not args.commit and git("status", "--porcelain", "--", "src", "configs", "scripts"):
        sys.exit("LỖI: src/configs/scripts có thay đổi chưa commit — kernel sẽ không chạy đúng code đang thấy")

    code = (HERE / "train_full.py").read_text(encoding="utf-8")
    subs = {"CONFIG": json.dumps(MODELS[args.model]), "SESSION": str(args.session), "COMMIT": json.dumps(commit),
            "SESSION_MINUTES": str(args.session_minutes), "EXTRA_ARGS": json.dumps(args.extra)}
    for key, value in subs.items():
        code, n = re.subn(rf"^{key}(: [^=]+)? = .*$", lambda m: f"{key}{m.group(1) or ''} = {value}", code,
                          count=1, flags=re.M)
        if n != 1:
            sys.exit(f"LỖI: không tìm thấy hằng {key} trong train_full.py")

    out = BUILD / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_full.py").write_text(code, encoding="utf-8")
    meta = {
        "id": f"{user}/{name}", "title": name, "code_file": "train_full.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": DATASETS, "competition_sources": [],
        "kernel_sources": [f"{user}/{slug(args.model, args.session - 1, args.tag)}"] if args.session > 1 else [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "build.json").write_text(json.dumps({"commit": commit, **subs}, indent=2), encoding="utf-8")
    print(f"{out}: {meta['id']} | commit {commit[:8]} | nguồn {meta['kernel_sources'] or '-'} | "
          f"{args.session_minutes} phút | extra {args.extra}")

    if args.push:
        import os
        env = {**os.environ, "PYTHONUTF8": "1", **({"KAGGLE_CONFIG_DIR": config_dir} if config_dir else {})}
        timeout = (args.session_minutes + 20) * 60
        subprocess.run(["kaggle", "kernels", "push", "-p", str(out), "-t", str(timeout)], env=env, check=True)
        print(f"Theo dõi: python scripts/kaggle/watch_kernel.py {meta['id']}")


if __name__ == "__main__":
    main()
