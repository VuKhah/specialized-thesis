"""Sinh + push kernel một phiên train full (`train_full.py`), nối phiên qua `kernel_sources`.

    python scripts/kaggle/train_full/launch.py --model conformer --session 1            # chỉ sinh, xem lại
    python scripts/kaggle/train_full/launch.py --model conformer --session 1 --push
    python scripts/kaggle/train_full/launch.py --model mamba --session 2 --account B --push

Công thức chung v2a (2026-10-12, "A + 40 ms" — `configs/v2a/`): bắt buộc `--account` (người dùng phân bổ):

    # (1) rẽ nhánh 8 epoch giảm từ latest.pt v1 sau epoch 29 (BRANCH: nguồn mặc định = output kernel v1)
    python scripts/kaggle/train_full/launch.py --model v2a-mamba --session 1 --account B
    python scripts/kaggle/train_full/launch.py --model v2a-conextbimamba --session 1 --account A
    python scripts/kaggle/train_full/launch.py --model v2a-conformer --session 1 --account A
    # (2) Conformer relPE từ đầu 38 epoch; (3) nhánh 40 ms từ đầu 38 epoch
    python scripts/kaggle/train_full/launch.py --model v2a-conformer-relpos --session 1 --account ?
    python scripts/kaggle/train_full/launch.py --model v2a-s4-mamba --session 1 --account ?   # tương tự s4-conextbimamba, s4-conformer-relpos

Kernel output riêng tư chỉ gắn được trong cùng tài khoản → nhánh (1) chạy ở tài khoản sở hữu kernel v1, hoặc
truyền `--branch_dataset <owner>/<slug>` (dataset chứa `<experiment v1>/latest.pt`, chia sẻ cho tài khoản chạy).

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
    # C (2026-10-12): người dùng cấp thêm; điền username + thư mục khoá khi có (không đọc nội dung khoá).
    "C": (None, "C:/Users/Dell/.kaggle_C"),
}
MODELS = {"conformer": "configs/model_conformer.yaml", "mamba": "configs/model_mamba.yaml",
          "conextbimamba": "configs/model_conextbimamba.yaml",
          # Công thức v2 (2026-10-12): bảng chính 10 ms, ablation PE, nhánh 40 ms.
          "v2-mamba": "configs/v2/model_mamba.yaml", "v2-conextbimamba": "configs/v2/model_conextbimamba.yaml",
          "v2-conformer-relpos": "configs/v2/model_conformer_relpos.yaml",
          "v2-conformer-nope": "configs/v2/model_conformer_nope.yaml",
          "v2s4-mamba": "configs/v2/model_mamba_sub4.yaml",
          "v2s4-conextbimamba": "configs/v2/model_conextbimamba_sub4.yaml",
          "v2s4-conformer-relpos": "configs/v2/model_conformer_relpos_sub4.yaml",
          # Công thức chung v2a (2026-10-12, thay v2): v1 + pha giảm 1-sqrt 8 epoch cuối, 38 epoch, không InterCTC.
          "v2a-mamba": "configs/v2a/model_mamba.yaml", "v2a-conextbimamba": "configs/v2a/model_conextbimamba.yaml",
          "v2a-conformer": "configs/v2a/model_conformer.yaml",
          "v2a-conformer-relpos": "configs/v2a/model_conformer_relpos.yaml",
          "v2a-s4-mamba": "configs/v2a/model_mamba_sub4.yaml",
          "v2a-s4-conextbimamba": "configs/v2a/model_conextbimamba_sub4.yaml",
          "v2a-s4-conformer-relpos": "configs/v2a/model_conformer_relpos_sub4.yaml"}
# Rẽ nhánh v2a: experiment + epoch của checkpoint v1 (latest.pt cuối epoch 29, có optimizer — đã kiểm
# `kaggle kernels files` 2026-10-12) và kernel v1 có output đó. Conformer còn bản trong dataset
# vuvanduc1/conformer-s1-ckpt (checkpoints/conformer_ctc_baseline/latest.pt, ep29) — dùng với --branch_dataset.
BRANCH = {"v2a-mamba": {"experiment": "mamba_ctc", "epoch": 29, "kernel": "vuvanduc1/train-full-mamba-s2"},
          "v2a-conextbimamba": {"experiment": "conextbimamba_ctc", "epoch": 29,
                                "kernel": "tieunhi/train-full-conextbimamba-s2"},
          "v2a-conformer": {"experiment": "conformer_ctc_baseline", "epoch": 29,
                            "kernel": "tieunhi/train-full-conformer-s1"}}
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
    ap.add_argument("--account", default=None, choices=ACCOUNTS, help="Mặc định A; bắt buộc với v2a-*")
    ap.add_argument("--branch_kernel", default=None, help="v2a nhánh (1): kernel v1 khác mặc định trong BRANCH")
    ap.add_argument("--branch_dataset", default=None,
                    help="v2a nhánh (1): gắn dataset chứa checkpoint v1 thay vì output kernel v1")
    ap.add_argument("--commit", default=None, help="Mặc định: phiên 1 = HEAD; phiên sau = commit của phiên 1")
    ap.add_argument("--session_minutes", type=int, default=540)
    ap.add_argument("--tag", default="", help="Tiền tố slug cho lần thử (vd. 'chain') — tách khỏi train thật")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Tham số thêm cho train.py (để cuối)")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    if args.account is None:
        if args.model.startswith("v2a-"):
            sys.exit("LỖI: v2a-* phải chỉ định --account (phân bổ tài khoản do người dùng chốt)")
        args.account = "A"
    user, config_dir = ACCOUNTS[args.account]
    if user is None:
        sys.exit(f"LỖI: tài khoản {args.account} chưa có username — điền ACCOUNTS trong launch.py")
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
            "SESSION_MINUTES": str(args.session_minutes), "EXTRA_ARGS": json.dumps(args.extra),
            "BRANCH_FROM": "None"}
    branch = BRANCH.get(args.model)
    branch_kernels, branch_datasets = [], []
    if branch:
        subs["BRANCH_FROM"] = json.dumps({"experiment": branch["experiment"], "epoch": branch["epoch"]})
        if args.session == 1:  # phiên sau resume latest.pt của chính nhánh, không cần nguồn v1
            if args.branch_dataset:
                branch_datasets = [args.branch_dataset]
            else:
                branch_kernels = [args.branch_kernel or branch["kernel"]]
                if branch_kernels[0].split("/")[0] != user:
                    print(f"CẢNH BÁO: {branch_kernels[0]} thuộc tài khoản khác {user} — kernel riêng tư không gắn "
                          "được; chạy bằng tài khoản sở hữu hoặc dùng --branch_dataset", flush=True)
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
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": DATASETS + branch_datasets, "competition_sources": [],
        "kernel_sources": ([f"{user}/{slug(args.model, args.session - 1, args.tag)}"] if args.session > 1 else [])
                          + branch_kernels,
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
