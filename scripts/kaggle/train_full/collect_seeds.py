"""Gom 6 run seed 43/44 nhóm 40 ms (L3, chốt 16/10) sau khi kernel xong — chạy local, từ gốc repo.

1. Trạng thái 6 kernel (`launch.py --tag seed<N>`); run chưa COMPLETE thì bỏ qua, báo.
2. Tải file nhỏ (json/jsonl/log/csv) vào `reports/results/train_full/<model>/seed<N>/` (CLAUDE.md: tải ngay khi xong).
3. Kiểm toàn vẹn từ `session_summary.json` — returncode, đủ 38 epoch, `finished`, step không hữu hạn, GPU-giờ.
   **Không in WER** (chỉ đọc trường không chứa "wer"): số chỉ xem ở lần chấm chung (note gap 9.6).
4. Tải `epoch37.pt` (chỉ trọng số, ~58 MB) vào `--ckpt_dir` (ngoài repo hoặc `checkpoints/`, gitignore).
5. Dựng thư mục dataset `<model>_seed<N>_ep37.pt` + `sets.json` (203 câu clean-test theo manifest + 100 câu val_unseen
   đầu để đối chiếu greedy) cho kernel `scripts/kaggle/dump_seeds_40ms/`; `--push` mới tạo/cập nhật dataset trên A.

    python scripts/kaggle/train_full/collect_seeds.py                      # 1–3, chỉ báo
    python scripts/kaggle/train_full/collect_seeds.py --ckpt --push        # + 4–5
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from launch import ACCOUNTS, slug  # noqa: E402

RUNS = [  # (model launch, tên ngắn trong dataset, tài khoản) — phân bổ chốt 16/10
    ("v2a-s4-conextbimamba", "conextbimamba", "A"),
    ("v2a-s4-conformer-relpos", "conformer_relpos", "B"),
    ("v2a-s4-mamba", "mamba", "C"),
]
SEEDS = (43, 44)
DATASET_SLUG = "kltn-v2a-s4-seeds-ep37"
EPOCHS = 38


def account_env(acc: str) -> tuple[str, dict]:
    user, config_dir = ACCOUNTS[acc]
    if user is None:
        user = json.loads((Path(config_dir) / "kaggle.json").read_text(encoding="utf-8"))["username"]
    env = {**os.environ, "PYTHONUTF8": "1", **({"KAGGLE_CONFIG_DIR": config_dir} if config_dir else {})}
    return user, env


def kaggle(args: list[str], env: dict) -> str:
    return subprocess.run(["kaggle", *args], env=env, capture_output=True, text=True, encoding="utf-8").stdout


def check_summary(path: Path) -> dict:
    s = json.loads(path.read_text(encoding="utf-8"))
    last = s.get("last_metrics", {})
    return {"returncode": s.get("returncode"), "epochs_done": s.get("epochs_done"), "finished": s.get("finished"),
            "latest_epoch": s.get("latest", {}).get("epoch"), "nonfinite": last.get("nonfinite_loss_steps"),
            "gpu_h": round(last.get("wall_seconds_total", 0) / 3600, 2), "commit": s.get("run", {}).get("commit", "")[:8],
            "extra": s.get("extra_args")}


def write_sets(path: Path) -> None:
    from src.data.vietsuperspeech_dataset import manifest_rows
    ct = json.loads((REPO / "data/processed/clean_test_manifest.json").read_text(encoding="utf-8"))["samples"]
    sets = {"clean_test": [{k: s[k] for k in ("split", "index", "audio")} for s in ct],
            "val_unseen_check": [{k: r[k] for k in ("split", "index", "audio")} for r in manifest_rows(["val_unseen"])[:100]]}
    path.write_text(json.dumps(sets, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="store_true", help="Tải epoch37.pt và dựng thư mục dataset")
    ap.add_argument("--ckpt_dir", type=Path, default=REPO / "checkpoints/v2a_s4_seeds")
    ap.add_argument("--push", action="store_true", help="Tạo/cập nhật dataset trên tài khoản A")
    args = ap.parse_args()
    os.chdir(REPO)

    ready, problems = [], []
    for model, short, acc in RUNS:
        user, env = account_env(acc)
        for seed in SEEDS:
            name = slug(model, 1, f"seed{seed}")
            status = re.search(r'status "?([\w.]+)', kaggle(["kernels", "status", f"{user}/{name}"], env))
            status = status.group(1).split(".")[-1] if status else "?"
            if status != "COMPLETE":
                print(f"{short} seed{seed} [{acc}]: {status} — bỏ qua")
                continue
            out = REPO / "reports/results/train_full" / model / f"seed{seed}"
            if not (out / "session_summary.json").exists():
                kaggle(["kernels", "output", f"{user}/{name}", "-p", str(out), "--file-pattern",
                        r".*\.(json|jsonl|log|csv)$"], env)
            info = check_summary(out / "session_summary.json")
            ok = (info["returncode"] == 0 and info["epochs_done"] == EPOCHS and info["finished"]
                  and info["latest_epoch"] == EPOCHS - 1 and f"--seed {seed}" in " ".join(info["extra"] or []))
            print(f"{short} seed{seed} [{acc}]: {'OK' if ok else 'CÓ VẤN ĐỀ'} {info}")
            (ready if ok else problems).append((short, seed, user, name, env))

    print(f"\n{len(ready)}/6 run sẵn sàng; {len(problems)} có vấn đề (chạy lại cùng seed — note gap 9.6 điều 1)")
    if not args.ckpt or not ready:
        return
    ds = args.ckpt_dir / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    for short, seed, user, name, env in ready:
        dst = ds / f"{short}_seed{seed}_ep37.pt"
        if dst.exists():
            continue
        tmp = args.ckpt_dir / f"{short}_seed{seed}"
        kaggle(["kernels", "output", f"{user}/{name}", "-p", str(tmp), "--file-pattern", r"epoch37\.pt$"], env)
        hits = sorted(tmp.rglob("epoch37.pt"))
        if not hits:
            print(f"LỖI: không tải được epoch37.pt của {short} seed{seed}")
            continue
        shutil.copy2(hits[0], dst)
        print(f"{dst.name}: {dst.stat().st_size / 2**20:.0f} MB")
    write_sets(ds / "sets.json")
    a_user, a_env = account_env("A")
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": DATASET_SLUG, "id": f"{a_user}/{DATASET_SLUG}", "licenses": [{"name": "CC0-1.0"}]}), encoding="utf-8")
    n = len(list(ds.glob("*_ep37.pt")))
    print(f"Dataset {ds}: {n} checkpoint + sets.json")
    if args.push:
        exists = "ready" in kaggle(["datasets", "status", f"{a_user}/{DATASET_SLUG}"], a_env)
        cmd = ["datasets", "version", "-p", str(ds), "-m", f"{n} checkpoint"] if exists else ["datasets", "create", "-p", str(ds)]
        print(kaggle(cmd, a_env).strip().splitlines()[-1:])


if __name__ == "__main__":
    main()
