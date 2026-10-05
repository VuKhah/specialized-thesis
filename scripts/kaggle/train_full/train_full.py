"""Kernel Kaggle **GPU** (2x T4) — train full 1 mô hình, 1 phiên; phiên sau resume từ output phiên trước.

Mỗi phiên là một kernel riêng `<user>/train-full-<model>-s<k>`: Kaggle không cho kernel gắn output của
chính nó, nên phiên k gắn output phiên k-1 qua `kernel_sources` (D7, `training_plan_kaggle.md`). Đừng
push tay — dùng `launch.py` (điền các hằng dưới đây, sinh `kernel-metadata.json`, push):

    python scripts/kaggle/train_full/launch.py --model conformer --session 1 --push
    python scripts/kaggle/watch_kernel.py <user>/train-full-conformer-s1

Mọi phiên của một mô hình chạy **cùng commit** (`COMMIT`, launcher ghim = HEAD local đã push) để code
không đổi giữa chừng. Phiên k > 1 không thấy `latest.pt` của phiên trước thì dừng ngay — không bao giờ
lặng lẽ train lại từ đầu.

Output (`/kaggle/working`): `checkpoints/<experiment>/` (latest.pt, best.pt, metrics.jsonl,
eval_*.jsonl — cộng dồn qua các phiên), `runs/` (tensorboard), `logs/` (log text, gpu_util.csv),
`session_summary.json`.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# --- launch.py điền các hằng này ---
CONFIG = "configs/model_conformer.yaml"
SESSION = 1
COMMIT = ""  # rỗng = HEAD của master (chỉ dùng khi thử tay)
SESSION_MINUTES = 540  # D7: phiên GPU ~9 h; train.py dừng sớm hơn TAIL_MARGIN để kịp lưu + ghi output
EXTRA_ARGS: list[str] = []  # vd. ["--limit_eval", "64"] khi thử nối phiên
# ------------------------------------

REPO_URL = "https://github.com/VuKhah/specialized-thesis"
REPO_DIR = Path("/tmp/specialized-thesis")
AUDIO_DIR = "/tmp/audio_cache"
OUT = Path("/kaggle/working")
# Eval cuối epoch có thể bắt đầu ngay trước hạn (train.py chỉ xét giờ giữa các step); B1 fp32 eval
# val + val_unseen ~10 phút (train thử ×4 dữ liệu không đổi tập eval) → chừa 30 phút.
TAIL_MARGIN = 30
T_KERNEL = time.time()


def bootstrap() -> None:
    subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO_DIR)], check=True)
    if COMMIT:
        subprocess.run(["git", "checkout", "-q", COMMIT], cwd=REPO_DIR, check=True)
    sys.path.insert(0, str(REPO_DIR))


def find_previous(experiment: str) -> Path | None:
    """Thư mục checkpoint của phiên trước trong /kaggle/input. Glob có giới hạn độ sâu: rglob cả
    /kaggle/input duyệt ~67 nghìn wav (~3 phút GPU chạy không)."""
    hits = [h for d in range(1, 5) for h in glob.glob("/kaggle/input/" + "*/" * d + f"checkpoints/{experiment}/latest.pt")]
    return Path(hits[0]).parent if hits else None


def main() -> None:
    from scripts.kaggle.train_trial.train_trial import (LOG_DIR, check_env, find_wheels, run, stamp)
    import yaml

    check_env()
    run(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)
    cfg = yaml.safe_load((REPO_DIR / CONFIG).read_text(encoding="utf-8"))
    experiment, epochs = cfg["experiment_name"], cfg["training"]["epochs"]
    if "--epochs" in EXTRA_ARGS:  # thử nối phiên ghi đè số epoch — tổng kết phải theo cùng con số
        epochs = int(EXTRA_ARGS[EXTRA_ARGS.index("--epochs") + 1])
    ckpt_root = OUT / "checkpoints"
    print(f"{stamp()} TRAIN FULL {experiment} — phiên {SESSION}, {CONFIG}, {epochs} epoch", flush=True)

    if SESSION > 1:
        prev = find_previous(experiment)
        if prev is None:
            run(["find", "/kaggle/input", "-maxdepth", "5", "-name", "*.pt"], check=False)
            sys.exit(f"LỖI: phiên {SESSION} nhưng không thấy checkpoints/{experiment}/latest.pt của phiên trước "
                     "— kiểm tra kernel_sources")
        shutil.copytree(prev, ckpt_root / experiment, dirs_exist_ok=True)
        prev_runs = prev.parent.parent / "runs"
        if prev_runs.is_dir():
            shutil.copytree(prev_runs, OUT / "runs", dirs_exist_ok=True)
        lines = (ckpt_root / experiment / "metrics.jsonl").read_text(encoding="utf-8").splitlines() \
            if (ckpt_root / experiment / "metrics.jsonl").exists() else []
        print(f"{stamp()} chép phiên trước từ {prev}: {sorted(p.name for p in prev.iterdir())}; "
              f"{len(lines)} epoch đã xong", flush=True)
        if lines and json.loads(lines[-1])["epoch"] >= epochs - 1:
            print(f"{stamp()} TRAIN FULL ĐÃ XONG từ phiên trước — không train thêm", flush=True)
            return

    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run(pip + ["einops", "librosa", "soundfile", "sentencepiece", "datasets", "jiwer", "pyyaml", "tensorboard"])
    wheels = find_wheels()
    if not wheels:
        sys.exit("LỖI: không thấy wheel mamba trong /kaggle/input — kiểm tra dataset mamba-wheels-v2")
    run(pip + ["--no-deps"] + wheels)
    run([sys.executable, "-m", "src.data.extract_audio", "--input", "/kaggle/input", "--dest", AUDIO_DIR],
        cwd=REPO_DIR)
    run([sys.executable, "-m", "src.data.vietsuperspeech_dataset"], cwd=REPO_DIR)  # làm ấm cache HF
    env = {**os.environ, "AUDIO_CACHE_DIR": AUDIO_DIR, "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}

    # --max_minutes của train.py tính từ lúc train.py khởi động → trừ phần setup đã tiêu.
    budget = SESSION_MINUTES - (time.time() - T_KERNEL) / 60 - TAIL_MARGIN
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "-m", "src.training.train",
           "--config", CONFIG, "--max_minutes", f"{budget:.0f}", "--ckpt_dir", str(ckpt_root),
           "--log_dir", str(OUT / "runs"), "--ckpt_every_minutes", "20", *EXTRA_ARGS]
    code, out = run(cmd, cwd=REPO_DIR, env=env, check=False, log_name=f"{experiment}_s{SESSION}")
    if SESSION > 1 and "resume từ" not in out:
        print("!!! phiên sau không resume — kiểm tra log", flush=True)

    (ckpt_root / experiment / "latest.tmp").unlink(missing_ok=True)
    metrics_path = ckpt_root / experiment / "metrics.jsonl"
    metrics = [json.loads(l) for l in metrics_path.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if metrics_path.exists() else []
    state_epoch = None
    if (ckpt_root / experiment / "latest.pt").exists():
        import torch
        st = torch.load(ckpt_root / experiment / "latest.pt", map_location="cpu", weights_only=False)
        state_epoch = {"epoch": st["epoch"], "step_in_epoch": st["step_in_epoch"], "global_step": st["global_step"],
                       "best_wer": st.get("best_wer")}
    done = bool(metrics) and metrics[-1]["epoch"] >= epochs - 1
    summary = {"experiment": experiment, "session": SESSION, "config": CONFIG, "returncode": code,
               "epochs_target": epochs, "epochs_done": len(metrics), "latest": state_epoch, "finished": done,
               "kernel_minutes": (time.time() - T_KERNEL) / 60, "last_metrics": metrics[-1] if metrics else None}
    (OUT / "session_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\nTRAIN FULL XONG" if done else f"\nPHIÊN {SESSION} XONG — chạy phiên {SESSION + 1}", flush=True)
    if code != 0:
        sys.exit(f"LỖI: train.py trả về {code} (checkpoint đã lưu vẫn nằm trong output)")


if __name__ == "__main__":
    bootstrap()
    from scripts.kaggle.train_trial.train_trial import LOG_DIR, Tee, start_gpu_monitor, stamp
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(LOG_DIR / f"train_full_s{SESSION}.log", "a", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.stdout, _log_file), Tee(sys.stderr, _log_file)
    monitor = start_gpu_monitor()
    try:
        main()
    finally:
        if monitor:
            monitor.terminate()
        print(f"{stamp()} kết thúc kernel", flush=True)
