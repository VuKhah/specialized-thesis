"""Kernel Kaggle **GPU** (2x T4) — train thử 3 mô hình nhóm train từ đầu.

Mục đích (Plan.md mục 4, 2026-10-03; điều chỉnh 2026-10-04): 1 shard
(`train_shard0`, 11.364 câu sau lọc), 5 epoch, 1 seed cho Mamba B1,
ConExtBiMamba và Conformer-12M → xem cả 3 có train ổn định không, rồi người
dùng chốt giữ 2 hay 3 mô hình. Đồng thời là lần đầu chạy `train.py` mới trên GPU
(DDP/NCCL, SyncBN, AMP, khối Mamba thật).

    kaggle kernels push -p scripts/kaggle/train_trial -t 19800   (trường hợp xấu nhất ~5 h: setup + 3 × trần 90 phút)
    kaggle kernels status <username>/train-trial-asr
    kaggle kernels output <username>/train-trial-asr -p <thư mục>   (PYTHONUTF8=1)

Ước ~2,5-3,5 GPU-giờ (số đo benchmark lần 2 + eval). Dữ liệu: output 5 kernel
make-dataset-asr-* (kernel_sources) — đủ 5 tar vì val_unseen nằm rải cả 4 shard.
Wheel mamba: output kernel verify-mamba-asr.

**Ghim image (2026-10-04):** image GPU mặc định của Kaggle đã lên Python 3.13 → pip từ chối
wheel cp312 (lần chạy đầu lỗi ở bước cài wheel). `kernel-metadata.json` (gitignore) phải có
`docker_image` = image của kernel verify-mamba-asr (Python 3.12, torch 2.10+cu128 — lấy bằng
`kaggle kernels pull <user>/verify-mamba-asr -m`). Dữ liệu + wheel gắn qua `dataset_sources`
(`vss-asr-*`, `mamba-wheels`). Trạng thái RUNNING của CLI gồm cả lúc chờ cấp máy — không
suy ra script đã chạy tới đâu.

Thứ tự (dừng sớm nếu lỗi code, không đốt quota):
1. CHECK_CODE: tham số + bất biến padding với kernel CUDA thật cho B1 và
   ConExtBiMamba (fp32 và AMP).
2. Chạy ngắn DDP: ConExtBiMamba `--max_minutes 3` rồi resume thêm 3 phút →
   kiểm checkpoint giữa epoch + resume trên GPU (có cả BatchNorm và Mamba).
3. Train thử lần lượt B1 → ConExtBiMamba → Conformer, mỗi mô hình có trần
   `TRIAL_MAX_MINUTES` (hết trần thì lưu latest.pt, chạy lại kernel để resume).
   Một mô hình lỗi thì ghi lại rồi chạy tiếp mô hình sau.
4. Bảng so sánh (chất lượng, học/ổn định, tài nguyên) + cổng loại cứng →
   trial_report.md / .json (src/evaluation/trial_report.py — chạy lại được ở
   local trên output đã tải về).
"""

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_URL = "https://github.com/VuKhah/specialized-thesis"
REPO_DIR = Path("/tmp/specialized-thesis")
AUDIO_DIR = "/tmp/audio_cache"
OUT = Path("/kaggle/working")
EPOCHS = 5
TRAIN_MANIFESTS = ["train_shard0"]
CONFIGS = ["configs/model_mamba.yaml", "configs/model_conextbimamba.yaml", "configs/model_conformer.yaml"]
TRIAL_MAX_MINUTES = 90  # trần mỗi mô hình; ước 30-45 phút train + ~10 phút eval
# Lệnh im lặng quá chừng này phút thì coi là treo (vd. DDP/NCCL kẹt) → kill cả nhóm tiến trình.
# train.py in tiến độ mỗi 50 step (~0,5-1 phút) và mỗi 50 batch eval, nên 20 phút im lặng là bất thường.
SILENCE_MINUTES = 20
# Môi trường mà wheel mamba được build cho (CHƯA CHỐT sau sự cố 2026-10-04 — đổi theo phương án
# người dùng chọn: A build lại trên image mặc định, B cài torch 2.10 + wheel dựng sẵn).
EXPECT_PY, EXPECT_TORCH = "3.12", "2.10"
# Ghi đè file của repo sau khi clone (như benchmark.py) — để trống khi code đã push.
OVERLAY: dict[str, str] = {}

CHECK_CODE = r'''
import sys, yaml, torch
from src.training.train import build_encoder, build_optimizer
sys.stdout.reconfigure(encoding="utf-8")
# (config, số tham số tính trên khối giả cùng shape mamba-ssm v2.3.1, số tham số miễn weight decay)
EXPECTED = [("configs/model_mamba.yaml", 12_285_696, 56), ("configs/model_conextbimamba.yaml", 12_241_536, 24)]
ok = True
for path, n_expected, nd_expected in EXPECTED:
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    torch.manual_seed(0)
    enc = build_encoder(cfg).cuda().eval()
    n, nd = enc.num_parameters(), len(build_optimizer(enc, cfg).param_groups[1]["params"])
    print(f"{path}: {n:,} tham số (kỳ vọng {n_expected:,}), no_decay {nd} (kỳ vọng {nd_expected})")
    ok = ok and n == n_expected and nd == nd_expected
    T = 1500
    feats = torch.randn(3, T, 80, device="cuda")
    lengths = torch.tensor([T, 1100, 900], device="cuda")
    feats[1, 1100:] = 1e3 * torch.randn(T - 1100, 80, device="cuda")
    for amp in (False, True):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            batch, _ = enc(feats, lengths)
            alone, _ = enc(feats[1:2, :1100].contiguous(), lengths[1:2])
            f2 = feats.clone(); f2[0, 1000] += 1.0
            moved, _ = enc(f2, lengths)
        d_pad = (batch[1, :1100] - alone[0]).abs().max().item()
        d_future = (moved[0, 500] - batch[0, 500]).abs().max().item()
        finite = bool(torch.isfinite(batch[:, :900]).all())
        tol = 5e-2 if amp else 1e-3
        print(f"  amp={amp}: |riêng - trong batch| = {d_pad:.2e} (< {tol}), khung 1000 → khung 500 lệch "
              f"{d_future:.2e} (> 0), hữu hạn {finite}")
        ok = ok and d_pad < tol and d_future > 0 and finite
print("CHECK " + ("OK" if ok else "FAIL"), flush=True)
sys.exit(0 if ok else 1)
'''


T0 = time.time()


def stamp() -> str:
    return f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:.0f}′]"


def run(cmd, cwd=None, check=True, env=None) -> tuple[int, str]:
    """In output ngay khi có (xem trực tiếp bằng `kaggle kernels logs -f`), trả về (returncode, output).
    Watchdog: im lặng quá SILENCE_MINUTES thì kill cả nhóm tiến trình (torchrun + 2 worker)."""
    print(f"\n{stamp()} $ {' '.join(map(str, cmd))}", flush=True)
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", start_new_session=True)
    lines_q: queue.Queue = queue.Queue()

    def pump():
        for line in proc.stdout:
            lines_q.put(line)
        lines_q.put(None)

    threading.Thread(target=pump, daemon=True).start()
    lines, last = [], time.time()
    while True:
        try:
            line = lines_q.get(timeout=30)
        except queue.Empty:
            if time.time() - last > SILENCE_MINUTES * 60:
                print(f"{stamp()} !!! im lặng {SILENCE_MINUTES} phút — coi là treo, kill nhóm tiến trình", flush=True)
                os.killpg(proc.pid, signal.SIGKILL)
                break
            continue
        if line is None:
            break
        last = time.time()
        # Bỏ cảnh báo lặp của tensorflow/oneDNN khi import tensorboard.
        if "oneDNN" not in line and "tensorflow" not in line:
            print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    print(f"{stamp()} → returncode {proc.returncode}", flush=True)
    if check and proc.returncode != 0:
        sys.exit(f"LỖI: lệnh trên trả về {proc.returncode}")
    return proc.returncode, "".join(lines)


def train_cmd(config, ckpt_dir, max_minutes, epochs=EPOCHS, log_dir=OUT / "runs", extra=()):
    return [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "-m", "src.training.train",
            "--config", config, "--train_manifests", *TRAIN_MANIFESTS, "--epochs", str(epochs),
            "--max_minutes", str(max_minutes), "--ckpt_dir", str(ckpt_dir), "--log_dir", str(log_dir),
            "--ckpt_every_minutes", "10", *extra]


def check_env() -> None:
    """Dòng đầu tiên của log: phiên bản môi trường; lệch EXPECT_* thì dừng ngay (vài giây quota)
    thay vì lỗi ở bước cài wheel. Kaggle tự đổi image mặc định giữa các lần chạy (sự cố 2026-10-04)."""
    import platform
    import torch
    py = ".".join(platform.python_version_tuple()[:2])
    print(f"{stamp()} MÔI TRƯỜNG: python {platform.python_version()} | torch {torch.__version__} | "
          f"CUDA torch {torch.version.cuda} | GPU {torch.cuda.device_count()}x "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}", flush=True)
    if py != EXPECT_PY or not torch.__version__.startswith(EXPECT_TORCH) or torch.cuda.device_count() != 2:
        sys.exit(f"LỖI MÔI TRƯỜNG: cần python {EXPECT_PY}, torch {EXPECT_TORCH}.*, 2 GPU — "
                 "image đã đổi hoặc chưa ghim đúng; wheel mamba phải build cho đúng bộ này")


def main():
    check_env()
    run(["nvidia-smi"], check=False)
    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    run(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)
    for rel, content in OVERLAY.items():
        (REPO_DIR / rel).write_text(content, encoding="utf-8")
        print(f"OVERLAY: ghi đè {rel} ({len(content)} ký tự)", flush=True)
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run(pip + ["einops", "librosa", "soundfile", "sentencepiece", "datasets", "jiwer", "pyyaml", "tensorboard"])
    wheels = sorted(str(p) for p in Path("/kaggle/input").rglob("*.whl"))
    if not wheels:
        run(["find", "/kaggle/input", "-maxdepth", "4"], check=False)
        sys.exit("LỖI: không thấy wheel mamba trong /kaggle/input — kiểm tra kernel_sources")
    run(pip + ["--no-deps"] + wheels)
    run([sys.executable, "-c", CHECK_CODE], cwd=REPO_DIR)

    run([sys.executable, "-m", "src.data.extract_audio", "--input", "/kaggle/input", "--dest", AUDIO_DIR],
        cwd=REPO_DIR)
    run([sys.executable, "-m", "src.data.vietsuperspeech_dataset"], cwd=REPO_DIR)  # làm ấm cache HF
    env = {**os.environ, "AUDIO_CACHE_DIR": AUDIO_DIR, "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}

    # Chạy ngắn: dừng giữa epoch 0 rồi resume — kiểm DDP + checkpoint + resume trên GPU.
    smoke = Path("/tmp/smoke")
    conext = "configs/model_conextbimamba.yaml"
    # TensorBoard của lần chạy ngắn để ở /tmp — cùng experiment_name nên ghi vào OUT/runs sẽ lẫn với train thử.
    run(train_cmd(conext, smoke, 3, epochs=1, log_dir=smoke / "runs"), cwd=REPO_DIR, env=env)
    _, out = run(train_cmd(conext, smoke, 3, epochs=1, log_dir=smoke / "runs"), cwd=REPO_DIR, env=env)
    if "resume từ" not in out:
        sys.exit("LỖI: lần chạy thứ hai không resume từ checkpoint")

    for config in CONFIGS:
        code, _ = run(train_cmd(config, OUT / "trial", TRIAL_MAX_MINUTES), cwd=REPO_DIR, env=env, check=False)
        if code != 0:
            print(f"!!! {config} lỗi (returncode {code}) — chạy tiếp mô hình sau", flush=True)

    # Bảng so sánh + cổng G0-G3 (src/evaluation/trial_report.py) → trial_report.{md,json}.
    run([sys.executable, "-m", "src.evaluation.trial_report", str(OUT / "trial"), "--out", str(OUT)],
        cwd=REPO_DIR, env=env, check=False)
    print("\nTRAIN THỬ XONG", flush=True)


if __name__ == "__main__":
    main()
