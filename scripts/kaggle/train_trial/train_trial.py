"""Kernel Kaggle **GPU** (2x T4) — train thử 3 mô hình nhóm train từ đầu.

Mục đích (Plan.md mục 4, 2026-10-03; điều chỉnh 2026-10-04): 1 shard
(`train_shard0`, 11.364 câu sau lọc), 5 epoch, 1 seed cho Mamba B1,
ConExtBiMamba và Conformer-12M → xem cả 3 có train ổn định không, rồi người
dùng chốt giữ 2 hay 3 mô hình. Đồng thời là lần đầu chạy `train.py` mới trên GPU
(DDP/NCCL, SyncBN, AMP, khối Mamba thật).

    kaggle kernels push -p scripts/kaggle/train_trial -t 14400
    kaggle kernels status <username>/train-trial-asr
    kaggle kernels output <username>/train-trial-asr -p <thư mục>   (PYTHONUTF8=1)

Ước ~2,5-3,5 GPU-giờ (số đo benchmark lần 2 + eval). Dữ liệu: output 5 kernel
make-dataset-asr-* (kernel_sources) — đủ 5 tar vì val_unseen nằm rải cả 4 shard.
Wheel mamba: output kernel verify-mamba-asr.

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
import subprocess
import sys
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


def run(cmd, cwd=None, check=True, env=None) -> tuple[int, str]:
    """In output ngay khi có (train dài hàng giờ — log Kaggle phải thấy tiến độ), trả về (returncode, output)."""
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    lines = []
    for line in proc.stdout:
        # Bỏ cảnh báo lặp của tensorflow/oneDNN khi import tensorboard.
        if "oneDNN" not in line and "tensorflow" not in line:
            print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    if check and proc.returncode != 0:
        sys.exit(f"LỖI: lệnh trên trả về {proc.returncode}")
    return proc.returncode, "".join(lines)


def train_cmd(config, ckpt_dir, max_minutes, epochs=EPOCHS, extra=()):
    return [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "-m", "src.training.train",
            "--config", config, "--train_manifests", *TRAIN_MANIFESTS, "--epochs", str(epochs),
            "--max_minutes", str(max_minutes), "--ckpt_dir", str(ckpt_dir), "--log_dir", str(OUT / "runs"),
            "--ckpt_every_minutes", "10", *extra]


def main():
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
    run(train_cmd(conext, smoke, 3, epochs=1), cwd=REPO_DIR, env=env)
    _, out = run(train_cmd(conext, smoke, 3, epochs=1), cwd=REPO_DIR, env=env)
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
