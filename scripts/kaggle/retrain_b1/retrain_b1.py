"""Kernel Kaggle **GPU** (2x T4) — train thử LẠI Mamba B1 sau chẩn đoán NaN (2026-10-05).

Train thử 2026-10-05 (`train_trial.py`): B1 bỏ 1318/3555 step vì loss không hữu hạn, 41,6% câu
val_unseen rỗng; ConExtBiMamba và Conformer ổn. Chẩn đoán: kernel `diag-b1-asr`
(`scripts/kaggle/diag_b1/`). Kernel này train lại **chỉ B1** với cùng điều kiện train thử (shard0, 5
epoch, seed 42, 2 GPU DDP + AMP) để so được với hai mô hình kia, cộng phần sửa trong `OVERLAY`
(file của repo bị ghi đè sau khi clone — chưa commit).

    kaggle kernels push -p scripts/kaggle/retrain_b1 -t 7200
    python scripts/kaggle/watch_kernel.py <user>/retrain-b1-asr

`kernel-metadata.json` (gitignore): `dataset_sources` như train_trial; `kernel_sources` =
`<user>/train-trial-asr` → chép metrics/eval của ConExtBiMamba + Conformer (không chép checkpoint)
vào `OUT/trial` để `trial_report` so cả 3 như lần trước. Ước ~45 phút GPU (setup ~12′ + B1 ~30′).
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
CONFIGS = ["configs/model_mamba.yaml"]
PREV_EXPERIMENTS = ["conextbimamba_ctc", "conformer_ctc_baseline"]  # lấy kết quả từ train thử 2026-10-05
TRIAL_MAX_MINUTES = 90  # trần mỗi mô hình; ước 30-45 phút train + ~10 phút eval
# Lệnh im lặng quá chừng này phút thì coi là treo (vd. DDP/NCCL kẹt) → kill cả nhóm tiến trình.
# train.py in tiến độ mỗi 50 step (~0,5 phút) và mỗi 50 batch eval; preflight B 2026-10-04 im lâu nhất
# ~1,6 phút (làm ấm cache HF). Hạ 20 → 10 phút: mỗi lần treo đốt ít GPU chạy không hơn.
SILENCE_MINUTES = 10
LOG_DIR = OUT / "logs"
# Môi trường wheel mamba được build cho — kernel env-check-asr 2026-10-04 (phương án A: image mặc
# định, causal-conv1d 1.5.4 + mamba-ssm 2.3.1 build từ mã nguồn, chỉ sm_75). Image đổi → chạy lại env_check.
EXPECT_PY, EXPECT_TORCH = "3.13", "2.11"
# True = chỉ chạy phần kiểm tra (môi trường → CHECK_CODE → dữ liệu → pipeline tí hon → resume) rồi
# dừng, không train: dùng cho tài khoản mới / môi trường mới (quy trình mục 6, ~15 phút GPU).
PREFLIGHT_ONLY = False
# Pipeline tí hon: mỗi mô hình 1 epoch trên N câu, eval N câu/tập — đi qua train → eval DDP →
# best.pt → metrics.jsonl → trial_report như thật; hỏng thì dừng sau ~1 phút/mô hình.
PIPELINE_ROWS = 64
# Ghi đè file của repo sau khi clone (như benchmark.py) — để trống khi code đã push.
# Sửa 2026-10-05 (chưa commit): khối Mamba chạy fp32 — mamba_encoder.mamba_fp32.
OVERLAY: dict[str, str] = {}  # mamba_fp32 đã commit — để trống khi code đã push

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


class Tee:
    """Mọi print vừa ra log Kaggle vừa vào file text, flush từng dòng. Gắn cho cả stderr: thông báo
    `sys.exit("LỖI ...")` và traceback đi ra stderr — thiếu thì file log mất đúng dòng cần đọc."""

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
        self.file.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):  # isatty, fileno, encoding... của stream gốc
        return getattr(self.stream, name)


def stamp() -> str:
    return f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:.0f}′]"


def find_wheels(root: Path = Path("/kaggle/input")) -> list[str]:
    """Chỉ tìm trong thư mục dataset mamba-wheels*: rglob cả /kaggle/input duyệt ~67 nghìn file wav trên
    ổ mount, mất ~3 phút GPU chạy không (preflight B 2026-10-04). Dataset gắn ở /kaggle/input/<slug>/
    hoặc /kaggle/input/datasets/<user>/<slug>/ tuỳ phiên bản Kaggle."""
    return sorted({str(p) for pat in ("mamba-wheels*/**/*.whl", "*/*/mamba-wheels*/**/*.whl")
                   for p in root.glob(pat)})


def start_gpu_monitor() -> subprocess.Popen | None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        return subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
             "--format=csv", "-l", "30"], stdout=open(LOG_DIR / "gpu_util.csv", "w"), stderr=subprocess.DEVNULL)
    except OSError as e:
        print(f"Không chạy được nvidia-smi để ghi gpu_util.csv: {e}", flush=True)
        return None


def run(cmd, cwd=None, check=True, env=None, log_name: str | None = None) -> tuple[int, str]:
    """In output ngay khi có (xem trực tiếp bằng `kaggle kernels logs -f`), trả về (returncode, output).
    Watchdog: im lặng quá SILENCE_MINUTES thì kill cả nhóm tiến trình (torchrun + 2 worker).
    `log_name`: chép thêm output vào LOG_DIR/<log_name>.log (log riêng từng lần train)."""
    print(f"\n{stamp()} $ {' '.join(map(str, cmd))}", flush=True)
    log_file = open(LOG_DIR / f"{log_name}.log", "a", encoding="utf-8") if log_name else None
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
            if log_file:
                log_file.write(line)
                log_file.flush()
        lines.append(line)
    proc.wait()
    print(f"{stamp()} → returncode {proc.returncode}", flush=True)
    if log_file:
        log_file.write(f"{stamp()} → returncode {proc.returncode}\n")
        log_file.close()
    if check and proc.returncode != 0:
        sys.exit(f"LỖI: lệnh trên trả về {proc.returncode}")
    return proc.returncode, "".join(lines)


def train_cmd(config, ckpt_dir, max_minutes, epochs=EPOCHS, log_dir=OUT / "runs", extra=()):
    return [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "-m", "src.training.train",
            "--config", config, "--train_manifests", *TRAIN_MANIFESTS, "--epochs", str(epochs),
            "--max_minutes", str(max_minutes), "--ckpt_dir", str(ckpt_dir), "--log_dir", str(log_dir),
            "--ckpt_every_minutes", "10", *extra]


def copy_previous_results(dest: Path) -> None:
    """metrics.jsonl + eval_*.jsonl của hai mô hình đã train ổn ở train thử 2026-10-05 (không chép .pt)."""
    import glob
    import shutil
    for exp in PREV_EXPERIMENTS:
        hits = [h for d in range(1, 4) for h in glob.glob("/kaggle/input/" + "*/" * d + f"trial/{exp}/metrics.jsonl")]
        if not hits:
            print(f"!!! không thấy kết quả cũ của {exp} trong /kaggle/input (kernel_sources)", flush=True)
            continue
        src = Path(hits[0]).parent
        (dest / exp).mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.jsonl"):
            shutil.copyfile(f, dest / exp / f.name)
        print(f"chép kết quả cũ {exp}: {len(list(src.glob('*.jsonl')))} file", flush=True)


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
    wheels = find_wheels()
    if not wheels:
        run(["find", "/kaggle/input", "-maxdepth", "4"], check=False)
        sys.exit("LỖI: không thấy wheel mamba trong /kaggle/input — kiểm tra dataset mamba-wheels-v2")
    run(pip + ["--no-deps"] + wheels)
    run([sys.executable, "-c", CHECK_CODE], cwd=REPO_DIR)

    run([sys.executable, "-m", "src.data.extract_audio", "--input", "/kaggle/input", "--dest", AUDIO_DIR],
        cwd=REPO_DIR)
    run([sys.executable, "-m", "src.data.vietsuperspeech_dataset"], cwd=REPO_DIR)  # làm ấm cache HF
    env = {**os.environ, "AUDIO_CACHE_DIR": AUDIO_DIR, "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}

    # Pipeline tí hon cho cả 3 mô hình (check=True: hỏng là dừng kernel, chưa tốn giờ train).
    pipe = Path("/tmp/pipeline")
    for config in CONFIGS:
        _, out = run(train_cmd(config, pipe, 10, epochs=1, log_dir=pipe / "runs",
                               extra=("--limit_train", str(PIPELINE_ROWS), "--limit_eval", str(PIPELINE_ROWS),
                                      "--log_every_steps", "1", "--no_resume")), cwd=REPO_DIR, env=env)
        if "eval val" not in out or "WER=" not in out:
            sys.exit(f"LỖI: pipeline tí hon {config} không đi tới eval")
    run([sys.executable, "-m", "src.evaluation.trial_report", str(pipe), "--out", str(pipe / "report")],
        cwd=REPO_DIR, env=env)
    for exp in sorted(p.name for p in pipe.iterdir() if (p / "metrics.jsonl").exists()):
        print(f"  pipeline {exp}: " + ", ".join(sorted(f.name for f in (pipe / exp).iterdir())), flush=True)

    if PREFLIGHT_ONLY:
        print(f"\n{stamp()} PREFLIGHT OK — dừng, không train (PREFLIGHT_ONLY)", flush=True)
        return

    for config in CONFIGS:
        code, _ = run(train_cmd(config, OUT / "trial", TRIAL_MAX_MINUTES), cwd=REPO_DIR, env=env, check=False,
                      log_name=Path(config).stem.removeprefix("model_"))
        if code != 0:
            print(f"!!! {config} lỗi (returncode {code}) — chạy tiếp mô hình sau", flush=True)

    copy_previous_results(OUT / "trial")
    # Bảng so sánh + cổng G0-G3 (src/evaluation/trial_report.py) → trial_report.{md,json}.
    run([sys.executable, "-m", "src.evaluation.trial_report", str(OUT / "trial"), "--out", str(OUT)],
        cwd=REPO_DIR, env=env, check=False)
    print("\nTRAIN THỬ XONG", flush=True)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(LOG_DIR / "train_trial.log", "a", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.stdout, _log_file), Tee(sys.stderr, _log_file)
    monitor = start_gpu_monitor()
    try:
        main()
    finally:
        # Kể cả khi sys.exit vì lỗi: không để nvidia-smi -l chạy tiếp sau khi script xong.
        if monitor:
            monitor.terminate()
        print(f"{stamp()} kết thúc kernel", flush=True)
