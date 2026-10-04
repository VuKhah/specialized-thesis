"""Kernel Kaggle **GPU** (2x T4) — kiểm môi trường + build wheel mamba trước train thử.

Sau sự cố 2026-10-04/05 (`docs/notes/training_plan_kaggle.md` mục 6): image GPU mặc định
đã lên Python 3.13, wheel cp312 cũ không cài được; ghim image cũ thì máy GPU không chạy
được script. Phương án A (người dùng chọn 2026-10-05): **giữ mamba-ssm v2.3.1 +
causal-conv1d v1.5.4**, build lại từ mã nguồn trên image mặc định.

    kaggle kernels push -p scripts/kaggle/env_check -t 3600
    python scripts/kaggle/watch_kernel.py <username>/env-check-asr

Thứ tự (dừng ở bước hỏng đầu tiên, log cho biết vì sao):
1. In môi trường thật của image GPU (python, torch, CUDA torch, nvcc, số GPU).
2. Build 2 wheel từ tag đã ghim, `--no-build-isolation` như lần 2026-09-24, nhưng **chỉ
   sm_75** (T4): `setup.py` viết cứng ~10 kiến trúc → vá danh sách cờ `-gencode`, không
   đụng mã CUDA. Bắt buộc build (FORCE_BUILD): torch của image không có wheel dựng sẵn.
3. CHECK_CODE của kernel train thử (đọc từ repo — một nguồn duy nhất) cho B1 + ConExt.
4. Mỗi mô hình (B1, ConExt, Conformer) chạy DDP 2 GPU + AMP ~3 phút trên train_shard0 —
   kiểm cả torchaudio/Conformer trên image mới.
5. `env.json` (phiên bản → EXPECT_* của kernel train) + wheel trong /kaggle/working/wheels.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
T0 = time.time()
REPO_URL = "https://github.com/VuKhah/specialized-thesis"
REPO_DIR = Path("/tmp/specialized-thesis")
OUT = Path("/kaggle/working")
WHEEL_DIR = OUT / "wheels"
BUILDS = [("causal-conv1d", "https://github.com/Dao-AILab/causal-conv1d", "v1.5.4", "CAUSAL_CONV1D_FORCE_BUILD"),
          ("mamba", "https://github.com/state-spaces/mamba", "v2.3.1", "MAMBA_FORCE_BUILD")]
CONFIGS = ["configs/model_mamba.yaml", "configs/model_conextbimamba.yaml", "configs/model_conformer.yaml"]
ONLY_SM75 = ('    cc_flag = [f for f in cc_flag if f != "-gencode" and not f.startswith("arch=")]'
             ' + ["-gencode", "arch=compute_75,code=sm_75"]  # vá: chỉ T4\n')
result: dict = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:.0f}′] {msg}", flush=True)


def sh(cmd, cwd=None, env=None, check=True) -> tuple[int, str]:
    log("$ " + " ".join(map(str, cmd)))
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    lines = []
    for line in proc.stdout:
        if "oneDNN" not in line and "tensorflow" not in line and "ptxas info" not in line:
            print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    log(f"→ returncode {proc.returncode}")
    if check and proc.returncode != 0:
        finish(f"lệnh lỗi: {' '.join(map(str, cmd))[:200]}")
    return proc.returncode, "".join(lines)


def finish(error: str | None = None) -> None:
    result["error"], result["minutes"] = error, round((time.time() - T0) / 60, 1)
    (OUT / "env.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(("ENV CHECK FAIL: " + error) if error else "ENV CHECK OK")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    sys.exit(1 if error else 0)


def main() -> None:
    import platform
    import torch
    _, nvcc = sh(["nvcc", "--version"], check=False)
    result.update(python=platform.python_version(), torch=torch.__version__, torch_cuda=torch.version.cuda,
                  nvcc=(re.findall(r"release ([\d.]+)", nvcc) or ["?"])[0], gpus=torch.cuda.device_count(),
                  gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    log(f"MÔI TRƯỜNG: {result}")
    sh(["nvidia-smi"], check=False)
    if result["gpus"] != 2:
        finish(f"cần 2 GPU, thấy {result['gpus']}")

    pip = [sys.executable, "-m", "pip", "install", "-q"]
    sh(pip + ["packaging", "ninja", "einops", "librosa", "soundfile", "sentencepiece", "datasets", "jiwer",
              "pyyaml", "tensorboard"])
    _, ta = sh([sys.executable, "-c", "import torchaudio; print('torchaudio', torchaudio.__version__)"], check=False)
    result["torchaudio"] = ta.strip().split()[-1] if "torchaudio" in ta else f"LỖI: {ta.strip()[-200:]}"

    WHEEL_DIR.mkdir(parents=True, exist_ok=True)
    result["build_minutes"] = {}
    for name, url, tag, force_var in BUILDS:
        src = Path(f"/tmp/build/{name}")
        sh(["git", "clone", "--depth", "1", "--branch", tag, url, str(src)])
        setup = (src / "setup.py").read_text(encoding="utf-8")
        anchor = "    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI"
        if setup.count(anchor) != 1:
            finish(f"{name} {tag}: không tìm thấy chỗ vá danh sách kiến trúc trong setup.py")
        (src / "setup.py").write_text(setup.replace(anchor, ONLY_SM75 + anchor), encoding="utf-8")
        t = time.time()
        env = {**os.environ, "MAX_JOBS": "4", force_var: "TRUE"}
        sh([sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "-v", "-w", str(WHEEL_DIR),
            str(src)], env=env)
        result["build_minutes"][name] = round((time.time() - t) / 60, 1)
    result["wheels"] = sorted(p.name for p in WHEEL_DIR.glob("*.whl"))
    sh(pip + ["--no-deps"] + sorted(str(p) for p in WHEEL_DIR.glob("*.whl")))
    sh([sys.executable, "-c", "import causal_conv1d, mamba_ssm; from mamba_ssm import Mamba; "
                              "print('mamba_ssm', mamba_ssm.__version__, 'causal_conv1d', causal_conv1d.__version__)"])

    sh(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    _, head = sh(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)
    result["repo"] = head.strip()
    spec = importlib.util.spec_from_file_location("train_trial", REPO_DIR / "scripts/kaggle/train_trial/train_trial.py")
    trial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trial)  # chỉ định nghĩa hằng/hàm (main() có guard)
    sh([sys.executable, "-c", trial.CHECK_CODE], cwd=REPO_DIR)
    result["check_code"] = "OK"

    sh([sys.executable, "-m", "src.data.extract_audio", "--input", "/kaggle/input", "--dest", "/tmp/audio_cache",
        "--only", "train_shard0"], cwd=REPO_DIR)
    sh([sys.executable, "-m", "src.data.vietsuperspeech_dataset"], cwd=REPO_DIR, check=False)  # làm ấm cache HF
    env = {**os.environ, "AUDIO_CACHE_DIR": "/tmp/audio_cache", "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}
    result["ddp_amp_steps"] = {}
    for config in CONFIGS:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "-m", "src.training.train",
               "--config", config, "--train_manifests", "train_shard0", "--epochs", "1", "--max_minutes", "3",
               "--log_every_steps", "10", "--ckpt_dir", "/tmp/envck", "--log_dir", "/tmp/envck_runs", "--no_resume"]
        code, out = sh(cmd, cwd=REPO_DIR, env=env, check=False)
        steps = re.findall(r"step (\d+)/\d+ loss tb ([\d.na]+) \| ([\d.]+) s/step.*?VRAM đỉnh ([\d.]+) GiB", out)
        result["ddp_amp_steps"][config] = {"returncode": code, "exited_on_time": "Hết 3 phút" in out,
                                           "last": steps[-1] if steps else None}
        if code != 0 or not steps:
            finish(f"{config}: DDP + AMP không chạy được (returncode {code})")
    finish()


if __name__ == "__main__":
    main()
