"""Kernel Kaggle **GPU** — đo RQ2 cuối: độ trễ, RTF, VRAM đỉnh, số tham số theo độ dài audio ghép (batch 1, 1 T4).

Đội hình (người dùng duyệt 2026-10-12): 10 ms — Mamba B1, ConExtBiMamba, Conformer v1 (không PE), Conformer
PE tương đối; 40 ms — mamba_sub4, conextbimamba_sub4, conformer_relpos_sub4 (`FINAL_LINEUP` trong
`src/evaluation/rtf.py`, InterCTC tắt như phạm vi "A + 40 ms"); cuối cùng Parakeet-CTC-0.6B-vi (pipeline NeMo
riêng, fp32) trong tiến trình con riêng — cài NeMo hỏng không làm mất kết quả nhóm train từ đầu.

Đo bằng `python -m src.evaluation.rtf` (làm ấm từng shape, trung vị + IQR, OOM ghi như kết quả, chia đôi tìm độ
dài tối đa). Trọng số ngẫu nhiên seed 0 khi `CHECKPOINTS` rỗng — chi phí không phụ thuộc trọng số; có checkpoint
thật (gắn output kernel train làm input) thì điền `CHECKPOINTS` rồi chạy lại. Chỉ GPU 0: phép đo 1 máy, không
chạy song song 2 mô hình trên 2 GPU (tránh tranh CPU/PCIe làm lệch độ trễ).

**Chỉ đọc** dataset `vss-asr-*` đã gắn (audio đọc thẳng, không giải nén), chỉ ghi `/kaggle/working`.

    kaggle kernels push -p scripts/kaggle/rq2_final -t 9000   (CHỈ khi người dùng duyệt; điền COMMIT trước)
    python scripts/kaggle/watch_kernel.py <user>/rq2-final-asr

`kernel-metadata.json` (gitignore): 5 `vss-asr-*` + `mamba-wheels-v2` (như rq2_pilot). Ước ~60-80 phút GPU
(xem `ESTIMATE` bên dưới). Output: `rq2_final/` (`results.jsonl`, `results.csv`, `env.json`,
`audio_source.json`), `rq2_final_parakeet/`, `logs/`.
"""

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

COMMIT = "0090bd75535c980542885aa680a50789b9991630"  # ĐIỀN commit đã push chứa src/evaluation/rtf.py + src/data/concat_audio.py mới (kernel tự dừng nếu rỗng)
REPO_URL = "https://github.com/VuKhah/specialized-thesis"
REPO_DIR = Path("/tmp/specialized-thesis")
OUT = Path("/kaggle/working")
# None = mặc định DEFAULT_LENGTHS_S của rtf.py (đề xuất, chờ duyệt); hoặc chuỗi "5,10,...".
LENGTHS: str | None = None
NOISE_CHECK_LENGTHS = "30,240"  # đo thêm nhiễu Gauss: kiểm chi phí không phụ thuộc nội dung audio
# tên trong FINAL_LINEUP → đường dẫn .pt dưới /kaggle/input (best.pt / trung bình checkpoint). Rỗng = ngẫu nhiên.
CHECKPOINTS: dict[str, str] = {}
MEASURE_PARAKEET = True
PARAKEET_PRECISION = "fp32"  # fp16 tràn số trên vài câu clean-test (zero-shot 2026-10) → số chính thức fp32
ESTIMATE = ("pilot 2026-10-05 nội suy theo danh sách mức mới: mamba ~4′, conext ~3′, conformer v1 ~18′ (3.600 s ≈ "
            "157 s/lượt × 4), relpos ≤ 5′ (OOM sớm), 3 nhánh 40 ms ~7′, kiểm nhiễu ~1′, Parakeet ~15′ (cài NeMo + "
            "tải .nemo + đo), setup ~8′ → ~60-80′")


def bootstrap() -> None:
    import subprocess
    if not COMMIT:
        sys.exit("LỖI: COMMIT rỗng — commit + push code RQ2 rồi điền COMMIT (ghim phiên bản đo)")
    subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run(["git", "checkout", "-q", COMMIT], cwd=REPO_DIR, check=True)
    sys.path.insert(0, str(REPO_DIR))


def main() -> None:
    from scripts.kaggle.train_trial.train_trial import check_env, find_wheels, run

    check_env()
    run(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)
    print("ước thời gian:", ESTIMATE, flush=True)
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run(pip + ["einops", "librosa", "soundfile", "sentencepiece", "datasets", "jiwer", "pyyaml", "tensorboard"])
    wheels = find_wheels()
    if not wheels:
        sys.exit("LỖI: không thấy wheel mamba trong /kaggle/input — kiểm tra dataset mamba-wheels-v2")
    run(pip + ["--no-deps"] + wheels)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"}

    cmd = [sys.executable, "-m", "src.evaluation.rtf", "--models", "final", "--out", str(OUT / "rq2_final"),
           "--audio", "concat", "--input_dir", "/kaggle/input", "--noise_check_lengths", NOISE_CHECK_LENGTHS]
    if LENGTHS:
        cmd += ["--lengths", LENGTHS]
    if CHECKPOINTS:
        cmd += ["--checkpoints", ",".join(f"{k}={v}" for k, v in CHECKPOINTS.items())]
    code, _ = run(cmd, cwd=REPO_DIR, env=env, check=False, log_name="rq2_scratch")
    print(f"nhóm train từ đầu: returncode {code}", flush=True)

    if MEASURE_PARAKEET:
        from scripts.kaggle.finetune_parakeet.finetune_parakeet_kernel import (NEMO_FILE, NEMO_REPO,
                                                                               NEMO_REVISION, install_nemo)
        install_nemo(run)
        nemo_dl = ("from huggingface_hub import hf_hub_download; "
                   f"print(hf_hub_download('{NEMO_REPO}', '{NEMO_FILE}', revision='{NEMO_REVISION}'))")
        _, out = run([sys.executable, "-c", nemo_dl], check=False, env={**env, "HF_HUB_OFFLINE": "0"})
        nemo_path = out.strip().splitlines()[-1] if out.strip() else ""
        cmd = [sys.executable, "-m", "src.evaluation.rtf", "--models", "none", "--parakeet_nemo", nemo_path,
               "--parakeet_precision", PARAKEET_PRECISION, "--out", str(OUT / "rq2_final_parakeet"),
               "--audio", "concat", "--input_dir", "/kaggle/input", "--noise_check_lengths", NOISE_CHECK_LENGTHS]
        if LENGTHS:
            cmd += ["--lengths", LENGTHS]
        code, _ = run(cmd, cwd=REPO_DIR, env=env, check=False, log_name="rq2_parakeet")
        print(f"Parakeet: returncode {code}", flush=True)


if __name__ == "__main__":
    bootstrap()
    from scripts.kaggle.train_trial.train_trial import LOG_DIR, Tee, start_gpu_monitor, stamp
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log = open(LOG_DIR / "rq2_final.log", "a", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.stdout, _log), Tee(sys.stderr, _log)
    monitor = start_gpu_monitor()
    try:
        main()
    finally:
        if monitor:
            monitor.terminate()
        print(f"{stamp()} kết thúc kernel", flush=True)
