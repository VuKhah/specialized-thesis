"""Kernel Kaggle **GPU** (2x T4) — P1 kế hoạch 2026-10-12: eval trọng số trung bình epoch 20-29 của 3 mô hình
v1 (B1 mamba, conextbimamba, conformer) trên val, val_unseen, clean-test 203 câu; kèm best.pt của từng mô hình
đo lại bằng CÙNG đường đo (`src.evaluation.average_checkpoints`) để hiệu avg − best không lẫn khác biệt cách đo.

Trung bình đã làm sẵn ở local (CPU, không cần mamba_ssm) rồi upload thành dataset `tieunhi/kltn-v1-avg20-29`
(`<model>_avg20-29.pt`, ~50 MB/file) — đỡ upload 30 file epoch (~1,5 GB). best.pt lấy từ `tieunhi/kltn-best-ckpt`.

    kaggle kernels push -p scripts/kaggle/avg_eval -t 3600
    python scripts/kaggle/watch_kernel.py tieunhi/avg-eval-v1-asr

Hai GPU chạy song song hai hàng: GPU0 mamba (chậm nhất: khối Mamba fp32), GPU1 conformer rồi conextbimamba.
Mỗi lần eval một thư mục riêng `parts/<name>/` rồi gộp ở cuối — tránh hai tiến trình cùng ghi một file.
Output (`/kaggle/working`): `results.json`, `predictions.tsv` (định dạng final_compare), `eval_<tập>_<name>.jsonl`
(đầu vào bootstrap.py), `avg_eval_summary.json` (WER từng tập, hiệu avg − best), `logs/`.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Ghim commit có `src/evaluation/average_checkpoints.py` bản đã test CPU (2026-10-12). Commit + push trước rồi
# điền hash vào đây — để trống thì kernel dừng ngay (không chạy code HEAD trôi nổi).
COMMIT = ""
REPO_URL = "https://github.com/VuKhah/specialized-thesis"
REPO_DIR = Path("/tmp/specialized-thesis")
AUDIO_DIR = "/tmp/audio_cache"
OUT = Path("/kaggle/working")
TAG = "avg20-29"
EVAL_BEST = True  # False: chỉ eval avg (~ một nửa thời gian GPU), so với số best.pt cũ (khác đường đo val)
MODELS = {"mamba": "configs/model_mamba.yaml", "conextbimamba": "configs/model_conextbimamba.yaml",
          "conformer": "configs/model_conformer.yaml"}
QUEUES = [["mamba"], ["conformer", "conextbimamba"]]  # mỗi phần tử = một GPU
T_KERNEL = time.time()


def bootstrap() -> None:
    if not COMMIT:
        sys.exit("LỖI: chưa điền COMMIT (commit chứa average_checkpoints.py đã push)")
    subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run(["git", "checkout", "-q", COMMIT], cwd=REPO_DIR, check=True)
    sys.path.insert(0, str(REPO_DIR))


def find_input(filename: str, dataset_slug: str) -> Path:
    """Glob giới hạn độ sâu (rglob cả /kaggle/input duyệt ~67 nghìn wav, ~3 phút GPU chạy không); dataset gắn
    ở /kaggle/input/<slug>/ hoặc /kaggle/input/datasets/<user>/<slug>/ tuỳ phiên bản Kaggle."""
    root = Path("/kaggle/input")
    hits = sorted({p for pat in (f"{dataset_slug}/**/{filename}", f"*/*/{dataset_slug}/**/{filename}")
                   for p in root.glob(pat)})
    if not hits:
        sys.exit(f"LỖI: không thấy {filename} trong dataset {dataset_slug}")
    return hits[0]


def main() -> None:
    from scripts.kaggle.train_trial.train_trial import check_env, find_wheels, run, stamp

    check_env()
    _, head = run(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run(pip + ["einops", "librosa", "soundfile", "sentencepiece", "datasets", "jiwer", "pyyaml", "tensorboard"])
    wheels = find_wheels()
    if not wheels:
        sys.exit("LỖI: không thấy wheel mamba trong /kaggle/input — kiểm tra dataset mamba-wheels-v2")
    run(pip + ["--no-deps"] + wheels)

    jobs = []  # (model, name, checkpoint)
    for model in MODELS:
        jobs.append((model, f"{model}_{TAG}", find_input(f"{model}_{TAG}.pt", "kltn-v1-avg20-29")))
        if EVAL_BEST:
            jobs.append((model, f"{model}_best", find_input(f"{model}.pt", "kltn-best-ckpt")))

    # val_unseen + clean-test lấy câu từ cả hai split → cần cả 4 shard train lẫn val.
    run([sys.executable, "-m", "src.data.extract_audio", "--input", "/kaggle/input", "--dest", AUDIO_DIR],
        cwd=REPO_DIR)
    run([sys.executable, "-m", "src.data.vietsuperspeech_dataset"], cwd=REPO_DIR)  # làm ấm cache HF
    # Tải nốt audio thiếu (nếu có) khi còn mạng; các tiến trình eval sau đó chạy HF_HUB_OFFLINE.
    prefetch = ("import os, json; os.environ['AUDIO_CACHE_DIR'] = %r\n"
                "from src.data.vietsuperspeech_dataset import manifest_rows\n"
                "from src.evaluation.average_checkpoints import _prefetch_audio, clean_test_samples\n"
                "_prefetch_audio([r['audio'] for r in manifest_rows(['val', 'val_unseen'])]"
                " + [s['audio'] for s in clean_test_samples('data/processed/clean_test_manifest.json')])\n"
                % AUDIO_DIR)
    run([sys.executable, "-c", prefetch], cwd=REPO_DIR)
    setup_minutes = (time.time() - T_KERNEL) / 60

    status: dict[str, dict] = {}

    def worker(gpu: int, models: list[str]) -> None:
        env = {**os.environ, "AUDIO_CACHE_DIR": AUDIO_DIR, "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8",
               "CUDA_VISIBLE_DEVICES": str(gpu)}
        for model, name, ckpt in [j for j in jobs if j[0] in models]:
            t0 = time.time()
            code, _ = run([sys.executable, "-m", "src.evaluation.average_checkpoints", "--config", MODELS[model],
                           "--checkpoint", str(ckpt), "--name", name, "--out_dir", str(OUT / "parts" / name),
                           "--sets", "clean_test", "val", "val_unseen"],
                          cwd=REPO_DIR, env=env, check=False, log_name=name)
            status[name] = {"returncode": code, "minutes": round((time.time() - t0) / 60, 1), "gpu": gpu}

    threads = [threading.Thread(target=worker, args=(g, q)) for g, q in enumerate(QUEUES)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Gộp: thứ tự cột theo `jobs` (mamba, conextbimamba, conformer; avg trước best).
    os.chdir(REPO_DIR)  # clean_test_samples đọc đường dẫn tương đối của repo
    import csv

    from src.evaluation.average_checkpoints import clean_test_samples, merge_predictions_tsv

    samples = clean_test_samples("data/processed/clean_test_manifest.json")
    runs = {}
    for _, name, _ in jobs:
        part = OUT / "parts" / name
        if not (part / "results.json").exists():
            continue
        runs[name] = json.loads((part / "results.json").read_text(encoding="utf-8"))["runs"][name]
        for f in part.glob("eval_*.jsonl"):
            shutil.copyfile(f, OUT / f.name)
        if (part / "predictions.tsv").exists():
            with open(part / "predictions.tsv", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"))
            merge_predictions_tsv(OUT / "predictions.tsv", samples, name,
                                  [r[f"hyp_{name}"] for r in rows], [r[f"raw_{name}"] for r in rows])
    (OUT / "results.json").write_text(json.dumps({"commit": head.strip(), "runs": runs}, ensure_ascii=False,
                                                 indent=2), encoding="utf-8")

    table = {name: {s: round(m["wer"], 4) for s, m in r["sets"].items()} for name, r in runs.items()}
    diff = {}
    for model in MODELS:
        a, b = table.get(f"{model}_{TAG}"), table.get(f"{model}_best")
        if a and b:
            diff[model] = {s: round(a[s] - b[s], 4) for s in a if s in b}
    summary = {"commit": head.strip(), "setup_minutes": round(setup_minutes, 1),
               "kernel_minutes": round((time.time() - T_KERNEL) / 60, 1), "jobs": status, "wer": table,
               "avg_minus_best": diff}
    (OUT / "avg_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    failed = [n for n, s in status.items() if s["returncode"] != 0]
    if failed:
        sys.exit(f"LỖI: {failed} — xem logs/<name>.log")
    print(f"{stamp()} AVG EVAL XONG", flush=True)


if __name__ == "__main__":
    bootstrap()
    from scripts.kaggle.train_trial.train_trial import LOG_DIR, Tee, start_gpu_monitor, stamp
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(LOG_DIR / "avg_eval.log", "a", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.stdout, _log_file), Tee(sys.stderr, _log_file)
    monitor = start_gpu_monitor()
    try:
        main()
    finally:
        if monitor:
            monitor.terminate()
        print(f"{stamp()} kết thúc kernel", flush=True)
