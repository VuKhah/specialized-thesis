"""Kernel Kaggle **GPU** (2x T4) — kiểm tra cuối trước train full (2026-10-05).

Kiểm mọi thứ đổi sau train thử mà local không kiểm được (CUDA, DDP, VRAM), rồi đo tốc độ để lập lịch
quota. Code mới cần kiểm: speed perturbation + bucketing + kiểm định dạng (`b871cc8`), khối Mamba fp32,
giải phóng đồ thị khi bỏ step, `@torch.no_grad()` của `evaluate`.

    kaggle kernels push -p scripts/kaggle/preflight_full -t 7200
    python scripts/kaggle/watch_kernel.py <user>/preflight-full-asr

Mục (K = kiểm, dừng kernel nếu hỏng; Đ = đo, ghi số):
  K1 môi trường + CHECK_CODE (tham số, bất biến padding với kernel CUDA thật) — như train thử
  K2 định dạng mọi file audio train/val (sf.info): 16 kHz mono, không thiếu file
  Đ3 DataLoader CPU: s/batch có / không speed perturbation (2 worker, batch 8, bucketing)
  K4 pipeline tí hon 3 mô hình (train → eval DDP → best.pt → metrics.jsonl)
  K5 bỏ step NaN dưới DDP (B1): NaN chỉ ở rank 1 rồi cả hai rank → không treo, đếm đúng, VRAM không phình
  K6 dừng giữa epoch + resume (ConExtBiMamba, có bucketing + SP)
  Đ7 tốc độ trên đủ 4 shard: B1, ConExtBiMamba, Conformer (có SP) + Conformer không SP (nghẽn dữ liệu?)
  Đ8 thời gian eval đủ val + val_unseen (B1 — chậm nhất) có no_grad
Kết quả: `/kaggle/working/preflight_summary.json` + log text ở `logs/`. Ước ~70-80 phút GPU.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_URL = "https://github.com/VuKhah/specialized-thesis"
REPO_DIR = Path("/tmp/specialized-thesis")
AUDIO_DIR = "/tmp/audio_cache"
OUT = Path("/kaggle/working")
CONFIGS = {"mamba": "configs/model_mamba.yaml", "conextbimamba": "configs/model_conextbimamba.yaml",
           "conformer": "configs/model_conformer.yaml"}
SPEED_MINUTES = 9  # Đ7: mỗi mô hình (gồm ~1-2 phút khởi động) → ~400-1500 step
FULL_STEPS_PER_EPOCH = 2841  # 45.442 câu train / batch toàn cục 16
SUMMARY: dict = {"checks": {}, "measures": {}}

FORMAT_CHECK = r'''
import sys, json, soundfile as sf, os
from pathlib import Path
from src.data.vietsuperspeech_dataset import manifest_rows, AUDIO_CACHE_DIR
sys.stdout.reconfigure(encoding="utf-8")
names = ["train_shard0", "train_shard1", "train_shard2", "train_shard3", "val", "val_unseen"]
bad, missing, n = [], [], 0
for name in names:
    for r in manifest_rows([name]):
        p = Path(AUDIO_CACHE_DIR) / r["audio"]; n += 1
        if not p.exists():
            missing.append(str(p)); continue
        i = sf.info(str(p))
        if i.samplerate != 16000 or i.channels != 1:
            bad.append((str(p), i.samplerate, i.channels))
print(json.dumps({"files": n, "missing": len(missing), "bad_format": len(bad), "examples": (missing + bad)[:5]}))
'''

LOADER_BENCH = r'''
import sys, time, json, torch
from torch.utils.data import DataLoader
from src.data.vietsuperspeech_dataset import VietSuperSpeechDataset, collate_fn, manifest_rows, speed_factor
from src.training.train import ResumableSampler
sys.stdout.reconfigure(encoding="utf-8")
rows = manifest_rows(["train_shard0", "train_shard1", "train_shard2", "train_shard3"])
items = [(r["split"], r["index"]) for r in rows]
F = [0.9, 1.0, 1.1]
res = {}
for sp in (False, True):
    ds = VietSuperSpeechDataset(items=items, speed_factors=F if sp else None, seed=42)
    lengths = [r["duration"] / (speed_factor(F, 42, 0, i) if sp else 1.0) for i, r in enumerate(rows)]
    sampler = ResumableSampler(len(ds), 0, 2, 42, 0, 0, batch_size=8, lengths=lengths, pool_batches=50)
    loader = DataLoader(ds, sampler=sampler, batch_size=8, collate_fn=collate_fn, num_workers=2)
    it = iter(loader); next(it); t = time.time(); k = 0
    for _ in range(150):
        next(it); k += 1
    res["sp" if sp else "no_sp"] = (time.time() - t) / k
print(json.dumps({"s_per_batch": res}))
'''

# K5: thay ctc_loss bằng bản trả NaN ở vài step train (eval chạy no_grad → không đụng tới).
NAN_WRAPPER = r'''
import os, sys, torch
sys.path.insert(0, "/tmp/specialized-thesis")
import src.training.train as T
rank = int(os.environ.get("RANK", 0))
orig, count = T.ctc_loss, {"n": 0}
def fake(*a, **k):
    loss = orig(*a, **k)
    if torch.is_grad_enabled():
        count["n"] += 1
        c = count["n"]
        if (31 <= c <= 35 and rank == 1) or 36 <= c <= 45:
            return loss * float("nan")
    return loss
T.ctc_loss = fake
if __name__ == "__main__":
    T.main()
'''


def bootstrap() -> None:
    subprocess.run(["git", "clone", "-q", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
    sys.path.insert(0, str(REPO_DIR))


def last_progress(out: str) -> dict:
    """Dòng tiến độ cuối của train.py: '... step 300/2841 loss tb 5.1 | 0.512 s/step, ... | bỏ 0+1, VRAM đỉnh 3.3 GiB'."""
    pat = re.compile(r"step (\d+)/(\d+) loss tb ([\d.na]+) \| ([\d.]+) s/step.*bỏ (\d+)\+(\d+)(?:, VRAM đỉnh ([\d.]+) GiB)?")
    hits = pat.findall(out)
    if not hits:
        return {}
    s, total, loss, sps, nf, amp, vram = hits[-1]
    return {"step": int(s), "loss": loss, "s_per_step": float(sps), "nonfinite": int(nf), "amp_skipped": int(amp),
            "peak_vram_gib": float(vram) if vram else None}


def gpu_util_between(t0: float, t1: float) -> float | None:
    """Trung bình utilization (2 GPU) trong khoảng thời gian, từ gpu_util.csv (nvidia-smi -l 30)."""
    import csv
    from datetime import datetime
    vals = []
    try:
        with open(OUT / "logs" / "gpu_util.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f, skipinitialspace=True):
                ts = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                if t0 <= ts <= t1:
                    vals.append(float(row["utilization.gpu [%]"].split()[0]))
    except (OSError, KeyError, ValueError) as e:
        print(f"gpu_util.csv: {e}", flush=True)
    return sum(vals) / len(vals) if vals else None


def record(section: str, key: str, value) -> None:
    SUMMARY[section][key] = value
    (OUT / "preflight_summary.json").write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== {section}.{key}: {json.dumps(value, ensure_ascii=False)}", flush=True)


def main() -> None:
    from scripts.kaggle.train_trial.train_trial import CHECK_CODE, check_env, find_wheels, run, stamp

    check_env()
    _, head = run(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)
    SUMMARY["commit"] = head.strip()
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run(pip + ["einops", "librosa", "soundfile", "sentencepiece", "datasets", "jiwer", "pyyaml", "tensorboard"])
    wheels = find_wheels()
    if not wheels:
        sys.exit("LỖI: không thấy wheel mamba trong /kaggle/input — kiểm tra dataset mamba-wheels-v2")
    run(pip + ["--no-deps"] + wheels)
    run([sys.executable, "-c", CHECK_CODE], cwd=REPO_DIR)
    record("checks", "K1_env_check_code", "OK")

    run([sys.executable, "-m", "src.data.extract_audio", "--input", "/kaggle/input", "--dest", AUDIO_DIR],
        cwd=REPO_DIR)
    run([sys.executable, "-m", "src.data.vietsuperspeech_dataset"], cwd=REPO_DIR)
    env = {**os.environ, "AUDIO_CACHE_DIR": AUDIO_DIR, "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}

    _, out = run([sys.executable, "-c", FORMAT_CHECK], cwd=REPO_DIR, env=env)
    fmt = json.loads(out.strip().splitlines()[-1])
    record("checks", "K2_audio_format", fmt)
    if fmt["missing"] or fmt["bad_format"]:
        sys.exit("LỖI K2: thiếu file hoặc sai định dạng")

    _, out = run([sys.executable, "-c", LOADER_BENCH], cwd=REPO_DIR, env=env, check=False)
    try:
        record("measures", "D3_loader_cpu", json.loads(out.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        record("measures", "D3_loader_cpu", "LỖI — xem log")

    def train(config, ckpt, minutes, extra=(), script=None, log_name=None, check=True):
        launch = [script] if script else ["-m", "src.training.train"]
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", *launch,
               "--config", config, "--max_minutes", str(minutes), "--ckpt_dir", str(ckpt),
               "--log_dir", str(Path(ckpt) / "runs"), "--log_every_steps", "25", *extra]
        return run(cmd, cwd=REPO_DIR, env=env, check=check, log_name=log_name)

    pipe = Path("/tmp/pipeline")
    for name, config in CONFIGS.items():
        _, out = train(config, pipe, 10, ("--epochs", "1", "--limit_train", "64", "--limit_eval", "64",
                                          "--log_every_steps", "1", "--no_resume"))
        if "eval val" not in out or "WER=" not in out:
            sys.exit(f"LỖI K4: pipeline tí hon {name} không đi tới eval")
    record("checks", "K4_pipeline_3_models", "OK")

    wrapper = Path("/tmp/nan_wrapper.py")
    wrapper.write_text(NAN_WRAPPER, encoding="utf-8")
    _, out = train(CONFIGS["mamba"], Path("/tmp/nan"), 4, ("--log_every_steps", "5", "--no_resume"),
                   script=str(wrapper), log_name="k5_nan", check=False)
    prog = [m.groupdict() for m in re.finditer(
        r"step (?P<step>\d+)/\d+ .*?bỏ (?P<nf>\d+)\+\d+, VRAM đỉnh (?P<vram>[\d.]+) GiB", out)]
    before = max((float(p["vram"]) for p in prog if int(p["step"]) <= 30), default=None)
    after = max((float(p["vram"]) for p in prog if int(p["step"]) > 45), default=None)
    nf = max((int(p["nf"]) for p in prog), default=0)
    # Lỗi cũ giữ thêm nguyên một đồ thị forward (~+80% VRAM ở train thử B1); bucketing làm độ dài batch
    # dao động ~10% giữa các step nên ngưỡng 25% chứ không chặt hơn.
    k5 = {"nonfinite_counted": nf, "expected": 15, "vram_before_gib": before, "vram_after_gib": after,
          "last_step": int(prog[-1]["step"]) if prog else 0}
    k5["ok"] = bool(nf == 15 and k5["last_step"] > 50 and before and after and after <= before * 1.25)
    record("checks", "K5_nan_skip_ddp", k5)
    if not k5["ok"]:
        print("!!! K5 không đạt — xem logs/k5_nan.log (vẫn chạy tiếp để có số đo)", flush=True)

    smoke = Path("/tmp/smoke")
    train(CONFIGS["conextbimamba"], smoke, 3, ("--epochs", "1"))
    _, out = train(CONFIGS["conextbimamba"], smoke, 3, ("--epochs", "1"))
    resumed = re.search(r"resume từ .*: epoch (\d+) step (\d+)", out)
    record("checks", "K6_resume", {"ok": bool(resumed), "resumed_at": resumed.group(0) if resumed else None})
    if not resumed:
        sys.exit("LỖI K6: lần chạy thứ hai không resume")

    no_sp = REPO_DIR / "configs" / "_conformer_no_sp.yaml"
    no_sp.write_text(re.sub(r"^  speed_perturb: .*$", "", (REPO_DIR / CONFIGS["conformer"]).read_text(
        encoding="utf-8"), flags=re.M), encoding="utf-8")
    runs = [*CONFIGS.items(), ("conformer_no_sp", str(no_sp.relative_to(REPO_DIR)))]
    for name, config in runs:
        t0 = time.time()
        code, out = train(config, Path(f"/tmp/speed_{name}"), SPEED_MINUTES, ("--no_resume",),
                          log_name=f"d7_{name}", check=False)
        m = last_progress(out)
        m.update({"returncode": code, "gpu_util_pct": gpu_util_between(t0 + 120, time.time())})
        if m.get("s_per_step"):
            m["train_hours_30_epochs"] = m["s_per_step"] * FULL_STEPS_PER_EPOCH * 30 / 3600
        record("measures", f"D7_speed_{name}", m)

    # Đ8: train.py không có chế độ chỉ-eval → 1 epoch tí hon (16 câu train) rồi eval đủ 2 tập, đọc eval_seconds.
    t0 = time.time()
    _, out = train(CONFIGS["mamba"], Path("/tmp/eval_b1"), 30, ("--epochs", "1", "--limit_train", "16",
                                                                 "--no_resume", "--log_every_steps", "50"),
                   log_name="d8_eval", check=False)
    mfile = Path("/tmp/eval_b1/mamba_ctc/metrics.jsonl")
    if mfile.exists():
        m = json.loads(mfile.read_text(encoding="utf-8").splitlines()[-1])
        record("measures", "D8_eval_b1", {k: v for k, v in m.items() if k.startswith(("eval_seconds", "wer_"))}
               | {"wall_minutes": (time.time() - t0) / 60})
    else:
        record("measures", "D8_eval_b1", "LỖI — xem logs/d8_eval.log")

    print(f"\n{stamp()} PREFLIGHT FULL XONG\n" + json.dumps(SUMMARY, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    bootstrap()
    from scripts.kaggle.train_trial.train_trial import LOG_DIR, Tee, start_gpu_monitor, stamp
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(LOG_DIR / "preflight_full.log", "a", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.stdout, _log_file), Tee(sys.stderr, _log_file)
    monitor = start_gpu_monitor()
    try:
        main()
    finally:
        if monitor:
            monitor.terminate()
        print(f"{stamp()} kết thúc kernel", flush=True)
