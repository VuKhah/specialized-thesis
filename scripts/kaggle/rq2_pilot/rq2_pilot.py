"""Kernel Kaggle **GPU** — pilot RQ2: chi phí suy luận theo độ dài audio (2026-10-05, người dùng duyệt).

Câu hỏi (Plan.md nhật ký 2026-10-05): tốc độ train (s/step, câu 10-15 s) không trả lời được ưu thế O(T)
của Mamba; Speech Slytherin (arXiv:2407.09732) báo ngưỡng thời lượng tỉ lệ nghịch với độ phân giải
token. Pilot đo, với **trọng số ngẫu nhiên** (chi phí không phụ thuộc giá trị trọng số):
1. Độ trễ, RTF, VRAM đỉnh của Conformer-12M, ConExtBiMamba, Mamba B1 theo độ dài 5 s → 3600 s.
2. Độ dài tối đa chạy được trên 1 T4 (tăng gấp đôi tới OOM rồi chia đôi khoảng).
3. Cùng phép đo khi chèn **subsampling Conv2d 4×** trước encoder — chỉ để đo xem điểm giao dịch chuyển
   thế nào; khối này dựng trong script, **không** phải code mô hình, không khớp lại số tham số.
4. Tham khảo: B1 với khối Mamba fp16 (không `mamba_fp32`) — tốc độ nếu fp16 dùng được.

Điều kiện chung: 1 GPU, batch 1, `no_grad`, eval, autocast fp16 như lúc train (khối Mamba fp32 qua
`mamba_fp32` — OVERLAY vì chưa commit), warm-up, `cuda.synchronize`, trung vị. Audio: "audio ghép
nhân tạo" (RQ2 chốt 2026-10-03) — nối các đoạn liên tiếp cùng video (đã bỏ excluded/heldout qua
`manifest_rows`), video dài nhất trước. **Chỉ đọc** dataset đã gắn, chỉ ghi `/kaggle/working`.

    kaggle kernels push -p scripts/kaggle/rq2_pilot -t 3600
    python scripts/kaggle/watch_kernel.py <user>/rq2-pilot-asr

`kernel-metadata.json` (gitignore): `dataset_sources` = 5 `vss-asr-*` + `mamba-wheels-v2`. Ước ~25-35 phút GPU.
Output: `rq2_pilot/results.jsonl` (mỗi dòng 1 phép đo), `env.json`, `audio_source.json`, `rq2_pilot.log`.
"""

import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_URL = "https://github.com/VuKhah/specialized-thesis"
REPO_DIR = Path("/tmp/specialized-thesis")
OUT = Path("/kaggle/working")
EXPECT_PY, EXPECT_TORCH = "3.13", "2.11"
T0 = time.time()
# Ghi đè file của repo sau khi clone — khối Mamba fp32 (mamba_fp32), đã dùng ở retrain-b1-asr, chưa commit.
OVERLAY: dict[str, str] = {}  # mamba_fp32 đã commit — để trống khi code đã push

BENCH_CODE = r'''
import gc, glob, json, os, platform, re, statistics, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, soundfile as sf, torch, yaml
import src.models.mamba_encoder as me
import src.models.conextbimamba_encoder as ce
from src.training.train import build_model
from src.data.vietsuperspeech_dataset import manifest_rows, video_of

sys.stdout.reconfigure(encoding="utf-8")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda"
OUT = Path(os.environ.get("RQ2_OUT", "/kaggle/working/rq2_pilot")); OUT.mkdir(parents=True, exist_ok=True)
INPUT = os.environ.get("RQ2_INPUT", "/kaggle/input")
LENGTHS_S = [float(x) for x in os.environ.get("RQ2_LENGTHS", "5,10,15,30,60,120,240,480,960,1920,3600").split(",")]
MAX_S = LENGTHS_S[-1]
FAKE_OOM_S = float(os.environ.get("RQ2_FAKE_OOM_S", "0"))  # chỉ để test CPU nhánh chia đôi
BISECT_STEPS = 4
SR = 16000
t0 = time.time()
def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

# ---------- audio ghép nhân tạo ----------
rows = manifest_rows(["train_shard0", "train_shard1", "train_shard2", "train_shard3", "val"])
by_video = defaultdict(list)
for r in rows:
    m = re.search(r"_seg(\d+)\.wav$", r["audio"])
    by_video[video_of(r["audio"])].append((int(m.group(1)), r))
parts = [Path(p).with_suffix("") for d in range(1, 4) for p in glob.glob(INPUT + "/" + "*/" * d + "*.tsv")]
parts = [p for p in parts if (p / "audio").is_dir()]
def wav_path(rel):
    for p in parts:
        if (p / rel).exists():
            return p / rel
    if os.environ.get("RQ2_SKIP_MISSING"):  # chỉ test local (cache audio không đủ)
        return None
    raise FileNotFoundError(rel)
chunks, total, used = [], 0.0, []
for video, segs in sorted(by_video.items(), key=lambda kv: -sum(r["duration"] for _, r in kv[1])):
    n = 0
    for _, r in sorted(segs, key=lambda x: x[0]):
        path = wav_path(r["audio"])
        if path is None:
            continue
        x, sr = sf.read(path, dtype="float32")
        assert sr == SR and x.ndim == 1
        chunks.append(x); total += len(x) / SR; n += 1
        if total >= MAX_S:
            break
    used.append({"video": video, "segments": n})
    if total >= MAX_S:
        break
long_wav = torch.from_numpy(np.concatenate(chunks))
del chunks
log(f"audio ghép: {total:.0f} s từ {len(used)} video, {sum(u['segments'] for u in used)} đoạn")
json.dump({"total_s": total, "videos": used}, open(OUT / "audio_source.json", "w"), ensure_ascii=False, indent=1)

env = {"python": platform.python_version(), "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
       "gpu_mem_gib": torch.cuda.get_device_properties(0).total_memory / 2**30 if DEV == "cuda" else None,
       "sdp_flash": torch.backends.cuda.flash_sdp_enabled(), "sdp_mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
       "sdp_math": torch.backends.cuda.math_sdp_enabled(), "amp": AMP, "lengths_s": LENGTHS_S}
json.dump(env, open(OUT / "env.json", "w"), indent=1)
log("môi trường:", env)

# ---------- subsampling 4× chỉ để đo (kiểu Conv2dSubsampling ESPnet/Conformer: 2 conv k3 s2 + linear) ----------
class Subsample4(torch.nn.Module):
    def __init__(self, n_mels=80, ch=256):
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, ch, 3, 2), torch.nn.ReLU(), torch.nn.Conv2d(ch, ch, 3, 2), torch.nn.ReLU())
        self.out = torch.nn.Linear(ch * (((n_mels - 1) // 2 - 1) // 2), n_mels)  # → input_dim của encoder, encoder giữ nguyên
    def forward(self, feats):
        x = self.conv(feats.unsqueeze(1))  # [B, ch, T', F']
        b, c, t, f = x.shape
        return self.out(x.transpose(1, 2).reshape(b, t, c * f))

CONFIGS = {"conformer": "configs/model_conformer.yaml", "conextbimamba": "configs/model_conextbimamba.yaml",
           "mamba_b1": "configs/model_mamba.yaml"}
VARIANTS = [(m, 1, "fp32") for m in CONFIGS] + [(m, 4, "fp32") for m in CONFIGS] + [("mamba_b1", 1, "fp16")]

class OOM(Exception):
    pass

_mamba_fp32 = me.mamba_fp32
def set_mamba_precision(p):
    """fp16 = gọi khối trực tiếp dưới autocast (như trước 2026-10-05) — chỉ để tham khảo tốc độ."""
    f = _mamba_fp32 if p == "fp32" else (lambda block, x: block(x))
    me.mamba_fp32 = f
    ce.mamba_fp32 = f

def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()

@torch.no_grad()
def run_once(model, sub, wav):
    """Trả (giây e2e, giây encoder, số khung vào encoder). e2e = front-end + (subsampling) + encoder + CTC head."""
    w, L = wav[None].to(DEV), torch.tensor([wav.numel()], device=DEV)
    with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
        sync(); a = time.perf_counter()
        feats, fl = model.feature_extractor(w, L)
        sync(); b = time.perf_counter()
        if sub is not None:
            feats = sub(feats); fl = torch.tensor([feats.size(1)], device=DEV)
        hidden, ol = model.encoder(feats, fl)
        sync(); c = time.perf_counter()
        lp = torch.log_softmax(model.ctc_head(hidden).float(), -1)
        sync(); d = time.perf_counter()
    assert torch.isfinite(lp).all(), "đầu ra không hữu hạn"
    return d - a, c - b, int(fl[0])

def measure(model, sub, seconds):
    if FAKE_OOM_S and seconds >= FAKE_OOM_S:
        raise OOM()
    wav = long_wav[: int(seconds * SR)]
    try:
        if DEV == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        first = run_once(model, sub, wav)  # warm-up (cuDNN/kernels theo shape mới)
        n_runs = 5 if first[0] < 2 else (3 if first[0] < 10 else 1)
        if first[0] < 0.5:
            run_once(model, sub, wav)
        res = [run_once(model, sub, wav) for _ in range(n_runs)]
        peak = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else None
    except torch.cuda.OutOfMemoryError:
        raise OOM()
    e2e = statistics.median(r[0] for r in res); enc = statistics.median(r[1] for r in res)
    return {"seconds": seconds, "frames": res[0][2], "latency_s": e2e, "encoder_s": enc, "rtf": e2e / seconds,
            "peak_vram_gib": peak, "n_runs": n_runs}

out_f = open(OUT / "results.jsonl", "a", encoding="utf-8")
def record(row):
    out_f.write(json.dumps(row, ensure_ascii=False) + "\n"); out_f.flush()

tok_vocab = 1001  # = BPETokenizer(configs/tokenizer.model).vocab_size; chỉ ảnh hưởng CTC head
for name, factor, prec in VARIANTS:
    tag = dict(model=name, subsample=factor, mamba_precision=prec)
    set_mamba_precision(prec)
    torch.manual_seed(0)
    model = build_model(yaml.safe_load(open(CONFIGS[name], encoding="utf-8")), tok_vocab).to(DEV).eval()
    sub = Subsample4().to(DEV).eval() if factor == 4 else None
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    log(f"=== {name} subsample {factor}× khối Mamba {prec} (encoder {n_enc:,} tham số)")
    last_ok, fail = None, None
    for s in LENGTHS_S:
        try:
            r = measure(model, sub, s)
        except OOM:
            fail = s; log(f"  {s:>6.0f} s: OOM"); record({**tag, "seconds": s, "oom": True}); break
        record({**tag, **r, "oom": False}); last_ok = s
        log(f"  {s:>6.0f} s ({r['frames']:>7} khung): e2e {r['latency_s']*1000:9.1f} ms, encoder {r['encoder_s']*1000:9.1f} ms, "
            f"RTF {r['rtf']:.5f}, VRAM {r['peak_vram_gib'] if r['peak_vram_gib'] is None else round(r['peak_vram_gib'], 2)} GiB")
    limit = {"model": name, "subsample": factor, "mamba_precision": prec, "kind": "limit"}
    if fail is None:
        limit.update(max_ok_s=last_ok, oom_s=None, note=f"không OOM tới {MAX_S:.0f} s")
    else:
        lo, hi = (last_ok or 0.0), fail
        for _ in range(BISECT_STEPS):
            mid = round((lo + hi) / 2, 1)
            try:
                r = measure(model, sub, mid); record({**tag, **r, "oom": False, "bisect": True}); lo = mid
            except OOM:
                record({**tag, "seconds": mid, "oom": True, "bisect": True}); hi = mid
        limit.update(max_ok_s=lo, oom_s=hi)
    record(limit); log("  giới hạn:", limit)
    del model, sub; gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
set_mamba_precision("fp32")
log("RQ2 PILOT XONG")
'''


def stamp() -> str:
    return f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:.0f}′]"


def run(cmd, cwd=None, check=True, env=None):
    print(f"\n{stamp()} $ {' '.join(map(str, cmd))}", flush=True)
    log = open(OUT / "rq2_pilot.log", "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        if "oneDNN" in line or "tensorflow" in line:
            continue
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    proc.wait()
    print(f"{stamp()} → returncode {proc.returncode}", flush=True)
    if check and proc.returncode != 0:
        sys.exit(f"LỖI: lệnh trên trả về {proc.returncode}")


def main():
    import os
    import platform
    import torch
    print(f"{stamp()} MÔI TRƯỜNG: python {platform.python_version()} | torch {torch.__version__} | "
          f"GPU {torch.cuda.device_count()}x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}",
          flush=True)
    if not platform.python_version().startswith(EXPECT_PY) or not torch.__version__.startswith(EXPECT_TORCH):
        sys.exit(f"LỖI MÔI TRƯỜNG: cần python {EXPECT_PY}, torch {EXPECT_TORCH}.* (wheel mamba build cho bộ này)")
    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    run(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)
    for rel, content in OVERLAY.items():
        (REPO_DIR / rel).write_text(content, encoding="utf-8")
        print(f"OVERLAY: ghi đè {rel} ({len(content)} ký tự)", flush=True)
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run(pip + ["einops", "librosa", "soundfile", "sentencepiece", "datasets", "jiwer", "pyyaml", "tensorboard"])
    root = Path("/kaggle/input")
    wheels = sorted({str(p) for pat in ("mamba-wheels*/**/*.whl", "*/*/mamba-wheels*/**/*.whl") for p in root.glob(pat)})
    if not wheels:
        sys.exit("LỖI: không thấy wheel mamba (dataset mamba-wheels-v2)")
    run(pip + ["--no-deps"] + wheels)
    # Chỉ GPU 0: đo batch 1 trên 1 máy, như phép đo RQ2.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"}
    run([sys.executable, "-c", BENCH_CODE], cwd=REPO_DIR, env=env)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
