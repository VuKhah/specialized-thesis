"""Kernel Kaggle **GPU** — chẩn đoán loss/forward không hữu hạn của Mamba B1 (train thử 2026-10-05).

Bối cảnh (Plan.md nhật ký 2026-10-05 chiều): B1 bỏ 1318/3555 step vì loss NaN/Inf, val loss NaN,
41,6% câu val_unseen ra rỗng; CTC đã fp32 nên NaN có sẵn trong forward dưới autocast fp16. Câu rỗng
dồn theo video và đi theo độ to (RMS) của audio. Kernel này trả lời:
1. Câu nào ra không hữu hạn khi chạy riêng (batch 1, không padding) — fp16 và fp32.
2. Tầng nào, nhánh nào (xuôi/ngược) ra Inf đầu tiên; cùng đầu vào chạy fp32 thì đầu ra khối lớn tới
   đâu (vượt 65504 = max fp16 → tràn số, không phải lỗi logic).
3. Đổi độ to (nhân waveform ×0,25…×4) có làm đổi tỉ lệ không hữu hạn không.
4. Trọng số B1: best.pt (epoch 2, chưa rỗng) so với latest.pt (epoch 4).
5. Đối chứng: ConExtBiMamba (cũng có khối Mamba, train ổn) trên cùng câu.

    kaggle kernels push -p scripts/kaggle/diag_b1 -t 1800
    python scripts/kaggle/watch_kernel.py <user>/diag-b1-asr

`kernel-metadata.json` (gitignore): `dataset_sources` = 5 `vss-asr-*` + `mamba-wheels-v2`;
`kernel_sources` = `<user>/train-trial-asr` (checkpoint + eval jsonl). Chỉ 1 GPU, không DDP, không
chép audio (đọc thẳng file wav từ dataset gắn sẵn). Ước ~10-15 phút GPU.
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

DIAG_CODE = r'''
import csv, glob, json, random, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, yaml
from src.training.train import build_model
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.models.ctc_model import greedy_collapse
from src.models.mamba_encoder import reverse_padded
from src.evaluation.wer import compute_wer

sys.stdout.reconfigure(encoding="utf-8")
DEV, FP16_MAX = "cuda", 65504.0
N_PER_GROUP, GAINS = 40, (0.25, 0.5, 1.0, 2.0, 4.0)
CONFIGS = {"mamba_ctc": "configs/model_mamba.yaml", "conextbimamba_ctc": "configs/model_conextbimamba.yaml"}
torch.backends.cuda.matmul.allow_tf32 = False  # fp32 "thật" để so với fp16
torch.backends.cudnn.allow_tf32 = False
t0 = time.time()
def log(*a):
    print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)

# Glob độ sâu cố định: rglob cả /kaggle/input duyệt ~67 nghìn wav (~3 phút GPU chạy không).
def find_one(rel):
    for d in range(1, 5):
        hits = glob.glob("/kaggle/input/" + "*/" * d + rel)
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(rel)
trial_root = find_one("trial/mamba_ctc/latest.pt").parents[2]
log("output train thử:", trial_root)
parts = [Path(p).with_suffix("") for d in range(1, 4) for p in glob.glob("/kaggle/input/" + "*/" * d + "*.tsv")]
parts = [p for p in parts if (p / "audio").is_dir()]
log("thư mục audio:", [str(p) for p in parts])
def wav_path(rel):
    for p in parts:
        if (p / rel).exists():
            return p / rel
    raise FileNotFoundError(rel)

# Mẫu: 40 câu B1 rỗng + 40 câu không rỗng ở eval val_unseen epoch 4 (seed 0).
audio_of = {(r["split"], r["index"]): r["audio"]
            for r in csv.DictReader(open("data/splits/val_unseen.tsv", encoding="utf-8"), delimiter="\t")}
recs = [json.loads(l) for l in open(trial_root / "trial/mamba_ctc/eval_val_unseen_epoch4.jsonl", encoding="utf-8")]
random.seed(0)
samples = []
for empty in (True, False):
    group = [r for r in recs if (not r["hyp"].strip()) == empty]
    for r in random.sample(group, N_PER_GROUP):
        x, sr = sf.read(wav_path(audio_of[(r["split"], str(r["index"]))]), dtype="float32")
        assert sr == 16000 and x.ndim == 1, (sr, x.shape)
        rms_db = 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12)
        samples.append(dict(ref=r["ref"], b1_empty=empty, wav=torch.from_numpy(x), rms_db=float(rms_db),
                            video=r["video"]))
log(f"{len(samples)} mẫu; RMS dB tb: rỗng {np.mean([s['rms_db'] for s in samples if s['b1_empty']]):.1f}, "
    f"không rỗng {np.mean([s['rms_db'] for s in samples if not s['b1_empty']]):.1f}")

tok = BPETokenizer("configs/tokenizer.model")
def load(exp, which):
    cfg = yaml.safe_load(open(CONFIGS[exp], encoding="utf-8"))
    model = build_model(cfg, tok.vocab_size)
    state = torch.load(trial_root / f"trial/{exp}/{which}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    log(f"nạp {exp}/{which}.pt (epoch {state.get('epoch')})")
    return model.to(DEV).eval()

def amax(t):
    """max |t|, NaN tính là inf (để 'không hữu hạn' luôn lớn nhất)."""
    return float(t.detach().float().abs().nan_to_num(nan=float("inf")).max())

@torch.no_grad()
def b1_trace(model, wav, amp):
    """Như MambaEncoder.forward (bidirectional, eval: dropout = identity) nhưng ghi max |.| từng tầng."""
    enc = model.encoder
    w, L = wav[None].to(DEV), torch.tensor([wav.numel()], device=DEV)
    rows = []
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        feats, fl = model.feature_extractor(w, L)
        x = enc.input_proj(feats).float()
        rows.append(("input_proj", amax(feats), amax(x), None))
        for i, (bf, bb, norm) in enumerate(zip(enc.layers, enc.layers_bwd, enc.norms)):
            h = norm(x)
            of = bf(h)
            ob = reverse_padded(bb(reverse_padded(h, fl)), fl)
            x = x + of + ob
            rows.append((i, amax(of), amax(ob), amax(x)))
        hidden = enc.norm_f(x)
        log_probs = torch.log_softmax(model.ctc_head(hidden).float(), dim=-1)
    hyp = tok.decode(greedy_collapse(log_probs, fl)[0])
    return rows, bool(torch.isfinite(log_probs).all()), hyp

@torch.no_grad()
def plain(model, wav, amp, gain=1.0):
    w, L = (wav * gain)[None].to(DEV), torch.tensor([wav.numel()], device=DEV)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        lp, ol = model(w, L)
    return bool(torch.isfinite(lp).all()), tok.decode(greedy_collapse(lp, ol)[0])

def first_bad(rows):
    for name, a, b, c in rows[1:]:
        if not np.isfinite(a):
            return name, "xuôi"
        if not np.isfinite(b):
            return name, "ngược"
    return None

print("\n===== 1-2. B1 latest.pt (epoch 4): từng câu, batch 1 =====", flush=True)
b1 = load("mamba_ctc", "latest")
res = []
for s in samples:
    r16, fin16, hyp16 = b1_trace(b1, s["wav"], True)
    r32, fin32, hyp32 = b1_trace(b1, s["wav"], False)
    peak32 = max(max(a, b) for _, a, b, _ in r32[1:])
    res.append(dict(s=s, fin16=fin16, fin32=fin32, hyp16=hyp16, hyp32=hyp32, bad=first_bad(r16), peak32=peak32,
                    resid32=max(c for *_, c in r32[1:]), r32=r32))
for empty in (True, False):
    g = [r for r in res if r["s"]["b1_empty"] == empty]
    refs = [r["s"]["ref"] for r in g]
    print(f"nhóm {'RỖNG' if empty else 'KHÔNG RỖNG'} ở eval (n={len(g)}): không hữu hạn fp16 "
          f"{sum(not r['fin16'] for r in g)}, fp32 {sum(not r['fin32'] for r in g)} | hyp rỗng fp16 "
          f"{sum(not r['hyp16'].strip() for r in g)}, fp32 {sum(not r['hyp32'].strip() for r in g)} | "
          f"WER fp16 {compute_wer(refs, [r['hyp16'] for r in g]):.3f}, fp32 {compute_wer(refs, [r['hyp32'] for r in g]):.3f}",
          flush=True)
    pk = np.array([r["peak32"] for r in g])
    print(f"   max |đầu ra khối Mamba| chạy fp32: p50 {np.median(pk):.3g}, max {pk.max():.3g} "
          f"(> {FP16_MAX:.0f}: {(pk > FP16_MAX).sum()} câu); max |residual| fp32 {max(r["resid32"] for r in g):.3g}",
          flush=True)
from collections import Counter
print("tầng/nhánh ra không hữu hạn đầu tiên (fp16):", Counter(r["bad"] for r in res if r["bad"]).most_common(10))
bad = [r for r in res if r["bad"]]
if bad:
    layer, branch = bad[0]["bad"]
    print(f"vd. câu đầu tiên hỏng (RMS {bad[0]['s']['rms_db']:.1f} dB) — fp32 cùng câu, max |đầu ra| từng tầng:")
    for name, a, b, c in bad[0]["r32"]:
        print(f"   {name}: xuôi/feats {a:.1f}  ngược/x {b:.1f}  residual {c if c is None else round(c, 1)}")
print("tương quan RMS dB ↔ max đầu ra fp32:",
      f"{np.corrcoef([r['s']['rms_db'] for r in res], [np.log10(min(r['peak32'], 1e30)) for r in res])[0, 1]:.2f}")
print("max đầu ra fp32 theo nhóm RMS:", [(lo, float("%.3g" % np.median([r["peak32"] for r in res if lo <= r["s"]["rms_db"] < lo + 5])))
                                         for lo in (-35, -30, -25, -20, -15) if any(lo <= r['s']['rms_db'] < lo + 5 for r in res)])

print("\n===== 3. Đổi độ to (B1 latest, fp16): số câu không hữu hạn / tổng =====", flush=True)
sub = [s for s in samples if s["b1_empty"]][:20] + [s for s in samples if not s["b1_empty"]][:20]
for gain in GAINS:
    n_bad = {e: sum(not plain(b1, s["wav"], True, gain)[0] for s in sub if s["b1_empty"] == e) for e in (True, False)}
    print(f"   ×{gain:<5} ({20 * np.log10(gain):+.0f} dB): nhóm rỗng {n_bad[True]}/20, nhóm không rỗng {n_bad[False]}/20",
          flush=True)

print("\n===== 4. Trọng số B1: best.pt (epoch 2) vs latest.pt (epoch 4) =====", flush=True)
def wstats(model):
    out = {}
    for name, blocks in (("xuôi", model.encoder.layers), ("ngược", model.encoder.layers_bwd)):
        for i, blk in enumerate(blocks):
            out[(name, i)] = dict(in_proj=amax(blk.in_proj.weight), out_proj=amax(blk.out_proj.weight),
                                  x_proj=amax(blk.x_proj.weight), dt_bias=amax(blk.dt_proj.bias),
                                  A_min=float((-torch.exp(blk.A_log.float())).min()), D=amax(blk.D))
    return out
b1_best = load("mamba_ctc", "best")
ws_best, ws_last = wstats(b1_best), wstats(b1)
for k in sorted(ws_last, key=lambda k: (k[1], k[0])):
    a, b = ws_best[k], ws_last[k]
    print(f"   {k[0]:5} {k[1]:2}: " + "  ".join(f"{n} {a[n]:.2f}→{b[n]:.2f}" for n in a), flush=True)
res_best = [plain(b1_best, s["wav"], True) for s in samples]
print(f"best.pt fp16 trên 80 câu: không hữu hạn {sum(not f for f, _ in res_best)}, rỗng {sum(not h.strip() for _, h in res_best)}")
peaks_best = [max(max(a, b) for _, a, b, _ in b1_trace(b1_best, s["wav"], False)[0][1:]) for s in samples]
print(f"best.pt fp32 max |đầu ra khối|: p50 {np.median(peaks_best):.3g}, max {max(peaks_best):.3g}")
del b1_best

print("\n===== 5. Đối chứng ConExtBiMamba latest.pt trên cùng 80 câu =====", flush=True)
cx = load("conextbimamba_ctc", "latest")
r16 = [plain(cx, s["wav"], True) for s in samples]
r32 = [plain(cx, s["wav"], False) for s in samples]
refs = [s["ref"] for s in samples]
print(f"không hữu hạn fp16 {sum(not f for f, _ in r16)}, fp32 {sum(not f for f, _ in r32)}; rỗng fp16 "
      f"{sum(not h.strip() for _, h in r16)}; WER fp16 {compute_wer(refs, [h for _, h in r16]):.3f}, "
      f"fp32 {compute_wer(refs, [h for _, h in r32]):.3f}")
print("\nDIAG XONG", flush=True)
'''


def stamp() -> str:
    return f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:.0f}′]"


def run(cmd, cwd=None, check=True):
    print(f"\n{stamp()} $ {' '.join(map(str, cmd))}", flush=True)
    log = open(OUT / "diag_b1.log", "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    proc.wait()
    print(f"{stamp()} → returncode {proc.returncode}", flush=True)
    if check and proc.returncode != 0:
        sys.exit(f"LỖI: lệnh trên trả về {proc.returncode}")


def main():
    import platform
    import torch
    print(f"{stamp()} MÔI TRƯỜNG: python {platform.python_version()} | torch {torch.__version__} | "
          f"GPU {torch.cuda.device_count()}x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}",
          flush=True)
    if not platform.python_version().startswith(EXPECT_PY) or not torch.__version__.startswith(EXPECT_TORCH):
        sys.exit(f"LỖI MÔI TRƯỜNG: cần python {EXPECT_PY}, torch {EXPECT_TORCH}.* (wheel mamba build cho bộ này)")
    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    run(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run(pip + ["einops", "librosa", "soundfile", "sentencepiece", "datasets", "jiwer", "pyyaml", "tensorboard"])
    root = Path("/kaggle/input")
    wheels = sorted({str(p) for pat in ("mamba-wheels*/**/*.whl", "*/*/mamba-wheels*/**/*.whl") for p in root.glob(pat)})
    if not wheels:
        sys.exit("LỖI: không thấy wheel mamba (dataset mamba-wheels-v2)")
    run(pip + ["--no-deps"] + wheels)
    run([sys.executable, "-c", DIAG_CODE], cwd=REPO_DIR)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
