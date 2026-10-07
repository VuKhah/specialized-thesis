"""Kernel Kaggle GPU (T4) — đo **tốc độ** train của 3 encoder nhóm train từ đầu khi có/không subsampling
và khi khối Mamba chạy fp16 (thêm 2026-10-08, người dùng duyệt). Chỉ đo thời gian/VRAM, KHÔNG đo WER.

Câu hỏi: Mamba B1 chậm ~2,6× Conformer mỗi step (train full) trong khi 2405.12609 (Bảng XIII) và
Zevallos 2025 báo Mamba nhanh hơn/ngang Conformer — do không subsampling (chuỗi 10 ms, dài gấp 4 bài
khác), do khối Mamba ép fp32 (`mamba_fp32`), hay do kiến trúc?

Biến thể (mỗi encoder, cùng yaml của train full):
  a  nguyên bản: T' = T, khối Mamba fp32 (như train full)
  b  Conv2d subsampling 4× trước encoder (T' ≈ T/4, khung 40 ms như ConMamba/Zevallos), Mamba vẫn fp32
  c  T' = T, khối Mamba chạy dưới autocast fp16 như phần còn lại (Conformer không có khối Mamba → bỏ)

**Độc lập với code train:** clone repo ở commit đã train (`COMMIT`), không sửa file nào trong repo;
subsampling và bản fp16 chỉ gắn vào model trong tiến trình con của kernel này (monkey-patch / bọc encoder).

Hạn chế (ghi khi báo cáo): 1 GPU, batch 8 (= mỗi GPU lúc train DDP 2×8), không DDP, audio giả (nhiễu
Gauss, độ dài 10-15 s × speed perturb, xếp theo độ dài như bucketing) đã nạp sẵn trên GPU → chỉ so được
**tỉ lệ** giữa các cấu hình, không so số giây tuyệt đối với log train full. Trọng số khởi tạo ngẫu nhiên:
biến thể c không nói được fp16 có tràn số với trọng số đã train hay không (lỗi 2026-10-05 xuất hiện sau
vài epoch) — chỉ đo tốc độ. Subsampling theo kiểu Conv2dSubsampling4 của ESPnet (2 conv 3×3 stride 2,
kênh = d_model) → encoder thêm ~1,8M tham số (ghi trong kết quả).

    kaggle kernels push -p scripts/kaggle/subsample_speed -t 3600
    PYTHONUTF8=1 kaggle kernels output tieunhi/subsample-speed-asr -p reports/results/subsample_speed
    python scripts/kaggle/subsample_speed/subsample_speed.py --local    # thử nhanh trên CPU (MambaRef)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding != "utf-8":
        _stream.reconfigure(encoding="utf-8")

REPO_URL = "https://github.com/VuKhah/specialized-thesis"
COMMIT = "bb8d0ebd885c6ba21f67373cfd44cb67bb67c988"  # code của Conformer s1, ConExt s1-s2 (Mamba s1 = 7660a6b, encoder y hệt)
REPO_DIR = Path("/tmp/specialized-thesis")
OUT = Path("/kaggle/working")
CONFIGS = ["configs/model_conformer.yaml", "configs/model_mamba.yaml", "configs/model_conextbimamba.yaml"]
VARIANTS = ["a", "b", "c"]
# v2 chạy 2 lượt (lượt 2 ngược thứ tự): mọi cấu hình lệch < 3% giữa hai lượt → v3 chỉ 1 lượt.
REPEATS = 1

WORKER_CODE = r'''"""Đo 1 cấu hình (encoder × biến thể) — gọi từ subsample_speed.py, cwd = gốc repo. In `RESULT {json}`."""

import argparse
import json
import math
import sys
import time

import torch
import yaml

sys.stdout.reconfigure(encoding="utf-8")

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--variant", choices=["a", "b", "c"], required=True)
p.add_argument("--device", default="cuda")
p.add_argument("--ref_mamba", action="store_true", help="chỉ để thử trên CPU: MambaRef thay mamba_ssm")
p.add_argument("--batch", type=int, default=8)
p.add_argument("--warmup", type=int, default=5)  # chỉ bỏ ở lượt 1 (lượt 1 không dùng làm số đo)
p.add_argument("--steps", type=int, default=40)
p.add_argument("--eval_batches", type=int, default=20)
p.add_argument("--min_sec", type=float, default=10.0)
p.add_argument("--max_sec", type=float, default=15.0)
args = p.parse_args()

if args.ref_mamba:
    from src.models.mamba_ref import use_reference_mamba
    use_reference_mamba()
from src.models import conextbimamba_encoder, mamba_encoder
from src.training.train import build_model, build_optimizer
from src.models.ctc_model import ctc_loss

if args.variant == "c":
    # Cả hai encoder gọi mamba_fp32 qua tên trong module của mình → vá cả hai chỗ.
    def mamba_amp(block, x):
        return block(x)
    mamba_encoder.mamba_fp32 = mamba_amp
    conextbimamba_encoder.mamba_fp32 = mamba_amp


class Conv2dSubsampling4(torch.nn.Module):
    """Theo Conv2dSubsampling của ESPnet: 2 × (Conv2d 3×3 stride 2 + ReLU), không padding, rồi Linear."""

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, d_model, 3, 2), torch.nn.ReLU(),
            torch.nn.Conv2d(d_model, d_model, 3, 2), torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(d_model * (((input_dim - 1) // 2 - 1) // 2), d_model)

    def forward(self, feats, lengths):
        x = self.conv(feats.unsqueeze(1))  # [B, C, T', F']
        x = self.out(x.transpose(1, 2).flatten(2))
        return x, ((lengths - 1) // 2 - 1) // 2


class SubsampledEncoder(torch.nn.Module):
    """Subsampling thay input_proj của encoder (subsampling đã chiếu lên d_model)."""

    def __init__(self, encoder, input_dim: int):
        super().__init__()
        self.subsampling = Conv2dSubsampling4(input_dim, encoder.output_dim)
        encoder.input_proj = torch.nn.Identity()
        self.encoder = encoder

    @property
    def output_dim(self):
        return self.encoder.output_dim

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, feats, lengths):
        return self.encoder(*self.subsampling(feats, lengths))


cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
device = torch.device(args.device)
amp = device.type == "cuda"
torch.manual_seed(0)
model = build_model(cfg, vocab_size=cfg["tokenizer"]["vocab_size"])
if args.variant == "b":
    model.encoder = SubsampledEncoder(model.encoder, cfg["encoder"]["input_dim"])
model = model.to(device)
opt = build_optimizer(model, cfg)
scaler = torch.amp.GradScaler(device.type, enabled=amp)

# Batch giả, giống nhau ở mọi cấu hình (seed cố định): độ dài 10-15 s chia speed perturb {0,9; 1; 1,1},
# xếp theo độ dài rồi cắt batch (≈ bucketing của train full), xáo thứ tự batch. ~4 token/s.
g = torch.Generator().manual_seed(1234)
n_train = args.warmup + args.steps
n_total = n_train + args.eval_batches
base = args.min_sec + (args.max_sec - args.min_sec) * torch.rand(n_total * args.batch, generator=g)
speed = torch.tensor([0.9, 1.0, 1.1])[torch.randint(0, 3, (n_total * args.batch,), generator=g)]
secs = (base / speed).sort().values.view(n_total, args.batch)
secs = secs[torch.randperm(n_total, generator=g)]
batches = []
for row in secs:
    lens = (row * 16000).long()
    wav = 0.1 * torch.randn(args.batch, int(lens.max()), generator=g)
    wav[torch.arange(wav.size(1)).unsqueeze(0) >= lens.unsqueeze(1)] = 0.0
    tlen = (row * 4).long()
    tgt = torch.randint(1, cfg["tokenizer"]["vocab_size"], (int(tlen.sum()),), generator=g)
    batches.append(tuple(t.to(device) for t in (wav, lens, tgt, tlen)))
audio_seconds_eval = float(secs[n_train:].sum())


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
# Hai lượt qua CÙNG các batch, chỉ đo lượt 2. v2 (2026-10-08): mỗi batch một shape mới → Conformer/ConExt
# tốn ~1,1 s/step cố định, không đổi theo T' (a ≈ b), còn Mamba B1 khớp log train. Giả thuyết: cuDNN dựng
# kế hoạch conv cho mỗi shape mới (depthwise Conv1d của conv module; Mamba dùng causal_conv1d, không qua
# cuDNN). Lúc train 2841 step/epoch shape lặp lại → trạng thái đã cache. Lượt 1 ghi riêng để kiểm giả thuyết.
model.train()
pass_times, losses, skipped, frames_out = [[], []], [], 0, []
for pass_ in range(2):
    for i, (wav, lens, tgt, tlen) in enumerate(batches[:n_train]):
        sync()
        t0 = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            log_probs, out_lengths = model(wav, lens)
        loss = ctc_loss(log_probs, out_lengths, tgt, tlen)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        scale_before = scaler.get_scale() if amp else 1.0
        scaler.step(opt)
        scaler.update()
        sync()
        if pass_ == 1 or i >= args.warmup:
            pass_times[pass_].append(time.perf_counter() - t0)
        if pass_ == 1:
            skipped += int(amp and scaler.get_scale() < scale_before)
        losses.append(loss.item())
        frames_out.append(int(out_lengths.max()))
times = pass_times[1]
peak_train = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
model.eval()
eval_times = []
with torch.no_grad():
    for pass_ in range(2):
        for wav, lens, _, _ in batches[n_train:]:
            sync()
            t0 = time.perf_counter()
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                model(wav, lens)
            sync()
            if pass_ == 1:
                eval_times.append(time.perf_counter() - t0)
peak_eval = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None

times_sorted = sorted(times)
print("RESULT " + json.dumps({
    "config": cfg["experiment_name"], "variant": args.variant, "device": str(device),
    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    "encoder_params": sum(p.numel() for p in model.encoder.parameters()),
    "batch": args.batch, "steps_timed": len(times),
    "s_per_step_mean": round(sum(times) / len(times), 4),
    "s_per_step_median": round(times_sorted[len(times) // 2], 4),
    "s_per_step_min": round(times_sorted[0], 4), "s_per_step_max": round(times_sorted[-1], 4),
    "first_pass_s_per_step_mean": round(sum(pass_times[0]) / len(pass_times[0]), 4),
    "encoder_frames_max_mean": round(sum(frames_out) / len(frames_out), 1),
    "peak_vram_train_gib": round(peak_train, 3) if peak_train is not None else None,
    "eval_s_per_batch": round(sum(eval_times) / len(eval_times), 4),
    "eval_rtf": round(sum(eval_times) / audio_seconds_eval, 5),
    "peak_vram_eval_gib": round(peak_eval, 3) if peak_eval is not None else None,
    "loss_first": round(losses[0], 3), "loss_last": round(losses[-1], 3),
    "loss_finite": all(math.isfinite(x) for x in losses), "amp_skipped_steps": skipped,
    "torch": torch.__version__,
}), flush=True)
'''


def run(cmd, cwd=None, check=True, env=None):
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    if check and r.returncode != 0:
        sys.exit(f"LỖI: lệnh trên trả về {r.returncode}")
    return r


def plan():
    order = [(cfg, v) for cfg in CONFIGS for v in VARIANTS if not (v == "c" and "conformer" in cfg)]
    return [(rep, cfg, v) for rep in range(REPEATS) for cfg, v in (order if rep == 0 else order[::-1])]


def measure(repo: Path, extra: list[str], env: dict, out_json: Path, stop_on_first_error: bool):
    worker = repo / "_subsample_speed_worker.py"
    worker.write_text(WORKER_CODE, encoding="utf-8")
    results = []
    try:
        for rep, cfg, v in plan():
            r = run([sys.executable, str(worker), "--config", cfg, "--variant", v] + extra, cwd=repo, check=False,
                    env=env)
            line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
            res = json.loads(line[len("RESULT "):]) if line else {
                "config": cfg, "variant": v, "error": f"returncode {r.returncode}", "stderr_tail": r.stderr[-1500:]}
            res["repeat"] = rep
            results.append(res)
            out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            if "error" in res and stop_on_first_error and len(results) == 1:
                sys.exit("LỖI: cấu hình đầu tiên thất bại — dừng để khỏi đốt quota")
    finally:
        worker.unlink(missing_ok=True)
    return results


def summarize(results):
    ok = [r for r in results if "error" not in r]
    keys = sorted({(r["config"], r["variant"]) for r in ok})
    med = {}
    for k in keys:
        vals = sorted(r["s_per_step_median"] for r in ok if (r["config"], r["variant"]) == k)
        med[k] = vals[len(vals) // 2]
    ref = med.get(("conformer_ctc_baseline", "a"))
    print("\n" + "=" * 100)
    print(f"{'encoder':<24}{'bt':<4}{'tham số':>12}{'s/step':>9}{'× Conf-a':>10}{'khung':>8}"
          f"{'VRAM train':>12}{'RTF eval':>10}{'loss ok':>9}{'amp bỏ':>8}")
    for k in keys:
        r = next(x for x in ok if (x["config"], x["variant"]) == k)
        ratio = f"{med[k] / ref:.2f}" if ref else "-"
        vram = f"{r['peak_vram_train_gib']:.2f}" if r["peak_vram_train_gib"] is not None else "-"
        print(f"{k[0]:<24}{k[1]:<4}{r['encoder_params']:>12,}{med[k]:>9.3f}{ratio:>10}"
              f"{r['encoder_frames_max_mean']:>8.0f}{vram:>12}{r['eval_rtf']:>10.4f}{str(r['loss_finite']):>9}"
              f"{r['amp_skipped_steps']:>8}")
    for r in results:
        if "error" in r:
            print(f"!!! LỖI {r['config']} {r['variant']} lượt {r['repeat']}: {r['error']}\n{r['stderr_tail']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="thử trên CPU với code đang có ở máy (MambaRef)")
    args = ap.parse_args()

    if args.local:
        global REPEATS
        REPEATS = 1
        repo = Path(__file__).resolve().parents[3]
        out_json = Path(os.environ.get("TEMP", "/tmp")) / "subsample_speed_local.json"
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        extra = ["--device", "cpu", "--ref_mamba", "--batch", "2", "--warmup", "1", "--steps", "2",
                 "--eval_batches", "1", "--min_sec", "2", "--max_sec", "3"]
        summarize(measure(repo, extra, env, out_json, stop_on_first_error=False))
        print(f"\n(local) kết quả: {out_json}")
        return

    run(["nvidia-smi"], check=False)
    run(["git", "clone", "-q", REPO_URL, str(REPO_DIR)])
    run(["git", "checkout", "-q", COMMIT], cwd=REPO_DIR)
    run(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    # Đủ danh sách như train_full.py: worker import src.training.train → kéo theo jiwer, datasets, tensorboard
    # (v1 2026-10-08 thiếu jiwer, lỗi ngay cấu hình đầu).
    run(pip + ["einops", "librosa", "soundfile", "sentencepiece", "datasets", "jiwer", "pyyaml", "tensorboard"])
    wheels = sorted(str(p) for p in Path("/kaggle/input").rglob("*.whl"))
    if not wheels:
        run(["find", "/kaggle/input", "-maxdepth", "4"], check=False)
        sys.exit("LỖI: không thấy wheel mamba trong /kaggle/input — kiểm tra dataset mamba-wheels-v2")
    run(pip + ["--no-deps"] + wheels)
    run([sys.executable, "-c", "from mamba_ssm import Mamba; print('import mamba_ssm OK')"])

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": "0"}
    out_json = OUT / "subsample_speed_result.json"
    results = measure(REPO_DIR, [], env, out_json, stop_on_first_error=True)
    summarize(results)
    print(f"\nkết quả: {out_json}\nSUBSAMPLE SPEED XONG", flush=True)


if __name__ == "__main__":
    main()
