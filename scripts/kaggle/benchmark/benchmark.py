"""Kernel Kaggle **GPU** (2x T4) — bước 2 của D9 trong
docs/notes/training_plan_kaggle.md mục 5: đo tốc độ train trước khi sửa code
dùng chung, để chọn tăng tốc theo thứ tự D8 (num_workers → AMP → 2 GPU).

    kaggle kernels push -p scripts/kaggle/benchmark -t 7200
    kaggle kernels status <username>/benchmark-asr
    kaggle kernels output <username>/benchmark-asr -p <thư mục>   (PYTHONUTF8=1)

Tốn ~30-40 phút quota GPU. Wheel mamba lấy từ output kernel verify-mamba-asr
(gắn qua kernel_sources) để khỏi build lại 13 phút. Mỗi cấu hình chạy trong
tiến trình con riêng (DDP cần torchrun; tách tiến trình cũng tránh VRAM/ cache
của cấu hình trước ảnh hưởng cấu hình sau).

Hạn chế của phép đo: audio đọc từ đĩa local của kernel (sau lần đầu còn nằm
trong page cache), không phải từ /kaggle/input như lúc train thật → lợi ích
của num_workers có thể bị đánh giá thấp.
"""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_URL = "https://github.com/VuKhah/specialized-thesis"
REPO_DIR = Path("/tmp/specialized-thesis")
HF_REVISION = "cbf624ae9b30e1c2793a27e95b262115c69601f3"
WARMUP, STEPS = 3, 40
CONFIGS = ["configs/model_conformer.yaml", "configs/model_mamba.yaml"]
# (workers mỗi tiến trình, amp, ddp, sync_bn). Lần 1 (2026-09-30, 20 step):
# workers 0/4 × fp32/AMP × 1/2 GPU — kết quả ở training_plan_kaggle.md mục 5.
# Lần 2 (D8 phương án c): chỉ AMP; tách nguyên nhân Conformer DDP chậm
# (worker tranh CPU? SyncBN?) và đo lại 1 GPU làm mốc với 40 step.
VARIANTS = [(4, True, False, False), (2, True, True, False), (2, True, True, True)]
# Ghi đè file của repo sau khi clone: {đường dẫn tương đối: nội dung}. Để trống
# khi code cần đo đã có trên GitHub; khi chưa push thì điền lúc đẩy kernel
# (không sửa tay ở đây) — xem nhật ký Plan.md 2026-09-30.
OVERLAY: dict[str, str] = {}

# Chép nguyên văn vào repo lúc chạy: kernel dạng script chỉ upload 1 file.
WORKER_CODE = r'''"""Đo tốc độ train 1 cấu hình (gọi từ benchmark.py, chạy trong thư mục repo).

    python bench_worker.py --config configs/model_mamba.yaml --workers 4 --amp
    torchrun --nproc_per_node 2 bench_worker.py --config ... --workers 4 --amp --ddp

Tự viết vòng step thay vì gọi train.py: đây là phép đo *trước khi* sửa code
dùng chung (D9), các tuỳ chọn num_workers/AMP/DDP chưa có trong train.py.
In 1 dòng `RESULT {json}` để benchmark.py gom lại.
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from src.data.vietsuperspeech_dataset import VietSuperSpeechDataset, collate_fn
from src.models.ctc_model import CTCASRModel
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.train import build_encoder, build_optimizer

sys.stdout.reconfigure(encoding="utf-8")

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--workers", type=int, default=0)
p.add_argument("--amp", action="store_true")
p.add_argument("--ddp", action="store_true")
p.add_argument("--sync_bn", action="store_true")
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--steps", type=int, default=20)
args = p.parse_args()

cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
global_bs = cfg["training"]["batch_size"]
if args.ddp:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
else:
    rank, world = 0, 1
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
# Giữ batch toàn cục = batch_size của config (chia đều cho các GPU) để DDP
# không đổi bài toán tối ưu so với 1 GPU.
bs = global_bs // world

tok = BPETokenizer("configs/tokenizer.model")
n_samples = global_bs * (args.warmup + args.steps)
ds = Subset(VietSuperSpeechDataset(split="train", tokenizer=tok), range(n_samples))
sampler = DistributedSampler(ds, shuffle=False) if args.ddp else None
loader = DataLoader(ds, batch_size=bs, sampler=sampler, shuffle=False, collate_fn=collate_fn,
                    num_workers=args.workers, pin_memory=args.workers > 0,
                    persistent_workers=args.workers > 0)

torch.manual_seed(0)
model = CTCASRModel(encoder=build_encoder(cfg), vocab_size=tok.vocab_size).to(device)
if args.amp:
    # Front-end log-mel giữ fp32: phổ công suất có thể vượt 65504 (max fp16) → inf.
    fe = model.feature_extractor
    fe_forward = fe.forward

    def fe_fp32(*a, **k):
        with torch.autocast("cuda", enabled=False):
            return fe_forward(*a, **k)

    fe.forward = fe_fp32
if args.sync_bn:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
n_bn = sum(isinstance(m, (torch.nn.BatchNorm1d, torch.nn.SyncBatchNorm)) for m in model.modules())
if args.ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
core = model.module if args.ddp else model
opt = build_optimizer(core, cfg)
scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

torch.cuda.reset_peak_memory_stats(device)
times, losses = [], []
t_prev = None
model.train()
for step, batch in enumerate(loader):
    # Thời gian 1 step tính cả chờ dữ liệu (từ cuối step trước) — thứ num_workers cải thiện.
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
    with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
        log_probs, out_lengths = model(batch["waveform"], batch["waveform_lengths"])
    loss = torch.nn.functional.ctc_loss(log_probs.float().transpose(0, 1), batch["targets"], out_lengths,
                                        batch["target_lengths"], blank=0, zero_infinity=True)
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
    scaler.step(opt)
    scaler.update()
    torch.cuda.synchronize(device)
    now = time.time()
    if t_prev is not None and step > args.warmup:
        times.append(now - t_prev)
    t_prev = now
    losses.append(loss.item())

if rank == 0:
    s_per_step = sum(times) / len(times)
    steps_per_epoch = math.ceil(60_656 / global_bs)
    print("RESULT " + json.dumps({
        "config": cfg["experiment_name"], "workers": args.workers, "amp": args.amp, "gpus": world,
        "sync_bn": args.sync_bn, "n_batchnorm": n_bn, "encoder_params": core.encoder.num_parameters(),
        "no_decay_params": len(opt.param_groups[1]["params"]),
        "global_batch": global_bs, "s_per_step": round(s_per_step, 3),
        "epoch_minutes": round(s_per_step * steps_per_epoch / 60, 1),
        "30_epoch_hours": round(s_per_step * steps_per_epoch * 30 / 3600, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
        "loss_first": round(losses[0], 3), "loss_last": round(losses[-1], 3),
        "loss_finite": all(math.isfinite(x) for x in losses),
        "amp_scale": scaler.get_scale() if args.amp else None,
    }), flush=True)
if args.ddp:
    dist.destroy_process_group()
'''


def run(cmd, cwd=None, check=True):
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-4000:], flush=True)
    if check and r.returncode != 0:
        sys.exit(f"LỖI: lệnh trên trả về {r.returncode}")
    return r


def prefetch(n_files):
    # Tải đúng các file mà Subset(range(n)) của split train cần, vào đường dẫn
    # cache mà VietSuperSpeechDataset đọc (tương đối theo cwd = repo).
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    ds = load_dataset("thanhnew2001/VietSuperSpeech", split="train")
    paths = [ds[i]["audio"] for i in range(n_files)]
    cache = REPO_DIR / "data/raw/audio_cache"

    def one(rel):
        hf_hub_download("thanhnew2001/VietSuperSpeech", rel, repo_type="dataset",
                        revision=HF_REVISION, local_dir=cache)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, paths))
    print(f"Đã tải {len(paths)} file audio.", flush=True)


def main():
    run(["nvidia-smi"], check=False)
    # Số lõi CPU (worker tranh nhau?) và đĩa /tmp (đủ giải nén 4 shard tar ~25 GB?).
    run(["nproc"], check=False)
    run(["df", "-h", "/tmp", "/kaggle/working"], check=False)
    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    run(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)
    for rel, content in OVERLAY.items():
        (REPO_DIR / rel).write_text(content, encoding="utf-8")
        print(f"OVERLAY: ghi đè {rel} ({len(content)} ký tự)", flush=True)
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run(pip + ["einops", "librosa", "soundfile", "sentencepiece", "datasets", "jiwer", "pyyaml", "tensorboard"])
    # Tìm đệ quy: đường dẫn gắn kernel_sources không cố định (lần chạy
    # 2026-09-30 không khớp /kaggle/input/*/wheels/ như lúc verify-mamba).
    wheels = sorted(str(p) for p in Path("/kaggle/input").rglob("*.whl"))
    if not wheels:
        run(["find", "/kaggle/input", "-maxdepth", "4"], check=False)
        sys.exit("LỖI: không thấy wheel mamba trong /kaggle/input — kiểm tra kernel_sources")
    run(pip + ["--no-deps"] + wheels)
    run([sys.executable, "-c", "from mamba_ssm import Mamba; print('import mamba_ssm OK')"])

    (REPO_DIR / "bench_worker.py").write_text(WORKER_CODE, encoding="utf-8")
    sys.path.insert(0, str(REPO_DIR))
    prefetch(16 * (WARMUP + STEPS))

    results = []
    for cfg in CONFIGS:
        for workers, amp, ddp, sync_bn in VARIANTS:
            if sync_bn and "mamba" in cfg:
                continue  # Mamba không có BatchNorm — SyncBN không đổi gì
            launcher = ([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2"] if ddp
                        else [sys.executable])
            cmd = launcher + ["bench_worker.py", "--config", cfg, "--workers", str(workers),
                              "--warmup", str(WARMUP), "--steps", str(STEPS)]
            cmd += ["--amp"] * amp + ["--ddp"] * ddp + ["--sync_bn"] * sync_bn
            r = run(cmd, cwd=REPO_DIR, check=False)
            line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
            if line:
                results.append(json.loads(line[len("RESULT "):]))
            else:
                results.append({"config": cfg, "workers": workers, "amp": amp, "ddp": ddp,
                                "error": f"returncode {r.returncode}", "stderr_tail": r.stderr[-1500:]})
                if len(results) == 1:
                    # Cấu hình đầu đã lỗi thì gần như chắc là lỗi code, không phải
                    # cấu hình → dừng để khỏi đốt quota (lần 2026-09-30 mất ~9 phút).
                    Path("/kaggle/working/benchmark_result.json").write_text(
                        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                    sys.exit("LỖI: cấu hình đầu tiên thất bại — dừng benchmark")

    Path("/kaggle/working/benchmark_result.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 70 + "\nKẾT QUẢ\n" + "=" * 70)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("\nBENCHMARK XONG", flush=True)


if __name__ == "__main__":
    main()
