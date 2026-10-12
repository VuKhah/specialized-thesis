"""Fine-tune Parakeet-CTC-0.6B-vi (#4, nhóm pre-train) trên train VietSuperSpeech.

Chạy (Kaggle 2x T4, qua kernel `scripts/kaggle/finetune_parakeet/`):
    AUDIO_CACHE_DIR=/tmp/audio_cache HF_HUB_OFFLINE=1 torchrun --nproc_per_node 2 -m src.training.finetune_parakeet \\
        --nemo_path <file .nemo> --out_dir /kaggle/working/parakeet_ft --max_minutes 380 --total_steps 6000
    ... --probe   # 1 GPU: đo s/step + VRAM đỉnh trên batch dài nhất, không train thật

Chốt 2026-10-11 (người dùng): chỉ fine-tune Parakeet (PhoWhisper giữ zero-shot), trần ~7 GPU-h cho lần chính
(≈ ngân sách Conformer-12M 6,6 h → so được trên trục tài nguyên), chạy ngay, chấm clean-test lại sau hiệu đính.

Nhóm pre-train giữ pipeline riêng (luật hai mức so sánh, CLAUDE.md): front-end (`model.preprocessor`),
SpecAugment (`model.spec_augmentation`, tự bật khi `train()`), tokenizer và CTC head của .nemo — không dùng
log-mel/CMVN/BPE 1000 của nhóm train từ đầu, không speed perturbation. Chỉ dùng chung: chọn câu
(`manifest_rows` — đã bỏ excluded.tsv và video giữ riêng), sampler bucketing, `ctc_loss`/`greedy_collapse`
(blank của NeMo là chỉ số CUỐI, không phải 0), `compute_wer` (chuẩn hóa chung) và tập đánh giá.

Nhãn train đi qua `normalize_text` (chữ thường, bỏ dấu câu) trước khi token hóa: nhãn gốc VIẾT HOA, còn
tokenizer của Parakeet có phân biệt hoa/thường (train trên nhãn Qwen3 thêm hoa + dấu câu) — chữ thường gần
phân bố pre-train hơn, và WER vốn tính sau `normalize_text`.

Không resume giữa phiên: một lần chính ≤ 7 h nằm gọn trong phiên Kaggle 9 h, còn trạng thái optimizer của
600M tham số (~4,8 GB) ghi định kỳ thì tốn đĩa /kaggle/working (20 GB). Chỉ lưu trọng số `best.pt` (theo WER
val) + `last.pt`.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from src.data.vietsuperspeech_dataset import VietSuperSpeechDataset, collate_fn, manifest_rows, video_of
from src.evaluation.text_normalize import normalize_text
from src.evaluation.wer import compute_wer
from src.models.ctc_model import ctc_loss, greedy_collapse
from src.training.train import ResumableSampler, _log

TRAIN_MANIFESTS = ["train_shard0", "train_shard1", "train_shard2", "train_shard3"]
EVAL_MANIFESTS = ["val", "val_unseen"]  # tập đầu chọn best.pt — như nhóm train từ đầu


class NemoTextTokenizer:
    """Cầu nối `VietSuperSpeechDataset(tokenizer=...)` (gọi `.encode(text)`) với tokenizer của .nemo."""

    def __init__(self, nemo_tokenizer):
        self.tok = nemo_tokenizer

    def encode(self, text: str) -> list[int]:
        return self.tok.text_to_ids(normalize_text(text))

    def decode(self, ids: list[int]) -> str:
        return self.tok.ids_to_text(ids) if ids else ""


def load_parakeet(nemo_path: str, device):
    import nemo.collections.asr as nemo_asr

    return nemo_asr.models.ASRModel.restore_from(nemo_path, map_location=device)


def forward(model, core, waveform, lengths, amp: bool):
    """Front-end luôn fp32 (như zero-shot); encoder theo `amp`. Gọi qua `model` (wrapper DDP) để có hook
    đồng bộ gradient; `core` là mô hình gốc."""
    feats, feats_len = core.preprocessor(input_signal=waveform, length=lengths)
    with torch.autocast(waveform.device.type, dtype=torch.float16, enabled=amp):
        log_probs, enc_len, _ = model(processed_signal=feats, processed_signal_length=feats_len)
    return log_probs.float(), enc_len


@torch.no_grad()
def evaluate(model, core, loader, rows, tok, blank, device, amp) -> tuple[float, float, list[dict]]:
    core.eval()
    refs, hyps, loss_sum, n = [], [], 0.0, 0
    t0 = time.time()
    for i, batch in enumerate(loader, 1):
        if i % 100 == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            _log(f"  eval batch {i}/{len(loader)} ({time.time() - t0:.0f}s)")  # watchdog kernel: im 10 phút là kill
        log_probs, enc_len = forward(core, core, batch["waveform"].to(device), batch["waveform_lengths"].to(device), amp)
        loss = ctc_loss(log_probs, enc_len, batch["targets"].to(device), batch["target_lengths"].to(device), blank)
        loss_sum += loss.item() * len(batch["text"])
        n += len(batch["text"])
        hyps.extend(tok.decode(ids) for ids in greedy_collapse(log_probs, enc_len, blank))
        refs.extend(batch["text"])
    core.train()
    records = [{"split": r["split"], "index": r["index"], "video": video_of(r["audio"]), "ref": ref, "hyp": hyp}
               for r, ref, hyp in zip(rows, refs, hyps, strict=True)]
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (records, loss_sum, n))
        records = [rec for recs, _, _ in gathered for rec in recs]
        loss_sum, n = sum(g[1] for g in gathered), sum(g[2] for g in gathered)
    return compute_wer([r["ref"] for r in records], [r["hyp"] for r in records]), loss_sum / max(n, 1), records


def lr_factor(step: int, warmup: int, total: int, min_ratio: float) -> float:
    """Warmup tuyến tính rồi cosine về `min_ratio` tại `total` (total = 0 → hằng sau warmup) — cùng dạng
    `CosineAnnealing` của NeMo. Mặc định theo `asr_finetune/speech_to_text_finetune.yaml` của NeMo (AdamW 1e-4,
    cosine, min_lr 5e-6 = 0,05×; người dùng chốt LR 1e-4 ngày 2026-10-12). Bản 1e-5 của linto_stt_fr_fastconformer
    đi kèm 100k step trên 9.500 h — ngân sách ~7 GPU-h ở đây ít step hơn nhiều. Khác nhóm train từ đầu (warmup +
    hằng) vì ở đây ngân sách cố định, không báo cáo theo epoch. `total` phải khớp số step chạy được trong
    `--max_minutes` (tính từ s/step của trial), nếu không cosine bị cắt giữa chừng."""
    if step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    progress = min(1.0, (step - warmup) / (total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def probe(core, train_ds, rows, args, device, blank) -> None:
    """Đo trên 1 GPU, batch gồm các câu DÀI NHẤT (VRAM xấu nhất): s/step, VRAM đỉnh, loss hữu hạn hay không,
    tham số không nhận gradient (quyết `find_unused_parameters` của DDP)."""
    longest = sorted(range(len(rows)), key=lambda i: -rows[i]["duration"])[: args.per_gpu_batch]
    batch = collate_fn([train_ds[i] for i in longest])
    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    torch.cuda.reset_peak_memory_stats(device)
    times, losses = [], []
    for step in range(args.probe_steps):
        torch.cuda.synchronize(device)
        t = time.time()
        log_probs, enc_len = forward(core, core, batch["waveform"].to(device), batch["waveform_lengths"].to(device),
                                     args.amp)
        loss = ctc_loss(log_probs, enc_len, batch["targets"].to(device), batch["target_lengths"].to(device), blank)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if step == 0:
            unused = [n for n, p in core.named_parameters() if p.requires_grad and p.grad is None]
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(core.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize(device)
        times.append(time.time() - t)
        losses.append(loss.item())
    out = {"amp": args.amp, "per_gpu_batch": args.per_gpu_batch, "max_duration_s": rows[longest[0]]["duration"],
           "s_per_step_median": sorted(times[1:])[len(times[1:]) // 2] if len(times) > 1 else times[0],
           "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30, "losses": losses,
           "all_finite": all(math.isfinite(x) for x in losses), "unused_params": unused[:20],
           "n_unused_params": len(unused)}
    print("PROBE " + json.dumps(out, ensure_ascii=False), flush=True)


def main():
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr_ratio", type=float, default=0.05)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--total_steps", type=int, default=0, help="Cuối lịch cosine; 0 = hằng sau warmup")
    ap.add_argument("--weight_decay", type=float, default=1e-3)  # = NeMo speech_to_text_finetune.yaml
    ap.add_argument("--grad_clip", type=float, default=5.0)  # = nhóm train từ đầu (lựa chọn dự án)
    ap.add_argument("--global_batch", type=int, default=16)  # = nhóm train từ đầu
    ap.add_argument("--per_gpu_batch", type=int, default=8)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast cho encoder (06/10: fp16 tràn ở 16/203 câu)")
    ap.add_argument("--max_minutes", type=float, default=0)
    ap.add_argument("--eval_every_steps", type=int, default=0, help="0 = chỉ eval đầu (zero-shot) + cuối")
    ap.add_argument("--no_eval_at_start", action="store_true")
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every_steps", type=int, default=50)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe_steps", type=int, default=6)
    args = ap.parse_args()

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group("nccl")
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    is_main = rank == 0
    if torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    else:  # chỉ để test vòng lặp ở local bằng mô hình giả
        device = torch.device("cpu")
    if args.global_batch % (world * args.per_gpu_batch):
        raise ValueError(f"global_batch {args.global_batch} không chia hết cho {world} GPU × {args.per_gpu_batch}")
    accum = args.global_batch // (world * args.per_gpu_batch)

    torch.manual_seed(args.seed)
    core = load_parakeet(args.nemo_path, device)
    blank = core.decoder.num_classes_with_blank - 1
    tok = NemoTextTokenizer(core.tokenizer)
    items = lambda rows: [(r["split"], r["index"]) for r in rows]  # noqa: E731

    train_rows = manifest_rows(TRAIN_MANIFESTS)
    train_rows = train_rows[: args.limit_train] if args.limit_train else train_rows
    train_ds = VietSuperSpeechDataset(tokenizer=tok, items=items(train_rows))
    if args.probe:
        core.train()
        probe(core, train_ds, train_rows, args, device, blank)
        return

    loader_kw = dict(collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    eval_rows, eval_loaders = {}, {}
    for name in EVAL_MANIFESTS:
        rows = manifest_rows([name])
        rows = rows[: args.limit_eval] if args.limit_eval else rows
        eval_rows[name] = rows[rank::world]
        eval_loaders[name] = DataLoader(VietSuperSpeechDataset(tokenizer=tok, items=items(eval_rows[name])),
                                        batch_size=args.eval_batch, shuffle=False, **loader_kw)

    if ddp:
        core = torch.nn.SyncBatchNorm.convert_sync_batchnorm(core)  # như nhóm train từ đầu (D8)
    core.train()
    # NeMo có thể giữ module không dùng ở CTC thuần (vd. nhánh phụ); probe in danh sách — True cho chắc.
    model = (torch.nn.parallel.DistributedDataParallel(core, device_ids=[device.index], find_unused_parameters=True,
                                                       gradient_as_bucket_view=True) if ddp else core)
    optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, args.warmup_steps, args.total_steps, args.min_lr_ratio))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    steps_per_epoch = math.ceil(math.ceil(len(train_ds) / world) / args.per_gpu_batch) // accum
    if is_main:
        n_params = sum(p.numel() for p in core.parameters())
        print(f"[parakeet_ft] {n_params:,} tham số | train {len(train_ds)} câu ({steps_per_epoch} step/epoch, "
              f"batch toàn cục {args.global_batch} = {world}×{args.per_gpu_batch}×{accum}) | lr {args.lr} "
              f"warmup {args.warmup_steps} total {args.total_steps} | AMP={args.amp} | blank {blank}", flush=True)
        (out / "run_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    best_wer = float("inf")

    def rank0_decides(flag: int) -> int:
        if not ddp:
            return flag
        t = torch.tensor([flag], device=device)
        dist.broadcast(t, 0)
        return int(t.item())

    def save_weights(path: Path, **meta) -> None:
        if is_main:
            tmp = path.with_suffix(".tmp")
            torch.save({"model": core.state_dict(), **meta}, tmp)
            os.replace(tmp, path)
            _log(f"lưu {path.name} ({path.stat().st_size / 2**30:.2f} GiB)")

    def run_eval(step: int, epoch: float, extra: dict) -> None:
        nonlocal best_wer
        m = {"global_step": step, "epoch": epoch, "lr": scheduler.get_last_lr()[0],
             "wall_seconds": time.time() - t_start, **extra}
        for name, loader in eval_loaders.items():
            t = time.time()
            wer, loss, records = evaluate(model, core, loader, eval_rows[name], tok, blank, device, args.amp)
            m[f"wer_{name}"], m[f"loss_{name}"], m[f"eval_seconds_{name}"] = wer, loss, time.time() - t
            if is_main:
                with open(out / f"eval_{name}_step{step}.jsonl", "w", encoding="utf-8") as f:
                    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        if is_main:
            _log("eval step {}: ".format(step) + "  ".join(f"{n}: WER={m[f'wer_{n}']:.4f}" for n in eval_loaders))
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        if step > 0 and m[f"wer_{EVAL_MANIFESTS[0]}"] < best_wer:
            best_wer = m[f"wer_{EVAL_MANIFESTS[0]}"]
            save_weights(out / "best.pt", global_step=step, **{k: v for k, v in m.items() if k.startswith("wer_")})

    if not args.no_eval_at_start:
        run_eval(0, 0.0, {"note": "zero-shot, trước fine-tune"})

    global_step, epoch, done = 0, 0, False
    t_train = time.time()
    stats = {"nonfinite": 0, "amp_skipped": 0, "loss_sum": 0.0, "loss_n": 0}
    peak_vram = (lambda: torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else (lambda: 0.0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while not done:
        sampler = ResumableSampler(len(train_ds), rank, world, args.seed, epoch, 0, batch_size=args.per_gpu_batch * accum,
                                   lengths=[r["duration"] for r in train_rows], pool_batches=50)
        loader = DataLoader(train_ds, sampler=sampler, batch_size=args.per_gpu_batch, **loader_kw)
        micro, window_has_grad = 0, False
        for batch in loader:
            log_probs, enc_len = forward(model, core, batch["waveform"].to(device, non_blocking=True),
                                         batch["waveform_lengths"].to(device, non_blocking=True), args.amp)
            loss = ctc_loss(log_probs, enc_len, batch["targets"].to(device), batch["target_lengths"].to(device), blank)
            ok = torch.isfinite(loss.detach()).float()
            if ddp:
                dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if ok.item():
                scaler.scale(loss / accum).backward()
                stats["loss_sum"] += loss.item()
                stats["loss_n"] += 1
                window_has_grad = True
            else:
                stats["nonfinite"] += 1
            del loss, log_probs, enc_len
            micro += 1
            if micro % accum:
                continue
            # Cả cửa sổ bị bỏ thì không gọi scaler.step: GradScaler báo lỗi khi chưa có gradient nào (AMP).
            if window_has_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(core.parameters(), args.grad_clip)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if args.amp and scaler.get_scale() < scale_before:
                    stats["amp_skipped"] += 1
            window_has_grad = False
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if is_main and args.log_every_steps and global_step % args.log_every_steps == 0:
                sps = (time.time() - t_train) / global_step
                _log(f"[parakeet_ft] step {global_step} (epoch {epoch}) loss tb "
                     f"{stats['loss_sum'] / max(stats['loss_n'], 1):.3f} | {sps:.3f} s/step | lr "
                     f"{scheduler.get_last_lr()[0]:.2e} | bỏ {stats['nonfinite']}+{stats['amp_skipped']} | "
                     f"VRAM đỉnh {peak_vram():.1f} GiB | "
                     f"{(time.time() - t_start) / 60:.1f} phút")
            finished = args.total_steps and global_step >= args.total_steps
            stop = rank0_decides(int(bool(finished) or (args.max_minutes > 0 and time.time() - t_start > args.max_minutes * 60)))
            due = args.eval_every_steps and global_step % args.eval_every_steps == 0
            if stop or due:
                extra = {"train_loss_mean": stats["loss_sum"] / max(stats["loss_n"], 1),
                         "nonfinite_steps": stats["nonfinite"], "amp_skipped_steps": stats["amp_skipped"],
                         "s_per_step": (time.time() - t_train) / global_step,
                         "peak_vram_gib": peak_vram(),
                         "train_seconds": time.time() - t_train}
                run_eval(global_step, global_step / max(steps_per_epoch, 1), extra)
            if stop:
                done = True
                break
        epoch += 1

    save_weights(out / "last.pt", global_step=global_step)
    if is_main:
        print(f"[parakeet_ft] xong: {global_step} step, best WER val {best_wer:.4f}, "
              f"{(time.time() - t_start) / 60:.1f} phút", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
