"""Training loop dùng chung cho cả Mamba-CTC và Conformer-CTC.

Chạy: python -m src.training.train --config configs/model_conformer.yaml
      python -m src.training.train --config configs/model_mamba.yaml
Kaggle (D8, 2 GPU + SyncBN + AMP; audio giải nén bằng src/data/extract_audio.py):
      python -m src.data.vietsuperspeech_dataset   # làm ấm cache HF một lần (có mạng)
      AUDIO_CACHE_DIR=/tmp/audio_cache HF_HUB_OFFLINE=1 torchrun --nproc_per_node 2 -m src.training.train \\
          --config configs/model_mamba.yaml --max_minutes 510 --ckpt_dir /kaggle/working/checkpoints

Cùng một script cho cả hai — chỉ config khác nhau ở mục `encoder.type`.
Đây là điều kiện bắt buộc để so sánh có kiểm soát (không được viết hai script
train riêng, dễ lệch nhau ở tiểu tiết và làm mất tính công bằng so sánh).

Checkpoint/resume tự động (mặc định bật, D7): Kaggle cắt phiên ~9-12 h, nên
`<ckpt_dir>/<experiment_name>/latest.pt` được ghi đè cuối mỗi epoch **và** mỗi
`--ckpt_every_minutes` giữa epoch; `--max_minutes` lưu rồi thoát êm trước khi bị
cắt. Thứ tự dữ liệu mỗi epoch chỉ phụ thuộc (seed, epoch) — resume giữa epoch
bỏ đúng các batch đã học. Phiên Kaggle mới: gắn output phiên trước làm input,
truyền `--resume_from .../latest.pt`. `--no_resume` để train lại từ đầu.

Lịch LR giữ warmup + hằng số (không decay): dừng ở epoch bất kỳ vẫn là một
điểm so sánh hợp lệ giữa các mô hình (train full, báo cáo theo epoch — 2026-09-28).
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

from src.data.vietsuperspeech_dataset import VietSuperSpeechDataset, collate_fn, manifest_indices
from src.evaluation.wer import compute_wer
from src.features.log_mel import LogMelFeatureExtractor
from src.models.conformer_encoder import ConformerEncoder
from src.models.ctc_model import CTCASRModel, ctc_loss
from src.models.mamba_encoder import MambaEncoder
from src.tokenizer.bpe_tokenizer import BPETokenizer

CHECKPOINT_ROOT = Path("checkpoints")
TENSORBOARD_ROOT = Path("runs")


def build_encoder(cfg: dict):
    enc_cfg = dict(cfg["encoder"])
    enc_type = enc_cfg.pop("type")
    if enc_type == "conformer":
        return ConformerEncoder(**enc_cfg)
    if enc_type == "mamba":
        return MambaEncoder(**enc_cfg)
    raise ValueError(f"Encoder type không hỗ trợ: {enc_type}")


def build_model(cfg: dict, vocab_size: int) -> CTCASRModel:
    """Chỗ duy nhất dựng model đầy đủ (train, eval_clean_test, kernel benchmark)
    → front-end (CMVN, SpecAugment) đọc từ mục `features` của yaml giống nhau."""
    fe = LogMelFeatureExtractor(**cfg["features"])
    return CTCASRModel(encoder=build_encoder(cfg), vocab_size=vocab_size, feature_extractor=fe)


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.AdamW:
    """AdamW, trừ weight decay cho tham số gắn cờ `_no_weight_decay`.

    mamba_ssm.Mamba gắn cờ này lên `A_log`, `D` (phạt về 0 là kéo A = -exp(A_log)
    về -1, làm lệch tốc độ "quên" của trạng thái). Conformer không có tham số
    nào mang cờ → với Conformer, kết quả y hệt AdamW trên toàn bộ tham số.
    Rà 2026-09-30 theo mã nguồn mamba-ssm v2.3.1 (TODO.md).
    """
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if getattr(p, "_no_weight_decay", False) else decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["training"]["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["training"]["lr"],
    )


def warmup_lr_lambda(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


class ResumableSampler(Sampler[int]):
    """Như `DistributedSampler(shuffle=True)` (đệm cho chia đều số GPU) nhưng bỏ
    qua được `skip` mẫu đầu phần của rank này — để resume giữa epoch đúng thứ
    tự dữ liệu (D7). Hoán vị chỉ phụ thuộc (seed, epoch) nên mọi rank, mọi phiên
    đều ra cùng một thứ tự."""

    def __init__(self, n: int, rank: int, world: int, seed: int, epoch: int, skip: int = 0):
        self.n, self.rank, self.world, self.seed, self.epoch, self.skip = n, rank, world, seed, epoch, skip
        self.per_rank = math.ceil(n / world)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=g).tolist()
        perm += perm[: self.per_rank * self.world - self.n]
        return iter(perm[self.rank :: self.world][self.skip :])

    def __len__(self):
        return self.per_rank - self.skip


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, **meta) -> None:
    """Ghi file tạm rồi đổi tên: Kaggle cắt phiên giữa lúc ghi thì latest.pt cũ vẫn nguyên."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            **meta,
        },
        tmp,
    )
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model: CTCASRModel, eval_loader: DataLoader, tokenizer: BPETokenizer, device, amp: bool = False) -> float:
    """WER greedy; với DDP mỗi rank giải mã phần của mình rồi gom lại (WER tính
    trên toàn bộ tập, không lặp mẫu nào)."""
    model.eval()
    references, hypotheses = [], []
    for batch in eval_loader:
        waveform = batch["waveform"].to(device)
        waveform_lengths = batch["waveform_lengths"].to(device)
        with torch.autocast(torch.device(device).type, dtype=torch.float16, enabled=amp):
            pred_ids_batch = model.greedy_decode(waveform, waveform_lengths)
        hypotheses.extend(tokenizer.decode(ids) for ids in pred_ids_batch)
        references.extend(batch["text"])
    model.train()
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (references, hypotheses))
        references = [r for refs, _ in gathered for r in refs]
        hypotheses = [h for _, hyps in gathered for h in hyps]
    return compute_wer(references, hypotheses)


def main():
    t_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer_model", default="configs/tokenizer.model")
    parser.add_argument("--eval_every", type=int, default=1, help="Số epoch giữa 2 lần eval WER")
    parser.add_argument("--no_resume", action="store_true", help="Bỏ qua checkpoint cũ, train lại từ đầu")
    parser.add_argument("--resume_from", default=None, help="latest.pt của phiên trước, dùng khi ckpt_dir chưa có")
    parser.add_argument("--train_manifests", nargs="+", default=None,
                        help="Ghi đè data.train_manifests, vd. train_shard0 cho train thử 1 shard")
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè training.epochs")
    parser.add_argument("--max_minutes", type=float, default=0,
                        help="Lưu rồi thoát sau chừng này phút tính từ lúc khởi động (0 = không giới hạn)")
    parser.add_argument("--ckpt_every_minutes", type=float, default=20)
    parser.add_argument("--ckpt_dir", default=str(CHECKPOINT_ROOT))
    parser.add_argument("--log_dir", default=str(TENSORBOARD_ROOT))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["training"]
    epochs = args.epochs or tcfg["epochs"]
    seed = tcfg["seed"]
    experiment_name = cfg["experiment_name"]

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    is_main = rank == 0
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    amp = tcfg["amp"] and device.type == "cuda"
    # Batch toàn cục giữ = batch_size của config (chia đều cho các GPU) để DDP
    # không đổi bài toán tối ưu so với 1 GPU (D8).
    if tcfg["batch_size"] % world:
        raise ValueError(f"batch_size {tcfg['batch_size']} không chia hết cho {world} tiến trình")
    batch_size = tcfg["batch_size"] // world

    tokenizer = BPETokenizer(args.tokenizer_model)
    train_split, train_indices = manifest_indices(args.train_manifests or cfg["data"]["train_manifests"])
    eval_split, eval_indices = manifest_indices([cfg["data"]["eval_manifest"]])
    train_ds = VietSuperSpeechDataset(split=train_split, tokenizer=tokenizer, indices=train_indices)
    eval_ds = VietSuperSpeechDataset(split=eval_split, tokenizer=tokenizer, indices=eval_indices[rank::world])
    loader_kw = dict(batch_size=batch_size, collate_fn=collate_fn, num_workers=tcfg["num_workers"],
                     pin_memory=device.type == "cuda")
    eval_loader = DataLoader(eval_ds, shuffle=False, **loader_kw)
    steps_per_epoch = math.ceil(math.ceil(len(train_ds) / world) / batch_size)

    torch.manual_seed(seed)
    model = build_model(cfg, tokenizer.vocab_size).to(device)
    encoder = model.encoder
    if is_main:
        print(f"[{experiment_name}] encoder params: {encoder.num_parameters():,} | train {len(train_ds)} câu "
              f"({steps_per_epoch} step/epoch), val {len(eval_indices)} câu | {world} tiến trình, "
              f"batch {batch_size}/tiến trình, AMP={amp}", flush=True)

    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_lr_lambda(step, tcfg["warmup_steps"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    ckpt_dir = Path(args.ckpt_dir) / experiment_name
    latest_ckpt = ckpt_dir / "latest.pt"
    best_ckpt = ckpt_dir / "best.pt"

    start_epoch, skip_steps, global_step, best_wer = 0, 0, 0, float("inf")
    resume_path = latest_ckpt if latest_ckpt.exists() else (Path(args.resume_from) if args.resume_from else None)
    if not args.no_resume and resume_path is not None:
        state = torch.load(resume_path, map_location=device)
        if state.get("n_train", len(train_ds)) != len(train_ds):
            raise ValueError(f"Checkpoint train trên {state['n_train']} câu, lần này {len(train_ds)} — "
                             "đổi dữ liệu thì dùng experiment_name khác hoặc --no_resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        global_step = state["global_step"]
        best_wer = state.get("best_wer", best_wer)
        # Checkpoint cũ (trước 2026-10-04) chỉ lưu cuối epoch, không có step_in_epoch.
        done = state.get("step_in_epoch", steps_per_epoch)
        start_epoch, skip_steps = (state["epoch"] + 1, 0) if done >= steps_per_epoch else (state["epoch"], done)
        if is_main:
            print(f"[{experiment_name}] resume từ {resume_path}: epoch {start_epoch} step {skip_steps} "
                  f"(global_step={global_step})", flush=True)

    # SyncBN chỉ chạy trên GPU; Conformer có BatchNorm trong conv module, Mamba không có (D8).
    if ddp and device.type == "cuda":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    core = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )

    writer = SummaryWriter(log_dir=str(Path(args.log_dir) / experiment_name)) if is_main else None

    def checkpoint(path: Path, epoch: int, step_in_epoch: int) -> None:
        if is_main:
            save_checkpoint(path, core, optimizer, scheduler, scaler, epoch=epoch, step_in_epoch=step_in_epoch,
                            global_step=global_step, best_wer=best_wer, n_train=len(train_ds))

    def rank0_decides(flag: int) -> int:
        """Quyết định theo đồng hồ phải giống nhau ở mọi rank, nếu không DDP treo."""
        if not ddp:
            return flag
        t = torch.tensor([flag], device=device)
        dist.broadcast(t, 0)
        return int(t.item())

    def out_of_time() -> bool:
        return args.max_minutes > 0 and time.time() - t_start > args.max_minutes * 60

    last_ckpt_time = time.time()
    for epoch in range(start_epoch, epochs):
        sampler = ResumableSampler(len(train_ds), rank, world, seed, epoch, skip_steps * batch_size)
        train_loader = DataLoader(train_ds, sampler=sampler, **loader_kw)
        step_in_epoch, skip_steps = skip_steps, 0
        loss_sum, loss_n = 0.0, 0
        model.train()
        for batch in train_loader:
            waveform = batch["waveform"].to(device, non_blocking=True)
            waveform_lengths = batch["waveform_lengths"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)

            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                log_probs, out_lengths = model(waveform, waveform_lengths)
            loss = ctc_loss(log_probs, out_lengths, targets, target_lengths)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            step_in_epoch += 1
            loss_value = loss.item()
            loss_sum += loss_value
            loss_n += 1
            if is_main:
                writer.add_scalar("train/loss", loss_value, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            # Bước cuối epoch không dừng ở đây: để eval + lưu cuối epoch chạy trước.
            if step_in_epoch < steps_per_epoch:
                ckpt_due = time.time() - last_ckpt_time > args.ckpt_every_minutes * 60
                action = rank0_decides(2 if out_of_time() else 1 if ckpt_due else 0)
                if action:
                    checkpoint(latest_ckpt, epoch, step_in_epoch)
                    last_ckpt_time = time.time()
                if action == 2:
                    if is_main:
                        print(f"Hết {args.max_minutes:g} phút: lưu epoch {epoch} step {step_in_epoch}/"
                              f"{steps_per_epoch}, thoát", flush=True)
                        writer.close()
                    if ddp:
                        dist.destroy_process_group()
                    return

        # Trung bình các step của phiên này trong epoch (thiếu phần trước resume nếu resume giữa epoch).
        if is_main:
            mean_loss = loss_sum / max(loss_n, 1)
            print(f"epoch {epoch}: loss trung bình={mean_loss:.4f} ({loss_n} step, "
                  f"{(time.time() - t_start) / 60:.1f} phút từ đầu phiên)", flush=True)
            writer.add_scalar("train/epoch_loss", mean_loss, epoch)

        is_last_epoch = epoch == epochs - 1
        if (epoch + 1) % args.eval_every == 0 or is_last_epoch:
            wer = evaluate(core, eval_loader, tokenizer, device, amp)
            if is_main:
                print(f"epoch {epoch}: eval WER={wer:.4f}", flush=True)
                writer.add_scalar("eval/wer", wer, epoch)
            if wer < best_wer:
                best_wer = wer
                checkpoint(best_ckpt, epoch, steps_per_epoch)
        checkpoint(latest_ckpt, epoch, steps_per_epoch)
        last_ckpt_time = time.time()
        if rank0_decides(int(out_of_time())) and not is_last_epoch:
            if is_main:
                print(f"Hết {args.max_minutes:g} phút sau epoch {epoch}, thoát", flush=True)
            break

    if is_main:
        writer.close()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
