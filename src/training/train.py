"""Training loop dùng chung cho cả Mamba-CTC và Conformer-CTC.

Chạy: python -m src.training.train --config configs/model_conformer.yaml
      python -m src.training.train --config configs/model_mamba.yaml

Cùng một script cho cả hai — chỉ config khác nhau ở mục `encoder.type`.
Đây là điều kiện bắt buộc để so sánh có kiểm soát (không được viết hai script
train riêng, dễ lệch nhau ở tiểu tiết và làm mất tính công bằng so sánh).

Checkpoint/resume tự động (mặc định bật): Kaggle free-tier bị ngắt session
sau ~9-12h, nên mỗi epoch đều ghi đè `checkpoints/<experiment_name>/latest.pt`
— chạy lại đúng lệnh trên là tự tiếp tục từ epoch dở dang, không cần cờ gì
thêm. Dùng `--no_resume` khi muốn train lại từ đầu (vd. đổi kiến trúc/config
nhưng giữ nguyên experiment_name).
"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.vietsuperspeech_dataset import VietSuperSpeechDataset, collate_fn
from src.evaluation.wer import compute_wer
from src.models.conformer_encoder import ConformerEncoder
from src.models.ctc_model import CTCASRModel
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


def warmup_lr_lambda(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, global_step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def evaluate(model: CTCASRModel, eval_loader: DataLoader, tokenizer: BPETokenizer, device: str) -> float:
    model.eval()
    references, hypotheses = [], []
    for batch in eval_loader:
        waveform = batch["waveform"].to(device)
        waveform_lengths = batch["waveform_lengths"].to(device)
        pred_ids_batch = model.greedy_decode(waveform, waveform_lengths)
        hypotheses.extend(tokenizer.decode(ids) for ids in pred_ids_batch)
        references.extend(batch["text"])
    model.train()
    return compute_wer(references, hypotheses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer_model", default="configs/tokenizer.model")
    parser.add_argument("--eval_every", type=int, default=1, help="Số epoch giữa 2 lần eval WER")
    parser.add_argument("--no_resume", action="store_true", help="Bỏ qua checkpoint cũ, train lại từ đầu")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BPETokenizer(args.tokenizer_model)
    experiment_name = cfg["experiment_name"]

    train_ds = VietSuperSpeechDataset(split=cfg["data"]["train_split"], tokenizer=tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    eval_ds = VietSuperSpeechDataset(split=cfg["data"]["eval_split"], tokenizer=tokenizer)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    encoder = build_encoder(cfg)
    model = CTCASRModel(encoder=encoder, vocab_size=tokenizer.vocab_size).to(device)
    print(f"[{experiment_name}] encoder params: {encoder.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_lr_lambda(step, cfg["training"]["warmup_steps"]),
    )

    ckpt_dir = CHECKPOINT_ROOT / experiment_name
    latest_ckpt = ckpt_dir / "latest.pt"
    best_ckpt = ckpt_dir / "best.pt"

    start_epoch = 0
    global_step = 0
    best_wer = float("inf")
    if not args.no_resume and latest_ckpt.exists():
        state = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        print(f"[{experiment_name}] resume từ checkpoint epoch {state['epoch']} (global_step={global_step})")

    writer = SummaryWriter(log_dir=str(TENSORBOARD_ROOT / experiment_name))

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        model.train()
        for batch in train_loader:
            waveform = batch["waveform"].to(device)
            waveform_lengths = batch["waveform_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            loss = model.compute_loss(waveform, waveform_lengths, targets, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            optimizer.step()
            scheduler.step()

            global_step += 1
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        print(f"epoch {epoch}: loss={loss.item():.4f}")
        writer.add_scalar("train/epoch_loss", loss.item(), epoch)

        is_last_epoch = epoch == cfg["training"]["epochs"] - 1
        if (epoch + 1) % args.eval_every == 0 or is_last_epoch:
            wer = evaluate(model, eval_loader, tokenizer, device)
            print(f"epoch {epoch}: eval WER={wer:.4f}")
            writer.add_scalar("eval/wer", wer, epoch)
            if wer < best_wer:
                best_wer = wer
                save_checkpoint(best_ckpt, model, optimizer, scheduler, epoch, global_step)

        save_checkpoint(latest_ckpt, model, optimizer, scheduler, epoch, global_step)

    writer.close()


if __name__ == "__main__":
    main()
