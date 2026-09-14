"""Training loop dùng chung cho cả Mamba-CTC và Conformer-CTC.

Chạy: python -m src.training.train --config configs/model_conformer.yaml
      python -m src.training.train --config configs/model_mamba.yaml

Cùng một script cho cả hai — chỉ config khác nhau ở mục `encoder.type`.
Đây là điều kiện bắt buộc để so sánh có kiểm soát (không được viết hai script
train riêng, dễ lệch nhau ở tiểu tiết và làm mất tính công bằng so sánh).
"""

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.vietsuperspeech_dataset import VietSuperSpeechDataset, collate_fn
from src.models.conformer_encoder import ConformerEncoder
from src.models.ctc_model import CTCASRModel
from src.models.mamba_encoder import MambaEncoder
from src.tokenizer.bpe_tokenizer import BPETokenizer


def build_encoder(cfg: dict):
    enc_cfg = dict(cfg["encoder"])
    enc_type = enc_cfg.pop("type")
    if enc_type == "conformer":
        return ConformerEncoder(**enc_cfg)
    if enc_type == "mamba":
        return MambaEncoder(**enc_cfg)
    raise ValueError(f"Encoder type không hỗ trợ: {enc_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer_model", default="configs/tokenizer.model")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BPETokenizer(args.tokenizer_model)

    train_ds = VietSuperSpeechDataset(split=cfg["data"]["train_split"], tokenizer=tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    encoder = build_encoder(cfg)
    model = CTCASRModel(encoder=encoder, vocab_size=tokenizer.vocab_size).to(device)
    print(f"[{cfg['experiment_name']}] encoder params: {encoder.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])

    # TODO (Tuần 4-5): scheduler warmup, checkpoint/resume (quan trọng vì Kaggle
    # free-tier bị ngắt session), logging (tensorboard), eval loop định kỳ.
    for epoch in range(cfg["training"]["epochs"]):
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

        print(f"epoch {epoch}: loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
