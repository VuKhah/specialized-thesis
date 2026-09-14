"""Model CTC dùng chung: front-end -> encoder (Mamba hoặc Conformer) -> CTC head.

Đây là lớp lắp ráp thể hiện trực tiếp thiết kế "chỉ hoán đổi encoder" trong
đề cương (Chương 2). Không được thêm logic khác biệt giữa hai encoder ở đây.
"""

import torch
import torch.nn.functional as F

from src.features.log_mel import LogMelFeatureExtractor
from src.models.encoder_base import ASREncoder


class CTCASRModel(torch.nn.Module):
    def __init__(self, encoder: ASREncoder, vocab_size: int, feature_extractor: LogMelFeatureExtractor | None = None):
        super().__init__()
        self.feature_extractor = feature_extractor or LogMelFeatureExtractor()
        self.encoder = encoder
        self.ctc_head = torch.nn.Linear(encoder.output_dim, vocab_size)

    def forward(self, waveform: torch.Tensor, waveform_lengths: torch.Tensor):
        feats, feat_lengths = self.feature_extractor(waveform, waveform_lengths)
        hidden, out_lengths = self.encoder(feats, feat_lengths)
        logits = self.ctc_head(hidden)  # [B, T, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, out_lengths

    def compute_loss(self, waveform, waveform_lengths, targets, target_lengths, blank_id: int = 0):
        log_probs, out_lengths = self.forward(waveform, waveform_lengths)
        # CTCLoss cần [T, B, C]
        log_probs_tbc = log_probs.transpose(0, 1)
        return F.ctc_loss(
            log_probs_tbc,
            targets,
            out_lengths,
            target_lengths,
            blank=blank_id,
            zero_infinity=True,
        )
