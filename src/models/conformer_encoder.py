"""Conformer encoder — baseline so sánh, dùng khối Conformer chuẩn của torchaudio
(torchaudio.models.Conformer) để tránh tự cài đặt lại self-attention + conv
module từ đầu (rủi ro bug thấp hơn, tập trung effort vào phần Mamba/so sánh).
"""

import torch
import torchaudio

from src.models.encoder_base import ASREncoder


class ConformerEncoder(ASREncoder):
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 4,
        conv_kernel_size: int = 31,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._d_model = d_model
        self.input_proj = torch.nn.Linear(input_dim, d_model)
        self.conformer = torchaudio.models.Conformer(
            input_dim=d_model,
            num_heads=n_heads,
            ffn_dim=ff_dim,
            num_layers=n_layers,
            depthwise_conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

    @property
    def output_dim(self) -> int:
        return self._d_model

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor):
        x = self.input_proj(feats)  # [B, T, d_model]
        out, out_lengths = self.conformer(x, feat_lengths)
        return out, out_lengths
