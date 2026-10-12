"""Interface chung cho encoder — điểm hoán đổi DUY NHẤT giữa hai thí nghiệm.

MambaEncoder và ConformerEncoder đều implement class này. Nhờ vậy
training loop, CTC head, front-end, tokenizer dùng chung 100% code — chỉ
đổi encoder khi tạo model, đảm bảo so sánh có kiểm soát (ablation study
đúng nghĩa, không lẫn biến nhiễu).
"""

from abc import ABC, abstractmethod

import torch


class ASREncoder(torch.nn.Module, ABC):
    """input: [B, T, n_mels] log-mel features + độ dài thực.
    output: [B, T', d_model] biểu diễn ẩn + độ dài thực tương ứng.
    """

    @abstractmethod
    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor, layer_hook=None):
        """layer_hook(i, x) → x, gọi sau lớp thứ i (1-based) với luồng residual — cho InterCTC +
        self-conditioning (2026-10-12, ESPnet `interctc_layer_idx`/`interctc_use_conditioning`)."""
        raise NotImplementedError

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def normalize_intermediate(self, x: torch.Tensor) -> torch.Tensor:
        """Biểu diễn trung gian đưa vào CTC head. Mặc định giữ nguyên (mỗi lớp đã kết thúc bằng LayerNorm);
        chồng pre-norm (B1) ghi đè bằng norm cuối — như ESPnet chuẩn hóa intermediate bằng `after_norm`."""
        return x

    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
