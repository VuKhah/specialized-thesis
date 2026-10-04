"""ConExtBiMamba — khung Conformer, thay MHSA bằng ExtBiMamba (ứng viên #5).

Theo Zhang và cs., *Mamba in Speech*, arXiv:2405.12609 (v6), Hình 2c, Bảng XII
và XIV: giữ macaron FFN + conv module (Swish) của Conformer, chỗ MHSA đặt cặp
Mamba xuôi/ngược kiểu "ngoài" (chung LayerNorm, cộng — giống hệt một lớp B1),
không positional encoding (Bảng XIV: thêm PE không đổi WER).

Lệch bài có chủ đích (ghi ở docs/notes/lineup_preparation.md mục 5):
- Bài giữ nguyên kích thước Conformer nên ConExtBiMamba nhiều tham số hơn
  (34,23M → 41,59M, Bảng XII). Ở đây khớp ~12M với Conformer-12M bằng cách
  giảm số lớp + ffn_dim (yaml), giữ khối Mamba mặc định như B1.
- Bài khởi tạo A bằng ma trận chéo + nhiễu Gauss (Bảng XIV); ở đây dùng khởi
  tạo mặc định S4D-Real của mamba_ssm.Mamba v2.3.1 — cùng khối với B1, để hai
  ứng viên Mamba chỉ khác khung chứ không khác khối.

FFN và conv module lấy nguyên lớp của torchaudio (cùng mã với ConformerEncoder)
để chênh lệch với Conformer-12M chỉ nằm ở chỗ MHSA ↔ ExtBiMamba.
"""

import torch
from torchaudio.models.conformer import _ConvolutionModule, _FeedForwardModule

from src.models.encoder_base import ASREncoder
from src.models import mamba_encoder
from src.models.mamba_encoder import reverse_padded


class ConExtBiMambaLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        ff_dim: int,
        conv_kernel_size: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ):
        super().__init__()
        self.ffn1 = _FeedForwardModule(d_model, ff_dim, dropout=dropout)
        self.mamba_norm = torch.nn.LayerNorm(d_model)
        # Tra qua module (không import thẳng tên) để test CPU thay được bằng khối giả.
        self.mamba_fwd = mamba_encoder._MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = mamba_encoder._MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_dropout = torch.nn.Dropout(dropout)
        self.conv_module = _ConvolutionModule(
            input_dim=d_model,
            num_channels=d_model,
            depthwise_kernel_size=conv_kernel_size,
            dropout=dropout,
            bias=True,
        )
        self.ffn2 = _FeedForwardModule(d_model, ff_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(d_model)

    def _conv(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # Gọi lại từng bước của _ConvolutionModule.forward để xoá padding ngay
        # trước depthwise conv (kernel 31, không nhân quả): không xoá thì giá trị
        # ở khung padding lọt vào ~15 khung thật cuối mỗi câu, mất tính bất biến
        # padding mà cặp Mamba giữ được. torchaudio Conformer không xoá (attention
        # có key_padding_mask, conv thì không) — lệch nhỏ so với Conformer-12M.
        # BatchNorm lúc train vẫn tính thống kê cả khung padding, như Conformer.
        m = self.conv_module
        h = m.layer_norm(x).transpose(1, 2)  # [B, D, T]
        h = m.sequential[1](m.sequential[0](h))  # pointwise + GLU
        h = h.masked_fill(pad_mask.unsqueeze(1), 0.0)
        h = m.sequential[2:](h)
        return h.transpose(1, 2)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        h = self.mamba_norm(x)
        h_bwd = reverse_padded(self.mamba_bwd(reverse_padded(h, lengths)), lengths)
        x = x + self.mamba_dropout(self.mamba_fwd(h) + h_bwd)
        x = x + self._conv(x, pad_mask)
        x = x + 0.5 * self.ffn2(x)
        return self.final_layer_norm(x)


class ConExtBiMambaEncoder(ASREncoder):
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        n_layers: int = 6,
        ff_dim: int = 1024,
        conv_kernel_size: int = 31,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not mamba_encoder.MAMBA_SSM_AVAILABLE:
            raise ImportError("mamba-ssm chưa được cài (cần CUDA) — xem notebooks/00_setup_environment.ipynb.")
        self._d_model = d_model
        self.input_proj = torch.nn.Linear(input_dim, d_model)
        # Không chia out_proj cho √(số khối) như MambaEncoder: hệ số đó của
        # mamba-ssm bù cho residual cộng dồn qua cả chồng pre-norm, còn ở đây mỗi
        # lớp kết thúc bằng final_layer_norm (như MHSA/FFN của torchaudio, cũng
        # không chia). Suy luận của dự án — bài không nói.
        self.layers = torch.nn.ModuleList(
            [
                ConExtBiMambaLayer(d_model, ff_dim, conv_kernel_size, d_state, d_conv, expand, dropout)
                for _ in range(n_layers)
            ]
        )

    @property
    def output_dim(self) -> int:
        return self._d_model

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor):
        x = self.input_proj(feats)  # [B, T, d_model]
        pad_mask = torch.arange(x.size(1), device=x.device) >= feat_lengths.to(x.device).unsqueeze(1)  # [B, T]
        for layer in self.layers:
            x = layer(x, feat_lengths, pad_mask)
        return x, feat_lengths
