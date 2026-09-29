"""Mamba (Selective State Space Model / S6) encoder — kiến trúc Mamba-1 cổ
điển (mamba_ssm.Mamba, pin tag v2.3.1), đúng như trích dẫn trong đề cương
(Gu & Dao, arXiv:2312.00752). KHÔNG dùng Mamba-2/Mamba-3 (khác kiến trúc,
khác mục tiêu thiết kế, ngoài phạm vi đề cương đã đăng ký — xem
docs/notes/mamba_versions.md).

Xem notebooks/00_setup_environment.ipynb để xác nhận `mamba-ssm` CUDA kernel
có chạy được trên T4 hay không. Nếu không, dùng selective_scan_ref (thuần
PyTorch, chậm hơn nhưng không cần build kernel) — đúng phương án dự phòng
đã ghi trong đề cương.
"""

import math

import torch

from src.models.encoder_base import ASREncoder

try:
    from mamba_ssm import Mamba as _MambaBlock

    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False


class MambaEncoder(ASREncoder):
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        n_layers: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        if not MAMBA_SSM_AVAILABLE:
            raise ImportError(
                "mamba-ssm chưa được cài. Chạy notebooks/00_setup_environment.ipynb "
                "trên Kaggle T4 trước, hoặc cài theo hướng dẫn trong README.md."
            )

        self._d_model = d_model
        self.input_proj = torch.nn.Linear(input_dim, d_model)
        self.layers = torch.nn.ModuleList(
            [
                _MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(n_layers)
            ]
        )
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(d_model) for _ in range(n_layers)])
        # Ba chỗ dưới theo MixerModel/_init_weights của mamba-ssm v2.3.1 (rà
        # 2026-09-30): dùng khối Mamba lẻ thì không tự có. norm_f: chồng pre-norm
        # để residual chưa chuẩn hoá đi thẳng vào CTC head — Conformer torchaudio
        # thì mỗi lớp đã kết thúc bằng LayerNorm.
        self.norm_f = torch.nn.LayerNorm(d_model)
        # rescale_prenorm_residual: 28 nhánh cộng dồn vào residual → chia out_proj
        # (đã khởi tạo kaiming_uniform a=√5, giống _init_weights) cho √n_layers.
        with torch.no_grad():
            for layer in self.layers:
                layer.out_proj.weight /= math.sqrt(n_layers)

    @property
    def output_dim(self) -> int:
        return self._d_model

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor):
        # residual_in_fp32: dưới autocast, input_proj trả fp16 → cộng dồn 28 lớp
        # ở fp16. Ép fp32 ở đây là đủ: LayerNorm chạy fp32 và fp32 + fp16 → fp32.
        x = self.input_proj(feats).float()  # [B, T, d_model]
        # Không mask padding là đúng *với Mamba đơn hướng*: conv1d và scan đều
        # nhân quả, padding nằm cuối chuỗi nên không lan ngược về khung thật;
        # CTC chỉ tính trên feat_lengths. Chuyển sang hai chiều thì phải mask.
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))
        return self.norm_f(x), feat_lengths
