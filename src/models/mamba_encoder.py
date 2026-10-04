"""Mamba (Selective State Space Model / S6) encoder — kiến trúc Mamba-1 cổ
điển (mamba_ssm.Mamba, pin tag v2.3.1), đúng như trích dẫn trong đề cương
(Gu & Dao, arXiv:2312.00752). KHÔNG dùng Mamba-2/Mamba-3 (khác kiến trúc,
khác mục tiêu thiết kế, ngoài phạm vi đề cương đã đăng ký — xem
docs/notes/mamba_versions.md).

Mặc định hai chiều kiểu "ngoài" (phương án B1, chốt 2026-10-02 — lý do và
cách tính tham số ở docs/notes/mamba_bidirectional.md): Conformer nhìn cả câu,
Mamba một chiều chỉ nhìn quá khứ → RQ1 lẫn biến "ngữ cảnh một phía". Cờ
`bidirectional=False` giữ lại đường một chiều cũ (28 lớp) cho phần thảo luận.

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


def reverse_padded(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Đảo [B, T, D] theo trục thời gian trong phạm vi độ dài thật của từng mẫu,
    padding giữ nguyên ở cuối. Phép đảo là tự nghịch đảo: gọi hai lần ra x.

    Không dùng torch.flip cả tensor: padding (ở cuối) sẽ thành đầu chuỗi và
    lọt vào trạng thái của nhánh quét ngược trước khi gặp khung thật.
    """
    t = torch.arange(x.size(1), device=x.device)
    lengths = lengths.to(x.device).unsqueeze(1)
    index = torch.where(t < lengths, lengths - 1 - t, t)  # [B, T]
    return torch.gather(x, 1, index.unsqueeze(-1).expand_as(x))


class MambaEncoder(ASREncoder):
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        n_layers: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not MAMBA_SSM_AVAILABLE:
            raise ImportError(
                "mamba-ssm chưa được cài. Chạy notebooks/00_setup_environment.ipynb "
                "trên Kaggle T4 trước, hoặc cài theo hướng dẫn trong README.md."
            )

        self._d_model = d_model
        self.bidirectional = bidirectional

        def make_blocks():
            return torch.nn.ModuleList(
                [
                    _MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                    for _ in range(n_layers)
                ]
            )

        self.input_proj = torch.nn.Linear(input_dim, d_model)
        # Nhánh xuôi giữ tên `layers` để checkpoint/state_dict của bản một chiều
        # vẫn nạp được khi tắt cờ.
        self.layers = make_blocks()
        # B1: khối ngược có tham số riêng nhưng dùng chung LayerNorm với khối
        # xuôi — đúng con số 12.285.696 đã chốt trong note.
        self.layers_bwd = make_blocks() if bidirectional else None
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(d_model) for _ in range(n_layers)])
        # Ba chỗ dưới theo MixerModel/_init_weights của mamba-ssm v2.3.1 (rà
        # 2026-09-30): dùng khối Mamba lẻ thì không tự có. norm_f: chồng pre-norm
        # để residual chưa chuẩn hoá đi thẳng vào CTC head — Conformer torchaudio
        # thì mỗi lớp đã kết thúc bằng LayerNorm.
        self.norm_f = torch.nn.LayerNorm(d_model)
        # Thêm 2026-10-04 (người dùng duyệt): mamba-ssm gốc không dropout (công
        # thức LM dữ liệu lớn), còn 2405.12609 và ConMamba cấu hình dropout 0,1
        # cho mô hình ASR có Mamba — bằng Conformer ở đây. Bài không nói vị trí;
        # đặt trên đầu ra mỗi khối trước khi cộng vào residual là lựa chọn của dự án.
        self.dropout = torch.nn.Dropout(dropout)
        # rescale_prenorm_residual: _init_weights chia out_proj cho
        # √(n_residuals_per_layer × n_layer), tức √(số đầu ra khối độc lập cộng
        # vào residual). B1 cộng 2 out_proj mỗi lớp (phương sai cộng dồn như 2
        # nhánh) → √(2 × 14) = √28, trùng hệ số của bản một chiều 28 lớp.
        # out_proj đã khởi tạo kaiming_uniform a=√5, giống _init_weights.
        n_branches = 2 if bidirectional else 1
        scale = math.sqrt(n_branches * n_layers)
        with torch.no_grad():
            for blocks in (self.layers, self.layers_bwd or []):
                for block in blocks:
                    block.out_proj.weight /= scale

    @property
    def output_dim(self) -> int:
        return self._d_model

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor):
        # residual_in_fp32: dưới autocast, input_proj trả fp16 → cộng dồn 28
        # khối ở fp16. Ép fp32 ở đây là đủ: LayerNorm chạy fp32 và fp32 + fp16 → fp32.
        x = self.input_proj(feats).float()  # [B, T, d_model]
        # Không cần mask padding vì padding luôn nằm *sau* khung thật theo chiều
        # quét của mọi khối: nhánh xuôi nhân quả (conv1d + scan) nên padding ở
        # cuối không lan ngược; nhánh ngược đảo theo feat_lengths nên padding vẫn
        # ở cuối. Giá trị rác ở vị trí padding qua các lớp cũng không lan vào
        # khung thật vì cùng lý do; CTC chỉ tính trên feat_lengths.
        if not self.bidirectional:
            for block, norm in zip(self.layers, self.norms):
                x = x + self.dropout(block(norm(x)))
            return self.norm_f(x), feat_lengths

        for block_fwd, block_bwd, norm in zip(self.layers, self.layers_bwd, self.norms):
            h = norm(x)
            h_bwd = reverse_padded(block_bwd(reverse_padded(h, feat_lengths)), feat_lengths)
            x = x + self.dropout(block_fwd(h)) + self.dropout(h_bwd)
        return self.norm_f(x), feat_lengths
