"""Khối Mamba-1 thuần PyTorch chạy CPU — chỉ để **suy luận** khi không có `mamba-ssm` (demo trên máy
không GPU). Không dùng để train hay đo tốc độ (RQ2): quét tuần tự bằng vòng Python, chậm hơn kernel CUDA nhiều.

Chép đúng đường chậm của `mamba_ssm.Mamba` (mamba_simple.py) + `selective_scan_ref`
(ops/selective_scan_interface.py) ở tag v2.3.1, cùng tên/shape tham số → nạp thẳng state_dict của checkpoint
train trên Kaggle. Đây cũng là phương án dự phòng "selective_scan_ref" đã ghi trong đề cương.

Dùng: `mamba_encoder._MambaBlock = MambaRef; mamba_encoder.MAMBA_SSM_AVAILABLE = True` trước khi `build_model`
(như cách test CPU thay khối giả) — xem `use_reference_mamba`.
"""

import math

import torch
import torch.nn.functional as F


class MambaRef(torch.nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = torch.nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = torch.nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
                                      padding=d_conv - 1, bias=True)
        self.x_proj = torch.nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = torch.nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # Giá trị thật đến từ checkpoint; khởi tạo chỉ cần đúng shape.
        self.A_log = torch.nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.D = torch.nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = torch.nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [B, L, d_model] → [B, L, d_model]."""
        length = hidden.shape[1]
        x, z = self.in_proj(hidden).transpose(1, 2).chunk(2, dim=1)  # [B, d_inner, L] mỗi cái
        x = F.silu(self.conv1d(x)[..., :length])
        dt, b, c = torch.split(self.x_proj(x.transpose(1, 2)), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # mamba_simple: dt_proj.weight nhân ở đây, bias đưa vào scan làm delta_bias rồi softplus.
        delta = F.softplus(dt @ self.dt_proj.weight.T + self.dt_proj.bias).transpose(1, 2)  # [B, d_inner, L]
        a = -torch.exp(self.A_log.float())  # [d_inner, N]
        y = self._scan(x, delta, a, b, c)
        y = (y + x * self.D[None, :, None]) * F.silu(z)
        return self.out_proj(y.transpose(1, 2))

    @staticmethod
    def _scan(u, delta, a, b, c):
        """selective_scan_ref: h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t u_t ; y_t = C_t · h_t.
        Xếp trục thời gian ra ngoài cùng cho lát cắt liền bộ nhớ + TorchScript: nhanh ~2× vòng gốc, lệch ≤ 2e-5."""
        delta_a = torch.exp(delta.unsqueeze(-1) * a[:, None, :]).permute(2, 0, 1, 3).contiguous()  # [L, B, D, N]
        delta_b_u = (delta.unsqueeze(-1) * b.unsqueeze(1) * u.unsqueeze(-1)).permute(2, 0, 1, 3).contiguous()
        return torch.einsum("lbdn,bln->bdl", _recurrence(delta_a, delta_b_u), c)


@torch.jit.script
def _recurrence(delta_a: torch.Tensor, delta_b_u: torch.Tensor) -> torch.Tensor:
    h = torch.zeros_like(delta_a[0])
    hs = torch.empty_like(delta_b_u)
    for t in range(delta_a.shape[0]):
        h = torch.addcmul(delta_b_u[t], delta_a[t], h)
        hs[t] = h
    return hs


def use_reference_mamba() -> None:
    from src.models import mamba_encoder

    if not mamba_encoder.MAMBA_SSM_AVAILABLE:
        mamba_encoder._MambaBlock = MambaRef
        mamba_encoder.MAMBA_SSM_AVAILABLE = True
