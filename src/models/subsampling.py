"""Conv2d subsampling 4× (khung 10 ms → 40 ms) cho nhánh 40 ms (kế hoạch 2026-10-12).

Kiểu `Conv2dSubsampling` của ESPnet — cùng dạng với phép đo tốc độ 2026-10-08
(`docs/notes/ket_qua/subsample_speed_2026-10-08.md`, biến thể b): 2 conv 3×3 stride 2 không padding, ReLU,
rồi Linear về d_model. ConMamba dùng CNN 2 lớp stride 2 tương tự (4×). Đặt giữa front-end và encoder,
dùng chung cho cả ba encoder → mọi encoder cùng nhận chuỗi T' = ((T - 1) // 2 - 1) // 2.
"""

import torch


class Conv2dSubsampling4(torch.nn.Module):
    def __init__(self, input_dim: int = 80, output_dim: int = 256, channels: int = 256):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, channels, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, 3, 2),
            torch.nn.ReLU(),
        )
        freq = ((input_dim - 1) // 2 - 1) // 2
        self.out = torch.nn.Linear(channels * freq, output_dim)
        self.output_dim = output_dim

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(torch.div(lengths - 1, 2, rounding_mode="floor") - 1, 2, rounding_mode="floor")

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor):
        """feats [B, T, F] → [B, T', output_dim]."""
        x = self.conv(feats.unsqueeze(1))  # [B, C, T', F']
        b, c, t, f = x.shape
        x = self.out(x.transpose(1, 2).reshape(b, t, c * f))
        return x, self.output_lengths(feat_lengths).clamp(min=1, max=t)
