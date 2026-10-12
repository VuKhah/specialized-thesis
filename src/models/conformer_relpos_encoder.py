"""Conformer có mã hóa vị trí tương đối (Gulati và cs., 2020) — Conformer của bảng chính từ 2026-10-12.

Vì sao có file này: `torchaudio.models.Conformer` (v1, `conformer_encoder.py`) dùng
`nn.MultiheadAttention` thường và không cộng PE nào — lệch bài gốc (Conformer dùng PE tương đối kiểu
Transformer-XL), phát hiện khi viết Ch1 sau khi đã train v1. Bản v1 giữ lại làm ablation "không PE".

Cài đặt theo ESPnet (`RelPositionalEncoding` + `RelPositionMultiHeadedAttention`, bản "latest"):
- PE sin/cos cho vị trí tương đối +(T-1) … -(T-1); đầu vào nhân √d_model (xscale, như ESPnet/NeMo).
- score = (q + u)·kᵀ + rel_shift((q + v)·pᵀ), chia √d_k; u, v học được theo từng head.
- Dropout trên đầu vào sau xscale (ESPnet positional_dropout_rate), không dropout trên PE (như NeMo
  dropout_emb = 0).
Còn lại giữ nguyên `ConformerLayer` của torchaudio (cùng FFN/conv module, cùng thứ tự khối), thêm
một sửa: xoá padding ngay trước depthwise conv (kernel 31, không nhân quả) như ConExtBiMamba — torchaudio
không xoá nên ~15 khung thật cuối câu nhận giá trị padding, WER phụ thuộc cách ghép batch (đo 2026-10-12:
lệch tới 0,38 giữa câu riêng và trong batch ở v1). `pos_enc: none` = cùng lớp, bỏ số hạng vị trí
(ablation PE của bảng chính; xscale giữ nguyên để hai bản chỉ khác đúng PE).

Lưu ý RQ2: ma trận vị trí [B, H, T, 2T-1] phải tạo tường minh (không dùng được SDPA mem-efficient như
v1) → VRAM bậc hai theo T; audio rất dài sẽ OOM sớm hơn Conformer v1 (đặc điểm thật của Conformer chuẩn).
"""

import math

import torch
import torch.nn.functional as F
from torchaudio.models.conformer import _ConvolutionModule, _FeedForwardModule

from src.models.encoder_base import ASREncoder


def relative_positions(length: int, d_model: int, device, dtype) -> torch.Tensor:
    """[2T-1, d_model]: hàng j ứng với vị trí tương đối (T-1) - j."""
    pos = torch.arange(length - 1, -length, -1, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * -(math.log(10000.0) / d_model))
    pe = torch.zeros(2 * length - 1, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


def rel_shift(x: torch.Tensor) -> torch.Tensor:
    """[B, H, T, 2T-1] (cột j = vị trí tương đối (T-1) - j) → [B, H, T, T] với [i, k] = vị trí i - k.
    Như ESPnet `RelPositionMultiHeadedAttention.rel_shift`."""
    b, h, t, n = x.shape
    x = torch.cat([x.new_zeros(b, h, t, 1), x], dim=-1).view(b, h, n + 1, t)
    return x[:, :, 1:].reshape(b, h, t, n)[:, :, :, : n // 2 + 1]


class RelPositionMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, rel_pos: bool = True):
        super().__init__()
        self.rel_pos = rel_pos
        self.h, self.d_k = n_heads, d_model // n_heads
        self.linear_q = torch.nn.Linear(d_model, d_model)
        self.linear_k = torch.nn.Linear(d_model, d_model)
        self.linear_v = torch.nn.Linear(d_model, d_model)
        self.linear_out = torch.nn.Linear(d_model, d_model)
        if rel_pos:
            self.linear_pos = torch.nn.Linear(d_model, d_model, bias=False)
            self.pos_bias_u = torch.nn.Parameter(torch.empty(n_heads, self.d_k))
            self.pos_bias_v = torch.nn.Parameter(torch.empty(n_heads, self.d_k))
            torch.nn.init.xavier_uniform_(self.pos_bias_u)
            torch.nn.init.xavier_uniform_(self.pos_bias_v)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor | None, pad_mask: torch.Tensor) -> torch.Tensor:
        """x [B, T, D], pos_emb [2T-1, D] (None khi rel_pos=False), pad_mask [B, T] (True = padding)."""
        b, t, _ = x.shape
        q = self.linear_q(x).view(b, t, self.h, self.d_k)
        k = self.linear_k(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        # SDPA tính softmax((q·kᵀ)/√d_k + bias)·v mà không lưu ma trận attention cho backward (smoke test CPU
        # 2026-10-12: bản tính tay tràn RAM ở batch 16 × ~1.500 khung). Số hạng vị trí đi vào qua bias:
        # (q+u)·kᵀ/√d_k + rel_shift((q+v)·pᵀ)/√d_k = đúng công thức ESPnet.
        bias = torch.zeros(b, 1, 1, t, device=x.device, dtype=q.dtype).masked_fill(
            pad_mask[:, None, None, :], float("-inf"))
        if self.rel_pos:
            p = self.linear_pos(pos_emb).view(-1, self.h, self.d_k).transpose(0, 1)  # [H, 2T-1, d_k]
            bd = rel_shift(torch.matmul((q + self.pos_bias_v).transpose(1, 2), p.transpose(-2, -1)))
            bias = bias + bd / math.sqrt(self.d_k)
            q = q + self.pos_bias_u
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k, v, attn_mask=bias.to(q.dtype),
                                             dropout_p=self.dropout.p if self.training else 0.0)
        return self.linear_out(out.transpose(1, 2).reshape(b, t, -1))


class ConformerRelPosLayer(torch.nn.Module):
    """= torchaudio `ConformerLayer` (convolution_first=False), MHSA thay bằng bản PE tương đối."""

    def __init__(self, d_model: int, ff_dim: int, n_heads: int, conv_kernel_size: int, dropout: float,
                 rel_pos: bool = True):
        super().__init__()
        self.ffn1 = _FeedForwardModule(d_model, ff_dim, dropout=dropout)
        self.self_attn_layer_norm = torch.nn.LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(d_model, n_heads, dropout, rel_pos)
        self.self_attn_dropout = torch.nn.Dropout(dropout)
        self.conv_module = _ConvolutionModule(input_dim=d_model, num_channels=d_model,
                                              depthwise_kernel_size=conv_kernel_size, dropout=dropout, bias=True)
        self.ffn2 = _FeedForwardModule(d_model, ff_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(d_model)

    def _conv(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # = ConExtBiMambaLayer._conv: các bước của _ConvolutionModule.forward, xoá padding trước depthwise conv.
        m = self.conv_module
        h = m.layer_norm(x).transpose(1, 2)  # [B, D, T]
        h = m.sequential[1](m.sequential[0](h))  # pointwise + GLU
        h = h.masked_fill(pad_mask.unsqueeze(1), 0.0)
        h = m.sequential[2:](h)
        return h.transpose(1, 2)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor | None, pad_mask: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x = x + self.self_attn_dropout(self.self_attn(self.self_attn_layer_norm(x), pos_emb, pad_mask))
        x = x + self._conv(x, pad_mask)
        x = x + 0.5 * self.ffn2(x)
        return self.final_layer_norm(x)


class ConformerRelPosEncoder(ASREncoder):
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 4,
        conv_kernel_size: int = 31,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        pos_enc: str = "rel",
    ):
        super().__init__()
        if pos_enc not in ("rel", "none"):
            raise ValueError(f"pos_enc phải là rel/none, nhận {pos_enc}")
        self._d_model = d_model
        self.rel_pos = pos_enc == "rel"
        self.input_proj = torch.nn.Linear(input_dim, d_model)
        self.input_dropout = torch.nn.Dropout(dropout)
        self.layers = torch.nn.ModuleList(
            [ConformerRelPosLayer(d_model, ff_dim, n_heads, conv_kernel_size, dropout, self.rel_pos)
             for _ in range(n_layers)]
        )

    @property
    def output_dim(self) -> int:
        return self._d_model

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor, layer_hook=None):
        x = self.input_dropout(self.input_proj(feats) * math.sqrt(self._d_model))
        t = x.size(1)
        pos_emb = relative_positions(t, self._d_model, x.device, x.dtype) if self.rel_pos else None
        pad_mask = torch.arange(t, device=x.device) >= feat_lengths.to(x.device).unsqueeze(1)
        for i, layer in enumerate(self.layers, 1):
            x = layer(x, pos_emb, pad_mask)
            if layer_hook is not None:
                x = layer_hook(i, x)
        return x, feat_lengths
