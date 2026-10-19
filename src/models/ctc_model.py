"""Model CTC dùng chung: front-end -> (subsampling) -> encoder (Mamba / ConExtBiMamba / Conformer) -> CTC head.

Thiết kế "chỉ hoán đổi encoder" (đề cương, Chương 2): mọi phần ngoài encoder ở đây dùng chung cho cả
ba encoder, không được thêm logic riêng theo encoder.

Thêm 2026-10-12 (công thức v2 — `docs/notes/ket_qua/conextbimamba_gap_2026-10-12.md` mục 7-8), đều tắt mặc định
để checkpoint v1 nạp và chạy y như cũ:
- `subsampling`: Conv2d 4× (nhánh 40 ms).
- `interctc`: CTC phụ ở lớp giữa (Lee & Watanabe, arXiv:2102.03216: w = 0,3, lớp ⌊L/2⌋) + self-conditioning
  (Nozaki & Komatsu, arXiv:2104.02724): softmax của dự đoán trung gian qua Linear cộng vào luồng residual
  trước lớp sau. Theo ESPnet (`interctc_layer_idx`, `interctc_use_conditioning`): dùng chung CTC head với
  đầu ra cuối, biểu diễn trung gian được chuẩn hóa (`after_norm`) trước head. Conditioning chạy cả lúc suy luận.
"""

import torch
import torch.nn.functional as F

from src.features.log_mel import LogMelFeatureExtractor
from src.models.encoder_base import ASREncoder


def ctc_loss(log_probs, out_lengths, targets, target_lengths, blank_id: int = 0) -> torch.Tensor:
    """Tách khỏi `compute_loss` để train.py gọi forward qua wrapper DDP (gọi
    thẳng `module.compute_loss` là bỏ qua hook đồng bộ gradient của DDP).
    Ép fp32: dưới AMP, CTC trên fp16 dễ tràn số."""
    return F.ctc_loss(
        log_probs.float().transpose(0, 1),  # CTCLoss cần [T, B, C]
        targets,
        out_lengths,
        target_lengths,
        blank=blank_id,
        zero_infinity=True,
    )


class CTCASRModel(torch.nn.Module):
    def __init__(self, encoder: ASREncoder, vocab_size: int, feature_extractor: LogMelFeatureExtractor | None = None,
                 subsampling: torch.nn.Module | None = None, interctc: dict | None = None):
        """interctc: {"weight": 0.3, "layer": None (= ⌊L/2⌋), "self_condition": True}; None = tắt."""
        super().__init__()
        self.feature_extractor = feature_extractor or LogMelFeatureExtractor()
        self.subsampling = subsampling
        self.encoder = encoder
        self.ctc_head = torch.nn.Linear(encoder.output_dim, vocab_size)
        self.interctc = dict(interctc) if interctc else None
        if self.interctc:
            self.interctc.setdefault("weight", 0.3)
            self.interctc.setdefault("self_condition", True)
            if not self.interctc.get("layer"):
                self.interctc["layer"] = encoder.n_layers // 2
            if not 1 <= self.interctc["layer"] < encoder.n_layers:
                raise ValueError(f"interctc.layer {self.interctc['layer']} ngoài [1, {encoder.n_layers})")
            self.conditioning = (torch.nn.Linear(vocab_size, encoder.output_dim)
                                 if self.interctc["self_condition"] else None)

    def forward(self, waveform: torch.Tensor, waveform_lengths: torch.Tensor, return_inter: bool = False):
        """Trả (log_probs [B, T', V], out_lengths); return_inter=True thêm list log_probs trung gian."""
        feats, feat_lengths = self.feature_extractor(waveform, waveform_lengths)
        return self.forward_features(feats, feat_lengths, return_inter)

    def forward_features(self, feats: torch.Tensor, feat_lengths: torch.Tensor, return_inter: bool = False):
        """Phần sau front-end (subsampling → encoder → head) — tách để test bất biến padding ở mức encoder."""
        if self.subsampling is not None:
            feats, feat_lengths = self.subsampling(feats, feat_lengths)
        inter = []
        hook = None
        if self.interctc:
            def hook(i, x):
                if i != self.interctc["layer"]:
                    return x
                logits = self.ctc_head(self.encoder.normalize_intermediate(x))
                inter.append(F.log_softmax(logits, dim=-1))
                if self.conditioning is not None:
                    x = x + self.conditioning(F.softmax(logits, dim=-1))
                return x
        hidden, out_lengths = self.encoder(feats, feat_lengths, layer_hook=hook)
        log_probs = F.log_softmax(self.ctc_head(hidden), dim=-1)  # [B, T', V]
        return (log_probs, out_lengths, inter) if return_inter else (log_probs, out_lengths)

    def total_ctc_loss(self, log_probs, out_lengths, inter, targets, target_lengths, blank_id: int = 0):
        """(1 - w)·CTC + w·InterCTC (Lee & Watanabe 2021, công thức 13); không InterCTC thì = CTC.
        Trả (loss tổng, loss CTC cuối, loss InterCTC hoặc None)."""
        main = ctc_loss(log_probs, out_lengths, targets, target_lengths, blank_id)
        if not inter:
            return main, main, None
        aux = sum(ctc_loss(lp, out_lengths, targets, target_lengths, blank_id) for lp in inter) / len(inter)
        w = self.interctc["weight"]
        return (1 - w) * main + w * aux, main, aux

    def compute_loss(self, waveform, waveform_lengths, targets, target_lengths, blank_id: int = 0):
        log_probs, out_lengths, inter = self.forward(waveform, waveform_lengths, return_inter=True)
        return self.total_ctc_loss(log_probs, out_lengths, inter, targets, target_lengths, blank_id)[0]

    @torch.no_grad()
    def greedy_decode(self, waveform: torch.Tensor, waveform_lengths: torch.Tensor, blank_id: int = 0) -> list[list[int]]:
        """Greedy CTC decode (argmax + collapse repeat + bỏ blank) — dùng cho
        eval loop (WER định kỳ khi train) và demo. Không phải beam search,
        chỉ đủ để theo dõi xu hướng WER qua các epoch."""
        log_probs, out_lengths = self.forward(waveform, waveform_lengths)
        return greedy_collapse(log_probs, out_lengths, blank_id)


def greedy_collapse(log_probs: torch.Tensor, out_lengths: torch.Tensor, blank_id: int = 0) -> list[list[int]]:
    """argmax + gộp lặp + bỏ blank. Tách riêng để eval của train.py giải mã và
    tính CTC loss trên cùng một lần forward."""
    pred_ids = log_probs.argmax(dim=-1)  # [B, T]
    results = []
    for ids, length in zip(pred_ids, out_lengths):
        collapsed = []
        prev = None
        for i in ids[:length].tolist():
            if i != prev and i != blank_id:
                collapsed.append(i)
            prev = i
        results.append(collapsed)
    return results
