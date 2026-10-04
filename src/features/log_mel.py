"""Front-end dùng chung cho cả Mamba-CTC và Conformer-CTC.

Chỉ có MỘT cài đặt trích xuất đặc trưng — không được để hai pipeline dùng
hai cách trích xuất khác nhau, nếu không phép so sánh matched-parameter sẽ
mất công bằng.

log-mel 80 kênh, cửa sổ 25 ms, bước 10 ms — giống Conformer (Gulati 2020),
Whisper, FastConformer/NeMo, ConMamba. Thêm 2026-10-04 (người dùng duyệt,
căn cứ: docs/notes/frontend_decoder_survey.md):
- CMVN **toàn cục** (như ConMamba `InputNormalization(global)`, recipe
  Conformer torchaudio): mean/std cố định tính trên train → không phụ thuộc độ
  dài câu, nên audio ghép dài của RQ2 nhận đúng phép biến đổi như lúc train
  (`per_feature` theo câu kiểu NeMo thì không).
- SpecAugment chỉ khi `self.training`, theo Conformer: 2 mask tần số F = 27,
  10 mask thời gian rộng tối đa 5% độ dài thật, **không** time warping (bài
  SpecAugment Bảng 6: đóng góp nhỏ nhất, nên bỏ đầu tiên). Mask sau CMVN, điền
  0 = giá trị trung bình.
"""

import json
from pathlib import Path

import torch
import torchaudio

CMVN_STATS_PATH = "configs/cmvn_stats.json"


class LogMelFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        cmvn_stats: str | None = None,
        spec_augment: dict | None = None,
    ):
        """cmvn_stats: đường dẫn json (`src/features/compute_cmvn.py`); None = không
        chuẩn hoá (chỉ dùng khi tính chính thống kê đó).
        spec_augment: {freq_masks, freq_width, time_masks, time_ratio}; None = tắt."""
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
        )
        self.use_cmvn = cmvn_stats is not None
        if self.use_cmvn:
            stats = json.loads(Path(cmvn_stats).read_text(encoding="utf-8"))
            if len(stats["mean"]) != n_mels:
                raise ValueError(f"{cmvn_stats} có {len(stats['mean'])} kênh, front-end {n_mels}")
            # Buffer → lưu cùng checkpoint: eval nạp checkpoint là đúng thống kê lúc train.
            self.register_buffer("cmvn_mean", torch.tensor(stats["mean"], dtype=torch.float32))
            self.register_buffer("cmvn_inv_std", 1.0 / torch.tensor(stats["std"], dtype=torch.float32))
        self.spec_augment = spec_augment

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor):
        """waveform: [B, T_samples], lengths: [B] (số sample thực, trước padding).

        Trả về:
          feats: [B, T_frames, n_mels]
          feat_lengths: [B] (số frame thực, trước padding)
        """
        # Giữ fp32 kể cả khi train dưới autocast (AMP, D8): phổ công suất có thể
        # vượt 65504 (max fp16) → inf. Không ảnh hưởng khi chạy fp32.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            mel = self.mel(waveform.float())  # [B, n_mels, T_frames]
            log_mel = torch.log(mel.clamp(min=1e-5))
            feats = log_mel.transpose(1, 2)  # [B, T_frames, n_mels]

            hop = self.mel.hop_length
            feat_lengths = torch.div(lengths, hop, rounding_mode="floor") + 1
            feat_lengths = feat_lengths.clamp(max=feats.size(1))

            if self.use_cmvn:
                feats = (feats - self.cmvn_mean) * self.cmvn_inv_std
            if self.training and self.spec_augment:
                feats = self._spec_augment(feats, feat_lengths)
        return feats, feat_lengths

    def _spec_augment(self, feats: torch.Tensor, feat_lengths: torch.Tensor) -> torch.Tensor:
        cfg = self.spec_augment
        B, T, F = feats.shape
        device = feats.device

        def band_mask(size: int, max_width: torch.Tensor, n_masks: int, limit: torch.Tensor) -> torch.Tensor:
            """n_masks dải [start, start + w) mỗi mẫu, w ~ U{0..max_width},
            start ~ U{0..limit - w}; trả [B, size] bool."""
            width = (torch.rand(B, n_masks, device=device) * (max_width + 1)).floor()
            start = (torch.rand(B, n_masks, device=device) * (limit - width + 1)).floor()
            pos = torch.arange(size, device=device)
            return ((pos >= start[..., None]) & (pos < (start + width)[..., None])).any(dim=1)

        freq = band_mask(F, torch.full((B, 1), float(cfg["freq_width"]), device=device), cfg["freq_masks"],
                         torch.full((B, 1), float(F), device=device))
        # Mask thời gian chỉ trong phần khung thật (độ dài khác nhau trong batch).
        lengths = feat_lengths.to(device).float()[:, None]
        time = band_mask(T, (lengths * cfg["time_ratio"]).floor(), cfg["time_masks"], lengths)
        return feats.masked_fill(freq[:, None, :] | time[:, :, None], 0.0)
