"""Front-end dùng chung cho cả Mamba-CTC và Conformer-CTC.

Chỉ có MỘT cài đặt trích xuất đặc trưng — không được để hai pipeline dùng
hai cách trích xuất khác nhau, nếu không phép so sánh matched-parameter sẽ
mất công bằng.
"""

import torch
import torchaudio


class LogMelFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
        )

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor):
        """waveform: [B, T_samples], lengths: [B] (số sample thực, trước padding).

        Trả về:
          feats: [B, T_frames, n_mels]
          feat_lengths: [B] (số frame thực, trước padding)
        """
        mel = self.mel(waveform)  # [B, n_mels, T_frames]
        log_mel = torch.log(mel.clamp(min=1e-5))
        feats = log_mel.transpose(1, 2)  # [B, T_frames, n_mels]

        hop = self.mel.hop_length
        feat_lengths = torch.div(lengths, hop, rounding_mode="floor") + 1
        feat_lengths = feat_lengths.clamp(max=feats.size(1))
        return feats, feat_lengths
