"""Đo RTF (Real-Time Factor) và latency theo các nhóm độ dài audio.

RTF = thời gian xử lý / thời gian audio. RTF < 1 nghĩa là nhanh hơn thời gian
thực. Đây là phép đo chính cho RQ2 (so sánh Mamba vs Conformer khi audio dài).
"""

import time

import torch


def measure_rtf(model, waveform: torch.Tensor, sample_rate: int = 16000, n_warmup: int = 3, n_runs: int = 10) -> dict:
    """waveform: [1, T] — đo 1 mẫu tại 1 thời điểm (không batch) để phản ánh
    đúng latency suy luận thực tế."""
    audio_duration_s = waveform.size(-1) / sample_rate
    lengths = torch.tensor([waveform.size(-1)])

    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(waveform, lengths)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            model(waveform, lengths)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_runs

    return {
        "audio_duration_s": audio_duration_s,
        "inference_time_s": elapsed,
        "rtf": elapsed / audio_duration_s,
    }


# Bucket độ dài audio dùng để đo RTF theo nhóm (khớp scope 3-30s của VietSuperSpeech)
AUDIO_LENGTH_BUCKETS_S = [(3, 8), (8, 15), (15, 22), (22, 30)]
