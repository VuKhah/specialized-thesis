"""Dataset VietSuperSpeech dùng chung cho cả hai thí nghiệm.

Nguồn: https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech
Lưu ý: transcript là pseudo-label (Zipformer-30M-RNNT-6000h qua Sherpa-ONNX),
chưa qua kiểm định người — xem src/data/survey.py (Tuần 3) cho tập clean-test
200-300 câu hiệu đính thủ công, seed cố định.

QUAN TRỌNG — cấu trúc audio (phát hiện khi xây notebooks/02_dataset_eda.ipynb,
2026-09-14): cột "audio" KHÔNG phải kiểu `datasets.Audio` tự giải mã — nó chỉ
là chuỗi đường dẫn tương đối (vd. "audio/asr_segments_.../xxx_seg047.wav").
Phải tự tải bằng `hf_hub_download` rồi đọc bằng soundfile.

Tải hàng loạt trước khi train (Tuần 4-5, ĐÃ GIẢI QUYẾT): chạy
`python -m src.data.prefetch_audio` trước khi train — tải song song toàn bộ
thư mục audio/ về AUDIO_CACHE_DIR một lần bằng `snapshot_download`, thay vì
tải từng file qua HTTP trong __getitem__ (~1-4s/file, không khả thi cho 60k+
mẫu). __getitem__ dưới đây đọc thẳng từ cache nếu đã prefetch; nếu chưa (vd.
dùng nhanh trong EDA) sẽ tự động fallback tải lẻ qua hf_hub_download.
"""

from pathlib import Path

import soundfile as sf
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

HF_DATASET_ID = "thanhnew2001/VietSuperSpeech"
AUDIO_CACHE_DIR = "data/raw/audio_cache"


class VietSuperSpeechDataset(Dataset):
    def __init__(self, split: str = "train", tokenizer=None):
        """split: "train" hoặc "validation" — tên split thật trên HF Hub
        (KHÔNG phải "dev-test" như ghi trong đề cương, xác nhận qua
        src/data/survey.py, Tuần 3 — xem docs/notes/dataset_discrepancy.md
        cho chênh lệch số liệu đầy đủ so với đề cương)."""
        self.hf_dataset = load_dataset(HF_DATASET_ID, split=split)
        self.tokenizer = tokenizer
        Path(AUDIO_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        item = self.hf_dataset[idx]
        # Schema: dict_keys(['audio', 'text', 'duration', 'source']) —
        # "audio" là đường dẫn tương đối (string), không tự giải mã, xem docstring đầu file.
        local_path = Path(AUDIO_CACHE_DIR) / item["audio"]
        if not local_path.exists():
            # Chưa chạy prefetch_audio.py — tải lẻ qua HTTP (chậm, chỉ nên
            # dùng cho khảo sát/EDA nhỏ, không phù hợp để train trực tiếp).
            local_path = Path(hf_hub_download(HF_DATASET_ID, item["audio"], repo_type="dataset", local_dir=AUDIO_CACHE_DIR))
        array, _sample_rate = sf.read(local_path)
        waveform = torch.from_numpy(array).float()
        text = item["text"]
        token_ids = self.tokenizer.encode(text) if self.tokenizer else None
        return {"waveform": waveform, "text": text, "token_ids": token_ids}


def collate_fn(batch):
    """Pad waveform + token_ids theo batch, trả về cùng với length thực."""
    waveforms = [b["waveform"] for b in batch]
    lengths = torch.tensor([w.size(0) for w in waveforms])
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)

    result = {"waveform": padded_waveforms, "waveform_lengths": lengths, "text": [b["text"] for b in batch]}

    if batch[0]["token_ids"] is not None:
        token_ids = [torch.tensor(b["token_ids"]) for b in batch]
        target_lengths = torch.tensor([t.size(0) for t in token_ids])
        result["targets"] = torch.cat(token_ids)  # CTC loss muốn targets dạng concat 1D
        result["target_lengths"] = target_lengths

    return result
