"""Dataset VietSuperSpeech dùng chung cho cả hai thí nghiệm.

Nguồn: https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech
Lưu ý: transcript là pseudo-label (Zipformer-30M-RNNT-6000h qua Sherpa-ONNX),
chưa qua kiểm định người — xem clean_test.py (Tuần 3) để tạo tập clean-test
200-300 câu hiệu đính thủ công, seed cố định.
"""

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

HF_DATASET_ID = "thanhnew2001/VietSuperSpeech"


class VietSuperSpeechDataset(Dataset):
    def __init__(self, split: str = "train", tokenizer=None):
        """split: "train" hoặc "validation" — tên split thật trên HF Hub
        (KHÔNG phải "dev-test" như ghi trong đề cương, xác nhận qua
        src/data/survey.py, Tuần 3 — xem docs/notes/dataset_discrepancy.md
        cho chênh lệch số liệu đầy đủ so với đề cương)."""
        self.hf_dataset = load_dataset(HF_DATASET_ID, split=split)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        item = self.hf_dataset[idx]
        # Schema xác nhận qua notebooks/00_setup_environment.ipynb (Kaggle, 2026-09-14):
        # dict_keys(['audio', 'text', 'duration', 'source']) — source = tên file video gốc.
        waveform = torch.tensor(item["audio"]["array"], dtype=torch.float32)
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
