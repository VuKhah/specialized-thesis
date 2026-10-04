"""Dataset VietSuperSpeech dùng chung cho cả hai thí nghiệm.

Nguồn: https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech
Lưu ý: transcript là pseudo-label (Zipformer-30M-RNNT-6000h qua Sherpa-ONNX),
chưa qua kiểm định người — xem src/data/survey.py (Tuần 3) cho tập clean-test
200-300 câu hiệu đính thủ công, seed cố định.

QUAN TRỌNG — cấu trúc audio (phát hiện khi xây notebooks/02_dataset_eda.ipynb,
2026-09-14): cột "audio" KHÔNG phải kiểu `datasets.Audio` tự giải mã — nó chỉ
là chuỗi đường dẫn tương đối (vd. "audio/asr_segments_.../xxx_seg047.wav").
Phải tự tải bằng `hf_hub_download` rồi đọc bằng soundfile.

Tải hàng loạt trước khi train (Tuần 4-5): chạy
`python -m src.data.prefetch_audio` trước khi train — tải song song đúng các
file audio mà train/validation dùng về AUDIO_CACHE_DIR một lần (không dùng
`snapshot_download` cả thư mục vì thừa ~43%), thay vì tải từng file qua HTTP
trong __getitem__ (~1-4s/file, không khả thi cho 60k+ mẫu). __getitem__ dưới đây đọc thẳng từ cache nếu đã prefetch; nếu chưa (vd.
dùng nhanh trong EDA) sẽ tự động fallback tải lẻ qua hf_hub_download.

`load_dataset` mất ~45 s mỗi lần gọi kể cả đã cache (liệt kê file repo HF có
118k file), lần đầu ~2 phút; train gọi 2 lần × mỗi tiến trình DDP. Đo
2026-10-04: offline đọc cache 0,2 s. Nên làm ấm cache một lần rồi chạy offline:
    python -m src.data.vietsuperspeech_dataset        # có mạng, tải metadata 2 split
    HF_HUB_OFFLINE=1 python -m src.training.train ...
Cache gắn theo revision (thư mục tên `HF_REVISION`) nên offline vẫn đúng bản pin.
"""

import csv
import os
import re
from pathlib import Path

import soundfile as sf
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

HF_DATASET_ID = "thanhnew2001/VietSuperSpeech"
# Pin revision (D10, docs/notes/training_plan_kaggle.md): manifest shard
# (data/splits/) và excluded.tsv lưu theo index split — tác giả đẩy commit mới
# là index lệch.
HF_REVISION = "cbf624ae9b30e1c2793a27e95b262115c69601f3"
# Kaggle giải nén tar dataset vào /tmp/audio_cache (src/data/extract_audio.py)
# → đặt biến môi trường thay vì sửa code; local giữ đường dẫn tương đối cũ.
AUDIO_CACHE_DIR = os.environ.get("AUDIO_CACHE_DIR", "data/raw/audio_cache")
SPLITS_DIR = Path("data/splits")
EXCLUDED_PATH = SPLITS_DIR / "excluded.tsv"
HELDOUT_PATH = SPLITS_DIR / "heldout_videos.tsv"
# Manifest của make_shards.py = đúng nội dung 5 Kaggle Dataset (tar).
SHARD_MANIFESTS = ["train_shard0", "train_shard1", "train_shard2", "train_shard3", "val"]


def manifest_split(name: str) -> str:
    """Split HF mà cột `index` của manifest shard trỏ vào (make_shards.py: train_shard* / val)."""
    return "train" if name.startswith("train") else "validation"


def video_of(audio: str) -> str:
    return re.sub(r"_seg\d+\.wav$", "", audio.rsplit("/", 1)[-1])


def load_excluded(path: Path = EXCLUDED_PATH) -> set[tuple[str, int]]:
    """(split, index) bị loại theo A1 (src/data/filter_language.py)."""
    with open(path, encoding="utf-8", newline="") as f:
        return {(r["split"], int(r["index"])) for r in csv.DictReader(f, delimiter="\t")}


def load_heldout_videos(path: Path = HELDOUT_PATH) -> set[str]:
    """Video giữ riêng cho test độc lập (src/data/make_heldout.py, chốt 2026-10-04)."""
    with open(path, encoding="utf-8", newline="") as f:
        return {r["video"] for r in csv.DictReader(f, delimiter="\t")}


def manifest_rows(names: list[str]) -> list[dict]:
    """Gộp các manifest `data/splits/<name>.tsv` → [{split, index, audio, duration}].

    Lọc ở bước đọc (manifest/tar đã tạo trước, giữ nguyên): luôn bỏ excluded.tsv
    (A1); với manifest shard (train_shard*, val) bỏ thêm video giữ riêng — để
    train và val "đã gặp" không chứa video của clean-test/val_unseen. File có cột
    `split` (val_unseen.tsv) lấy split theo từng dòng.
    """
    excluded = load_excluded()
    heldout = load_heldout_videos()
    rows = []
    for name in names:
        drop_heldout = name in SHARD_MANIFESTS
        with open(SPLITS_DIR / f"{name}.tsv", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                split = r.get("split") or manifest_split(name)
                key = (split, int(r["index"]))
                if key in excluded or (drop_heldout and video_of(r["audio"]) in heldout):
                    continue
                rows.append({"split": split, "index": key[1], "audio": r["audio"], "duration": float(r["duration"])})
    return rows


class VietSuperSpeechDataset(Dataset):
    def __init__(self, split: str = "train", tokenizer=None, indices: list[int] | None = None,
                 items: list[tuple[str, int]] | None = None):
        """split: "train" hoặc "validation" — tên split thật trên HF Hub
        (KHÔNG phải "dev-test" như ghi trong đề cương, xác nhận qua
        src/data/survey.py, Tuần 3 — xem docs/notes/dataset_discrepancy.md
        cho chênh lệch số liệu đầy đủ so với đề cương).

        items: [(split, index)] — trộn được hai split (val_unseen, clean-test lấy
        từ cả hai); thường từ `manifest_rows`. Nếu không có: dùng `split` +
        `indices` (None = cả split, **chưa** lọc excluded/heldout)."""
        if items is None:
            ds = load_dataset(HF_DATASET_ID, split=split, revision=HF_REVISION)
            self.hf = {split: ds}
            items = [(split, i) for i in (range(len(ds)) if indices is None else indices)]
        else:
            self.hf = {s: load_dataset(HF_DATASET_ID, split=s, revision=HF_REVISION) for s in {s for s, _ in items}}
        self.items = list(items)
        self.tokenizer = tokenizer
        Path(AUDIO_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        split, index = self.items[idx]
        item = self.hf[split][index]
        # Schema: dict_keys(['audio', 'text', 'duration', 'source']) —
        # "audio" là đường dẫn tương đối (string), không tự giải mã, xem docstring đầu file.
        local_path = Path(AUDIO_CACHE_DIR) / item["audio"]
        if not local_path.exists():
            # Chưa chạy prefetch_audio.py — tải lẻ qua HTTP (chậm, chỉ nên
            # dùng cho khảo sát/EDA nhỏ, không phù hợp để train trực tiếp).
            local_path = Path(hf_hub_download(HF_DATASET_ID, item["audio"], repo_type="dataset",
                                              revision=HF_REVISION, local_dir=AUDIO_CACHE_DIR))
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


if __name__ == "__main__":
    # Làm ấm cache metadata HF (xem docstring đầu file).
    for _split in ("train", "validation"):
        print(_split, len(load_dataset(HF_DATASET_ID, split=_split, revision=HF_REVISION)))
