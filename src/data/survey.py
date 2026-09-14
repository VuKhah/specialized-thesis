"""Khảo sát thống kê VietSuperSpeech + chuẩn bị dữ liệu cho tokenizer và
clean-test set (Tuần 3 theo đề cương).

Chạy: python -m src.data.survey
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from datasets import load_dataset

from src.data.vietsuperspeech_dataset import HF_DATASET_ID
from src.tokenizer.bpe_tokenizer import BPETokenizer

DURATION_BUCKETS_S = [(0, 3), (3, 8), (8, 15), (15, 22), (22, 30), (30, float("inf"))]


def _bucket(duration: float) -> str:
    for lo, hi in DURATION_BUCKETS_S:
        if lo <= duration < hi:
            return f"{lo}-{hi if hi != float('inf') else '+'}s"
    return "unknown"


def compute_split_stats(split: str, streaming: bool = True) -> dict:
    """Duyệt qua toàn bộ split, KHÔNG decode audio (chỉ đọc text/duration/
    source) để tránh tải/giải mã hàng trăm giờ audio chỉ để lấy thống kê."""
    ds = load_dataset(HF_DATASET_ID, split=split, streaming=streaming)

    n = 0
    total_duration = 0.0
    durations = []
    text_char_lens = []
    text_word_lens = []
    bucket_counts = Counter()
    source_counts = Counter()
    empty_text = 0
    chars = Counter()

    for item in ds:
        n += 1
        dur = item["duration"]
        text = item["text"]
        total_duration += dur
        durations.append(dur)
        bucket_counts[_bucket(dur)] += 1
        source_counts[item["source"]] += 1
        if not text or not text.strip():
            empty_text += 1
        else:
            text_char_lens.append(len(text))
            text_word_lens.append(len(text.split()))
            chars.update(text)

        if n % 5000 == 0:
            print(f"  ... {n} mẫu, {total_duration / 3600:.1f} giờ")

    durations.sort()

    def pct(p):
        if not durations:
            return None
        idx = min(len(durations) - 1, int(p * len(durations)))
        return durations[idx]

    return {
        "split": split,
        "n_samples": n,
        "total_duration_hours": total_duration / 3600,
        "duration_min_s": min(durations) if durations else None,
        "duration_max_s": max(durations) if durations else None,
        "duration_mean_s": total_duration / n if n else None,
        "duration_p50_s": pct(0.5),
        "duration_p90_s": pct(0.9),
        "duration_p99_s": pct(0.99),
        "duration_buckets": dict(bucket_counts),
        "n_unique_sources": len(source_counts),
        "top_10_sources_by_count": source_counts.most_common(10),
        "empty_text_count": empty_text,
        "text_char_len_mean": sum(text_char_lens) / len(text_char_lens) if text_char_lens else None,
        "text_word_len_mean": sum(text_word_lens) / len(text_word_lens) if text_word_lens else None,
        "n_unique_chars": len(chars),
        "char_frequency_top_50": chars.most_common(50),
    }


def build_text_corpus(split: str, out_path: str, streaming: bool = True) -> int:
    """Ghi toàn bộ transcript (1 dòng/câu) ra file — dùng để train BPE tokenizer."""
    ds = load_dataset(HF_DATASET_ID, split=split, streaming=streaming)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            text = item["text"].strip() if item["text"] else ""
            if text:
                f.write(text + "\n")
                n += 1
    return n


def sample_clean_test(split: str = "validation", n: int = 250, seed: int = 42, out_path: str = "data/processed/clean_test_manifest.json") -> list[dict]:
    """Lấy mẫu ngẫu nhiên (seed cố định) từ split validation để hiệu đính thủ công.

    Split thật trên HF Hub tên là "validation" (KHÔNG phải "dev-test" như
    ghi trong đề cương — xem docs/notes/dataset_discrepancy.md). Load không
    streaming (split nhỏ) để lấy theo index trực tiếp, đảm bảo tái lập seed.
    """
    ds = load_dataset(HF_DATASET_ID, split=split, streaming=False)
    total = len(ds)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(total), min(n, total)))

    manifest = []
    for idx in indices:
        item = ds[idx]
        manifest.append(
            {
                "index": idx,
                "source": item["source"],
                "duration_s": item["duration"],
                "pseudo_label": item["text"],
                "corrected_text": "",  # điền tay sau khi nghe lại audio
                "notes": "",
            }
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"split": split, "seed": seed, "n": len(manifest), "samples": manifest},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return manifest


if __name__ == "__main__":
    Path("reports/results").mkdir(parents=True, exist_ok=True)

    train_json_path = Path("reports/results/dataset_survey_train.json")
    if train_json_path.exists():
        print("Đã có reports/results/dataset_survey_train.json — bỏ qua, dùng lại.")
        with open(train_json_path, encoding="utf-8") as f:
            train_stats = json.load(f)
    else:
        print("Khảo sát tập train...")
        train_stats = compute_split_stats("train")
        with open(train_json_path, "w", encoding="utf-8") as f:
            json.dump(train_stats, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in train_stats.items() if k != "char_frequency_top_50"}, ensure_ascii=False, indent=2))

    print("\nKhảo sát tập validation...")
    val_stats = compute_split_stats("validation")
    with open("reports/results/dataset_survey_validation.json", "w", encoding="utf-8") as f:
        json.dump(val_stats, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in val_stats.items() if k != "char_frequency_top_50"}, ensure_ascii=False, indent=2))

    print("\nXây corpus text cho tokenizer (từ tập train)...")
    n_lines = build_text_corpus("train", "data/processed/train_transcripts.txt")
    print(f"  Đã ghi {n_lines} dòng.")

    print("\nLấy mẫu clean-test (n=250, seed=42) từ validation...")
    manifest = sample_clean_test("validation", n=250, seed=42)
    print(f"  Đã ghi {len(manifest)} mẫu vào data/processed/clean_test_manifest.json")

    print("\nTrain BPE tokenizer (vocab_size=1000) trên corpus vừa xây...")
    Path("configs").mkdir(parents=True, exist_ok=True)
    BPETokenizer.train("data/processed/train_transcripts.txt", "configs/tokenizer", vocab_size=1000)
    print("  Đã ghi configs/tokenizer.model + configs/tokenizer.vocab")
