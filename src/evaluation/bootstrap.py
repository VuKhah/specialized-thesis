"""Khoảng tin cậy WER bằng bootstrap theo khối video.

Câu cùng một video (cùng người nói, chủ đề, điều kiện thu) không độc lập → bootstrap
từng câu (Bisani & Ney, ICASSP 2004) cho khoảng tin cậy hẹp hơn thật. Theo Liu &
Peng, "Statistical Testing on ASR Performance via Blockwise Bootstrap"
(arXiv:1912.09508): lấy mẫu lại theo **khối** — ở đây một khối = một video.
WER của mỗi mẫu bootstrap = tổng lỗi / tổng số từ tham chiếu của các khối được
chọn (không phải trung bình WER từng câu), cùng định nghĩa với `compute_wer`.

So hai mô hình trên cùng tập (paired): mỗi lần lấy cùng tập khối cho cả hai, tính
khoảng tin cậy của hiệu WER.

    python -m src.evaluation.bootstrap A/eval_val_epoch4.jsonl
    python -m src.evaluation.bootstrap A/eval_val_epoch4.jsonl B/eval_val_epoch4.jsonl

Đầu vào: jsonl do train.py ghi (`split`, `index`, `video`, `ref`, `hyp` mỗi dòng).
"""

import argparse
import json
import sys
from collections import defaultdict

import jiwer
import numpy as np

from src.evaluation.text_normalize import normalize_text


def load_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def per_video_counts(records: list[dict]) -> dict[str, tuple[int, int]]:
    """{video: (số lỗi S+D+I, số từ tham chiếu)}, sau normalize_text như compute_wer."""
    errors, words = defaultdict(int), defaultdict(int)
    for r in records:
        ref, hyp = normalize_text(r["ref"]), normalize_text(r["hyp"])
        if not ref:
            continue  # jiwer không nhận ref rỗng; compute_wer cũng sẽ lỗi ở câu này
        out = jiwer.process_words(ref, hyp)
        errors[r["video"]] += out.substitutions + out.deletions + out.insertions
        words[r["video"]] += out.substitutions + out.deletions + out.hits
    return {v: (errors[v], words[v]) for v in words}


def _resample(n_videos: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_videos, size=(n_boot, n_videos))


def block_bootstrap_wer(records: list[dict], n_boot: int = 10_000, seed: int = 42, alpha: float = 0.05) -> dict:
    counts = per_video_counts(records)
    err = np.array([c[0] for c in counts.values()], dtype=np.float64)
    words = np.array([c[1] for c in counts.values()], dtype=np.float64)
    idx = _resample(len(counts), n_boot, seed)
    boot = err[idx].sum(1) / words[idx].sum(1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"wer": float(err.sum() / words.sum()), "ci_low": float(lo), "ci_high": float(hi),
            "n_videos": len(counts), "n_sentences": len(records), "n_boot": n_boot}


def paired_block_bootstrap(records_a: list[dict], records_b: list[dict], n_boot: int = 10_000, seed: int = 42,
                           alpha: float = 0.05) -> dict:
    """Hiệu WER(A) − WER(B) trên cùng tập câu; khoảng tin cậy không chứa 0 ⇒ chênh lệch có ý nghĩa."""
    key = lambda r: (r["split"], r["index"])  # noqa: E731
    if {key(r) for r in records_a} != {key(r) for r in records_b}:
        raise ValueError("Hai file không cùng tập câu — không so cặp được")
    ca, cb = per_video_counts(records_a), per_video_counts(records_b)
    videos = sorted(ca)
    ea, eb = (np.array([c[v][0] for v in videos], dtype=np.float64) for c in (ca, cb))
    words = np.array([ca[v][1] for v in videos], dtype=np.float64)
    idx = _resample(len(videos), n_boot, seed)
    diff = (ea[idx].sum(1) - eb[idx].sum(1)) / words[idx].sum(1)
    lo, hi = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    return {"wer_a": float(ea.sum() / words.sum()), "wer_b": float(eb.sum() / words.sum()),
            "diff": float((ea.sum() - eb.sum()) / words.sum()), "ci_low": float(lo), "ci_high": float(hi),
            "p_a_better": float((diff < 0).mean()), "n_videos": len(videos), "n_boot": n_boot}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+", help="1 file: CI của WER; 2 file: CI của hiệu WER (file1 − file2)")
    parser.add_argument("--n_boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(args.files) == 1:
        r = block_bootstrap_wer(load_records(args.files[0]), args.n_boot, args.seed)
        print(f"WER {r['wer']:.4f}  CI95 [{r['ci_low']:.4f}, {r['ci_high']:.4f}]  "
              f"({r['n_sentences']} câu, {r['n_videos']} video)")
    elif len(args.files) == 2:
        r = paired_block_bootstrap(load_records(args.files[0]), load_records(args.files[1]), args.n_boot, args.seed)
        print(f"WER A {r['wer_a']:.4f}  B {r['wer_b']:.4f}  A−B {r['diff']:+.4f}  "
              f"CI95 [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  P(A tốt hơn) {r['p_a_better']:.3f}  "
              f"({r['n_videos']} video)")
    else:
        parser.error("Nhận 1 hoặc 2 file")


if __name__ == "__main__":
    main()
