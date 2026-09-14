"""Tính WER trên dev-test gốc và clean-test (200-300 câu hiệu đính thủ công)."""

import jiwer


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    return jiwer.wer(references, hypotheses)


def compute_wer_report(references: list[str], hypotheses: list[str]) -> dict:
    """Trả về đầy đủ substitutions/deletions/insertions — dùng cho phân tích
    lỗi định tính (error taxonomy, Chương 3)."""
    out = jiwer.process_words(references, hypotheses)
    return {
        "wer": out.wer,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "hits": out.hits,
    }
