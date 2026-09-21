"""Tính WER: trên `validation` (eval loop trong train.py) và trên clean-test
(250 câu, xem eval_clean_test.py)."""

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
