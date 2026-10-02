"""Tính WER: trên `validation` (eval loop trong train.py) và trên clean-test
(250 câu, xem eval_clean_test.py).

Chuẩn hóa ref/hyp bằng `normalize_text` ngay tại đây (không để bên gọi tự làm)
để mọi đường tính WER — eval loop của train.py cho cả hai encoder, clean-test,
và mô hình pre-train ngoài — đi qua đúng một hàm (xem text_normalize.py).
"""

import jiwer

from src.evaluation.text_normalize import normalize_text


def _normalize_all(texts: list[str]) -> list[str]:
    return [normalize_text(t) for t in texts]


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    return jiwer.wer(_normalize_all(references), _normalize_all(hypotheses))


def compute_wer_report(references: list[str], hypotheses: list[str]) -> dict:
    """Trả về đầy đủ substitutions/deletions/insertions — dùng cho phân tích
    lỗi định tính (error taxonomy, Chương 3)."""
    out = jiwer.process_words(_normalize_all(references), _normalize_all(hypotheses))
    return {
        "wer": out.wer,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "hits": out.hits,
    }
