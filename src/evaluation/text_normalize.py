"""Chuẩn hóa văn bản dùng chung trước khi tính WER/CER — cho MỌI mô hình
(Mamba/Conformer của ta, Parakeet, PhoWhisper, wav2vec2, ...).

Vì sao cần: nhãn VietSuperSpeech viết HOA, không dấu câu, còn Parakeet (train
với nhãn có dấu câu + hoa/thường do Qwen3 sinh) và PhoWhisper ra chữ thường/hoa
lẫn dấu câu. Không chuẩn hóa chung thì WER đo khác biệt định dạng chứ không đo
nhận dạng. Chỉ được có MỘT hàm này: kernel `scripts/kaggle/zero_shot/zero_shot.py`
chép nguyên văn file này (có kiểm tra khớp khi chạy local) — sửa ở đây thì
chép lại sang đó.

Quyết định (đo trên `data/processed/train_transcripts.txt`, 60.656 nhãn, 2026-10-02):
- Nhãn đã NFC 100%, không có chữ số nào (số luôn viết bằng chữ: "BA ĐẾN NĂM
  TRĂM", "MỘT TRĂM PHẦN TRĂM"), ký tự ngoài chữ cái chỉ có `-` (29 lần, vd.
  CHECK-IN, WIN-WIN, X-QUANG, và `-` đứng riêng) và `<` (5 lần, rác).
- Gạch nối → khoảng trắng: nhãn tự mâu thuẫn ("CHECK-IN" / "CHECK IN" /
  "CHECKIN"), tách ra thì hai cách viết đầu khớp nhau.
- Dấu nháy đơn bị xóa (không thành khoảng trắng) để "don't" → "dont" — một từ,
  giống cách Zipformer viết tiếng Anh không dấu nháy.
- **Chữ số giữ nguyên, không đọc thành chữ.** Đọc số tiếng Việt có nhiều cách
  hợp lệ ("hai nghìn không trăm hai mươi tư" / "hai không hai bốn", "nghìn" /
  "ngàn", "mốt" / "một", "linh" / "lẻ"), bộ đọc số nào cũng tự sinh lỗi riêng
  và thiên vị cách đọc của nó. Vì nhãn không bao giờ có chữ số, mỗi chữ số trong
  hyp là một lỗi thật về *định dạng* — kernel zero-shot đếm số câu hyp có chữ số
  (`has_digit`) để lượng hóa phần WER này, thay vì che đi.
"""

import re
import unicodedata

_APOSTROPHES = "'’ʼ‘`"
_WHITESPACE = re.compile(r"\s+")
_DIGIT = re.compile(r"\d")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    for ch in _APOSTROPHES:
        text = text.replace(ch, "")
    # Mọi dấu câu (P*) và ký hiệu (S*: %, <, ⁇ của sentencepiece khi gặp unk, ...)
    # thành khoảng trắng — không xóa hẳn, kẻo dính hai từ "a,b" → "ab".
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return _WHITESPACE.sub(" ", text).strip()


def has_digit(text: str) -> bool:
    return bool(_DIGIT.search(text))


def diacritic_word_ratio(text: str) -> float:
    """Tỉ lệ từ mang dấu tiếng Việt (NFD rồi tìm ký tự combining; `đ` tính là có dấu).

    Heuristic CHƯA KIỂM CHỨNG (docs/notes/survey_model_candidates.md mục 4): đo
    ngôn ngữ của *nhãn*, không phải của audio."""
    words = normalize_text(text).split()
    if not words:
        return 0.0
    marked = sum(
        1 for w in words if "đ" in w or any(unicodedata.combining(ch) for ch in unicodedata.normalize("NFD", w))
    )
    return marked / len(words)


VIETNAMESE_LABEL_MIN_RATIO = 0.2


def is_vietnamese_label(text: str) -> bool:
    return diacritic_word_ratio(text) >= VIETNAMESE_LABEL_MIN_RATIO
