"""BPE tokenizer tiếng Việt, dùng chung cho cả hai thí nghiệm.

Huấn luyện bằng SentencePiece trên transcript của tập train VietSuperSpeech
(Tuần 3). blank_id=0 dành riêng cho CTC blank token.
"""

from pathlib import Path

import sentencepiece as spm


class BPETokenizer:
    BLANK_ID = 0

    def __init__(self, model_path: str | None = None):
        self.sp = spm.SentencePieceProcessor()
        if model_path:
            self.sp.load(model_path)

    @staticmethod
    def train(text_file: str, model_prefix: str, vocab_size: int = 1000):
        """text_file: 1 dòng = 1 transcript, trích từ tập train.
        vocab_size: bắt đầu thử 1000 — điều chỉnh theo kết quả khảo sát dữ liệu.
        Dành 1 id cho <blank> (CTC) — SentencePiece tự thêm <unk>/<s>/</s>,
        cần map thủ công id 0 -> blank khi build vocab cho CTC head.
        """
        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,  # tiếng Việt có dấu — cần coverage đầy đủ
            pad_id=-1,
            unk_id=1,
            bos_id=-1,
            eos_id=-1,
        )

    def encode(self, text: str) -> list[int]:
        # +1 để id 0 luôn dành cho CTC blank (SentencePiece id gốc dịch lên 1)
        return [i + 1 for i in self.sp.encode(text, out_type=int)]

    def decode(self, token_ids: list[int]) -> str:
        ids = [i - 1 for i in token_ids if i != self.BLANK_ID]
        return self.sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size() + 1  # +1 cho blank
