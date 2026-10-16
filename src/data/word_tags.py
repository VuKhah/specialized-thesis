"""Bảng loại từ cho clean-test: từ nào là tiếng Anh / tên riêng / viết tắt (thêm 2026-10-15).

Vì sao không dùng ô Ghi chú: ghi chú theo câu, gõ tay, nhiều câu quên ghi `tiếng Anh` (33 câu thiếu so với nội
dung đã hiệu đính). WER tách nhóm cần thông tin theo **từng từ** → lấy tự động từ `corrected_text`: mọi từ không
phải âm tiết Việt hợp lệ là ứng viên; người dùng chỉ duyệt cột `label` (không nghe lại audio).

Từ tiếng Anh trùng dạng âm tiết Việt (*in, to, so*) không tự phát hiện được → liệt kê từng lần xuất hiện của một
danh sách ngắn trong câu đã có từ ngoại, để người dùng quyết theo ngữ cảnh (khóa `từ@#câu:vị trí`).

Chạy lại an toàn: nhãn đã điền trong file cũ được giữ, chỉ thêm dòng mới / cập nhật đếm.

    python -m src.data.word_tags
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from src.evaluation.text_normalize import normalize_text
from src.evaluation.vi_syllable import is_valid_syllable

MANIFEST = Path("data/processed/clean_test_manifest.json")
ENTITIES = Path("docs/notes/clean_test_entities.md")
OUT = Path("data/processed/clean_test_word_tags.tsv")
LABELS = ("tiếng anh", "tên riêng", "viết tắt", "tiếng việt", "từ đệm", "lỗi gõ")
# Từ tiếng Anh hay gặp mà đồng dạng âm tiết Việt không dấu — chỉ xét trong câu đã có từ ngoại.
AMBIGUOUS = {"in", "on", "to", "so", "go", "no", "hi", "me", "an", "oh", "top", "my", "but", "u"}
FIELDS = ["key", "label", "guess", "count", "sentences", "context"]


def entity_words() -> set[str]:
    """Chữ trong bảng tên riêng (mục 1 cột "Ghi là" + mục 2) → đoán `tên riêng`."""
    text = ENTITIES.read_text(encoding="utf-8")
    sec1, sec2 = text.split("## 1.")[1].split("## 2.")
    sec2 = sec2.split("## 3.")[0].split("\n", 1)[1]
    parts = [ln.split("|")[2] for ln in sec1.splitlines() if ln.startswith("|") and not ln.startswith("|--")][1:]
    words = set()
    for part in parts + [sec2]:
        words.update(re.findall(r"[a-z]+", re.sub(r"\(.*?\)|#\d+", " ", part)))
    return words


def guess(word: str, ents: set[str]) -> str:
    if not re.search(r"[aeiouy]", word) or (len(word) <= 3 and word in {"mv", "nft", "dna", "aig", "id", "ok"}):
        return "viết tắt"
    return "tên riêng" if word in ents else "tiếng anh"


def context(words: list[str], i: int, k: int = 4) -> str:
    return " ".join(words[max(0, i - k):i] + [f"[{words[i]}]"] + words[i + 1:i + 1 + k])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    samples = json.loads(MANIFEST.read_text(encoding="utf-8"))["samples"]
    ents = entity_words()
    rows: dict[str, dict] = {}
    for n, s in enumerate(samples, 1):
        words = normalize_text(s["corrected_text"]).split()
        foreign = [i for i, w in enumerate(words) if not is_valid_syllable(w)]
        for i in foreign:
            r = rows.setdefault(words[i], {"key": words[i], "label": "", "guess": guess(words[i], ents), "count": 0,
                                           "sentences": [], "context": context(words, i)})
            r["count"] += 1
            if f"#{n}" not in r["sentences"]:
                r["sentences"].append(f"#{n}")
        if foreign:
            for i, w in enumerate(words):
                if w in AMBIGUOUS:
                    rows[f"{w}@#{n}:{i}"] = {"key": f"{w}@#{n}:{i}", "label": "", "guess": "?", "count": 1,
                                             "sentences": [f"#{n}"], "context": context(words, i)}
    old = {}
    if OUT.exists():
        with OUT.open(encoding="utf-8-sig", newline="") as f:
            old = {r["key"]: r for r in csv.DictReader(f, delimiter="\t")}
    for k, r in rows.items():
        # Giữ cả `guess` cũ: người dùng từng sửa thẳng cột guess (15/10), chạy lại không được xoá công sức đó.
        r["label"] = old.get(k, {}).get("label", "")
        r["guess"] = old.get(k, {}).get("guess") or r["guess"]
        r["sentences"] = " ".join(r["sentences"])
    # Từ đơn trước (nhiều lần xuất hiện lên đầu), từ mơ hồ theo ngữ cảnh xuống cuối.
    order = sorted(rows.values(), key=lambda r: ("@" in r["key"], -r["count"], r["key"]))
    # utf-8-sig để Excel mở đúng tiếng Việt.
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(order)
    bad = [r["key"] for r in order if r["label"] and r["label"] not in LABELS]
    single = sum("@" not in r["key"] for r in order)
    print(f"{OUT}: {single} từ ngoài âm tiết Việt + {len(order) - single} lần xuất hiện từ mơ hồ; "
          f"đã gán {sum(bool(r['label']) for r in order)}/{len(order)}")
    if bad:
        print(f"⚠️ nhãn lạ (chỉ dùng {', '.join(LABELS)}): {bad}")


if __name__ == "__main__":
    main()
