"""Danh sách đoạn loại khỏi train/val/clean-test vì audio không phải tiếng Việt
hoặc nhãn hỏng — quyết định A1 (người dùng chốt 2026-10-03, phương án C ∩ L,
docs/notes/survey_model_candidates.md mục 6).

Vì sao hai tín hiệu: nhãn do Zipformer chỉ-tiếng-Việt sinh tự động. Lọc theo
nhãn bắt được tiếng Anh bị phiên thành chuỗi giả tiếng Anh, nhưng để lọt tiếng
Nhật bị phiên thành âm tiết Việt có dấu (người dùng nghe kiểm Alive Kicking
Ep4). LID trên audio (Whisper-small, kernel `scripts/kaggle/lid/`) bắt được,
còn nhãn bắt thêm đoạn audio Việt nhưng nhãn rác. Một đoạn bị loại nếu dính
BẤT KỲ lý do nào:

- `lid`: Whisper không chọn `vi` là ngôn ngữ cao nhất. `p_vi` hai đỉnh rõ nên
  dùng top1 thay ngưỡng xác suất (đổi ngưỡng 0,5 chỉ lệch ~0,1 h).
- `video`: video có ≥ 80% đoạn nhãn < 20% từ có dấu (gộp train + validation)
  — các tập phỏng vấn khách nước ngoài, đã nghe mẫu đầu/giữa/cuối.
- `label`: nhãn < 20% từ có dấu (`is_vietnamese_label`).

Không sửa manifest shard (data/splits/train_shard*.tsv, val.tsv) hay tar trên
Kaggle: loại ở bước đọc dữ liệu, nên 5 Kaggle Dataset giữ nguyên.

    python -m src.data.filter_language                    # cần data/processed/lid.csv (output kernel lid-asr)
    python -m src.data.filter_language --train-tokenizer  # + train lại BPE trên nhãn train đã lọc

Tokenizer train lại 2026-10-03 (người dùng duyệt): bản cũ học cả ~11.800 nhãn
rác nên một phần vocab là mảnh từ giả tiếng Anh. Giữ nguyên tham số
(`BPETokenizer.train`, vocab 1000) để số tham số CTC head không đổi.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from datasets import load_dataset

from src.data.vietsuperspeech_dataset import HF_DATASET_ID, HF_REVISION, manifest_rows
from src.evaluation.text_normalize import is_vietnamese_label
from src.tokenizer.bpe_tokenizer import BPETokenizer

LID_PATH = Path("data/processed/lid.csv")
OUT_PATH = Path("data/splits/excluded.tsv")
CORPUS_PATH = Path("data/processed/train_transcripts_filtered.txt")
TOKENIZER_PREFIX = "configs/tokenizer"
VIDEO_BAD_FRACTION = 0.8
_VIDEO = re.compile(r"([^/]*)_seg\d+\.wav$")


def main(train_tokenizer: bool = False):
    lid = {}
    with open(LID_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lid[(r["split"], int(r["index"]))] = r["top1"]

    rows = []
    for split in ("train", "validation"):
        ds = load_dataset(HF_DATASET_ID, split=split, revision=HF_REVISION)
        for i, (a, t, d) in enumerate(zip(ds["audio"], ds["text"], ds["duration"])):
            rows.append({"split": split, "index": i, "audio": a, "duration": d,
                         "video": _VIDEO.search(a)[1], "vi_label": is_vietnamese_label(t), "text": t})
    missing = [(r["split"], r["index"]) for r in rows if (r["split"], r["index"]) not in lid]
    assert not missing, f"lid.csv thiếu {len(missing)} đoạn, vd. {missing[:3]}"

    per_video = defaultdict(list)
    for r in rows:
        per_video[r["video"]].append(not r["vi_label"])
    bad_videos = {v for v, b in per_video.items() if sum(b) / len(b) >= VIDEO_BAD_FRACTION}

    excluded = []
    for r in rows:
        reasons = []
        if lid[(r["split"], r["index"])] != "vi":
            reasons.append(f"lid:{lid[(r['split'], r['index'])]}")
        if r["video"] in bad_videos:
            reasons.append("video")
        if not r["vi_label"]:
            reasons.append("label")
        if reasons:
            excluded.append({**r, "reason": "+".join(reasons)})

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        f.write("split\tindex\taudio\treason\n")
        for r in excluded:
            f.write(f"{r['split']}\t{r['index']}\t{r['audio']}\t{r['reason']}\n")

    ex = {(r["split"], r["index"]) for r in excluded}
    print(f"video bị loại cả video: {len(bad_videos)}; đoạn bị loại: {len(excluded)} → {OUT_PATH}")
    # Chia train/val/clean-test sau lọc: xem src/data/make_heldout.py (video giữ riêng, 2026-10-04).
    for split in ("train", "validation"):
        part = [r for r in rows if r["split"] == split]
        kept = [r for r in part if (r["split"], r["index"]) not in ex]
        h = lambda xs: sum(r["duration"] for r in xs) / 3600  # noqa: E731
        print(f"  {split:12s} {len(part):6d} câu {h(part):7.2f} h → còn {len(kept):6d} câu {h(kept):7.2f} h")

    if train_tokenizer:
        # Đúng tập train lúc học: 4 shard, bỏ excluded + video giữ riêng (nhãn
        # của video test không được lọt vào vocab).
        train_keys = {(r["split"], r["index"]) for r in manifest_rows([f"train_shard{i}" for i in range(4)])}
        lines = [r["text"].strip() for r in rows if (r["split"], r["index"]) in train_keys]
        CORPUS_PATH.write_text("\n".join(l for l in lines if l) + "\n", encoding="utf-8")
        BPETokenizer.train(str(CORPUS_PATH), TOKENIZER_PREFIX, vocab_size=1000)
        print(f"tokenizer: {len(lines)} nhãn → {TOKENIZER_PREFIX}.model/.vocab")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tokenizer", action="store_true")
    main(parser.parse_args().train_tokenizer)
