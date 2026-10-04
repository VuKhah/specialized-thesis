"""Giữ riêng một số video khỏi train/val để có tập test **độc lập theo video**
— người dùng chốt 2026-10-04 (phương án B, giữ riêng theo video).

Vì sao: `validation` của VietSuperSpeech rải ~10% đoạn từ chính các video của
train (561/562 video trùng) → clean-test cũ (250 câu từ `validation`) 100% thuộc
video đã train: WER chỉ đo "người nói đã gặp", và so với mô hình pre-train bị
lệch. Chuẩn công bố tách theo người nói (LibriSpeech, VIVOS). Tách theo video
chưa chắc tách được người nói (MC chương trình lớn xuất hiện ở mọi tập) — nêu
hạn chế; khách mời thì chưa gặp.

Cách chọn (tất định): nguồn = các câu có audio trong 5 Kaggle Dataset (4 shard
+ val, tức đã trừ clean-test cũ) và chưa bị loại ở `excluded.tsv`. Video đủ
điều kiện = ≥ `MIN_SENTENCES` câu; xáo (seed 42) rồi lấy dần tới khi đạt
`HOLDOUT_FRACTION` số giờ. Từ mỗi video giữ riêng lấy `PER_VIDEO` câu làm
clean-test mới (để hiệu đính tay), phần còn lại là `val_unseen` (nhãn tự động).

Ghi ra (track git):
- `data/splits/heldout_videos.tsv` — video bị bỏ khỏi train và val khi đọc.
- `data/splits/val_unseen.tsv` — `split`, `index`, `audio`, `duration`.
- `data/processed/clean_test_manifest.json` — bản 2: mỗi câu có `split` riêng
  (câu đến từ cả `train` lẫn `validation`). Ghi đè bản cũ (cột hiệu đính của
  bản cũ còn trống; bản cũ vẫn trong lịch sử git).

Không sửa manifest shard / tar: lọc ở bước đọc như `excluded.tsv`.

    python -m src.data.make_heldout
"""

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from src.data.vietsuperspeech_dataset import (HF_DATASET_ID, HF_REVISION, SPLITS_DIR, SHARD_MANIFESTS,
                                              load_excluded, manifest_split, video_of)

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
HOLDOUT_FRACTION = 0.06
MIN_SENTENCES = 20
PER_VIDEO = 7
HELDOUT_PATH = SPLITS_DIR / "heldout_videos.tsv"
VAL_UNSEEN_PATH = SPLITS_DIR / "val_unseen.tsv"
CLEAN_TEST_PATH = Path("data/processed/clean_test_manifest.json")


def main() -> None:
    excluded = load_excluded()
    pool = []  # câu có audio trong 5 Kaggle Dataset, chưa bị loại A1
    for name in SHARD_MANIFESTS:
        split = manifest_split(name)
        with open(SPLITS_DIR / f"{name}.tsv", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if (split, int(r["index"])) not in excluded:
                    pool.append({"split": split, "index": int(r["index"]), "audio": r["audio"],
                                 "duration": float(r["duration"]), "video": video_of(r["audio"])})

    by_video = defaultdict(list)
    for r in pool:
        by_video[r["video"]].append(r)
    total_h = sum(r["duration"] for r in pool) / 3600

    eligible = sorted(v for v, rs in by_video.items() if len(rs) >= MIN_SENTENCES)
    random.Random(SEED).shuffle(eligible)
    heldout, held_h = [], 0.0
    for v in eligible:
        if held_h >= HOLDOUT_FRACTION * total_h:
            break
        heldout.append(v)
        held_h += sum(r["duration"] for r in by_video[v]) / 3600
    heldout.sort()

    rng = random.Random(SEED)
    test, unseen = [], []
    for v in heldout:
        rows = sorted(by_video[v], key=lambda r: (r["split"], r["index"]))
        picked = set(map(id, rng.sample(rows, PER_VIDEO)))
        for r in rows:
            (test if id(r) in picked else unseen).append(r)

    with open(HELDOUT_PATH, "w", encoding="utf-8", newline="") as f:
        f.write("video\tn_sentences\thours\n")
        for v in heldout:
            rs = by_video[v]
            f.write(f"{v}\t{len(rs)}\t{sum(r['duration'] for r in rs) / 3600:.3f}\n")
    with open(VAL_UNSEEN_PATH, "w", encoding="utf-8", newline="") as f:
        f.write("split\tindex\taudio\tduration\n")
        for r in unseen:
            f.write(f"{r['split']}\t{r['index']}\t{r['audio']}\t{r['duration']}\n")

    texts = {s: load_dataset(HF_DATASET_ID, split=s, revision=HF_REVISION) for s in ("train", "validation")}
    samples = []
    for r in test:
        item = texts[r["split"]][r["index"]]
        assert item["audio"] == r["audio"], f"manifest lệch HF ở {r['split']}#{r['index']}"
        samples.append({"split": r["split"], "index": r["index"], "audio": r["audio"],
                        "source": item["source"], "duration_s": r["duration"],
                        "pseudo_label": item["text"], "corrected_text": "", "notes": ""})
    CLEAN_TEST_PATH.write_text(json.dumps({
        "version": 2,
        "note": "clean-test độc lập theo video: lấy từ video giữ riêng (data/splits/heldout_videos.tsv), "
                "src/data/make_heldout.py. Thay bản 1 (250 câu từ validation, trùng video với train).",
        "seed": SEED, "per_video": PER_VIDEO, "n": len(samples), "hf_revision": HF_REVISION,
        "samples": samples,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    h = lambda rs: sum(r["duration"] for r in rs) / 3600  # noqa: E731
    held = set(heldout)
    train = [r for r in pool if r["split"] == "train" and r["video"] not in held]
    val = [r for r in pool if r["split"] == "validation" and r["video"] not in held]
    print(f"nguồn: {len(pool)} câu, {total_h:.2f} h, {len(by_video)} video ({len(eligible)} đủ điều kiện)")
    print(f"giữ riêng: {len(heldout)} video, {len(test) + len(unseen)} câu, {held_h:.2f} h "
          f"({100 * held_h / total_h:.1f}%) → {HELDOUT_PATH}")
    for name, rs in (("train", train), ("val (đã gặp)", val), ("val_unseen", unseen), ("clean-test", test)):
        print(f"  {name:14s} {len(rs):6d} câu {h(rs):7.2f} h ({100 * h(rs) / total_h:5.2f}%) "
              f"{len({r['video'] for r in rs}):4d} video")
    leak = {r["video"] for r in test + unseen} & {r["video"] for r in train + val}
    assert not leak, f"video giữ riêng lọt vào train/val: {sorted(leak)[:3]}"
    print("kiểm: không video nào của clean-test/val_unseen nằm trong train/val")


if __name__ == "__main__":
    main()
