"""Chia manifest 4 shard train + 1 val (D3, D5 đã duyệt 2026-09-28 —
docs/notes/training_plan_kaggle.md mục 5).

Shard chỉ là cách *lưu* thành Kaggle Dataset (D4: train gắn cả 4, xáo chung),
nên yêu cầu là mỗi shard đại diện đủ (độ dài, video, ký tự, token) — không
phải để train riêng từng shard.

Cách chia (D3): nhóm theo `source` (video), thứ tự video xáo bằng seed 42,
trong mỗi video sắp theo `duration`, rồi gán vòng tròn 0-1-2-3 *liên tục qua
các video* (điểm bắt đầu của video sau nối tiếp video trước) → mỗi video rải
đều ~1/4 sang mỗi shard và cỡ shard lệch nhau ≤ 1 câu.

Val (D5) = `validation` trừ 250 câu clean-test: clean-test là tập báo cáo, không
được dùng để chọn `best.pt`. Phải ghi rõ ở Chương 3 (điều kiện duyệt D5).

Manifest lưu theo index split ở revision đã pin → load_dataset phải cùng
HF_REVISION, nếu không index lệch âm thầm. Cột `audio` lưu kèm để notebook tạo
dataset và bước kiểm tra không phụ thuộc thứ tự dòng.

    python -m src.data.make_shards            # ghi data/splits/ + in thống kê
    python -m src.data.make_shards --check    # chỉ kiểm tra manifest đã ghi
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from datasets import load_dataset

from src.data.vietsuperspeech_dataset import HF_DATASET_ID, HF_REVISION

OUT_DIR = Path("data/splits")
CLEAN_TEST_PATH = Path("data/processed/clean_test_manifest.json")
TOKENIZER_PATH = "configs/tokenizer.model"
N_SHARDS = 4
SEED = 42
# Số liệu đã đo (Plan.md, dataset_discrepancy.md) — lệch là dữ liệu HF đã đổi.
EXPECTED = {"train": 60_656, "validation": 6_749, "clean_test": 250, "all_audio": 67_405}
# WAV 16 kHz, 16 bit, mono: khớp 220,84 h ↔ 25,44 GB đo qua HF API.
BYTES_PER_SECOND = 16_000 * 2


def load_split(split: str) -> list[dict]:
    ds = load_dataset(HF_DATASET_ID, split=split, revision=HF_REVISION)
    cols = ds.select_columns(["audio", "duration", "source", "text"]).to_dict()
    return [
        {"index": i, "audio": a, "duration": d, "source": s, "text": t}
        for i, (a, d, s, t) in enumerate(zip(cols["audio"], cols["duration"], cols["source"], cols["text"]))
    ]


def assign_shards(rows: list[dict], n_shards: int = N_SHARDS, seed: int = SEED) -> list[list[dict]]:
    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_video[r["source"]].append(r)
    videos = sorted(by_video)
    random.Random(seed).shuffle(videos)

    shards: list[list[dict]] = [[] for _ in range(n_shards)]
    k = 0
    for v in videos:
        # Phá hoà theo `audio` để kết quả không phụ thuộc thứ tự dòng trong HF.
        for r in sorted(by_video[v], key=lambda r: (r["duration"], r["audio"])):
            shards[k % n_shards].append(r)
            k += 1
    return [sorted(s, key=lambda r: r["index"]) for s in shards]


def split_val(val_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    manifest = json.loads(CLEAN_TEST_PATH.read_text(encoding="utf-8"))
    assert manifest["split"] == "validation", manifest["split"]
    clean = []
    for s in manifest["samples"]:
        r = val_rows[s["index"]]
        # Clean-test chỉ lưu index — đối chiếu nội dung để bắt trường hợp index lệch.
        assert (r["source"], r["duration"], r["text"]) == (s["source"], s["duration_s"], s["pseudo_label"]), \
            f"clean-test index {s['index']} không khớp dữ liệu HF ở revision {HF_REVISION}"
        clean.append(r)
    clean_idx = {r["index"] for r in clean}
    return [r for r in val_rows if r["index"] not in clean_idx], clean


def write_tsv(path: Path, rows: list[dict]) -> None:
    lines = ["index\taudio\tduration"] + [f"{r['index']}\t{r['audio']}\t{r['duration']}" for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_tsv(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    return [{"index": int(i), "audio": a, "duration": float(d)} for i, a, d in (l.split("\t") for l in lines)]


def check(train_rows, val_rows, shards, val, clean) -> None:
    """Các bất biến bước 4 của TODO.md; sai là dừng, không ghi đè manifest."""
    train_audio = [r["audio"] for r in train_rows]
    val_audio = {r["audio"] for r in val_rows}
    assert len(train_rows) == EXPECTED["train"], len(train_rows)
    assert len(val_rows) == EXPECTED["validation"], len(val_rows)
    assert len(set(train_audio)) == len(train_audio), "split train có file audio trùng"
    assert len(val_audio) == len(val_rows), "split validation có file audio trùng"

    shard_sets = [{r["audio"] for r in s} for s in shards]
    for i in range(len(shard_sets)):
        for j in range(i + 1, len(shard_sets)):
            assert not shard_sets[i] & shard_sets[j], f"shard {i} và {j} giao nhau"
    union = set().union(*shard_sets)
    assert union == set(train_audio) and sum(map(len, shards)) == EXPECTED["train"], "hợp các shard ≠ split train"
    assert not union & val_audio, "shard chứa file của validation/clean-test"
    assert max(map(len, shards)) - min(map(len, shards)) <= 1, [len(s) for s in shards]

    val_set, clean_set = {r["audio"] for r in val}, {r["audio"] for r in clean}
    assert len(clean_set) == EXPECTED["clean_test"], len(clean_set)
    assert not val_set & clean_set, "val chứa câu clean-test"
    assert val_set | clean_set == val_audio and len(val) == EXPECTED["validation"] - EXPECTED["clean_test"]
    assert len(union | val_audio) == EXPECTED["all_audio"], len(union | val_audio)
    print(f"KIỂM TRA ĐẠT: {N_SHARDS} shard rời nhau, hợp = {len(union)} file train, "
          f"không lẫn validation; val {len(val)} + clean-test {len(clean)} = {len(val_audio)}; "
          f"tổng {len(union | val_audio)} file.")


def stats(name: str, rows: list[dict], tok) -> dict:
    secs = sum(r["duration"] for r in rows)
    chars = set("".join(r["text"] for r in rows))
    tokens = set()
    for r in rows:
        tokens.update(tok.encode(r["text"]))
    buckets = defaultdict(int)
    for r in rows:
        buckets[min(int(r["duration"]), 15)] += 1
    return {
        "name": name, "n": len(rows), "hours": round(secs / 3600, 2),
        "gb_wav": round(secs * BYTES_PER_SECOND / 1e9, 2), "videos": len({r["source"] for r in rows}),
        "chars": len(chars), "bpe_tokens": len(tokens), "mean_s": round(secs / len(rows), 3),
        "mean_words": round(sum(len(r["text"].split()) for r in rows) / len(rows), 2),
        "pct_by_second": {f"{b}s": round(100 * c / len(rows), 1) for b, c in sorted(buckets.items())},
    }


def main():
    parser = argparse.ArgumentParser(description="Chia manifest 4 shard train + val (D3, D5)")
    parser.add_argument("--check", action="store_true", help="Chỉ kiểm tra manifest đã ghi trong data/splits/")
    args = parser.parse_args()

    train_rows, val_rows = load_split("train"), load_split("validation")
    val, clean = split_val(val_rows)

    if args.check:
        # Đọc lại từ đĩa và nối với dữ liệu HF theo index → bắt cả sửa tay lẫn index lệch.
        shards = []
        for k in range(N_SHARDS):
            rows = read_tsv(OUT_DIR / f"train_shard{k}.tsv")
            assert all(train_rows[r["index"]]["audio"] == r["audio"] for r in rows), f"shard {k} lệch index"
            shards.append([train_rows[r["index"]] for r in rows])
        disk_val = read_tsv(OUT_DIR / "val.tsv")
        assert [r["audio"] for r in disk_val] == [r["audio"] for r in val], "val.tsv khác val tính lại"
        assert shards == assign_shards(train_rows), "shard trên đĩa khác kết quả chia lại (seed/thuật toán đổi?)"
        check(train_rows, val_rows, shards, val, clean)
        return

    shards = assign_shards(train_rows)
    check(train_rows, val_rows, shards, val, clean)

    from src.tokenizer.bpe_tokenizer import BPETokenizer
    tok = BPETokenizer(TOKENIZER_PATH)
    table = [stats("train", train_rows, tok)] + [stats(f"shard{k}", s, tok) for k, s in enumerate(shards)]
    table += [stats("val", val, tok), stats("clean_test", clean, tok)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for k, s in enumerate(shards):
        write_tsv(OUT_DIR / f"train_shard{k}.tsv", s)
    write_tsv(OUT_DIR / "val.tsv", val)
    (OUT_DIR / "summary.json").write_text(json.dumps({
        "hf_dataset": HF_DATASET_ID, "hf_revision": HF_REVISION, "n_shards": N_SHARDS, "seed": SEED,
        "method": "vòng tròn liên tục qua video (thứ tự video xáo seed), trong video sắp theo duration",
        "val": "validation trừ clean-test (data/processed/clean_test_manifest.json)",
        "stats": table,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cols = ["name", "n", "hours", "gb_wav", "videos", "chars", "bpe_tokens", "mean_s", "mean_words"]
    print("\n" + " | ".join(cols) + " | %theo giây")
    for row in table:
        print(" | ".join(str(row[c]) for c in cols) + " | " + " ".join(f"{v}" for v in row["pct_by_second"].values()))
    print(f"\nĐã ghi {OUT_DIR}/train_shard0-{N_SHARDS - 1}.tsv, val.tsv, summary.json")


if __name__ == "__main__":
    main()
