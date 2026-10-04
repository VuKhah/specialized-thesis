"""Giải nén tar audio của các Kaggle Dataset vào cache local trước khi train.

Chốt 2026-09-30 (`PACK_TAR = True`, docs/notes/training_plan_kaggle.md mục 5):
5 dataset là tar do `scripts/kaggle/make_dataset/make_dataset.py` tạo, mỗi tar
chứa `audio/...` kèm `<part>.tsv` cùng thư mục. Giải nén vào `/tmp` (máy GPU
trống ~1,1 TB), **không** `/kaggle/working` (giới hạn 20 GB output); số đo tốc
độ train chỉ đúng khi đọc đĩa local.

    python -m src.data.extract_audio --input /kaggle/input --dest /tmp/audio_cache
    AUDIO_CACHE_DIR=/tmp/audio_cache torchrun --nproc_per_node 2 -m src.training.train ...

Chạy lại là bỏ qua tar đã giải nén (file đánh dấu trong `dest`).
"""

import argparse
import sys
import tarfile
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):  # sys.exit in lỗi ra stderr
    if _stream.encoding != "utf-8":
        _stream.reconfigure(encoding="utf-8")


def extract_tars(input_root: Path, dest: Path, only: list[str] | None = None) -> int:
    """Giải nén mọi `*.tar` dưới `input_root` (lọc theo tên part nếu có `only`);
    nếu cạnh tar có `<part>.tsv` thì kiểm đủ file. Trả về số tar đã xử lý."""
    dest.mkdir(parents=True, exist_ok=True)
    tars = sorted(input_root.rglob("*.tar"))
    if only:
        tars = [t for t in tars if t.stem in only]
    if not tars:
        sys.exit(f"LỖI: không thấy tar nào dưới {input_root}" + (f" khớp {only}" if only else ""))
    for tar_path in tars:
        marker = dest / f".extracted_{tar_path.stem}"
        if marker.exists():
            print(f"{tar_path.stem}: đã giải nén, bỏ qua", flush=True)
            continue
        t0 = time.time()
        with tarfile.open(tar_path) as tar:
            tar.extractall(dest, filter="data")
        manifest = tar_path.with_suffix(".tsv")
        if manifest.exists():
            rels = [line.split("\t")[1] for line in manifest.read_text(encoding="utf-8").splitlines()[1:]]
            missing = [r for r in rels if not (dest / r).exists()]
            if missing:
                sys.exit(f"LỖI: {tar_path.stem} thiếu {len(missing)}/{len(rels)} file, vd. {missing[:3]}")
            checked = f", đủ {len(rels)} file theo {manifest.name}"
        else:
            checked = ", không có manifest để kiểm"
        marker.touch()
        print(f"{tar_path.stem}: giải nén {time.time() - t0:.0f}s{checked}", flush=True)
    return len(tars)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--dest", default="/tmp/audio_cache")
    parser.add_argument("--only", nargs="*", help="Chỉ giải nén các part này, vd. train_shard0 val")
    args = parser.parse_args()
    extract_tars(Path(args.input), Path(args.dest), args.only)


if __name__ == "__main__":
    main()
