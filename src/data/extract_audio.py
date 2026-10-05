"""Giải nén tar audio của các Kaggle Dataset vào cache local trước khi train.

Chốt 2026-09-30 (`PACK_TAR = True`, docs/notes/training_plan_kaggle.md mục 5):
5 dataset là tar do `scripts/kaggle/make_dataset/make_dataset.py` tạo, mỗi tar
chứa `audio/...` kèm `<part>.tsv` cùng thư mục. Giải nén vào `/tmp` (máy GPU
trống ~1,1 TB), **không** `/kaggle/working` (giới hạn 20 GB output); số đo tốc
độ train chỉ đúng khi đọc đĩa local.

    python -m src.data.extract_audio --input /kaggle/input --dest /tmp/audio_cache
    AUDIO_CACHE_DIR=/tmp/audio_cache torchrun --nproc_per_node 2 -m src.training.train ...

Chạy lại là bỏ qua tar đã giải nén (file đánh dấu trong `dest`).

Hai dạng đầu vào: tar (output kernel make-dataset gắn qua `kernel_sources`), hoặc
thư mục `<part>/audio/...` cạnh `<part>.tsv` — Kaggle **tự giải nén tar khi tạo
Dataset từ output** (phát hiện 2026-10-04 với `vss-asr-train-shard*`), nên gắn
qua `dataset_sources` thì không còn tar. Dạng thư mục vẫn chép sang `dest`
(không symlink) để đọc đĩa local như lúc benchmark.
"""

import argparse
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):  # sys.exit in lỗi ra stderr
    if _stream.encoding != "utf-8":
        _stream.reconfigure(encoding="utf-8")


def _manifest_rels(tsv: Path) -> list[str]:
    return [line.split("	")[1] for line in tsv.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]


def _find(root: Path, pattern: str, max_depth: int = 5) -> list[Path]:
    """Glob giới hạn độ sâu thay cho `rglob`: rglob duyệt ~67 nghìn wav trên ổ mount của Kaggle, im lặng
    vài phút — tài khoản B 2026-10-05 vượt watchdog 10 phút của kernel và bị kill. tsv/tar nằm ở độ sâu ≤ 4."""
    return sorted({p for d in range(max_depth) for p in root.glob("*/" * d + pattern)})


def copy_extracted(input_root: Path, dest: Path, only: list[str] | None = None, workers: int = 16) -> int:
    """Chép thư mục `<part>/` (đã giải nén sẵn, cạnh `<part>.tsv`) vào `dest`
    theo đúng đường dẫn tương đối trong tsv; thiếu file thì dừng. Trả về số part."""
    parts = [t for t in _find(input_root, "*.tsv") if t.with_suffix("").is_dir()]
    if only:
        parts = [t for t in parts if t.stem in only]
    for tsv in parts:
        marker = dest / f".extracted_{tsv.stem}"
        if marker.exists():
            print(f"{tsv.stem}: đã chép, bỏ qua", flush=True)
            continue
        t0, src_root = time.time(), tsv.with_suffix("")
        rels = _manifest_rels(tsv)
        missing = [r for r in rels if not (src_root / r).exists()]
        if missing:
            sys.exit(f"LỖI: {tsv.stem} thiếu {len(missing)}/{len(rels)} file trong {src_root}, vd. {missing[:3]}")

        def copy_one(rel: str) -> None:
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_root / rel, out)

        print(f"{tsv.stem}: chép {len(rels)} file ...", flush=True)
        with ThreadPoolExecutor(workers) as pool:
            # In tiến độ: một part mất vài phút, watchdog của kernel kill tiến trình im quá 10 phút.
            for i, _ in enumerate(pool.map(copy_one, rels), 1):
                if i % 2000 == 0:
                    print(f"  {tsv.stem}: {i}/{len(rels)} ({time.time() - t0:.0f}s)", flush=True)
        marker.touch()
        print(f"{tsv.stem}: chép {len(rels)} file {time.time() - t0:.0f}s (đủ theo {tsv.name})", flush=True)
    return len(parts)


def extract_tars(input_root: Path, dest: Path, only: list[str] | None = None) -> int:
    """Giải nén mọi `*.tar` dưới `input_root` (lọc theo tên part nếu có `only`);
    nếu cạnh tar có `<part>.tsv` thì kiểm đủ file. Trả về số tar đã xử lý."""
    dest.mkdir(parents=True, exist_ok=True)
    tars = _find(input_root, "*.tar")
    if only:
        tars = [t for t in tars if t.stem in only]
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
    inp, dest = Path(args.input), Path(args.dest)
    n = extract_tars(inp, dest, args.only) + copy_extracted(inp, dest, args.only)
    if n == 0:
        sys.exit(f"LỖI: không thấy tar hay thư mục <part>/ cạnh <part>.tsv dưới {inp}"
                 + (f" khớp {args.only}" if args.only else ""))


if __name__ == "__main__":
    main()
