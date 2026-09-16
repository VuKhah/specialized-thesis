"""Tải trước toàn bộ audio VietSuperSpeech (Tuần 4-5, theo TODO trong
vietsuperspeech_dataset.py).

Tải từng file audio qua HTTP riêng lẻ trong __getitem__ mất ~1-4 giây/file —
với 60k+ mẫu train, epoch đầu sẽ mất hàng chục giờ nếu không có bước tải
trước. Script này tải song song nhiều luồng, trước khi chạy training loop.

QUAN TRỌNG: repo HF `audio/` có 118.259 file nhưng train+validation chỉ
tham chiếu 67.405 file (kiểm tra 2026-09-16 qua HfApi().list_repo_files) —
snapshot_download("audio/**") sẽ tải thừa ~43% dung lượng không cần thiết.
Nên script này KHÔNG dùng snapshot_download; thay vào đó duyệt qua text/
duration/source của cả 2 split (streaming, không decode audio) để lấy đúng
tập đường dẫn cần, rồi tải song song bằng hf_hub_download — vừa đủ, vừa
resume được (bỏ qua file đã tồn tại).

Chạy: python -m src.data.prefetch_audio
      python -m src.data.prefetch_audio --verify   (chỉ kiểm tra đủ file, không tải)

Bền vững qua nhiều session Kaggle (session cap ~9-12h): danh sách đường dẫn
cần tải được cache ra MANIFEST_PATH sau lần chạy đầu (khỏi phải stream lại
cả 2 split mỗi lần resume) và mỗi file tải lỗi không làm dừng cả batch — lỗi
được gom lại, in ra cuối, chạy lại script sẽ tự retry đúng các file còn
thiếu (không tải lại file đã có).
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from src.data.vietsuperspeech_dataset import AUDIO_CACHE_DIR, HF_DATASET_ID

MANIFEST_PATH = "data/processed/audio_manifest.json"


def needed_audio_paths(manifest_path: str = MANIFEST_PATH, refresh: bool = False) -> list[str]:
    """Trả về danh sách đường dẫn audio mà train+validation tham chiếu.

    Cache ra manifest_path sau lần tính đầu tiên — tránh phải stream lại
    (không decode audio, nhưng vẫn mất vài phút cho 67k+ mẫu) mỗi lần script
    được chạy lại để resume qua nhiều session Kaggle."""
    cache_file = Path(manifest_path)
    if cache_file.exists() and not refresh:
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)["paths"]

    paths: set[str] = set()
    for split in ("train", "validation"):
        ds = load_dataset(HF_DATASET_ID, split=split, streaming=True)
        for item in ds:
            paths.add(item["audio"])
    result = sorted(paths)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"n": len(result), "paths": result}, f)

    return result


def prefetch_audio(local_dir: str = AUDIO_CACHE_DIR, max_workers: int = 8, manifest_path: str = MANIFEST_PATH) -> Path:
    """Tải song song đúng các file audio được train+validation tham chiếu về
    local_dir. Bỏ qua file đã tồn tại nên chạy lại an toàn nếu bị ngắt giữa
    chừng (Kaggle session cap ~9-12h). Lỗi tải từng file (network timeout,
    v.v.) không làm dừng cả batch — được gom lại và in ra cuối để retry bằng
    cách chạy lại script (không tải lại file đã có).

    File tải về nằm ở local_dir/audio/... — cùng đường dẫn tương đối với cột
    "audio" trong dataset, nên VietSuperSpeechDataset đọc thẳng bằng
    Path(local_dir) / item["audio"] mà không cần gọi lại hf_hub_download.
    """
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    paths = needed_audio_paths(manifest_path)
    print(f"Cần {len(paths)} file audio (tham chiếu bởi train+validation).")

    def _download_one(rel_path: str) -> tuple[str, Exception | None]:
        if (Path(local_dir) / rel_path).exists():
            return rel_path, None
        try:
            hf_hub_download(HF_DATASET_ID, rel_path, repo_type="dataset", local_dir=local_dir)
            return rel_path, None
        except Exception as e:  # noqa: BLE001 - muốn gom mọi lỗi tải để không sập cả batch
            return rel_path, e

    done = 0
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_download_one, p) for p in paths]
        for future in as_completed(futures):
            rel_path, err = future.result()
            if err is not None:
                failures.append((rel_path, repr(err)))
            done += 1
            if done % 2000 == 0:
                print(f"  ... đã xử lý {done}/{len(paths)} (lỗi: {len(failures)})")

    if failures:
        print(f"\nCó {len(failures)} file tải lỗi (chạy lại script để retry):")
        for rel_path, err in failures[:20]:
            print(f"  - {rel_path}: {err}")
        if len(failures) > 20:
            print(f"  ... và {len(failures) - 20} file khác.")

    return Path(local_dir)


def verify_complete(local_dir: str = AUDIO_CACHE_DIR, manifest_path: str = MANIFEST_PATH) -> int:
    """Đếm số file còn thiếu so với local_dir sau khi prefetch. Trả về số
    file thiếu."""
    paths = needed_audio_paths(manifest_path)
    missing = [p for p in paths if not (Path(local_dir) / p).exists()]
    print(f"Kiểm tra xong {len(paths)} mẫu — thiếu {len(missing)} file.")
    return len(missing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải trước audio VietSuperSpeech (chỉ đúng file cần dùng)")
    parser.add_argument("--local-dir", default=AUDIO_CACHE_DIR)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--verify", action="store_true", help="Chỉ kiểm tra đủ file, không tải lại")
    args = parser.parse_args()

    if not args.verify:
        print(f"Tải audio VietSuperSpeech về {args.local_dir} ({args.max_workers} luồng song song)...")
        t0 = time.time()
        prefetch_audio(args.local_dir, args.max_workers)
        print(f"Xong tải sau {time.time() - t0:.1f}s")

    print("\nKiểm tra đầy đủ so với cả 2 split...")
    verify_complete(args.local_dir)
