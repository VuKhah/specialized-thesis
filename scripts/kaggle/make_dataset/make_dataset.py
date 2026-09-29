"""Kernel Kaggle **CPU** tạo 1 trong 5 dataset (4 shard train + 1 val) — bước 5
TODO.md, D6 trong docs/notes/training_plan_kaggle.md mục 5.

Mỗi lần chạy 1 phần (sửa PART rồi push): output 20 GB không chứa nổi cả 5
(~28 GB). Tải từ HF ẩn danh ~36 phút/shard (đo ở check-env-asr).

    kaggle kernels push -p scripts/kaggle/make_dataset
    kaggle kernels status <username>/make-dataset-asr

Manifest lấy từ repo GitHub (data/splits/, tạo bởi src/data/make_shards.py) →
**phải push manifest lên GitHub trước**. Trước khi tải, chạy lại
`make_shards --check` trong kernel để chắc manifest khớp HF ở revision đã pin.

Kiểm tra sau tải: đủ số file, mỗi file đọc được bằng soundfile, 16 kHz mono,
thời lượng lệch cột `duration` ≤ 0,1 s — dataset hỏng thì chỉ lộ ra giữa lúc
train (tốn GPU), nên bắt ở đây (CPU).
"""

import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PART = "train_shard0"  # train_shard0..3 | val
# Chốt tar 2026-09-30 (người dùng duyệt): số đo tốc độ train chỉ đúng khi đọc
# từ đĩa local (/tmp máy GPU trống 1,1 TB, chỉ 4 lõi CPU để bù I/O mạng), và
# tránh giới hạn số file khi tạo Dataset. False = WAV rời, giữ để so nếu cần.
PACK_TAR = True

REPO_URL = "https://github.com/VuKhah/specialized-thesis"
REPO_DIR = Path("/tmp/specialized-thesis")
HF_DATASET_ID = "thanhnew2001/VietSuperSpeech"
HF_REVISION = "cbf624ae9b30e1c2793a27e95b262115c69601f3"
WORK = Path("/kaggle/working")
STAGE = Path("/tmp/stage") if PACK_TAR else WORK / PART


def run(cmd, cwd=None):
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    if r.returncode != 0:
        sys.exit(f"LỖI: lệnh trên trả về {r.returncode}")


def read_manifest():
    lines = (REPO_DIR / "data/splits" / f"{PART}.tsv").read_text(encoding="utf-8").splitlines()[1:]
    return [(a, float(d)) for _, a, d in (l.split("\t") for l in lines)]


def download(rows):
    from huggingface_hub import hf_hub_download

    errors = []

    def one(rel):
        for attempt in range(5):
            try:
                hf_hub_download(HF_DATASET_ID, rel, repo_type="dataset", revision=HF_REVISION, local_dir=STAGE)
                return
            except Exception as e:  # noqa: BLE001 - lỗi mạng/429 thì thử lại, hết lượt mới ghi lỗi
                if attempt == 4:
                    errors.append(f"{rel}: {e!r}"[:300])
                time.sleep(2 ** attempt)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(one, [a for a, _ in rows]), 1):
            if i % 2000 == 0:
                print(f"  {i}/{len(rows)} file, {time.time() - t0:.0f}s, lỗi {len(errors)}", flush=True)
    print(f"Tải xong {len(rows)} file sau {time.time() - t0:.0f}s, lỗi {len(errors)}", flush=True)
    if errors:
        print("\n".join(errors[:20]))
        sys.exit("LỖI: còn file tải hỏng — chạy lại kernel")


def verify(rows):
    import soundfile as sf

    bad = []
    for rel, dur in rows:
        try:
            info = sf.info(STAGE / rel)
            if info.samplerate != 16_000 or info.channels != 1 or abs(info.duration - dur) > 0.1:
                bad.append(f"{rel}: sr={info.samplerate} ch={info.channels} dur={info.duration:.2f} vs {dur}")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{rel}: {e!r}")
    n_wav = sum(1 for _ in STAGE.rglob("*.wav"))
    print(f"Kiểm tra: {n_wav} WAV trên đĩa / {len(rows)} trong manifest, {len(bad)} file lỗi", flush=True)
    if bad or n_wav != len(rows):
        print("\n".join(bad[:20]))
        sys.exit("LỖI: dataset không khớp manifest")


def main():
    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    run(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)
    run([sys.executable, "-m", "pip", "install", "-q", "datasets", "soundfile", "sentencepiece"])
    run([sys.executable, "-m", "src.data.make_shards", "--check"], cwd=REPO_DIR)

    rows = read_manifest()
    print(f"{PART}: {len(rows)} file, {sum(d for _, d in rows) / 3600:.2f} h", flush=True)
    download(rows)
    verify(rows)
    # hf_hub_download(local_dir=...) để lại 1 file .metadata cho mỗi WAV → gấp
    # đôi số file trong output nếu không xoá.
    shutil.rmtree(STAGE / ".cache", ignore_errors=True)

    # Kèm manifest để dataset tự mô tả được (và train kiểm lại khi gắn).
    (WORK / f"{PART}.tsv").write_text((REPO_DIR / "data/splits" / f"{PART}.tsv").read_text(encoding="utf-8"),
                                      encoding="utf-8")
    if PACK_TAR:
        t0 = time.time()
        with tarfile.open(WORK / f"{PART}.tar", "w") as tar:
            tar.add(STAGE / "audio", arcname="audio")
        print(f"Đóng tar {time.time() - t0:.0f}s", flush=True)
    run(["du", "-sh", str(WORK)])
    print(f"\nMAKE DATASET {PART} XONG", flush=True)


if __name__ == "__main__":
    main()
