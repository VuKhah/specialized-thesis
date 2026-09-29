"""Kernel Kaggle **CPU** (không tốn quota GPU) — bước 1 của D9 trong
docs/notes/training_plan_kaggle.md mục 5: đo môi trường trước khi viết
notebook tạo 5 dataset (4 shard train + 1 val). Mọi con số trước đó chỉ là
ước lượng.

    kaggle kernels push -p scripts/kaggle/check_env
    kaggle kernels status <username>/check-env-asr
    kaggle kernels output <username>/check-env-asr -p <thư mục>   (PYTHONUTF8=1)

Đo 4 thứ:
1. Đĩa/RAM/CPU ở các vị trí có thể ghi tạm khi tạo shard (~6,4 GB/shard WAV,
   ~12,7 GB nếu phải giữ cả WAV lẫn tar).
2. Tốc độ tải audio từ HF (ẩn danh / có token) → ngoại suy thời gian tải
   67.405 file. HF giới hạn lượt resolve theo 5 phút nên đếm cả lỗi 429.
3. Output kernel có giữ đủ > 500 file không — sau khi kernel xong, đếm số file
   trong `filecount_test/` bằng `kaggle kernels output` ở máy local.
4. Tốc độ đóng gói tar các WAV đã tải.

Token HF lấy từ Kaggle Secrets tên `HF_TOKEN` — phải gắn secret vào kernel
trên giao diện web một lần; thiếu thì chỉ đo phần ẩn danh.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HF_DATASET_ID = "thanhnew2001/VietSuperSpeech"
# Pin theo D10; repo HF đứng yên từ 2026-02-22.
HF_REVISION = "cbf624ae9b30e1c2793a27e95b262115c69601f3"
N_TOTAL_FILES = 67_405
N_PER_TRIAL = 150
N_OUTPUT_FILES = 600
WORK = Path("/kaggle/working")
TMP = Path("/tmp/check_env")
result: dict = {}


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def check_disk():
    section("1. Đĩa / RAM / CPU")
    subprocess.run(["df", "-h"])
    subprocess.run(["free", "-h"])
    disks = {}
    for p in ["/kaggle/working", "/kaggle/input", "/kaggle/temp", "/tmp", "/dev/shm", "/"]:
        if Path(p).exists():
            u = shutil.disk_usage(p)
            disks[p] = {"total_gb": round(u.total / 1e9, 1), "free_gb": round(u.free / 1e9, 1)}
            print(f"{p:18s} total {disks[p]['total_gb']:7.1f} GB   free {disks[p]['free_gb']:7.1f} GB")
    result["disk"] = disks
    result["cpu_count"] = os.cpu_count()
    print(f"CPU: {os.cpu_count()} lõi")


def audio_paths(n):
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_ID, split="train", streaming=True, revision=HF_REVISION)
    paths = []
    for item in ds:
        paths.append(item["audio"])
        if len(paths) >= n:
            break
    return paths


def download_trial(name, paths, token, workers):
    from huggingface_hub import hf_hub_download

    out = TMP / name
    errors = []

    def one(p):
        try:
            hf_hub_download(HF_DATASET_ID, p, repo_type="dataset", revision=HF_REVISION,
                            local_dir=out, token=token)
        except Exception as e:  # noqa: BLE001 - chỉ đếm lỗi, không dừng phép đo
            errors.append(repr(e)[:200])

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, paths))
    dt = time.time() - t0
    nbytes = sum(f.stat().st_size for f in out.rglob("*.wav"))
    ok = len(paths) - len(errors)
    per_file = dt / max(ok, 1)
    r = {
        "files": len(paths), "ok": ok, "errors": len(errors),
        "n_429": sum("429" in e for e in errors), "workers": workers,
        "seconds": round(dt, 1), "mb_per_s": round(nbytes / 1e6 / dt, 1),
        "extrapolated_full_hours": round(per_file * N_TOTAL_FILES / 3600, 2),
        "error_samples": errors[:3],
    }
    print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)
    return r, out


def check_hf_speed():
    section("2. Tốc độ tải audio từ HF")
    paths = audio_paths(2 * N_PER_TRIAL)
    result["hf"] = {}
    r, anon_dir = download_trial("anon", paths[:N_PER_TRIAL], token=False, workers=8)
    result["hf"]["anonymous_8w"] = r

    token = None
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:  # noqa: BLE001
        print(f"Không lấy được secret HF_TOKEN ({type(e).__name__}) — bỏ qua phép đo có token.")
    if token:
        r, _ = download_trial("token", paths[N_PER_TRIAL:], token=token, workers=16)
        result["hf"]["token_16w"] = r
    else:
        result["hf"]["token_16w"] = "không có secret HF_TOKEN"
    return anon_dir


def check_tar(src_dir):
    section("4. Tốc độ đóng gói tar (không nén)")
    wavs = sorted(src_dir.rglob("*.wav"))
    tar_path = TMP / "test.tar"
    t0 = time.time()
    with tarfile.open(tar_path, "w") as tar:
        for w in wavs:
            tar.add(w, arcname=str(w.relative_to(src_dir)))
    dt = time.time() - t0
    size = tar_path.stat().st_size
    result["tar"] = {"files": len(wavs), "mb": round(size / 1e6, 1), "mb_per_s": round(size / 1e6 / dt, 1)}
    print(result["tar"])


def make_output_files():
    section("3. Tạo file để kiểm tra giới hạn số file trong output")
    d = WORK / "filecount_test"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(N_OUTPUT_FILES):
        (d / f"f{i:04d}.txt").write_text(str(i))
    result["output_files_written"] = N_OUTPUT_FILES
    print(f"Đã ghi {N_OUTPUT_FILES} file vào {d} — đếm lại ở local sau khi kernel xong.")


if __name__ == "__main__":
    TMP.mkdir(parents=True, exist_ok=True)
    check_disk()
    anon_dir = check_hf_speed()
    make_output_files()
    check_tar(anon_dir)
    (WORK / "check_env_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    section("KẾT QUẢ")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nCHECK ENV XONG", flush=True)
