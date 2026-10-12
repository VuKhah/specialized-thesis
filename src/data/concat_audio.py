"""Audio ghép nhân tạo cho RQ2 (chốt 2026-10-03, `docs/notes/dataset_discrepancy.md`): nối các đoạn liên tiếp
cùng video thành một waveform dài, **chỉ để đo** chi phí suy luận theo độ dài (không train, không đo WER).

Lấy từ train + val qua `manifest_rows` của 4 shard + val → đã bỏ `excluded.tsv` (A1) và video giữ riêng (để
clean-test/val_unseen không lọt vào bất kỳ phép đo nào). Video dài nhất trước, các đoạn theo số `_segN`; thiếu
độ dài thì nối tiếp video kế. Các đoạn liền số seg không liền thời gian (lặng bị cắt, seg khuyết) — hợp lệ vì chi
phí chỉ phụ thuộc số khung; nguồn ghép ghi vào `info` để nêu trong khóa luận.

    python -m src.data.concat_audio --seconds 3600 --out /tmp/long.wav --info /tmp/long.json
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from src.data.vietsuperspeech_dataset import AUDIO_CACHE_DIR, SHARD_MANIFESTS, manifest_rows, video_of

SAMPLE_RATE = 16000


def find_audio_roots(input_dir: str = "/kaggle/input") -> list[Path]:
    """Thư mục gốc chứa `audio/...`: AUDIO_CACHE_DIR (local / extract_audio) + các dataset `vss-asr-*` Kaggle đã
    giải nén sẵn (cạnh file .tsv, như pilot RQ2) — đọc thẳng, khỏi giải nén ~5 phút GPU chạy không."""
    roots = [Path(AUDIO_CACHE_DIR)]
    for depth in range(1, 4):
        for tsv in glob.glob(input_dir + "/" + "*/" * depth + "*.tsv"):
            p = Path(tsv).with_suffix("")
            if (p / "audio").is_dir():
                roots.append(p)
            if (Path(tsv).parent / "audio").is_dir():
                roots.append(Path(tsv).parent)
    return [r for r in dict.fromkeys(roots) if r.is_dir()]


def build_concat_audio(max_seconds: float, roots: list[Path], manifests: list[str] | None = None,
                       allow_missing: bool = False) -> tuple[np.ndarray, dict]:
    """Trả (waveform float32 [N] ở 16 kHz, dài ≥ max_seconds nếu đủ dữ liệu; info nguồn ghép).
    allow_missing: bỏ qua đoạn không có file (chỉ dùng local, cache audio không đủ)."""
    rows = manifest_rows(manifests or list(SHARD_MANIFESTS))
    by_video = defaultdict(list)
    for r in rows:
        m = re.search(r"_seg(\d+)\.wav$", r["audio"])
        by_video[video_of(r["audio"])].append((int(m.group(1)), r))

    def wav_path(rel: str) -> Path | None:
        for root in roots:
            if (root / rel).exists():
                return root / rel
        if allow_missing:
            return None
        raise FileNotFoundError(f"không thấy {rel} trong {', '.join(map(str, roots))}")

    chunks, total, used = [], 0.0, []
    for video, segs in sorted(by_video.items(), key=lambda kv: -sum(r["duration"] for _, r in kv[1])):
        seg_ids = []
        for seg, r in sorted(segs, key=lambda x: x[0]):
            path = wav_path(r["audio"])
            if path is None:
                continue
            x, sr = sf.read(path, dtype="float32")
            if sr != SAMPLE_RATE or x.ndim != 1:
                raise ValueError(f"{path}: cần mono {SAMPLE_RATE} Hz, nhận sr={sr}, shape={x.shape}")
            chunks.append(x)
            seg_ids.append(seg)
            total += len(x) / SAMPLE_RATE
            if total >= max_seconds:
                break
        if seg_ids:
            used.append({"video": video, "segments": len(seg_ids), "seg_first": seg_ids[0], "seg_last": seg_ids[-1]})
        if total >= max_seconds:
            break
    if not chunks:
        raise RuntimeError("không ghép được đoạn nào — kiểm tra roots")
    wav = np.concatenate(chunks)
    info = {"total_s": len(wav) / SAMPLE_RATE, "requested_s": max_seconds, "enough": total >= max_seconds,
            "n_videos": len(used), "n_segments": sum(u["segments"] for u in used), "videos": used,
            "manifests": manifests or list(SHARD_MANIFESTS), "allow_missing": allow_missing}
    return wav, info


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--out", required=True, help="file .wav đầu ra")
    ap.add_argument("--info", help="file .json nguồn ghép")
    ap.add_argument("--input_dir", default=os.environ.get("RQ2_INPUT", "/kaggle/input"))
    ap.add_argument("--allow_missing", action="store_true")
    args = ap.parse_args()
    wav, info = build_concat_audio(args.seconds, find_audio_roots(args.input_dir), allow_missing=args.allow_missing)
    sf.write(args.out, wav, SAMPLE_RATE)
    if args.info:
        Path(args.info).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ghép {info['total_s']:.1f} s từ {info['n_videos']} video, {info['n_segments']} đoạn → {args.out}")


if __name__ == "__main__":
    main()
