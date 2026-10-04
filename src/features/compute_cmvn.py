"""Tính mean/std toàn cục từng kênh log-mel cho CMVN → `configs/cmvn_stats.json`.

    python -m src.features.compute_cmvn            # 2000 câu, seed 42

Lấy mẫu ngẫu nhiên từ train **đã bỏ excluded.tsv và video giữ riêng** (4 shard),
không dùng validation. Không cần cả 48.340 câu: 2000 câu ≈ 2,6 triệu khung, sai số thống
kê nhỏ hơn nhiều so với chênh giữa các kênh. Chọn ngẫu nhiên thay vì lấy file
sẵn có trong cache local (cache tải theo part → lệch theo video). File thiếu thì
tải song song từ HF ở revision đã pin.

Tính trên đúng `LogMelFeatureExtractor` (tắt CMVN), chỉ khung thật, cộng dồn
float64. Đổi tham số front-end thì phải tính lại.
"""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

from src.data.vietsuperspeech_dataset import AUDIO_CACHE_DIR, HF_DATASET_ID, HF_REVISION, manifest_rows
from src.features.log_mel import CMVN_STATS_PATH, LogMelFeatureExtractor

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

TRAIN_MANIFESTS = [f"train_shard{i}" for i in range(4)]


def sample_audio_paths(n: int, seed: int) -> list[str]:
    # manifest_rows: đã bỏ excluded.tsv và video giữ riêng → đúng tập train.
    rows = sorted(r["audio"] for r in manifest_rows(TRAIN_MANIFESTS))  # không phụ thuộc thứ tự shard
    return random.Random(seed).sample(rows, n)


def ensure_local(rels: list[str]) -> None:
    missing = [r for r in rels if not (Path(AUDIO_CACHE_DIR) / r).exists()]
    print(f"{len(rels) - len(missing)}/{len(rels)} file có sẵn trong cache, tải {len(missing)}", flush=True)

    def one(rel):
        hf_hub_download(HF_DATASET_ID, rel, repo_type="dataset", revision=HF_REVISION, local_dir=AUDIO_CACHE_DIR)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(one, missing), 1):
            if i % 200 == 0:
                print(f"  tải {i}/{len(missing)}", flush=True)


@torch.no_grad()
def compute(rels: list[str]) -> dict:
    fe = LogMelFeatureExtractor(cmvn_stats=None).eval()
    total = torch.zeros(80, dtype=torch.float64)
    total_sq = torch.zeros(80, dtype=torch.float64)
    n_frames = 0
    for rel in rels:
        array, sr = sf.read(Path(AUDIO_CACHE_DIR) / rel)
        assert sr == 16_000, f"{rel}: {sr} Hz"
        wav = torch.from_numpy(array).float()[None]
        feats, lengths = fe(wav, torch.tensor([wav.size(1)]))
        f = feats[0, : lengths[0]].double()
        total += f.sum(0)
        total_sq += (f * f).sum(0)
        n_frames += f.size(0)
    mean = total / n_frames
    std = (total_sq / n_frames - mean * mean).clamp(min=1e-10).sqrt()
    return {"mean": mean.tolist(), "std": std.tolist(), "n_frames": n_frames}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=CMVN_STATS_PATH)
    args = parser.parse_args()

    rels = sample_audio_paths(args.n, args.seed)
    ensure_local(rels)
    stats = compute(rels)
    result = {
        "note": "CMVN toàn cục cho LogMelFeatureExtractor (80 mel, 25/10 ms, log clamp 1e-5); "
                "src/features/compute_cmvn.py",
        "source": f"{args.n} câu ngẫu nhiên (seed {args.seed}) từ train_shard0-3, đã bỏ excluded.tsv + video giữ riêng",
        "hf_revision": HF_REVISION,
        "n_utterances": args.n,
        **stats,
    }
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    m, s = torch.tensor(stats["mean"]), torch.tensor(stats["std"])
    print(f"{stats['n_frames']:,} khung | mean {m.min():.2f}..{m.max():.2f} | std {s.min():.2f}..{s.max():.2f}")
    print(f"Đã ghi {args.out}")


if __name__ == "__main__":
    main()
