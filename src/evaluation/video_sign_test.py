"""Kiểm định dấu theo video cho các so cặp encoder trên clean-test (2026-10-17).

Bổ sung **mô tả** cho bootstrap cặp của bảng chính (`review_fable_2026-10-15.md` mối đe doạ #5, người dùng duyệt
17/10): đếm số video mà A có WER thấp hơn B, kèm p hai phía của phép thử dấu chính xác (bỏ video hoà). Tiêu chí kết
luận "A tốt hơn B" **không đổi** (note gap 9.4/9.6: CI bootstrap); số ở đây chỉ để trình bày mức nhất quán theo video.

WER mỗi video = tổng lỗi / tổng từ ref của video đó, đọc từ `per_sentence.tsv` của lần chấm chung (cùng chuẩn hoá
bảng chính, không chấm lại). 40 ms: lỗi cộng qua 3 seed rồi chia 3 × số từ (WER trung bình seed của video).

    python -m src.evaluation.video_sign_test
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

SCORED = Path("reports/results/clean_test_scored_2026-10-16_chung/per_sentence.tsv")
OUT = Path("reports/results/video_sign_2026-10-17")
SEED_SUFFIXES = ("", "_seed43", "_seed44")

PAIRS_10 = [("mamba", "conformer_relpos"), ("conextbimamba", "conformer_relpos"), ("mamba", "conextbimamba"),
            ("conformer_relpos", "conformer"), ("mamba", "conformer"), ("conextbimamba", "conformer")]
PAIRS_40 = [("mamba", "conextbimamba"), ("mamba", "conformer_relpos"), ("conextbimamba", "conformer_relpos")]


def sign_test_p(wins: int, losses: int) -> float:
    """p hai phía của phép thử dấu chính xác (nhị thức, p = 0,5), video hoà đã bỏ."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_per_video():
    """{system: {video: [lỗi, từ ref]}}."""
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    with SCORED.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            c = acc[r["system"]][r["video"]]
            c[0] += int(r["err_vi"]) + int(r["err_fx"])
            c[1] += int(r["words_vi"]) + int(r["words_fx"])
    return acc


def combine(acc, systems):
    """Cộng lỗi và từ ref qua nhiều hệ thống (nhiều seed) → {video: WER}."""
    out = {}
    for v in acc[systems[0]]:
        e = sum(acc[s][v][0] for s in systems)
        w = sum(acc[s][v][1] for s in systems)
        out[v] = e / w
    return out


def compare(label, a_name, b_name, wa, wb):
    videos = sorted(wa)
    assert videos == sorted(wb)
    diffs = [wa[v] - wb[v] for v in videos]
    wins, losses = sum(d < 0 for d in diffs), sum(d > 0 for d in diffs)
    return {"config": label, "a": a_name, "b": b_name, "n_videos": len(videos), "a_better": wins,
            "b_better": losses, "ties": len(videos) - wins - losses, "p_sign": round(sign_test_p(wins, losses), 6)}, \
        [{"config": label, "a": a_name, "b": b_name, "video": v, "wer_a": round(wa[v], 6), "wer_b": round(wb[v], 6)}
         for v in videos]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    acc = load_per_video()
    summary, per_video = [], []
    for a, b in PAIRS_10:
        s, pv = compare("10 ms", a, b, combine(acc, [f"v2a10_{a}"]), combine(acc, [f"v2a10_{b}"]))
        summary.append(s)
        per_video += pv
    for mode in ("greedy", "lm"):
        for a, b in PAIRS_40:
            sa = [f"v2a_{a}{sfx}_{mode}" for sfx in SEED_SUFFIXES]
            sb = [f"v2a_{b}{sfx}_{mode}" for sfx in SEED_SUFFIXES]
            s, pv = compare(f"40 ms 3 seed {mode}", a, b, combine(acc, sa), combine(acc, sb))
            summary.append(s)
            per_video += pv

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("video_sign.csv", summary), ("per_video_wer.csv", per_video)):
        with (OUT / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for s in summary:
        print(f"{s['config']:<20} {s['a']:>16} vs {s['b']:<16} A thấp hơn {s['a_better']:>2} · B thấp hơn "
              f"{s['b_better']:>2} · hoà {s['ties']} / {s['n_videos']}  p = {s['p_sign']:.4f}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
