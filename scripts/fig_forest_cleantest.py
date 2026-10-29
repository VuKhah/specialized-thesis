"""Hình 3.1 khóa luận: forest plot hiệu WER giữa các cặp encoder trên clean-test (giải mã tham lam), hai khối 10 ms và
40 ms — đúng các cặp của Bảng 3.3. Đọc `results.json` của `src.evaluation.score_clean_test`, không tính lại gì.

    python scripts/fig_forest_cleantest.py [--res <results.json>] [--out <png>]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RES = Path("reports/results/clean_test_scored_2026-10-16_chung/results.json")
OUT = Path("reports/figures/clean_test/fig_forest_cap_encoder.png")
NAME = {"mamba": "B1", "conextbimamba": "ConExtBiMamba", "conformer_relpos": "Conformer PE", "conformer": "Conformer không PE"}
# 10 ms: một seed, mục `paired` (bootstrap theo video); 40 ms: ba seed, mục `paired_seeds` (hai tầng video × seed).
PAIRS_10 = [("mamba", "conformer_relpos"), ("mamba", "conextbimamba"), ("conextbimamba", "conformer_relpos"),
            ("conformer_relpos", "conformer")]
PAIRS_40 = [("mamba", "conextbimamba"), ("mamba", "conformer_relpos"), ("conextbimamba", "conformer_relpos")]


def vn(x: float, nd: int = 2) -> str:
    return f"{x:+.{nd}f}".replace(".", ",").replace("-", "−")


def rows(res: dict) -> list[tuple[str, str, float, float, float]]:
    pd = {(p["a"], p["b"]): p for p in res["paired"]}
    ps = {(p["a"], p["b"]): p for p in res["paired_seeds"] if p["mode"] == "greedy"}
    out = []
    for a, b in PAIRS_10:
        p = pd[(f"v2a10_{a}", f"v2a10_{b}")]
        out.append(("10 ms", f"{NAME[a]} − {NAME[b]}", p["diff"] * 100, p["ci"][0] * 100, p["ci"][1] * 100))
    for a, b in PAIRS_40:
        p = ps[(a, b)]
        out.append(("40 ms", f"{NAME[a]} − {NAME[b]}", p["diff"] * 100, p["ci_low"] * 100, p["ci_high"] * 100))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, default=RES)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    data = rows(json.loads(a.res.read_text(encoding="utf-8")))

    plt.rcParams.update({"font.size": 10.5, "axes.labelsize": 10.5})
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    # Một hàng trống ngăn hai khối; y giảm dần để hàng đầu Bảng 3.3 nằm trên cùng.
    ys, y = [], 0
    for i, r in enumerate(data):
        if i and r[0] != data[i - 1][0]:
            y -= 1
        ys.append(y); y -= 1
    for (blk, lab, d, lo, hi), yy in zip(data, ys):
        sig = lo > 0 or hi < 0
        ax.plot([lo, hi], [yy, yy], color="#222", lw=1.6, solid_capstyle="butt")
        ax.plot([d], [yy], "o", ms=7, mec="#222", mew=1.4, mfc="#222" if sig else "white", zorder=3)
        ax.text(1.02, yy, f"{vn(d)} [{vn(lo)}; {vn(hi)}]", transform=ax.get_yaxis_transform(), va="center",
                fontsize=9.5, color="#222")
    ax.axvline(0, color="#888", lw=1, ls="--", zorder=0)
    ax.set_yticks(ys, [r[1] for r in data])
    for blk in ("10 ms", "40 ms"):
        yb = [yy for r, yy in zip(data, ys) if r[0] == blk]
        ax.text(-0.02, max(yb) + 0.62, blk, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=10.5, fontweight="bold")
    ax.set_ylim(min(ys) - 0.6, max(ys) + 1.0)
    ax.set_xlabel("Hiệu WER A − B (điểm phần trăm), CI 95%")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−")))
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e5e5e5", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"đã ghi {a.out}")
    for r in data:
        print(f"  {r[0]} {r[1]}: {vn(r[2])} [{vn(r[3])}; {vn(r[4])}]")


if __name__ == "__main__":
    main()
