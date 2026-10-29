"""Hình (c) bổ sung B: khoảng cách hiệu dụng M3 theo độ sâu lớp cho 6 mô hình, từ `m3.csv` (không tính lại gì).

Trục hoành là độ sâu tương đối (lớp / số lớp) vì B1 14 lớp, ConExt 6, Conformer 8 — so theo chỉ số lớp sẽ lệch.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("reports/results/mamba_blackbox_2026-10-27/m3.csv")
OUT = Path("reports/figures/mamba_blackbox/fig_c_m3_theo_lop.png")
STYLE = {  # (encoder, chiều) → (nhãn, màu, kiểu nét)
    ("B1", "xuoi"): ("B1 xuôi", "#1f77b4", "-"),
    ("B1", "nguoc"): ("B1 ngược", "#1f77b4", "--"),
    ("ConExt", "xuoi"): ("ConExtBiMamba xuôi", "#2ca02c", "-"),
    ("ConExt", "nguoc"): ("ConExtBiMamba ngược", "#2ca02c", "--"),
    ("Conformer", "hai_chieu"): ("Conformer PE", "#d62728", "-"),
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    series = defaultdict(list)
    for r in csv.DictReader(open(SRC, encoding="utf-8")):
        enc = next(k for k in ("B1", "ConExt", "Conformer") if r["encoder"].startswith(k))
        series[(r["res"], enc, r["direction"])].append(
            (int(r["layer"]), float(r["M3_s"]), float(r["M3_lo"]), float(r["M3_hi"])))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, res in zip(axes, ("40 ms", "10 ms")):
        for (r, enc, d), pts in sorted(series.items()):
            if r != res:
                continue
            pts.sort()
            n = len(pts)
            x = [p[0] / n for p in pts]
            label, color, ls = STYLE[(enc, d)]
            ax.plot(x, [p[1] for p in pts], ls, color=color, marker="o", ms=3, label=label)
            ax.fill_between(x, [p[2] for p in pts], [p[3] for p in pts], color=color, alpha=0.12, lw=0)
        ax.set_title(f"Cấu hình {res}")
        ax.set_xlabel("Độ sâu tương đối (lớp / số lớp)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Khoảng cách hiệu dụng M3 (giây)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("ghi", OUT)


if __name__ == "__main__":
    main()
