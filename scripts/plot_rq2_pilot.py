"""Vẽ kết quả pilot RQ2 (kernel `rq2-pilot-asr`) từ `results.jsonl`.

    python scripts/plot_rq2_pilot.py reports/results/rq2_pilot_2026-10-05

Ra `rq2_pilot.png` (lưới 2×2: độ trễ / VRAM × không subsampling / subsampling 4×, trục log theo độ dài)
và `summary.md` (bảng số, độ dốc log-log, điểm giao, giới hạn độ dài) trong cùng thư mục.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# Bảng màu tham chiếu (dataviz skill, slot 1-3, thứ tự cố định theo mô hình — không theo thứ hạng).
COLORS = {"conformer": "#2a78d6", "conextbimamba": "#eb6834", "mamba_b1": "#1baf7a"}
NAMES = {"conformer": "Conformer-12M", "conextbimamba": "ConExtBiMamba", "mamba_b1": "Mamba B1"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(d: Path):
    rows = [json.loads(line) for line in open(d / "results.jsonl", encoding="utf-8")]
    meas = [r for r in rows if r.get("kind") != "limit" and not r.get("bisect")]
    limits = [r for r in rows if r.get("kind") == "limit"]
    return meas, limits


def series(meas, model, sub, prec):
    ok = sorted((r for r in meas if (r["model"], r["subsample"], r["mamba_precision"]) == (model, sub, prec)
                 and not r["oom"]), key=lambda r: r["seconds"])
    oom = [r["seconds"] for r in meas if (r["model"], r["subsample"], r["mamba_precision"]) == (model, sub, prec)
           and r["oom"]]
    return ok, oom


def slope(ok, key, min_s=60):
    pts = [(r["seconds"], r[key]) for r in ok if r["seconds"] >= min_s and r[key]]
    if len(pts) < 2:
        return None
    x, y = np.log10([p[0] for p in pts]), np.log10([p[1] for p in pts])
    return float(np.polyfit(x, y, 1)[0])


def crossover(a, b, key="latency_s"):
    """Độ dài nhỏ nhất (trong các mức đo) từ đó a nhanh hơn b ở mọi mức tiếp theo đều đo được."""
    da, db = {r["seconds"]: r[key] for r in a}, {r["seconds"]: r[key] for r in b}
    common = sorted(set(da) & set(db))
    for i, s in enumerate(common):
        if all(da[t] < db[t] for t in common[i:]):
            return s
    return None


def main():
    d = Path(sys.argv[1])
    meas, limits = load(d)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for col, sub in enumerate((1, 4)):
        for row, (key, label) in enumerate((("latency_s", "Độ trễ suy luận (s, log)"),
                                            ("peak_vram_gib", "VRAM đỉnh (GiB, log)"))):
            ax = axes[row, col]
            for model, color in COLORS.items():
                for prec, style in (("fp32", "-"), ("fp16", "--")):
                    if prec == "fp16" and (model != "mamba_b1" or sub != 1):
                        continue
                    ok, oom = series(meas, model, sub, prec)
                    if not ok:
                        continue
                    xs, ys = [r["seconds"] for r in ok], [r[key] for r in ok]
                    name = NAMES[model] + (" (khối Mamba fp16, tham khảo)" if prec == "fp16" else "")
                    ax.plot(xs, ys, style, color=color, lw=2, marker="o", ms=5, label=name)
                    if oom:
                        ax.plot([xs[-1]], [ys[-1]], marker="X", ms=11, color=color, mec="white", mew=1.5, ls="none")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.axvspan(10, 15, color=GRID, alpha=0.6, lw=0, zorder=0)
            if row == 0:
                conf, _ = series(meas, "conformer", sub, "fp32")
                notes = []
                for model in ("conextbimamba", "mamba_b1"):
                    c = crossover(series(meas, model, sub, "fp32")[0], conf)
                    notes.append(f"{NAMES[model]} nhanh hơn Conformer từ {c:g} s" if c is not None
                                 else f"{NAMES[model]}: không vượt Conformer trong dải đo")
                notes.append("Dải xám: độ dài câu train (10–15 s)")
                ax.text(0.02, 0.97, "\n".join(notes), transform=ax.transAxes, va="top", fontsize=9, color=MUTED)
            if row == 1 and sub == 4:
                ax.text(0.02, 0.97, "VRAM ở nhánh 4× bị chi phối bởi khối Conv2d\nsubsampling dựng tạm "
                        "(256 kênh × toàn bộ khung)\n— không phản ánh encoder; xem nhánh 1×.",
                        transform=ax.transAxes, va="top", fontsize=9, color=MUTED)
            ax.grid(True, which="major", color=GRID, lw=0.8)
            ax.set_ylabel(label)
            if row == 0:
                ax.set_title("Không subsampling (100 khung/s — cấu hình train)" if sub == 1
                             else "Subsampling 4× (25 khung/s — chỉ để đo)", color=TEXT, fontsize=11)
            if row == 1:
                ax.set_xlabel("Độ dài audio (s, log)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if any(r["oom"] for r in meas):
        handles.append(plt.Line2D([], [], marker="X", ms=10, color=MUTED, ls="none"))
        labels.append("Điểm đo cuối trước khi hết VRAM (OOM)")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Pilot RQ2 — chi phí suy luận theo độ dài audio (1× T4, batch 1, trọng số ngẫu nhiên)",
                 color=TEXT, fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(d / "rq2_pilot.png", dpi=150)

    # Bảng tóm tắt
    lines = ["# Pilot RQ2 — tóm tắt (sinh bởi scripts/plot_rq2_pilot.py)", ""]
    env = json.loads((d / "env.json").read_text(encoding="utf-8")) if (d / "env.json").exists() else {}
    if env:
        lines += [f"Môi trường: {env.get('gpu')} ({(env.get('gpu_mem_gib') or 0):.1f} GiB), torch {env.get('torch')}, "
                  f"SDPA flash={env.get('sdp_flash')} mem_efficient={env.get('sdp_mem_efficient')}.", ""]
    for sub in (1, 4):
        lines += [f"## Subsampling {sub}×", "", "| Mô hình | Độ dài (s) | Khung | Độ trễ (ms) | RTF | VRAM (GiB) |",
                  "|---|---|---|---|---|---|"]
        for model in COLORS:
            for prec in ("fp32", "fp16"):
                ok, _ = series(meas, model, sub, prec)
                for r in ok:
                    v = "–" if r["peak_vram_gib"] is None else f"{r['peak_vram_gib']:.2f}"
                    lines.append(f"| {NAMES[model]}{' (fp16)' if prec == 'fp16' else ''} | {r['seconds']:g} | "
                                 f"{r['frames']} | {r['latency_s'] * 1000:.1f} | {r['rtf']:.5f} | {v} |")
        lines.append("")
    lines += ["## Độ dốc log-log (≥ 60 s) và giới hạn độ dài", "",
              "| Mô hình | Subsampling | Khối Mamba | Độ dốc độ trễ | Độ dốc VRAM | Dài nhất chạy được (s) | OOM tại (s) |",
              "|---|---|---|---|---|---|---|"]
    for lim in limits:
        ok, _ = series(meas, lim["model"], lim["subsample"], lim["mamba_precision"])
        sl, sv = slope(ok, "latency_s"), slope(ok, "peak_vram_gib")
        lines.append(f"| {NAMES[lim['model']]} | {lim['subsample']}× | {lim['mamba_precision']} | "
                     f"{'–' if sl is None else f'{sl:.2f}'} | {'–' if sv is None else f'{sv:.2f}'} | "
                     f"{lim['max_ok_s']} | {lim['oom_s'] if lim['oom_s'] is not None else lim.get('note', '–')} |")
    lines += ["", "## Điểm giao độ trễ (mô hình Mamba nhanh hơn Conformer từ mức đo này trở đi)", ""]
    for sub in (1, 4):
        conf, _ = series(meas, "conformer", sub, "fp32")
        for model in ("mamba_b1", "conextbimamba"):
            ok, _ = series(meas, model, sub, "fp32")
            c = crossover(ok, conf)
            lines.append(f"- Subsampling {sub}×, {NAMES[model]}: "
                         + (f"từ {c:g} s" if c is not None else "không có trong dải đo chung (cả hai đều chạy được)"))
    (d / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
