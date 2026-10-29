"""Hình audio dài nguyên chuỗi: ΔWER theo độ dài cửa sổ cho 3 encoder 40 ms (+ RAM đỉnh, thời gian theo độ dài cửa sổ),
đọc từ `results.json` của `src.evaluation.long_audio_score` — không tính lại gì.

    python scripts/fig_audio_dai.py [--dir reports/results/long_audio_2026-10-27] [--out <png>]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

DEFAULT_DIR = Path("reports/results/long_audio_2026-10-27")
OUT = Path("reports/figures/audio_dai/fig_delta_wer_theo_L.png")
# (khoá trong results.json, nhãn, màu theo fig_mamba_m3.py)
MODELS = [
    ("b1_mamba_40ms", "B1 Mamba", "#1f77b4"),
    ("conextbimamba_40ms", "ConExtBiMamba", "#2ca02c"),
    ("conformer_relpe_40ms", "Conformer relPE", "#d62728"),
]
LEVELS = ["seg", "L1", "L5", "all"]
XLAB = ["đoạn gốc\n(mốc)", "cửa sổ\n≤ 1 phút", "cửa sổ\n≤ 5 phút", "cả chuỗi"]
DX = 0.14  # lệch ngang giữa các encoder để thanh CI không đè nhau


def vn(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def n_skip(md: dict, level: str) -> int:
    return sum(c[level]["skip_mem"] for c in md["status"].values())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    d = json.loads((a.dir / "results.json").read_text(encoding="utf-8"))
    models = [(m, lab, c, d["models"][m]) for m, lab, c in MODELS if d["models"].get(m, {}).get("delta")]
    if not models:
        raise SystemExit("results.json chưa có ΔWER nào")
    has_perf = any(md["perf_points"] for *_, md in models)

    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5})
    fig = plt.figure(figsize=(12.5, 5.6) if has_perf else (7.5, 5.6))
    fig.patch.set_facecolor("white")
    if has_perf:
        gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], hspace=0.12, wspace=0.22)
        ax = fig.add_subplot(gs[:, 0])
        rx = fig.add_subplot(gs[0, 1])
        tx = fig.add_subplot(gs[1, 1], sharex=rx)
    else:
        ax = fig.add_subplot(1, 1, 1)

    # Khung chính: điểm = Δ trung bình các seed (ghép cặp), thanh = CI 95% bootstrap theo video; điểm rỗng = chỉ phần
    # cửa sổ chạy được (có cửa sổ skip_mem) nên tập đoạn khác các encoder còn lại.
    ax.axhline(0, color="#555", lw=1, zorder=1)
    for k, (m, label, color, md) in enumerate(models):
        off = (k - (len(models) - 1) / 2) * DX
        xs, ys = [off], [0.0]
        ax.plot(off, 0, "o", ms=7, color=color, zorder=3)
        for i, level in enumerate(LEVELS[1:], 1):
            r = md["delta"].get(level)
            if not r:
                continue
            y, lo, hi = (v * 100 for v in (r["delta"], *r["ci"]))
            partial = n_skip(md, level) > 0
            ax.errorbar(i + off, y, yerr=[[y - lo], [hi - y]], fmt="o", ms=7, color=color, ecolor=color,
                        mfc="white" if partial else color, mew=1.8, elinewidth=1.8, capsize=4, capthick=1.5, zorder=3)
            xs.append(i + off)
            ys.append(y)
        ax.plot(xs, ys, "-", color=color, lw=1, alpha=0.45, zorder=2)
    ax.set_xticks(range(len(LEVELS)), XLAB)
    ax.set_xlim(-0.5, len(LEVELS) - 0.5)
    ax.set_ylabel("ΔWER so với giải mã từng đoạn gốc (điểm %)")
    ax.set_title("ΔWER khi encoder nhận nguyên cửa sổ dài (greedy, 40 ms ep37)")
    ax.yaxis.set_major_formatter(COMMA)
    ax.grid(axis="y", alpha=0.3)
    handles = [Line2D([], [], color=c, marker="o", ms=7, lw=1, label=f"{lab} ({len(md['seeds'])} seed)")
               for _, lab, c, md in models]
    if any(n_skip(md, lv) for *_, md in models for lv in LEVELS[1:]):
        handles.append(Line2D([], [], color="#666", marker="o", ms=7, mfc="white", mew=1.8, lw=0,
                              label="điểm rỗng: chỉ cửa sổ chạy được"))
    ax.legend(handles=handles, fontsize=9, loc="best", framealpha=0.9)

    # Khung phụ (mô tả, 1 seed): RAM đỉnh và thời gian từng cửa sổ dài; đoạn gốc = điểm trung bình.
    if has_perf:
        for (m, label, color, md), mk in zip(models, "os^"):  # hai Mamba có thể trùng điểm → khác dạng dấu
            pts = [p for p in md["perf_points"] if p[1]]
            seg = md["perf"].get("seg")
            for bx, col, scale in ((rx, 3, 1 / 1024), (tx, 2, 1.0)):
                xy = [(p[1] / 60, p[col] * scale) for p in pts if p[col] is not None]
                if seg and seg["duration_s"] and seg[("peak_mb", "sec")[col == 2]]:
                    xy.append((seg["duration_s"]["mean"] / 60, seg[("peak_mb", "sec")[col == 2]]["mean"] * scale))
                if xy:
                    bx.scatter(*zip(*xy), s=14, marker=mk, color=color, alpha=0.65, lw=0, label=label)
        rx.set_ylabel("RAM đỉnh (GB)")
        rx.set_title("Tài nguyên theo độ dài cửa sổ (mô tả, seed đầu)")
        tx.set_ylabel("Thời gian xử lý (giây)")
        tx.set_xlabel("Độ dài cửa sổ (phút)")
        rx.tick_params(labelbottom=False)
        for bx in (rx, tx):
            bx.grid(alpha=0.3)
            bx.xaxis.set_major_formatter(COMMA)
            bx.yaxis.set_major_formatter(COMMA)
            bx.set_xlim(left=0)
            bx.set_ylim(bottom=0)
        conf = next((md for m, *_, md in models if m.startswith("conformer")), None)
        if conf and conf["limit"]["min_skip_mem_duration_s"]:
            for bx in (rx, tx):
                bx.axvline(conf["limit"]["min_skip_mem_duration_s"] / 60, color="#d62728", ls=":", lw=1)
            rx.text(conf["limit"]["min_skip_mem_duration_s"] / 60, rx.get_ylim()[1] * 0.97, " Conformer bỏ qua\n từ đây",
                    color="#d62728", fontsize=8, va="top", ha="left")

    # Chú thích: mọi số đọc từ results.json.
    data = d["data"]
    n_vid = max(md["delta"][lv]["n_videos"] for *_, md in models for lv in md["delta"])
    notes = [f"Giải mã greedy CTC, 40 ms ep37; tham chiếu = nhãn giả val_unseen: {data['n_chains']} chuỗi, "
             f"{n_vid} video, {vn(data['hours'], 2)} giờ.",
             f"Điểm = trung bình seed (cùng checkpoint ở hai vế); thanh = CI 95% bootstrap theo video "
             f"({d['n_boot']:_} lần).".replace("_", ".")]
    for m, label, _, md in models:
        if not any(n_skip(md, lv) for lv in LEVELS[1:]):
            continue
        r = md["delta"].get("all")
        tail = (f"cả chuỗi chỉ tính {r['n_chains']}/{data['n_chains']} chuỗi ≤ 6 phút ({r['n_videos']} video, "
                f"{vn(r['hours'], 2)} giờ)" if r else "cả chuỗi: không chuỗi nào chạy được")
        notes.append(f"{label.split()[0]}: chỉ cửa sổ ≤ 6 phút (giới hạn RAM); {tail}.")
    missing = [m for m, *_ in MODELS if not d["models"].get(m, {}).get("delta")]
    few = [f"{lab} {len(md['seeds'])}/3 seed" for _, lab, _, md in models if len(md["seeds"]) < 3]
    if missing or few:
        notes.append("Chưa đủ dữ liệu: " + "; ".join([f"{m} chưa có" for m in missing] + few) + ".")
    # Căn tay thay tight_layout (không hợp với khung trái trải 2 hàng): chừa đáy cho nhãn trục 2 dòng + chú thích.
    fig.subplots_adjust(left=0.07 if has_perf else 0.11, right=0.985, top=0.93, bottom=0.13 + 0.032 * len(notes))
    fig.text(0.01, 0.01, "\n".join(notes), ha="left", va="bottom", fontsize=8.6, color="#444", linespacing=1.5)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")

    for m, label, _, md in models:
        print(label + ": " + " · ".join(f"{lv} {vn(r['delta'] * 100, 2)} [{vn(r['ci'][0] * 100, 2)}; "
                                       f"{vn(r['ci'][1] * 100, 2)}] {r['verdict']}" for lv, r in md["delta"].items()))
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
