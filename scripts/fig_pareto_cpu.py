"""Hình slide + Ch4: đánh đổi WER clean-test × thời gian CPU của 3 encoder 40 ms dùng trong công cụ phiên âm podcast.

Số đọc từ `reports/results/clean_test_scored_2026-10-16_chung/results.json` (WER beam + LM, 3 seed) và
`reports/results/longform_2026-10-27/p2_speed.json` (đo CPU P2); không tính lại mô hình nào.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

CLEAN = Path("reports/results/clean_test_scored_2026-10-16_chung/results.json")
SPEED = Path("reports/results/longform_2026-10-27/p2_speed.json")
OUT = Path("reports/figures/san_pham/fig_pareto_cpu.png")
# (khoá trong results.json, khoá trong p2_speed.json, nhãn, màu theo fig_mamba_m3.py)
MODELS = [
    ("mamba", "b1_mamba_40ms", "B1 Mamba", "#1f77b4"),
    ("conextbimamba", "conextbimamba_40ms", "ConExtBiMamba", "#2ca02c"),
    ("conformer_relpos", "conformer_relpe_40ms", "Conformer relPE", "#d62728"),
]
# Chỉ để chặn đọc nhầm cột: trung bình 3 seed đã công bố trong ghi chú kết quả (%).
EXPECTED_MEAN = {"mamba": 11.17, "conextbimamba": 10.95, "conformer_relpos": 10.59}


def vn(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def load_wer() -> dict:
    d = json.loads(CLEAN.read_text(encoding="utf-8"))
    sysd = d["systems"]
    out = {}
    for m, *_ in MODELS:
        seeds = [sysd[k]["wer"] * 100 for k in (f"v2a_{m}_lm", f"v2a_{m}_seed43_lm", f"v2a_{m}_seed44_lm")]
        mean = sum(seeds) / 3
        if abs(mean - EXPECTED_MEAN[m]) > 0.006:
            raise SystemExit(f"WER trung bình {m} = {mean:.3f}% lệch mốc {EXPECTED_MEAN[m]}% → dừng, kiểm lại cột")
        out[m] = (mean, min(seeds), max(seeds), seeds)
    # Chú thích "không phân biệt được" trên hình phải khớp CI cặp theo video đã lưu, không tự khẳng định.
    for p in d["paired_seeds"]:
        if p["mode"] == "lm" and not (p["ci_low"] <= 0 <= p["ci_high"]):
            raise SystemExit(f"CI {p['a']} vs {p['b']} (lm) không chứa 0 → chú thích trên hình sai, dừng")
    return out


def load_speed() -> tuple[dict, float]:
    d = json.loads(SPEED.read_text(encoding="utf-8"))
    dur = d["duration_s"]
    out = {}
    for _, key, *_ in MODELS:
        r = d["models"][key]
        if abs(r["total"] / dur - r["rtf"]) > 1e-6:
            raise SystemExit(f"{key}: rtf ≠ total/duration → định nghĩa RTF đã đổi, dừng")
        # phút xử lý cho 1 giờ audio = (giây xử lý / giây audio) × 60
        out[key] = {k: r[k] / dur * 60 for k in ("cut", "encoder", "decode", "total")}
    return out, dur


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    wer = load_wer()
    spd, dur = load_speed()
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
    comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor("white")

    # Khung trái: điểm = trung bình 3 seed, thanh = min–max; không vẽ đường biên Pareto vì WER không phân biệt được.
    offsets = {"mamba": (0, 14, "center", "bottom"), "conextbimamba": (12, 0, "left", "center"),
               "conformer_relpos": (12, 0, "left", "center")}
    for m, key, label, color in MODELS:
        mean, lo, hi, _ = wer[m]
        x = spd[key]["total"]
        ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", ms=9, color=color, ecolor=color,
                    elinewidth=2, capsize=6, capthick=2, zorder=3)
        dx, dy, ha, va = offsets[m]
        ax.annotate(f"{label}\n{vn(mean)} % · {vn(x)} phút", (x, hi if m == "mamba" else mean),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=10, color=color,
                    weight="bold", linespacing=1.3)
    ax.set_xlim(0, 8.4)
    ax.set_ylim(9.6, 12.2)
    ax.set_xlabel("Thời gian xử lý 1 giờ audio trên CPU (phút)")
    ax.set_ylabel("WER clean-test, beam + LM (%)")
    ax.set_title("WER × thời gian CPU (40 ms)")
    ax.xaxis.set_major_formatter(comma)
    ax.yaxis.set_major_formatter(comma)
    ax.grid(alpha=0.3)

    # Khung phải: thanh chồng encoder + giải mã; cắt đoạn quá nhỏ để thấy nên chỉ ghi trong chú thích.
    ys = list(range(len(MODELS)))[::-1]
    for y, (m, key, label, color) in zip(ys, MODELS):
        s = spd[key]
        bx.barh(y, s["encoder"], color=color, edgecolor="white", height=0.6)
        bx.barh(y, s["decode"], left=s["encoder"], color=color, alpha=0.3, hatch="//", edgecolor=color,
                lw=0, height=0.6)
        bx.text(s["encoder"] / 2, y, vn(s["encoder"]), ha="center", va="center", color="white", fontsize=10,
                weight="bold")
        bx.text(s["encoder"] + s["decode"] / 2, y, vn(s["decode"]), ha="center", va="center", color="#222",
                fontsize=9.5)
        bx.text(s["total"] + 0.12, y, f"{vn(s['total'])} phút", ha="left", va="center", fontsize=10.5,
                weight="bold", color=color)
    bx.set_yticks(ys, [lab for _, _, lab, _ in MODELS])
    bx.set_xlim(0, 8.9)
    bx.set_ylim(-0.6, 2.95)
    bx.set_xlabel("Thời gian cho 1 giờ audio trên CPU (phút)")
    bx.set_title("Tách theo bước")
    bx.xaxis.set_major_formatter(comma)
    bx.grid(axis="x", alpha=0.3)
    bx.legend(handles=[Patch(fc="#777", label="encoder"),
                       Patch(fc="#777", alpha=0.3, hatch="//", ec="#777", lw=0, label="giải mã beam + LM")],
              loc="upper right", fontsize=9.5, frameon=False, ncol=2, bbox_to_anchor=(1.0, 1.0))

    cut_max = max(s["cut"] for s in spd.values())
    notes = [
        "WER: clean-test, beam + LM, ep37; điểm = trung bình 3 seed (42/43/44), thanh = min–max 3 seed.  "
        "40 ms: ba encoder không phân biệt được về WER (CI 95% chứa 0).",
        "Mamba trên CPU = bản tham chiếu thuần PyTorch (mamba_ref), chưa tối ưu.",
        f"CPU Intel thế hệ 7, 2 luồng; 1 lần đo trên 1 chuỗi {vn(dur / 60, 1)} phút audio; "
        f"bước cắt đoạn ≤ {vn(cut_max, 3)} phút/giờ, không thấy trên hình.",
    ]
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.text(0.01, 0.06, "\n".join(notes), ha="left", va="center", fontsize=8.6, color="#444", linespacing=1.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")

    for m, key, label, _ in MODELS:
        mean, lo, hi, seeds = wer[m]
        s = spd[key]
        print(f"{label}: WER tb {mean:.2f} (min {lo:.2f}, max {hi:.2f}; seeds {[round(v, 2) for v in seeds]}) · "
              f"phút/giờ: encoder {s['encoder']:.2f} + giải mã {s['decode']:.2f} + cắt {s['cut']:.4f} "
              f"= {s['total']:.2f}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
