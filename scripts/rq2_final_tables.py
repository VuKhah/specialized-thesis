"""RQ2 hoàn chỉnh (2026-10-16): Bảng 3.6 chi phí huấn luyện, Bảng 3.7 chi phí suy luận theo độ dài + điểm giao,
bảng điều kiện số học, Hình 3.4 (độ trễ / VRAM theo độ dài, lưới 1×2, log-log) và Hình 3.5 (đánh đổi WER – RTF trên
câu 15 s).

Nguồn (chỉ đọc, không đo lại): `reports/results/rq2_final_2026-10-13/` (T4, batch 1, trọng số ngẫu nhiên — chi phí
không phụ thuộc trọng số; làm ấm từng độ dài; chia đôi tìm OOM), `reports/results/train_full/*/checkpoints/*/metrics.jsonl`
(40 ms: 3 seed `s1` = 42, `seed43`, `seed44`; 10 ms: 1 seed). Hình 3.5 lấy WER từ lần chấm chung 16/10:
`WER_SUMMARY` (trung bình 3 seed + CI 95% bootstrap theo video, cùng nguồn Bảng 3.8) và `RESULTS_WER` (từng seed).

    python scripts/rq2_final_tables.py            # bảng + Hình 3.4
    python scripts/rq2_final_tables.py --fig35    # thêm vẽ lại Hình 3.5

Hình 3.5 chỉ vẽ khi có `--fig35` để chạy mặc định không ghi đè PNG. Bản 17/10 (người dùng duyệt): 40 ms đổi từ seed 42
sang trung bình 3 seed — seed 42 của relPE là run tốt nhất trong 9 run, giữ nó sẽ làm lệch hình.

Output: `reports/results/rq2_tables_2026-10-16/` (`table_3_6_train_cost.csv` — 40 ms trung bình ± SD mẫu 3 seed,
`table_3_6_train_cost_per_seed.csv` — số từng seed 40 ms, `table_3_7_inference.csv`, `crossover.csv`, `numerics.md`),
`reports/figures/rq2/fig_3_4_latency_vram.png`, (`--fig35`) `fig_3_5_wer_rtf.png`.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines  # noqa: E402
import matplotlib.ticker  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
RQ2 = Path("reports/results/rq2_final_2026-10-13")
TRAIN = Path("reports/results/train_full")
RESULTS_WER = Path("reports/results/clean_test_scored_2026-10-16_chung/results.json")
WER_SUMMARY = Path("reports/results/error_analysis/clean_test_2026-10-16/summary.csv")
OUT = Path("reports/results/rq2_tables_2026-10-16")
FIG = Path("reports/figures/rq2")
SENTENCE_S = 15.0  # câu VietSuperSpeech 10–15 s → điểm đại diện cho Hình 3.5
TABLE_S = (15.0, 60.0, 240.0, 3600.0)

# Màu theo encoder (3 slot đầu palette tham chiếu — kiểm all-pairs đạt); nét theo cấu hình; Parakeet xám = mốc.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SEEDS_40 = (("s1", 42), ("seed43", 43), ("seed44", 44))  # thư mục run → seed; chỉ nhóm 40 ms có đủ 3 seed
CONFIGS = [  # (khoá RQ2, nhãn, màu, nét, marker, thư mục train — 40 ms là thư mục cha của các seed, cột WER)
    ("mamba_sub4", "B1 40 ms", BLUE, "-", "o", "v2a-s4-mamba", "v2a_mamba_greedy"),
    ("conextbimamba_sub4", "ConExt 40 ms", ORANGE, "-", "s", "v2a-s4-conextbimamba", "v2a_conextbimamba_greedy"),
    ("conformer_relpos_sub4", "Conformer PE 40 ms", AQUA, "-", "^", "v2a-s4-conformer-relpos",
     "v2a_conformer_relpos_greedy"),
    ("mamba", "B1 10 ms", BLUE, "--", "o", "v2a-mamba/s1", "v2a10_mamba"),
    ("conextbimamba", "ConExt 10 ms", ORANGE, "--", "s", "v2a-conextbimamba/s1", "v2a10_conextbimamba"),
    ("conformer_relpos", "Conformer PE 10 ms", AQUA, "--", "^", "v2a-conformer-relpos/s3", "v2a10_conformer_relpos"),
    ("conformer", "Conformer không PE 10 ms", AQUA, ":", "v", "v2a-conformer/s1", "v2a10_conformer"),
    ("parakeet_fp32", "Parakeet 0,6B (80 ms)", GRAY, "-.", "D", None, "raw_parakeet_fp32"),
]
MAMBA = ["mamba_sub4", "conextbimamba_sub4", "mamba", "conextbimamba"]
CONFORMER = ["conformer_relpos_sub4", "conformer_relpos", "conformer"]


def load_rq2():
    rows, params = {}, {}
    for d in ("rq2_final", "rq2_final_parakeet"):
        for r in csv.DictReader(open(RQ2 / d / "results.csv", encoding="utf-8")):
            rows.setdefault(r["model"], []).append(r)
        for line in open(RQ2 / d / "results.jsonl", encoding="utf-8"):
            j = json.loads(line)
            params.setdefault(j["model"], j)
    curves, oom = {}, {}
    for m, rs in rows.items():
        ok = {}
        for r in rs:  # 30 s và 240 s đo hai lần (lần sau là kiểm nhiễu) → giữ lần đầu
            if r["oom"] == "False" and r["bisect"] == "False":
                ok.setdefault(float(r["seconds"]), (float(r["seconds"]), float(r["latency_median_s"]),
                                                    float(r["rtf_median"]), float(r["peak_vram_gib"])))
        curves[m] = sorted(ok.values())
        fit = [float(r["seconds"]) for r in rs if r["oom"] == "False"]
        fail = [float(r["seconds"]) for r in rs if r["oom"] == "True"]
        oom[m] = (max(fit), min(fail)) if fail else None
    return curves, oom, params


def at(curve, s):
    hit = [c for c in curve if c[0] == s]
    return hit[0] if hit else None


def crossover(m, c, curves):
    """Độ dài nhỏ nhất mà từ đó trở lên Mamba luôn nhanh hơn (độ trễ trung vị); nội suy log–log giữa hai điểm đo.
    Conformer OOM ở độ dài nào thì tính là Mamba nhanh hơn ở đó."""
    lm = {s: lat for s, lat, *_ in curves[m]}
    lc = {s: lat for s, lat, *_ in curves[c]}
    grid = sorted(lm)
    faster = [lm[s] < lc.get(s, math.inf) for s in grid]
    if all(faster):
        return grid[0], "nhanh hơn ở mọi độ dài"
    if not faster[-1]:
        return None, "không giao trong 5–3.600 s"
    k = max(i for i, f in enumerate(faster) if not f) + 1
    s0, s1 = grid[k - 1], grid[k]
    if s1 not in lc:
        return s1, f"Conformer OOM từ ~{s1:.0f} s"
    r0, r1 = math.log(lm[s0] / lc[s0]), math.log(lm[s1] / lc[s1])
    x = math.exp(math.log(s0) + (math.log(s1) - math.log(s0)) * r0 / (r0 - r1))
    return x, f"giữa {s0:.0f} và {s1:.0f} s"


def clear_gain(m, c, curves, factor=1.5):
    """Mô tả (không phải quy tắc quyết định): độ dài nhỏ nhất mà từ đó trở lên Mamba nhanh hơn ≥ `factor` lần. Cần vì
    ở câu ngắn độ trễ cả hai bị chi phí cố định mỗi lần chạy (~20–30 ms) chi phối — "nhanh hơn" chỉ vài ms."""
    lm = {s: lat for s, lat, *_ in curves[m]}
    lc = {s: lat for s, lat, *_ in curves[c]}
    grid = sorted(lm)
    ok = [lc.get(s, math.inf) / lm[s] >= factor for s in grid]
    if not ok[-1]:
        return None
    k = max((i for i, f in enumerate(ok) if not f), default=-1) + 1
    return grid[k]


def run_cost(run_dir):
    R = [json.loads(x) for x in open(next(run_dir.glob("checkpoints/*/metrics.jsonl")), encoding="utf-8")]
    assert R[-1]["epoch"] == 37, run_dir
    return {"s/step": sorted(r["s_per_step"] for r in R)[len(R) // 2],
            "GPU-h": R[-1]["wall_seconds_total"] / 3600,
            "VRAM": max(r["peak_vram_gib"] for r in R),
            "nonfinite": sum(r["nonfinite_loss_steps"] for r in R),
            "encoder_params": {r["encoder_params"] for r in R},
            "branched": R[0]["epoch"] > 0}


def mean_sd(xs, nd):
    return f"{statistics.mean(xs):.{nd}f} ± {statistics.stdev(xs):.{nd}f}"


def train_table(params):
    """40 ms: trung bình ± SD mẫu (ddof=1) qua 3 seed; 10 ms chỉ có 1 seed nên báo số đơn, ghi rõ trong ghi chú."""
    out, per_seed = [], []
    for key, label, *_, tdir, _ in CONFIGS:
        if tdir is None:
            continue
        p = params[key]
        multi = key.endswith("_sub4")
        runs = [(seed, run_cost(TRAIN / tdir / d)) for d, seed in SEEDS_40] if multi else [(None, run_cost(TRAIN / tdir))]
        enc = set().union(*(c["encoder_params"] for _, c in runs))
        # tham số lấy từ lần đo RQ2 (trọng số ngẫu nhiên) → phải khớp số encoder các run train thật
        assert enc == {p["params_encoder"]}, (key, enc, p["params_encoder"])
        row = {"cấu hình": label, "số seed": "3 (42/43/44)" if multi else "1",
               "tham số encoder (M)": round(p["params_encoder"] / 1e6, 2),
               "subsampling (M)": round((p["params_subsampling"] or 0) / 1e6, 2),
               "tổng (M)": round(p["params_total"] / 1e6, 2)}
        if multi:
            col = lambda k: [c[k] for _, c in runs]  # noqa: E731
            row |= {"s/step": mean_sd(col("s/step"), 3), "GPU-h tới ep37 (2×T4)": mean_sd(col("GPU-h"), 2),
                    "VRAM đỉnh train (GiB/GPU)": mean_sd(col("VRAM"), 2),
                    "step không hữu hạn": mean_sd(col("nonfinite"), 0),
                    "ghi chú": "từ đầu; trung bình ± SD mẫu (ddof=1) 3 seed"}
            per_seed += [{"cấu hình": label, "seed": seed, "s/step": round(c["s/step"], 3),
                          "GPU-h tới ep37 (2×T4)": round(c["GPU-h"], 2), "VRAM đỉnh train (GiB/GPU)": round(c["VRAM"], 2),
                          "step không hữu hạn": c["nonfinite"], "tham số encoder": next(iter(c["encoder_params"]))}
                         for seed, c in runs]
        else:
            c = runs[0][1]
            row |= {"s/step": round(c["s/step"], 3), "GPU-h tới ep37 (2×T4)": round(c["GPU-h"], 2),
                    "VRAM đỉnh train (GiB/GPU)": round(c["VRAM"], 2), "step không hữu hạn": c["nonfinite"],
                    "ghi chú": ("rẽ nhánh v1 ep29: GPU-h gồm 30 epoch v1; s/step, VRAM chỉ pha giảm" if c["branched"]
                                else "từ đầu") + "; 1 seed"}
        out.append(row)
    return out, per_seed


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def style(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.grid(True, which="minor", color=GRID, linewidth=0.3, alpha=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


# Hình 3.4/3.5 sửa 16/10 sau phản biện (thiết kế + phương pháp): tách panel theo cấu hình, chỉ so trong cùng điều kiện;
# Parakeet (khác 4 điều kiện cùng lúc) bỏ khỏi trục tốc độ — số ở Bảng 3.7, WER thành đường tham chiếu ở Hình 3.5.
ENC = {  # encoder → (nhãn, màu, nét, marker); nét + marker khác nhau để đọc được khi in đen trắng
    "mamba": ("B1", BLUE, "-", "o"),
    "conextbimamba": ("ConExt", ORANGE, "--", "s"),
    "conformer_relpos": ("Conformer PE", AQUA, "-.", "^"),
    "conformer": ("Conformer không PE", AQUA, ":", "v"),
}
T4_GIB = 14.56
XTICKS = [(15, "15 s"), (60, "1 ph"), (240, "4 ph"), (900, "15 ph"), (3600, "1 giờ")]


def fig_latency(curves, oom):
    """Hình 3.4 lưới 1×2 (người dùng bác bản 2×2 16/10): (a) độ trễ, (b) VRAM, mỗi panel đủ 7 cấu hình ~12M để thấy
    ngay cùng một encoder đổi thế nào giữa 40 ms và 10 ms. Màu + hình marker = encoder; nét liền + marker đặc = 40 ms,
    nét đứt + marker rỗng = 10 ms (cùng quy ước đặc/rỗng với Hình 3.5). Bảy đường quá nhiều cho nhãn trực tiếp ở đầu
    đường → một legend chung (hàng = cấu hình, cột = encoder). Parakeet không vẽ (khác bốn điều kiện cùng lúc)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.55))
    series = [(e, suf, cfg) for suf, cfg, encs in (("_sub4", "40 ms", ["mamba", "conextbimamba", "conformer_relpos"]),
                                                   ("", "10 ms", ["mamba", "conextbimamba", "conformer_relpos",
                                                                  "conformer"]))
              for e in encs]
    # marker lệch pha giữa encoder: B1 và ConExt 40 ms trùng hẳn nhau ở VRAM → vẫn thấy cả tròn lẫn vuông
    phase = {"mamba": 0, "conextbimamba": 2, "conformer_relpos": 1, "conformer": 3}
    handles = {}
    for ax, idx in ((axes[0], 1), (axes[1], 3)):
        for e, suf, cfg in series:
            lab, color, _, mk = ENC[e]
            c = curves[e + suf]
            xs, ys = [x[0] for x in c], [x[idx] for x in c]
            solid = suf == "_sub4"
            (h,) = ax.plot(xs, ys, ls="-" if solid else (0, (4, 2)), color=color, lw=1.4, marker=mk, ms=4.2,
                           markevery=slice(phase[e], None, 4), mfc=color if solid else "white", mec=color, mew=0.9,
                           zorder=3 if solid else 2)
            handles[(e, cfg)] = (h, f"{lab} {cfg}")
            if oom[e + suf]:
                fail = oom[e + suf][1]
                if idx == 3:
                    # nối tới trần bộ nhớ T4 ở độ dài bắt đầu OOM: điểm × nằm đúng nơi mô hình vượt trần
                    ax.plot([xs[-1], fail], [ys[-1], T4_GIB], ls=(0, (1, 1.5)), color=color, lw=1.0, zorder=2)
                    ax.plot(fail, T4_GIB, marker="x", color=INK, ms=7, mew=1.5, zorder=5)
                    left = fail < 300  # hai điểm OOM cùng độ cao → nhãn tách sang hai phía
                    ax.annotate(f"OOM {fail:.0f} s", (fail, T4_GIB), xytext=(-6 if left else 6, 4),
                                textcoords="offset points", fontsize=8, color=INK, ha="right" if left else "left",
                                va="bottom")
                else:  # nhãn độ dài OOM chỉ ghi ở (b) — ở (a) vùng đó dày đường, chữ đè lên dữ liệu
                    ax.plot(xs[-1], ys[-1], marker="x", color=INK, ms=7, mew=1.5, zorder=5)
        ax.axvspan(10, 15, color=GRID, alpha=0.9, lw=0, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(4, 4500)
        ax.set_xticks([t for t, _ in XTICKS], [l for _, l in XTICKS])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: (f"{v:g}" if v >= 1 else f"{v:.2g}").replace(".", ",")))
        ax.set_xlabel("độ dài audio (thang log)", fontsize=8, color=INK2)
        style(ax)
    axes[0].set_title("(a) Độ trễ suy luận trung vị", fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("giây (thang log)", fontsize=8, color=INK2)
    axes[1].set_title("(b) VRAM đỉnh khi suy luận", fontsize=9, color=INK, loc="left")
    axes[1].set_ylabel("GiB (thang log)", fontsize=8, color=INK2)
    axes[1].set_ylim(0.05, 40)
    axes[1].axhline(T4_GIB, color=GRAY, lw=0.9, zorder=1)
    axes[1].text(5, T4_GIB / 1.12, "giới hạn bộ nhớ T4", fontsize=8, color=INK2, va="top")
    oom_h = matplotlib.lines.Line2D([], [], ls="none", marker="x", color=INK, ms=7, mew=1.5)
    # legend điền theo cột (ncol=4): xếp cặp (40 ms, 10 ms) theo encoder → hàng 1 là 40 ms, hàng 2 là 10 ms
    order = [handles[("mamba", "40 ms")], handles[("mamba", "10 ms")],
             handles[("conextbimamba", "40 ms")], handles[("conextbimamba", "10 ms")],
             handles[("conformer_relpos", "40 ms")], handles[("conformer_relpos", "10 ms")],
             (oom_h, "hết bộ nhớ (OOM)"), handles[("conformer", "10 ms")]]
    fig.legend([h for h, _ in order], [l for _, l in order], loc="lower center", ncol=4, fontsize=8, frameon=False,
               handlelength=2.6, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=2.0)
    fig.savefig(FIG / "fig_3_4_latency_vram.png", dpi=300)
    plt.close(fig)


def fig_tradeoff(curves, wer, summary):
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    # khoá encoder → (hàng summary.csv, cột results.json từng seed); 10 ms chỉ có 1 seed nên không vẽ chấm seed
    rows40 = {"mamba": ("40ms_b1", "v2a_mamba"), "conextbimamba": ("40ms_conext", "v2a_conextbimamba"),
              "conformer_relpos": ("40ms_relpe", "v2a_conformer_relpos")}
    rows10 = {"mamba": ("10ms_b1", None), "conextbimamba": ("10ms_conext", None),
              "conformer_relpos": ("10ms_relpe", None), "conformer": ("10ms_nope", None)}
    pts = []
    for suf, rows, filled, cfg in (("_sub4", rows40, True, "40 ms"), ("", rows10, False, "10 ms")):
        for e, (row, seed_col) in rows.items():
            lab, color, _, mk = ENC[e]
            x = 1000 * at(curves[e + suf], SENTENCE_S)[1]
            r = summary[row]
            m, lo, hi = (100 * float(r[k]) for k in ("wer_mean", "wer_ci_low", "wer_ci_high"))
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt=mk, color=color, ms=7,
                        mfc=color if filled else "white", mec=color, mew=1.4,
                        ecolor=color, elinewidth=1.0, capsize=2.5, zorder=3)
            if seed_col:
                cols = [f"{seed_col}_greedy", f"{seed_col}_seed43_greedy", f"{seed_col}_seed44_greedy"]
                ax.scatter([x + d for d in (-0.9, 0.0, 0.9)], [100 * wer[c]["wer"] for c in cols], s=7,
                           color=color, alpha=0.55, linewidths=0, zorder=2)
            pts.append((x, m, f"{lab} {cfg}"))
    for key, lab in (("raw_parakeet_fp32", "Parakeet 0,6B zero-shot"), ("v2a10_parakeet_ft", "Parakeet 0,6B fine-tune")):
        y = 100 * wer[key]["wer"]
        ax.axhline(y, color=GRAY, lw=0.8, ls="--", zorder=1)
        ax.text(0.5, y + 0.2, f"{lab} — tham chiếu, khác điều kiện", fontsize=6.5, color=INK2, ha="left", va="bottom")
    offsets = {"B1 40 ms": (14, -7), "ConExt 40 ms": (-8, 6), "Conformer PE 40 ms": (8, -9), "B1 10 ms": (-8, 0),
               "ConExt 10 ms": (-8, 6), "Conformer PE 10 ms": (7, 0), "Conformer không PE 10 ms": (7, 0)}
    for x, y, lab in pts:
        dx, dy = offsets.get(lab, (7, 0))
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=INK,
                    ha="right" if dx < 0 else "left", va="center")
    ax.set_xlim(0, 40)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.set_xlabel(f"độ trễ suy luận trung vị ở câu {SENTENCE_S:.0f} s (ms; T4, batch 1)", fontsize=8, color=INK2)
    ax.set_ylabel("WER clean-test (%), greedy, CI 95%", fontsize=8, color=INK2)
    ax.text(0.99, 0.97, "marker đặc: 40 ms, trung bình 3 seed (chấm nhỏ: từng seed) · rỗng: 10 ms, 1 seed",
            transform=ax.transAxes, fontsize=6.5, color=INK2, ha="right", va="top")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_3_5_wer_rtf.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig35", action="store_true", help="vẽ lại Hình 3.5 (mặc định giữ nguyên PNG hiện có)")
    args = ap.parse_args()
    plt.rcParams["font.family"] = "Times New Roman"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    curves, oom, params = load_rq2()

    t36, t36_seeds = train_table(params)
    write_csv(OUT / "table_3_6_train_cost.csv", t36)
    write_csv(OUT / "table_3_6_train_cost_per_seed.csv", t36_seeds)

    t37 = []
    for key, label, *_ in CONFIGS:
        row = {"cấu hình": label}
        for s in TABLE_S:
            p = at(curves[key], s)
            row[f"độ trễ {s:.0f} s (s)"] = round(p[1], 3) if p else "OOM"
            row[f"RTF {s:.0f} s"] = round(p[2], 4) if p else "OOM"
        row["VRAM 3.600 s (GiB)"] = round(at(curves[key], 3600.0)[3], 2) if at(curves[key], 3600.0) else "OOM"
        row["OOM (fit / hỏng, s)"] = f"{oom[key][0]:.0f} / {oom[key][1]:.0f}" if oom[key] else "không (tới 3.600 s)"
        t37.append(row)
    write_csv(OUT / "table_3_7_inference.csv", t37)

    cross = []
    for m in MAMBA:
        for c in CONFORMER:
            x, note = crossover(m, c, curves)
            cross.append({"Mamba": dict((k, lab) for k, lab, *_ in CONFIGS)[m],
                          "Conformer": dict((k, lab) for k, lab, *_ in CONFIGS)[c],
                          "Mamba nhanh hơn từ (s)": round(x) if x else "—", "ghi chú": note,
                          "nhanh hơn ≥ 1,5× từ (s)": (lambda g: round(g) if g else "—")(clear_gain(m, c, curves))})
    write_csv(OUT / "crossover.csv", cross)

    env = json.loads((RQ2 / "rq2_final/env.json").read_text(encoding="utf-8"))
    (OUT / "numerics.md").write_text(f"""# Điều kiện đo RQ2 (Bảng điều kiện số học — đưa vào Ch2 2.7.3 / Ch3 3.4)

| Mục | Nhóm train từ đầu (7 cấu hình) | Parakeet-CTC-0.6B-vi |
|---|---|---|
| Phần cứng | {env['gpu']} ({env['gpu_mem_gib']:.1f} GiB), 1 GPU | như bên trái |
| Phần mềm | Python {env['python']}, torch {env['torch']}, cuDNN {env['cudnn']}, commit `{env['commit']}` | NeMo (tiến trình riêng) |
| Số học | autocast fp16; front-end, log_softmax fp32; **khối Mamba fp32** | **fp32 toàn bộ** (fp16 tràn số ở 16/203 câu) |
| Attention | Conformer không PE: SDPA (flash/mem-efficient bật); Conformer PE tương đối: bias vị trí **tường minh** [B, H, T, 2T−1] → bộ nhớ O(T²) | FastConformer (subsampling 8×) |
| Trọng số | ngẫu nhiên seed 0 — chi phí không phụ thuộc giá trị trọng số | đã pre-train |
| Đầu vào | audio ghép các đoạn liên tiếp cùng một video ({json.loads((RQ2 / 'rq2_final/audio_source.json').read_text(encoding='utf-8'))['n_segments']} đoạn, 3.603 s), batch 1 | như bên trái |
| Đo | {env['args']['min_runs']}–{env['args']['max_runs']} lần/độ dài sau làm ấm (cuDNN dựng kế hoạch theo shape), báo trung vị; OOM tìm bằng chia đôi {env['args']['bisect_steps']} bước | như bên trái |

Kết luận phải gọi đúng tên: Conformer PE tương đối hết bộ nhớ ở audio dài là **thuộc tính cách cài bias vị trí
tường minh**, không phải của Conformer nói chung — bản không PE (SDPA) chạy hết 3.600 s.
""", encoding="utf-8")

    fig_latency(curves, oom)
    if args.fig35:
        with WER_SUMMARY.open(encoding="utf-8") as f:
            summary = {r["row"]: r for r in csv.DictReader(f)}
        fig_tradeoff(curves, json.loads(RESULTS_WER.read_text(encoding="utf-8"))["systems"], summary)

    for name, rows in (("Bảng 3.6", t36), ("Bảng 3.7", t37), ("Điểm giao", cross)):
        print(f"== {name}")
        for r in rows:
            print("  " + " | ".join(str(v) for v in r.values()))
    print(f"Hình → {FIG}")


if __name__ == "__main__":
    main()
