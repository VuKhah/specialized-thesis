"""RQ2 hoàn chỉnh (2026-10-16): Bảng 3.6 chi phí huấn luyện, Bảng 3.7 chi phí suy luận theo độ dài + điểm giao,
bảng điều kiện số học, Hình 3.4 (độ trễ / VRAM theo độ dài, log-log) và Hình 3.5 (đánh đổi WER – RTF trên câu 15 s).

Nguồn (chỉ đọc, không đo lại): `reports/results/rq2_final_2026-10-13/` (T4, batch 1, trọng số ngẫu nhiên — chi phí
không phụ thuộc trọng số; làm ấm từng độ dài; chia đôi tìm OOM), `reports/results/train_full/*/metrics.jsonl`, WER
clean-test ở `RESULTS_WER`. Chạy lại khi có thêm seed (cột GPU-h, WER 40 ms đổi theo).

    python scripts/rq2_final_tables.py

Output: `reports/results/rq2_tables_2026-10-16/*.csv|md`, `reports/figures/rq2/fig_3_4_*.png`, `fig_3_5_*.png`.
"""

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
RQ2 = Path("reports/results/rq2_final_2026-10-13")
TRAIN = Path("reports/results/train_full")
RESULTS_WER = Path("reports/results/clean_test_scored_2026-10-16/results.json")
OUT = Path("reports/results/rq2_tables_2026-10-16")
FIG = Path("reports/figures/rq2")
SENTENCE_S = 15.0  # câu VietSuperSpeech 10–15 s → điểm đại diện cho Hình 3.5
TABLE_S = (15.0, 60.0, 240.0, 3600.0)

# Màu theo encoder (3 slot đầu palette tham chiếu — kiểm all-pairs đạt); nét theo cấu hình; Parakeet xám = mốc.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
CONFIGS = [  # (khoá RQ2, nhãn, màu, nét, marker, thư mục train, cột WER)
    ("mamba_sub4", "B1 40 ms", BLUE, "-", "o", "v2a-s4-mamba/s1", "v2a_mamba_greedy"),
    ("conextbimamba_sub4", "ConExt 40 ms", ORANGE, "-", "s", "v2a-s4-conextbimamba/s1", "v2a_conextbimamba_greedy"),
    ("conformer_relpos_sub4", "Conformer PE 40 ms", AQUA, "-", "^", "v2a-s4-conformer-relpos/s1",
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


def train_table(params):
    out = []
    for key, label, *_, tdir, _ in CONFIGS:
        if tdir is None:
            continue
        R = [json.loads(x) for x in open(next((TRAIN / tdir).glob("checkpoints/*/metrics.jsonl")), encoding="utf-8")]
        last = R[-1]
        p = params[key]
        branched = R[0]["epoch"] > 0
        out.append({"cấu hình": label, "tham số encoder (M)": round(p["params_encoder"] / 1e6, 2),
                    "subsampling (M)": round((p["params_subsampling"] or 0) / 1e6, 2),
                    "tổng (M)": round(p["params_total"] / 1e6, 2),
                    "s/step": round(sorted(r["s_per_step"] for r in R)[len(R) // 2], 3),
                    "GPU-h tới ep37 (2×T4)": round(last["wall_seconds_total"] / 3600, 2),
                    "VRAM đỉnh train (GiB/GPU)": round(max(r["peak_vram_gib"] for r in R), 2),
                    "step không hữu hạn": sum(r["nonfinite_loss_steps"] for r in R),
                    "ghi chú": "rẽ nhánh v1 ep29: GPU-h gồm 30 epoch v1; s/step, VRAM chỉ pha giảm" if branched else "từ đầu"})
    return out


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
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.9), sharex=True, sharey="row")
    panels = [("40 ms (subsampling 4×)", "_sub4", ["mamba", "conextbimamba", "conformer_relpos"]),
              ("10 ms (không subsampling)", "", ["mamba", "conextbimamba", "conformer_relpos", "conformer"])]
    for col, (title, suf, encs) in enumerate(panels):
        for row, idx in ((0, 1), (1, 3)):
            ax = axes[row][col]
            ends = []
            for e in encs:
                lab, color, ls, mk = ENC[e]
                c = curves[e + suf]
                xs, ys = [x[0] for x in c], [x[idx] for x in c]
                ax.plot(xs, ys, ls=ls, color=color, lw=1.5, marker=mk, ms=3.5, markevery=3,
                        mfc="white" if e == "conformer" else color, mec=color, mew=0.8)
                if oom[e + suf]:
                    ax.plot(xs[-1], ys[-1], marker="x", color=INK, ms=6, mew=1.4, zorder=5)
                    if row == 0:
                        ax.annotate(f"OOM ~{oom[e + suf][1]:.0f} s", (xs[-1], ys[-1]), xytext=(-4, 4),
                                    textcoords="offset points", fontsize=6.5, color=INK, ha="right", va="bottom")
                else:
                    ends.append((ys[-1], lab, color))
            # nhãn trực tiếp ở đầu đường (đẩy dọc khi quá sát nhau, trên thang log)
            ends.sort()
            merged = []  # hai đường trùng nhau (B1 và ConExt 40 ms ở VRAM: cùng đỉnh do subsampling) → một nhãn
            for y, lab, color in ends:
                if merged and abs(math.log10(y) - math.log10(merged[-1][0])) < 0.01:
                    merged[-1] = (merged[-1][0], f"{merged[-1][1]} = {lab}", color)
                else:
                    merged.append((y, lab, color))
            placed = []
            for y, lab, color in merged:
                ly = y
                for py in placed:
                    if abs(math.log10(ly) - math.log10(py)) < 0.17:
                        ly = py * 10 ** 0.17
                placed.append(ly)
                ax.annotate(lab, (3600, y), xytext=(3900, ly), textcoords="data", fontsize=6.5, color=INK,
                            va="center", ha="left", annotation_clip=False)
            if row == 0:
                ax.axvspan(10, 15, color=GRID, alpha=0.9, lw=0)
                ax.set_title(f"({'ab'[col]}) Độ trễ trung vị — {title}", fontsize=8.5, color=INK, loc="left")
            else:
                ax.axhline(T4_GIB, color=GRAY, lw=0.8, ls="-")
                ax.text(6, T4_GIB * 0.82, "giới hạn bộ nhớ T4", fontsize=6.5, color=INK2, va="top")
                ax.set_title(f"({'cd'[col]}) VRAM đỉnh — {title}", fontsize=8.5, color=INK, loc="left")
                ax.set_xlabel("độ dài audio", fontsize=8, color=INK2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(4, 3600)
            ax.set_xticks([t for t, _ in XTICKS], [l for _, l in XTICKS])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            style(ax)
    axes[0][0].set_ylabel("giây", fontsize=8, color=INK2)
    axes[1][0].set_ylabel("GiB", fontsize=8, color=INK2)
    fig.tight_layout(w_pad=4.5, h_pad=1.2)
    fig.savefig(FIG / "fig_3_4_latency_vram.png", dpi=300)
    plt.close(fig)


def fig_tradeoff(curves, wer):
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    cols40 = {"mamba": "v2a_mamba_greedy", "conextbimamba": "v2a_conextbimamba_greedy",
              "conformer_relpos": "v2a_conformer_relpos_greedy"}
    cols10 = {e: f"v2a10_{e}" for e in ENC}
    pts = []
    for suf, cols, filled, cfg in (("_sub4", cols40, True, "40 ms"), ("", cols10, False, "10 ms")):
        for e, col in cols.items():
            lab, color, _, mk = ENC[e]
            x = 1000 * at(curves[e + suf], SENTENCE_S)[1]
            w = wer[col]
            ax.errorbar(x, 100 * w["wer"], yerr=[[100 * (w["wer"] - w["ci"][0])], [100 * (w["ci"][1] - w["wer"])]],
                        fmt=mk, color=color, ms=7, mfc=color if filled else "white", mec=color, mew=1.4,
                        ecolor=color, elinewidth=1.0, capsize=2.5, zorder=3)
            pts.append((x, 100 * w["wer"], f"{lab} {cfg}"))
    for key, lab in (("raw_parakeet_fp32", "Parakeet 0,6B zero-shot"), ("v2a10_parakeet_ft", "Parakeet 0,6B fine-tune")):
        y = 100 * wer[key]["wer"]
        ax.axhline(y, color=GRAY, lw=0.8, ls="--", zorder=1)
        ax.text(0.5, y + 0.2, f"{lab} — tham chiếu, khác điều kiện", fontsize=6.5, color=INK2, ha="left", va="bottom")
    offsets = {"B1 40 ms": (7, -2), "ConExt 40 ms": (-8, 6), "Conformer PE 40 ms": (8, -9), "B1 10 ms": (-8, 0),
               "ConExt 10 ms": (-8, 6), "Conformer PE 10 ms": (7, 0), "Conformer không PE 10 ms": (7, 0)}
    for x, y, lab in pts:
        dx, dy = offsets.get(lab, (7, 0))
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=INK,
                    ha="right" if dx < 0 else "left", va="center")
    ax.set_xlim(0, 40)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.set_xlabel(f"độ trễ suy luận trung vị ở câu {SENTENCE_S:.0f} s (ms; T4, batch 1)", fontsize=8, color=INK2)
    ax.set_ylabel("WER clean-test (%), greedy, CI 95%", fontsize=8, color=INK2)
    ax.text(0.99, 0.97, "marker đặc: 40 ms · rỗng: 10 ms", transform=ax.transAxes, fontsize=6.5, color=INK2,
            ha="right", va="top")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_3_5_wer_rtf.png", dpi=300)
    plt.close(fig)


def main():
    plt.rcParams["font.family"] = "Times New Roman"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    curves, oom, params = load_rq2()
    wer = json.loads(RESULTS_WER.read_text(encoding="utf-8"))["systems"]

    t36 = train_table(params)
    write_csv(OUT / "table_3_6_train_cost.csv", t36)

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
    fig_tradeoff(curves, wer)

    for name, rows in (("Bảng 3.6", t36), ("Bảng 3.7", t37), ("Điểm giao", cross)):
        print(f"== {name}")
        for r in rows:
            print("  " + " | ".join(str(v) for v in r.values()))
    print(f"Hình → {FIG}")


if __name__ == "__main__":
    main()
