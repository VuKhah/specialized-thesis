"""Hình WER val_unseen theo GPU-giờ tích luỹ (2026-10-17) — gộp "WER theo epoch" và "WER theo GPU-giờ" của khung Ch3
(`ket_qua_cuoi_khung_2026-10-15.md` mục 2, hình 1–2; quyết định #6 16/10: chỉ báo GPU-h, không train bù).

Nguồn (chỉ đọc): `metrics.jsonl` của các run trong `reports/results/train_full/`. WER là eval lúc train trên val_unseen
(nhãn giả, 29 video giữ riêng, khác clean-test) — dùng val_unseen thay val vì 561/562 video `validation` trùng train.
GPU-giờ = `wall_seconds_total` / 3600 của kernel 2×T4, cùng định nghĩa Bảng 3.6. Nhánh 10 ms rẽ từ v1 ep29: nối
`metrics.jsonl` v1 (ep0–29) với nhánh (ep30–37); `wall_seconds_total` của nhánh đã cộng dồn từ v1.

    python scripts/fig_wer_gpu_hours.py

Output: `reports/figures/rq2/fig_wer_gpu_hours.png`, `reports/results/rq2_tables_2026-10-16/wer_gpu_hours.csv`.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

from rq2_final_tables import ENC, FIG, INK, INK2, OUT, SEEDS_40, TRAIN, style  # noqa: E402

DECAY_FROM = 30  # epoch đầu pha giảm LR (công thức v2a, 38 epoch WSD)
Y_MAX = 30.0  # cắt trục y: vài epoch đầu WER 40–100% che mất phần đáng xem

RUNS_40 = {  # encoder → thư mục cha chứa các seed
    "mamba": "v2a-s4-mamba",
    "conextbimamba": "v2a-s4-conextbimamba",
    "conformer_relpos": "v2a-s4-conformer-relpos",
}
RUNS_10 = {  # encoder → các đoạn nối theo thứ tự (v1 rồi nhánh pha giảm); relPE 10 ms train từ đầu một mạch
    "mamba": ["mamba/s2", "v2a-mamba/s1"],
    "conextbimamba": ["conextbimamba/s2", "v2a-conextbimamba/s1"],
    "conformer_relpos": ["v2a-conformer-relpos/s3"],
    "conformer": ["conformer/s1", "v2a-conformer/s1"],
}


def read_metrics(run_dir):
    (path,) = (TRAIN / run_dir).glob("checkpoints/*/metrics.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def curve(parts):
    rows = [r for p in parts for r in read_metrics(p)]
    epochs = [r["epoch"] for r in rows]
    assert epochs == list(range(38)), (parts, epochs)  # nối đúng 0–37, không trùng/thiếu epoch
    walls = [r["wall_seconds_total"] for r in rows]
    assert walls == sorted(walls), parts  # GPU-giờ cộng dồn qua chỗ rẽ nhánh
    return [(r["epoch"], r["wall_seconds_total"] / 3600, 100 * r["wer_val_unseen"]) for r in rows]


def draw(ax, pts, e, emph):
    _, color, ls, mk = ENC[e]
    pre = [p for p in pts if p[0] < DECAY_FROM]
    post = [p for p in pts if p[0] >= DECAY_FROM - 1]  # nối liền với điểm cuối pha hằng
    ax.plot([p[1] for p in pre], [p[2] for p in pre], ls=ls, color=color, lw=0.9 * emph, alpha=0.85)
    ax.plot([p[1] for p in post], [p[2] for p in post], ls="-", color=color, lw=2.0 * emph, solid_capstyle="round")
    ax.plot(pts[-1][1], pts[-1][2], mk, color=color, ms=5.5, mec="white", mew=0.6, zorder=4)


def main():
    plt.rcParams["font.family"] = "Times New Roman"  # như Hình 3.4/3.5, khớp font quyển (Mẫu 5)
    rows_out = []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    ends = {0: [], 1: []}
    for e, parent in RUNS_40.items():
        for run, seed in SEEDS_40:
            pts = curve([f"{parent}/{run}"])
            draw(axes[0], pts, e, 0.8)
            rows_out += [{"config": "40 ms", "encoder": ENC[e][0], "seed": seed, "epoch": ep,
                          "gpu_h": round(h, 4), "wer_val_unseen": round(w / 100, 6)} for ep, h, w in pts]
            ends[0].append((e, pts[-1]))
    for e, parts in RUNS_10.items():
        pts = curve(parts)
        draw(axes[1], pts, e, 1.0)
        rows_out += [{"config": "10 ms", "encoder": ENC[e][0], "seed": 42, "epoch": ep,
                      "gpu_h": round(h, 4), "wer_val_unseen": round(w / 100, 6)} for ep, h, w in pts]
        ends[1].append((e, pts[-1]))
        if len(parts) > 1:  # đánh dấu chỗ rẽ nhánh (commit khác, RNG khác — khai ở Bảng 3.6)
            b = pts[DECAY_FROM - 1]
            axes[1].plot(b[1], b[2], "o", ms=4, mfc="white", mec=ENC[e][1], mew=1.0, zorder=4)

    for i, (ax, title) in enumerate(zip(axes, ("(a) 40 ms, 3 seed (42/43/44)", "(b) 10 ms, 1 seed"))):
        style(ax)
        ax.set_ylim(10, Y_MAX)
        ax.set_xlim(0, None)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("GPU-giờ tích luỹ (giờ tường, 2×T4)", fontsize=8, color=INK2)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    axes[0].set_ylabel("WER val_unseen (%), nhãn giả", fontsize=8, color=INK2)

    handles = [matplotlib.lines.Line2D([], [], color=ENC[e][1], ls=ENC[e][2], marker=ENC[e][3], ms=5, lw=1.2,
                                       label=ENC[e][0]) for e in ENC]
    handles += [matplotlib.lines.Line2D([], [], color=INK2, lw=2.0, label=f"pha giảm LR (ep{DECAY_FROM}–37)"),
                matplotlib.lines.Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec=INK2,
                                        label="rẽ nhánh từ v1 ep29")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7, frameon=False, handlelength=2.4,
               columnspacing=1.0, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_wer_gpu_hours.png", dpi=300)
    plt.close(fig)

    with (OUT / "wer_gpu_hours.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    for i, cfg in ((0, "40 ms"), (1, "10 ms")):
        for e, (ep, h, wer) in ends[i]:
            print(f"{cfg} {ENC[e][0]}: ep{ep} {h:.2f} GPU-h, WER val_unseen {wer:.2f}%")
    print(f"Hình → {FIG / 'fig_wer_gpu_hours.png'}")


if __name__ == "__main__":
    main()
