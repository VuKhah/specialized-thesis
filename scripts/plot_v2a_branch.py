"""Tổng hợp nhóm rẽ nhánh v2a (pha giảm 1-sqrt 8 epoch nối từ checkpoint ep29 của v1) cho Chương 3.

    python scripts/plot_v2a_branch.py reports/results/v2a_branch_2026-10-13

Đọc `reports/results/train_full/{<model>,v2a-<model>}/s*/checkpoints/*/metrics.jsonl` và
`reports/results/avg_eval_v1_2026-10-13/results.json`; ra `wer_curves.png` + `summary.md` trong thư mục đích.

Checkpoint bảng chính là ep37 (quy tắc 9.4 mục 9 `docs/notes/ket_qua/conextbimamba_gap_2026-10-12.md`, chốt trước khi có
số) — best/avg v1 chỉ là số tham chiếu. Không vẽ trục GPU-giờ: quy tắc trục cùng GPU-giờ chưa chốt (10. cùng note),
bảng chỉ ghi thời gian train đo được như một dữ kiện.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("reports/results")
# Cùng bảng màu/thứ tự với scripts/plot_rq2_pilot.py.
MODELS = {"mamba": ("Mamba B1", "#1baf7a"), "conextbimamba": ("ConExtBiMamba", "#eb6834"),
          "conformer": ("Conformer-12M (không PE)", "#2a78d6")}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SETS = ("val", "val_unseen")


def load_metrics(run: str) -> dict[int, dict]:
    """Gộp metrics.jsonl mọi phiên của một run; phiên sau ghi đè epoch trùng (resume)."""
    rows = {}
    for path in sorted((ROOT / "train_full" / run).glob("s*/checkpoints/*/metrics.jsonl")):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            rows[r["epoch"]] = r
    return rows


def train_hours(rows: dict[int, dict]) -> float:
    """Giờ đồng hồ train + eval trên 2×T4 (không gồm setup kernel) — dữ kiện đo, không phải quy tắc GPU-giờ."""
    return sum(r["steps"] * r["s_per_step"] + r.get("eval_seconds_val", 0) + r.get("eval_seconds_val_unseen", 0)
               for r in rows.values()) / 3600


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    avg = json.loads((ROOT / "avg_eval_v1_2026-10-13" / "results.json").read_text(encoding="utf-8"))["runs"]
    data = {m: (load_metrics(m), load_metrics(f"v2a-{m}")) for m in MODELS}

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [3, 2]})
    for m, (name, color) in MODELS.items():
        v1, v2a = data[m]
        e1 = sorted(v1)
        axes[0].plot(e1, [v1[e]["wer_val_unseen"] for e in e1], color=color, lw=1.6, label=f"{name} — v1 (LR hằng)")
        e2 = [29] + sorted(v2a)  # nối từ điểm rẽ nhánh
        y2 = [v1[29]["wer_val_unseen"]] + [v2a[e]["wer_val_unseen"] for e in sorted(v2a)]
        for ax in axes:
            ax.plot(e2, y2, color=color, lw=2.2, ls="--", marker="o", ms=3.5,
                    label=f"{name} — v2a (pha giảm)" if ax is axes[0] else None)
        axes[1].plot(e1[-10:], [v1[e]["wer_val_unseen"] for e in e1[-10:]], color=color, lw=1.6)
        axes[1].axhline(avg[f"{m}_avg20-29"]["sets"]["val_unseen"]["wer"], color=color, lw=1, ls=":")
    for ax in axes:
        ax.axvspan(29.5, 37.5, color=GRID, alpha=0.6, lw=0, zorder=0)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("WER val_unseen (nhãn giả)")
    axes[0].set_ylim(top=0.6)
    axes[0].set_title("Toàn bộ quá trình train", color=TEXT, fontsize=11)
    axes[1].set_title("Phóng to ep20–37\n(chấm: trung bình checkpoint v1 ep20–29)", color=TEXT, fontsize=11)
    axes[1].set_xticks(range(20, 38, 2))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Nhóm rẽ nhánh v2a: 30 epoch LR hằng (v1) + 8 epoch giảm 1-sqrt về 0 (vùng xám) — 1 seed",
                 color=TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(out / "wer_curves.png", dpi=150)

    lines = ["# Nhóm rẽ nhánh v2a — tóm tắt (sinh bởi scripts/plot_v2a_branch.py)", "",
             "WER trên nhãn giả, CTC greedy, 1 seed; chưa có CI. Checkpoint bảng chính = ep37 (chốt trước khi có số).",
             "", "| Mô hình | Tập | v1 ep29 (điểm rẽ) | v1 best (ep) | v1 avg ep20–29 | **v2a ep37** | ep37 − v1 best |",
             "|---|---|---|---|---|---|---|"]
    for m, (name, _) in MODELS.items():
        v1, v2a = data[m]
        for s in SETS:
            best_e = min(v1, key=lambda e: v1[e][f"wer_{s}"])
            b, last = v1[best_e][f"wer_{s}"], v2a[max(v2a)][f"wer_{s}"]
            lines.append(f"| {name} | {s} | {v1[29][f'wer_{s}']:.4f} | {b:.4f} (ep{best_e}) | "
                         f"{avg[f'{m}_avg20-29']['sets'][s]['wer']:.4f} | **{last:.4f}** | {(last - b) * 100:+.2f} điểm |")
    lines += ["", "clean_test (203 câu, mới hiệu đính 11 câu) chỉ có cho v1: "
              + "; ".join(f"{MODELS[m][0]} best {avg[f'{m}_best']['sets']['clean_test']['wer']:.4f} / avg "
                          f"{avg[f'{m}_avg20-29']['sets']['clean_test']['wer']:.4f}" for m in MODELS)
              + ". v2a ep37 chưa chấm clean_test.", "",
              "## Chi phí train đo được (2×T4, giờ đồng hồ train + eval, không gồm setup kernel)", "",
              "| Mô hình | s/step v1 (trung vị) | s/step pha giảm (trung vị) | Giờ v1 (30 ep) | Giờ pha giảm (8 ep) | "
              "VRAM đỉnh | Step loss không hữu hạn |", "|---|---|---|---|---|---|---|"]
    med = lambda xs: sorted(xs)[len(xs) // 2]
    for m, (name, _) in MODELS.items():
        v1, v2a = data[m]
        lines.append(f"| {name} | {med([r['s_per_step'] for r in v1.values()]):.3f} | "
                     f"{med([r['s_per_step'] for r in v2a.values()]):.3f} | {train_hours(v1):.2f} | "
                     f"{train_hours(v2a):.2f} | {max(r['peak_vram_gib'] for r in v2a.values()):.2f} GiB | "
                     f"{sum(r['nonfinite_loss_steps'] for r in v2a.values())} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
