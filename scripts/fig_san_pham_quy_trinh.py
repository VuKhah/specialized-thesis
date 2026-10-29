"""Hình Ch4: sơ đồ quy trình công cụ phiên âm podcast dài (`src/demo/longform.py`), kèm số đo P1–P3 từ
`reports/results/longform_2026-10-27/` (đọc file, không gõ tay số).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RES = Path("reports/results/longform_2026-10-27")
OUT = Path("reports/figures/san_pham/fig_quy_trinh.png")
BLUE, GREEN, GREY, ORANGE = "#dbe8f6", "#dff0d8", "#eeeeee", "#fbe5cc"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=color, ec="#555", lw=1))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.45, body, ha="center", va="top", fontsize=8.6, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.3))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p13 = json.loads((RES / "p1_p3.json").read_text(encoding="utf-8"))
    p2 = json.loads((RES / "p2_speed.json").read_text(encoding="utf-8"))["models"]
    p1, p3 = p13["p1"], p13["p3"]

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    w, h, y = 2.35, 1.6, 2.85
    xs = [0.1, 2.7, 5.3, 7.9, 10.5]
    box(ax, xs[0], y, w, h, "Audio dài", "podcast 10–60 phút\nwav/flac/mp3, tự đổi\n16 kHz mono", GREY)
    box(ax, xs[1], y, w, h, "Cắt tại chỗ lặng", "khung năng lượng thấp\nnhất trong [10; 15] s\n→ đoạn 10–15 s\n(như lúc train)", BLUE)
    box(ax, xs[2], y, w, h, "Encoder 40 ms + CTC", "Conformer relPE (mặc định,\nCPU) · B1 · ConExtBiMamba\nep37, không đổi gì\n→ log-prob từng đoạn", BLUE)
    box(ax, xs[3], y, w, h, "Beam + LM 3-gram", "từ điển + LM như Bảng 3.4\n+ thưởng b = 2 cho từ\ntrong ngữ cảnh (bổ sung A)\n+ mốc thời gian, độ tin cậy", BLUE)
    box(ax, xs[4], y, w, h, "Đầu ra", "SRT phụ đề · TXT\nJSON từng từ\nHTML nghe-đọc\n(bấm từ để nghe)", GREEN)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + w, y + h / 2, b, y + h / 2)
    box(ax, xs[3], 0.75, w, 1.3, "Ngữ cảnh (tuỳ chọn)", "tiêu đề + mô tả tập\nbỏ URL; giữ tên riêng,\ntừ tiếng Anh", ORANGE)
    arrow(ax, xs[3] + w / 2, 0.75 + 1.3, xs[3] + w / 2, y)

    r = {k: v["rtf"] for k, v in p2.items()}
    notes = (f"Đo trên val_unseen, nhãn giả ({p1['n_chains']} chuỗi, {p1['n_videos']} video, {p1['hours']:.1f} h):\n"
             f"P1  tự cắt vs đoạn gốc: WER {p1['wer_auto'] * 100:.2f}% vs {p1['wer_orig'] * 100:.2f}%, "
             f"hiệu {p1['diff'] * 100:+.2f} [{p1['ci'][0] * 100:+.2f}; {p1['ci'][1] * 100:+.2f}] điểm\n"
             f"P2  RTF CPU 2 luồng: Conformer {r['conformer_relpe_40ms']:.3f} · ConExt {r['conextbimamba_40ms']:.3f}"
             f" · B1 {r['b1_mamba_40ms']:.3f}  (1 giờ ≈ {r['conformer_relpe_40ms'] * 60:.1f} phút)\n"
             f"P3  AUROC độ tin cậy phát hiện từ sai: {p3['auroc_entropy']:.3f} (entropy)"
             f" · {p3['auroc_maxprob']:.3f} (max-prob)")
    notes = notes.replace(".", ",")  # dấu thập phân kiểu Việt như văn bản khóa luận
    ax.text(0.15, 1.9, notes, ha="left", va="top", fontsize=9, family="DejaVu Sans", linespacing=1.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
