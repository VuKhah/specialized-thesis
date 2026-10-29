"""Hình sơ đồ cho khóa luận (bản in, đơn giản): 1.1 ánh xạ gộp CTC, 2.1 hệ thống CTC dùng chung, 2.2 khối Mamba hai chiều
+ phép đảo theo độ dài thật, 2.3 ba loại lớp encoder. Số lớp/chiều lấy từ `configs/v2a/*_sub4.yaml`, không đọc lúc chạy.

    python scripts/fig_kien_truc.py [--out reports/figures/kien_truc]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path("reports/figures/kien_truc")
INK, MUTED = "#222222", "#666666"
C_SHARED = "#e8e8e8"  # mô-đun dùng chung mã (FFN, Conv, LN, front-end)
C_MAMBA = "#cfe0f3"
C_MHSA = "#f6d5cf"
C_ENC = "#fff1c7"
BLANK = "∅"


def box(ax, x, y, w, h, text, fc=C_SHARED, ls="-", fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=INK, lw=1.1, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.1, color=INK, ls=ls,
                                 shrinkA=0, shrinkB=0))


def blank_ax(fig_w, fig_h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- Hình 1.1
def ctc_rows(path: list[str], ext: list[str]) -> list[int]:
    """Hàng của từng khung trên lưới chuỗi mở rộng; mỗi bước đứng yên, sang hàng kế, hoặc nhảy qua một rỗng."""
    rows, s = [], None
    for sym in path:
        if s is not None and ext[s] == sym:
            rows.append(s); continue
        start = 0 if s is None else s + 1
        s = next(i for i in range(start, min(start + 2, len(ext))) if ext[i] == sym)
        rows.append(s)
    return rows


def collapse(path: list[str]) -> tuple[list[str], list[str]]:
    merged = [p for i, p in enumerate(path) if i == 0 or p != path[i - 1]]
    return merged, [p for p in merged if p != BLANK]


def fig_ctc(out: Path) -> None:
    ext = [BLANK, "a", BLANK, "b", BLANK]
    paths = [["a", "a", BLANK, BLANK, "b", "b", "b", BLANK],
             [BLANK, "a", "a", "a", BLANK, "b", BLANK, BLANK],
             ["a", BLANK, BLANK, "b", "b", "b", "b", "b"]]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for k, (ax, path) in enumerate(zip(axes, paths)):
        T = len(path)
        for t in range(T):
            for s in range(len(ext)):
                ax.plot(t, s, "o", ms=4, color="#bbbbbb", zorder=1)
        r = ctc_rows(path, ext)
        ax.plot(range(T), r, "-o", color=INK, lw=1.8, ms=7, mfc=INK, zorder=2)
        ax.set_xticks(range(T), [str(t + 1) for t in range(T)], fontsize=8.5)
        ax.set_yticks(range(len(ext)), ext if k == 0 else [""] * len(ext), fontsize=10)
        ax.set_ylim(len(ext) - 0.5, -0.5)
        ax.set_xlim(-0.5, T - 0.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel("khung $t$", fontsize=9)
        merged, final = collapse(path)
        ax.set_title(f"({chr(97 + k)})", fontsize=10)
        ax.text(0.5, -0.30, "π = (" + ", ".join(path) + ")", transform=ax.transAxes, ha="center", fontsize=9)
        ax.text(0.5, -0.40, "gộp lặp: (" + ", ".join(merged) + ")", transform=ax.transAxes, ha="center", fontsize=9,
                color=MUTED)
        ax.text(0.5, -0.50, "xoá rỗng: (" + ", ".join(final) + ")", transform=ax.transAxes, ha="center", fontsize=9)
    axes[0].set_ylabel("chuỗi mở rộng", fontsize=9)
    fig.subplots_adjust(bottom=0.32, wspace=0.12)
    fig.savefig(out / "fig_1_1_ctc_gop.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- Hình 2.1
def fig_pipeline(out: Path) -> None:
    fig, ax = blank_ax(10, 3.9, (0, 20), (0, 7.6))
    h, y1, y2 = 1.3, 5.2, 1.6
    # Hàng trên: đổi tốc độ áp lên sóng âm (dataset), SpecAugment áp lên log-mel; nét chấm = chỉ khi huấn luyện.
    box(ax, 0.2, y1, 2.4, h, "Sóng âm\n16 kHz")
    box(ax, 3.4, y1, 3.0, h, "Đổi tốc độ\n0,9 / 1,0 / 1,1", ls=":")
    box(ax, 7.2, y1, 3.6, h, "Log-mel 80 chiều\n+ CMVN toàn cục")
    box(ax, 11.6, y1, 2.8, h, "SpecAugment", ls=":")
    box(ax, 15.2, y1, 4.4, h, "Lấy mẫu xuống 4×\n(hai Conv2d bước 2)", ls="--")
    ax.text(17.4, y1 + h + 0.25, "chỉ cấu hình 40 ms", ha="center", fontsize=8.5, color=MUTED, style="italic")
    ax.text(8.9, y1 + h + 0.25, "nét chấm: chỉ khi huấn luyện", ha="center", fontsize=8.5, color=MUTED, style="italic")
    for a, b in [(2.6, 3.4), (6.4, 7.2), (10.8, 11.6), (14.4, 15.2)]:
        arrow(ax, (a, y1 + h / 2), (b, y1 + h / 2))
    ax.text(11.2, y1 - 0.3, "[T, 80]", ha="center", fontsize=8.5, color=MUTED)
    # Nối xuống hàng dưới
    ax.plot([17.4, 17.4, 2.0, 2.0], [y1, 4.25, 4.25, y2 + h + 0.02], color=INK, lw=1.1)
    arrow(ax, (2.0, 3.4), (2.0, y2 + h))
    # Hàng dưới: encoder → tuyến tính → hai nhánh
    box(ax, 0.2, y2, 6.6, h, "ENCODER\nB1 | ConExtBiMamba | Conformer PE", fc=C_ENC, fs=9, bold=False)
    box(ax, 8.0, y2, 4.4, h, "Tuyến tính 256 → 1.001\n+ log-softmax")
    arrow(ax, (6.8, y2 + h / 2), (8.0, y2 + h / 2))
    ax.text(7.4, y2 - 0.35, "[T′, 256]", ha="center", fontsize=8.5, color=MUTED)
    box(ax, 14.0, y2 + 0.85, 5.6, 1.15, "Hàm mất mát CTC\n(huấn luyện)", fs=9)
    box(ax, 14.0, y2 - 0.75, 5.6, 1.15, "Giải mã tham lam / beam + LM\n(suy luận)", fs=9)
    arrow(ax, (12.4, y2 + h / 2), (14.0, y2 + 1.42))
    arrow(ax, (12.4, y2 + h / 2), (14.0, y2 - 0.17))
    ax.text(12.9, y2 - 0.35, "[T′, 1.001]", ha="center", fontsize=8.5, color=MUTED)
    fig.savefig(out / "fig_2_1_he_thong.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- Hình 2.2
def frames(ax, x0, y, cells, label, w=0.9):
    ax.text(x0 - 0.25, y + w / 2, label, ha="right", va="center", fontsize=9)
    for i, c in enumerate(cells):
        pad = c == "đệm"
        ax.add_patch(Rectangle((x0 + i * w, y), w, w, fc="#c8c8c8" if pad else "white", ec=INK, lw=1))
        ax.text(x0 + i * w + w / 2, y + w / 2, "" if pad else c, ha="center", va="center", fontsize=9)


def fig_bimamba(out: Path) -> None:
    fig, ax = blank_ax(10.5, 4.3, (0, 25), (0, 10))
    # Trái: khối Mamba hai chiều (2.1)
    ax.text(5.6, 9.5, "(a) Khối Mamba hai chiều", ha="center", fontsize=10)
    box(ax, 0.3, 4.4, 1.6, 1.2, "$\\tilde{H}$", fc="white")
    box(ax, 3.4, 6.6, 4.4, 1.3, "Mamba$_f$", fc=C_MAMBA)
    box(ax, 2.6, 2.1, 1.3, 1.3, "R", fc="white")
    box(ax, 4.4, 2.1, 2.6, 1.3, "Mamba$_b$", fc=C_MAMBA)
    box(ax, 7.5, 2.1, 1.3, 1.3, "R", fc="white")
    ax.add_patch(plt.Circle((10.2, 5.0), 0.45, fc="white", ec=INK, lw=1.1))
    ax.text(10.2, 5.0, "+", ha="center", va="center", fontsize=12)
    arrow(ax, (1.9, 5.0), (2.4, 5.0))
    ax.plot([2.4, 2.4], [2.75, 7.25], color=INK, lw=1.1)
    arrow(ax, (2.4, 7.25), (3.4, 7.25)); arrow(ax, (2.4, 2.75), (2.6, 2.75))
    arrow(ax, (3.9, 2.75), (4.4, 2.75)); arrow(ax, (7.0, 2.75), (7.5, 2.75))
    ax.plot([7.8, 10.2], [7.25, 7.25], color=INK, lw=1.1); arrow(ax, (10.2, 7.25), (10.2, 5.45))
    ax.plot([8.8, 10.2], [2.75, 2.75], color=INK, lw=1.1); arrow(ax, (10.2, 2.75), (10.2, 4.55))
    ax.text(5.6, 0.9, "hai nhánh có phép chiếu vào–ra riêng", ha="center", fontsize=8.5, color=MUTED, style="italic")
    # Phải: đảo cả chuỗi so với đảo theo độ dài thật, câu 2 dài 4 khung + 2 khung đệm
    x0 = 17.0
    ax.text(19.7, 9.5, "(b) Đảo câu 2 trong lô (4 khung thật + 2 đệm)", ha="center", fontsize=10)
    frames(ax, x0, 7.2, ["x1", "x2", "x3", "x4", "đệm", "đệm"], "câu 2 trong lô")
    frames(ax, x0, 4.6, ["đệm", "đệm", "x4", "x3", "x2", "x1"], "đảo cả chuỗi")
    frames(ax, x0, 2.0, ["x4", "x3", "x2", "x1", "đệm", "đệm"], "đảo theo độ dài thật (2.2)")
    ax.text(x0 + 2.7, 1.2, "khung đệm (xám) luôn đứng sau mọi khung thật", ha="center", fontsize=8.5, color=MUTED,
            style="italic")
    fig.savefig(out / "fig_2_2_bimamba.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- Hình 2.3
def column(ax, x, title, mods, n_layers, w=5.3, h=1.0, gap=0.55, top=12.6):
    """`mods`: (nhãn, màu, có nối tắt cộng vào sau mô-đun không)."""
    ax.text(x + w / 2, top + 0.6, title, ha="center", fontsize=10.5, fontweight="bold")
    y = top
    arrow(ax, (x + w / 2, y + 0.45), (x + w / 2, y))
    for i, (lab, fc, res) in enumerate(mods):
        y -= h
        box(ax, x, y, w, h, lab, fc=fc, fs=9)
        if res:
            ax.text(x + w + 0.25, y + h / 2, "⊕", ha="left", va="center", fontsize=11, color=INK)
        if i < len(mods) - 1:
            arrow(ax, (x + w / 2, y), (x + w / 2, y - gap))
            y -= gap
    arrow(ax, (x + w / 2, y), (x + w / 2, y - 0.45))
    ax.text(x + w / 2, y - 1.05, f"× {n_layers} lớp", ha="center", fontsize=10)
    return y


def fig_layers(out: Path) -> None:
    fig, ax = blank_ax(10.5, 5.6, (-0.5, 20.5), (2.6, 13.8))
    column(ax, 0, "B1", [("LN", C_SHARED, False),
                         ("Mamba$_f$  ‖  R → Mamba$_b$ → R", C_MAMBA, True)], 14)
    column(ax, 7, "ConExtBiMamba", [("½ FFN", C_SHARED, True), ("LN → BiMamba", C_MAMBA, True),
                                    ("Conv (xoá đệm)", C_SHARED, True), ("½ FFN", C_SHARED, True),
                                    ("LN", C_SHARED, False)], 6)
    column(ax, 14, "Conformer PE", [("½ FFN", C_SHARED, True), ("LN → MHSA + PE tương đối", C_MHSA, True),
                                    ("Conv (xoá đệm)", C_SHARED, True), ("½ FFN", C_SHARED, True),
                                    ("LN", C_SHARED, False)], 8)
    ax.text(9.65, 2.75, "⊕: cộng nối tắt sau mô-đun   ·   xám: mô-đun dùng chung mã", ha="center", fontsize=8.5,
            color=MUTED, style="italic")
    fig.savefig(out / "fig_2_3_ba_lop.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    fig_ctc(a.out); fig_pipeline(a.out); fig_bimamba(a.out); fig_layers(a.out)
    print(f"đã ghi 4 hình vào {a.out}")


if __name__ == "__main__":
    main()
