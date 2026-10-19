"""L3 RQ3 — mô hình có "nhại" lỗi của nhãn giả Zipformer không (2026-10-17).

Quy tắc chốt trước khi tính: `docs/notes/rq3_phan_ra_loi_2026-10-16.md` mục 6. Tóm tắt: vị trí lỗi nhãn = từ ref bị
nhãn giả thay hoặc xoá (căn chỉnh jiwer ref → nhãn giả); tại mỗi vị trí, giả thuyết của hệ thống là *nhại* (cùng từ
thay / cùng xoá như nhãn giả), *đúng*, hoặc *sai khác*. Mốc: tỉ lệ lỗi của hệ thống trên vị trí nhãn giả đúng.
Token hoá dùng đúng hàm của bảng chính (`score_clean_test`).

    python -m src.evaluation.label_echo

Thêm L4 (thứ hạng trên nhãn giả và nhãn hiệu đính): WER mỗi hệ thống tính lại với **nhãn giả làm ref** (cùng
`hyp_tokens` hai phía, không áp bảng nhóm từ/từ vấp — chỉ để so thứ hạng) đặt cạnh WER bảng chính → `label_rank.csv`.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

import jiwer

from src.evaluation.score_clean_test import hyp_tokens, load_hypotheses, ref_tokens

OUT = Path("reports/results/label_echo_2026-10-17")
SCORED = Path("reports/results/clean_test_scored_2026-10-16_chung/results.json")
SEEDS = ("", "_seed43", "_seed44")
GROUPS = [  # (nhóm, nhãn, [cột từng seed]) — 40 ms gộp 3 seed
    ("10 ms", "Conformer relPE", ["v2a10_conformer_relpos"]),
    ("10 ms", "B1 Mamba", ["v2a10_mamba"]),
    ("10 ms", "ConExtBiMamba", ["v2a10_conextbimamba"]),
    ("10 ms", "Conformer không PE", ["v2a10_conformer"]),
    ("40 ms greedy", "Conformer relPE", [f"v2a_conformer_relpos{s}_greedy" for s in SEEDS]),
    ("40 ms greedy", "B1 Mamba", [f"v2a_mamba{s}_greedy" for s in SEEDS]),
    ("40 ms greedy", "ConExtBiMamba", [f"v2a_conextbimamba{s}_greedy" for s in SEEDS]),
    ("40 ms beam+LM", "Conformer relPE", [f"v2a_conformer_relpos{s}_lm" for s in SEEDS]),
    ("40 ms beam+LM", "B1 Mamba", [f"v2a_mamba{s}_lm" for s in SEEDS]),
    ("40 ms beam+LM", "ConExtBiMamba", [f"v2a_conextbimamba{s}_lm" for s in SEEDS]),
    ("pre-train, học nhãn giả", "Parakeet fine-tune", ["v2a10_parakeet_ft"]),
    ("đối chứng, chưa thấy nhãn giả", "Parakeet zero-shot", ["raw_parakeet_fp32"]),
    ("đối chứng, chưa thấy nhãn giả", "PhoWhisper zero-shot", ["raw_phowhisper"]),
]


def ref_ops(ref: list[str], hyp: list[str]) -> list[tuple[str, str | None]]:
    """Mỗi từ ref → ("ok", None) | ("sub", từ thay) | ("del", None). Đoạn substitute của jiwer luôn dài bằng nhau
    hai phía nên ghép 1-1 theo vị trí; chèn không gắn với từ ref nào nên bỏ qua."""
    ops = [("ok", None)] * len(ref)
    if not ref:
        return ops
    if not hyp:
        return [("del", None)] * len(ref)
    for ch in jiwer.process_words(" ".join(ref), " ".join(hyp)).alignments[0]:
        if ch.type == "substitute":
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                ops[ch.ref_start_idx + k] = ("sub", hyp[ch.hyp_start_idx + k])
        elif ch.type == "delete":
            for i in range(ch.ref_start_idx, ch.ref_end_idx):
                ops[i] = ("del", None)
    return ops


def wer_vs_pseudo(samples, hyps, alt, col: str) -> float:
    refs, hs = [], []
    for s in samples:
        key = f"{s['split']}:{s['index']}"
        refs.append(" ".join(t[0] for t in hyp_tokens(hyps[key]["pseudo_label"], alt)))
        hs.append(" ".join(t[0] for t in hyp_tokens(hyps[key][col], alt)))
    return jiwer.wer(refs, hs)


def label_rank(samples, hyps, alt) -> list[dict]:
    """L4: thứ hạng hệ thống (40 ms: trung bình 3 seed greedy) trên nhãn giả và trên bảng chính."""
    main_wer = json.loads(SCORED.read_text(encoding="utf-8"))["systems"]
    rows = []
    for group, label, cols in GROUPS:
        if group == "40 ms beam+LM":
            continue
        rows.append({"group": group, "system": label,
                     "wer_vs_pseudo": statistics.mean(wer_vs_pseudo(samples, hyps, alt, c) for c in cols),
                     "wer_vs_corrected": statistics.mean(main_wer[c]["wer"] for c in cols)})
    for key in ("wer_vs_pseudo", "wer_vs_corrected"):
        for rank, r in enumerate(sorted(rows, key=lambda r: r[key]), 1):
            r[f"rank_{key[7:]}"] = rank
    n = len(rows)
    rho = 1 - 6 * sum((r["rank_pseudo"] - r["rank_corrected"]) ** 2 for r in rows) / (n * (n * n - 1))
    for r in rows:
        r["wer_vs_pseudo"], r["wer_vs_corrected"] = round(r["wer_vs_pseudo"], 6), round(r["wer_vs_corrected"], 6)
        r["spearman_all"] = round(rho, 4)
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    samples, tags, alt, hyps, _ = load_hypotheses()
    sents = []  # (ref words, nhóm từ, ops nhãn giả, key)
    for n, s in enumerate(samples, 1):
        key = f"{s['split']}:{s['index']}"
        rt = ref_tokens(n, s["corrected_text"], tags, alt)
        ref = [t[0] for t in rt]
        lab = [t[0] for t in hyp_tokens(hyps[key]["pseudo_label"], alt)]
        sents.append((ref, [t[1] for t in rt], ref_ops(ref, lab), key))
    n_label_err = sum(op[0] != "ok" for _, _, ops, _ in sents for op in ops)
    n_ref = sum(len(r) for r, *_ in sents)

    per_col, examples = {}, []
    for _, _, cols in GROUPS:
        for col in cols:
            c = {g: {"echo": 0, "correct": 0, "other": 0} for g in ("việt", "ngoại")}
            base_err = base_n = 0
            for ref, grp, lops, key in sents:
                hops = ref_ops(ref, [t[0] for t in hyp_tokens(hyps[key][col], alt)])
                for i, (lo, ho) in enumerate(zip(lops, hops)):
                    if lo[0] == "ok":
                        base_n += 1
                        base_err += ho[0] != "ok"
                        continue
                    kind = "echo" if ho == lo else "correct" if ho[0] == "ok" else "other"
                    c[grp[i]][kind] += 1
                    if kind == "echo" and col in ("v2a_mamba_greedy", "v2a10_parakeet_ft"):
                        examples.append({"system": col, "key": key, "ref": ref[i], "label": lo[1] or "∅"})
            per_col[col] = {"counts": c, "base_err_rate": base_err / base_n}

    rows = []
    for group, label, cols in GROUPS:
        def share(col, kind, g=("việt", "ngoại")):
            c = per_col[col]["counts"]
            tot = sum(sum(c[x].values()) for x in g)
            return sum(c[x][kind] for x in g) / tot if tot else float("nan")

        row = {"group": group, "system": label, "n_seeds": len(cols)}
        metrics = {"echo": lambda c: share(c, "echo"), "correct": lambda c: share(c, "correct"),
                   "other": lambda c: share(c, "other"), "echo_vi": lambda c: share(c, "echo", ("việt",)),
                   "echo_fx": lambda c: share(c, "echo", ("ngoại",)),
                   "correct_fx": lambda c: share(c, "correct", ("ngoại",)),
                   "err_where_label_ok": lambda c: per_col[c]["base_err_rate"]}
        for m, f in metrics.items():
            vals = [f(col) for col in cols]
            row[f"{m}_mean"] = round(statistics.mean(vals), 6)
            row[f"{m}_sd"] = round(statistics.stdev(vals), 6) if len(vals) > 1 else ""
        rows.append(row)

    c0 = per_col[GROUPS[0][2][0]]["counts"]
    meta = {"n_ref_words": n_ref, "n_label_error_positions": n_label_err,
            "by_group": {g: sum(c0[g].values()) for g in c0}}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "label_echo.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (OUT / "echo_examples.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(examples[0]))
        w.writeheader()
        w.writerows(examples)
    ranks = label_rank(samples, hyps, alt)
    with (OUT / "label_rank.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ranks[0]))
        w.writeheader()
        w.writerows(ranks)
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Vị trí lỗi nhãn: {n_label_err} / {n_ref} từ ref ({meta['by_group']})")
    for r in rows:
        sd = lambda m: f" ± {100 * r[m + '_sd']:.1f}" if r[m + "_sd"] != "" else ""  # noqa: E731
        print(f"{r['group']:<30} {r['system']:<22} nhại {100 * r['echo_mean']:5.1f}{sd('echo')} · đúng "
              f"{100 * r['correct_mean']:5.1f} · khác {100 * r['other_mean']:5.1f} | nhại Việt {100 * r['echo_vi_mean']:5.1f}"
              f" ngoại {100 * r['echo_fx_mean']:5.1f} | lỗi nơi nhãn đúng {100 * r['err_where_label_ok_mean']:5.1f}")
    for r in ranks:
        print(f"L4 {r['group']:<30} {r['system']:<22} nhãn giả {100 * r['wer_vs_pseudo']:5.2f} (hạng {r['rank_pseudo']:>2})"
              f" · hiệu đính {100 * r['wer_vs_corrected']:5.2f} (hạng {r['rank_corrected']:>2})")
    print(f"Spearman thứ hạng: {ranks[0]['spearman_all']}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
