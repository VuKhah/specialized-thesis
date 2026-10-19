"""Phân rã lỗi clean-test cho Bảng 3.8 (RQ3, L6): S/D/I, loại thay thế, từ Việt / ngoại, thành phần âm tiết, từ dính.

Module mới thay vì sửa `error_analysis.py`: module đó chấm bằng `normalize_text` thô trên predictions.tsv dạng cũ
(08/10), nên tổng lỗi không khớp bảng chính hiện hành. Ở đây mọi token hoá, từ đệm, alt_spellings, `canon_tone`, từ vấp
dở, nhóm Việt/ngoại và nguồn giả thuyết đều lấy từ `score_clean_test` → S + D + I cộng đúng về WER của
`clean_test_scored_2026-10-16_chung/results.json` (kiểm trong `main`, sai thì dừng). Quy tắc phân loại chốt trước khi
tính: `docs/notes/rq3_phan_ra_loi_2026-10-16.md` mục 1.

    python -m src.evaluation.error_breakdown [--out_dir DIR] [--n_boot 10000] [--seed 42]
"""

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import jiwer
import numpy as np

from src.evaluation import score_clean_test as sct
from src.evaluation.error_analysis import classify_substitution
from src.evaluation.vi_syllable import ONSETS, RHYMES, is_valid_syllable, sticky_words, strip_tone

RESULTS = sct.OUT / "results.json"
OUT_DIR = Path("reports/results/error_analysis/clean_test_2026-10-16")
TONE_NAMES = {"̀": "huyền", "́": "sắc", "̉": "hỏi", "̃": "ngã", "̣": "nặng"}
_ONSET_LIST = sorted(ONSETS.split("|"), key=len, reverse=True) + [""]
_RHYME_SET = set(RHYMES)
# Cùng một âm, khác chữ do quy tắc chính tả (ngh/gh/k trước i, e, ê) — không tính là sai âm đầu.
_ONSET_CANON = {"NGH": "NG", "GH": "G", "K": "C"}

GROUPS = ("việt", "ngoại")
SUB_TYPES = ("sub_tone", "sub_diacritic", "sub_other")
SYL_KEYS = ("syl_pairs", "syl_bad_hyp", "syl_onset", "syl_rhyme", "syl_tone",
            "syl_only_tone", "syl_only_rhyme", "syl_only_onset", "syl_multi", "syl_spelling")
COUNT_KEYS = (["words", "words_vi", "words_fx", "errors", "S", "D", "I"]
              + [f"{k}_{g}" for k in ("S", "D", "I") for g in ("vi", "fx")] + list(SUB_TYPES) + list(SYL_KEYS)
              + ["glued", "hyp_words"])
# tên chỉ số → (tử số, mẫu số, hệ số)
METRICS = {
    "wer": ("errors", "words", 1), "S": ("S", "words", 1), "D": ("D", "words", 1), "I": ("I", "words", 1),
    "sub_tone": ("sub_tone", "words", 1), "sub_diacritic": ("sub_diacritic", "words", 1),
    "sub_other": ("sub_other", "words", 1),
    "wer_vi": ("err_vi", "words_vi", 1), "wer_fx": ("err_fx", "words_fx", 1),
    "S_vi": ("S_vi", "words_vi", 1), "D_vi": ("D_vi", "words_vi", 1), "I_vi": ("I_vi", "words_vi", 1),
    "S_fx": ("S_fx", "words_fx", 1), "D_fx": ("D_fx", "words_fx", 1), "I_fx": ("I_fx", "words_fx", 1),
    "glued_per_1k": ("glued", "words", 1000),
    "syl_bad_hyp_share": ("syl_bad_hyp", "S_vi", 1),
    "syl_onset_share": ("syl_onset", "syl_pairs", 1), "syl_rhyme_share": ("syl_rhyme", "syl_pairs", 1),
    "syl_tone_share": ("syl_tone", "syl_pairs", 1), "syl_only_tone_share": ("syl_only_tone", "syl_pairs", 1),
    "syl_only_rhyme_share": ("syl_only_rhyme", "syl_pairs", 1),
    "syl_only_onset_share": ("syl_only_onset", "syl_pairs", 1), "syl_multi_share": ("syl_multi", "syl_pairs", 1),
}

# Hàng của bảng: (khoá, tên, nhóm trình bày, các cột hệ thống — nhiều cột = các seed, gộp trung bình ± sd).
SEED_COLS = lambda m, mode: [f"v2a_{m}_{mode}", f"v2a_{m}_seed43_{mode}", f"v2a_{m}_seed44_{mode}"]  # noqa: E731
ROWS = [
    ("10ms_relpe", "Conformer relPE", "10ms", ["v2a10_conformer_relpos"]),
    ("10ms_b1", "B1 Mamba", "10ms", ["v2a10_mamba"]),
    ("10ms_conext", "ConExtBiMamba", "10ms", ["v2a10_conextbimamba"]),
    ("10ms_nope", "Conformer không PE (ablation)", "10ms", ["v2a10_conformer"]),
    ("40ms_relpe", "Conformer relPE", "40ms", SEED_COLS("conformer_relpos", "greedy")),
    ("40ms_b1", "B1 Mamba", "40ms", SEED_COLS("mamba", "greedy")),
    ("40ms_conext", "ConExtBiMamba", "40ms", SEED_COLS("conextbimamba", "greedy")),
    ("40lm_relpe", "Conformer relPE", "40lm", SEED_COLS("conformer_relpos", "lm")),
    ("40lm_b1", "B1 Mamba", "40lm", SEED_COLS("mamba", "lm")),
    ("40lm_conext", "ConExtBiMamba", "40lm", SEED_COLS("conextbimamba", "lm")),
    ("pt_parakeet_zs", "Parakeet-CTC-0.6B-vi zero-shot", "pretrain", ["raw_parakeet_fp32"]),
    ("pt_parakeet_ft", "Parakeet-CTC-0.6B-vi fine-tune (nhãn giả)", "pretrain", ["v2a10_parakeet_ft"]),
    ("pt_phowhisper", "PhoWhisper-small zero-shot", "pretrain", ["raw_phowhisper"]),
    ("label_zipformer", "Nhãn giả Zipformer", "label", ["pseudo_label"]),
    ("v1_conformer", "Conformer không PE v1 ep28", "v1", ["raw_conformer"]),
    ("v1_b1", "B1 Mamba v1 ep28", "v1", ["raw_mamba"]),
    ("v1_conext", "ConExtBiMamba v1 ep28", "v1", ["raw_conextbimamba"]),
]
SECTIONS = {
    "10ms": "Train từ đầu, cấu hình 10 ms, ep37, 1 seed, greedy",
    "40ms": "Train từ đầu, cấu hình 40 ms, ep37, 3 seed (tb ± sd), greedy — bảng chính",
    "40lm": "Train từ đầu, cấu hình 40 ms, ep37, 3 seed (tb ± sd), beam 10 + từ điển + LM 3-gram — bảng phụ",
    "pretrain": "Nhóm pre-train (mốc tham chiếu, khác điều kiện, không xếp hạng chung)",
    "label": "Nhãn giả Zipformer so với bản hiệu đính (tham khảo)",
    "v1": "Phụ lục: v1 10 ms ep28 (đối chiếu số từ dính 15/10)",
}


def syllable_parts(word: str):
    """(âm đầu, vần, thanh) của âm tiết Việt hợp lệ; None nếu không hợp lệ. Quy ước tách: mục 1 của note RQ3."""
    if not is_valid_syllable(word):
        return None
    base, _ = strip_tone(word)
    tone = next((TONE_NAMES[c] for c in unicodedata.normalize("NFD", word) if c in TONE_NAMES), "ngang")
    for onset in _ONSET_LIST:
        if base.startswith(onset) and base[len(onset):] in _RHYME_SET:
            rhyme = base[len(onset):]
            # Chữ G đứng trước I (gì, gìn, giếng) luôn là âm đầu GI viết gộp chữ i; để G thì trùng GH (ghi) sau chuẩn hoá.
            if onset == "G" and rhyme.startswith("I"):
                return "GI", rhyme, tone
            return _ONSET_CANON.get(onset, onset), rhyme, tone
    return None


def compare_syllables(r: str, h: str):
    """Các thành phần khác nhau giữa hai âm tiết hợp lệ, hoặc None nếu một bên không hợp lệ."""
    pr, ph = syllable_parts(r), syllable_parts(h)
    if pr is None or ph is None:
        return None
    return {name for name, a, b in zip(("onset", "rhyme", "tone"), pr, ph) if a != b}


def detail_errors(ref, hyp):
    """Như `score_clean_test.count_errors` nhưng tách S/D/I theo nhóm và giữ cặp thay thế.

    ref/hyp: list (từ, nhóm, tuỳ chọn). Trả về (Counter đếm "S_việt"…, list (r, h, nhóm ref))."""
    c, pairs = Counter(), []
    if not ref:
        for w in hyp:
            c[f"I_{w[1]}"] += 1
        return c, pairs
    if not hyp:
        for w in ref:
            c[f"D_{w[1]}"] += 1
        return c, pairs
    out = jiwer.process_words(" ".join(t[0] for t in ref), " ".join(t[0] for t in hyp))
    chunks = out.alignments[0]
    for k, ch in enumerate(chunks):
        if ch.type == "substitute":
            for t, h in zip(ref[ch.ref_start_idx:ch.ref_end_idx], hyp[ch.hyp_start_idx:ch.hyp_end_idx]):
                c[f"S_{t[1]}"] += 1
                pairs.append((t[0], h[0], t[1]))
        elif ch.type == "delete":
            for t in ref[ch.ref_start_idx:ch.ref_end_idx]:
                c[f"D_{t[1]}"] += 1
        elif ch.type == "insert":
            near = [x for x in (chunks[k - 1] if k else None, chunks[k + 1] if k + 1 < len(chunks) else None)
                    if x is not None and x.type == "substitute"]
            for t in hyp[ch.hyp_start_idx:ch.hyp_end_idx]:
                if near:
                    x = near[0]
                    c[f"I_{ref[x.ref_end_idx - 1 if x is chunks[k - 1] else x.ref_start_idx][1]}"] += 1
                else:
                    c[f"I_{t[1]}"] += 1
    return c, pairs


def score_sentence_detail(ref, hyp):
    """Chọn phương án từ vấp dở đúng như `score_clean_test.score_sentence`, rồi phân rã trên ref đã chọn."""
    chosen = ref
    if any(t[2] for t in ref):
        ref2 = [t for t in ref if not t[2]]
        if sum(sct.count_errors(ref2, hyp).values()) < sum(sct.count_errors(ref, hyp).values()):
            chosen = ref2
    c, pairs = detail_errors(chosen, hyp)
    ref_e, ref_w = sct.score_sentence(ref, hyp)
    for g in GROUPS:  # phân rã phải trùng từng câu với bảng chính
        assert sum(c[f"{k}_{g}"] for k in "SDI") == ref_e[g], (g, c, ref_e)
        assert sum(t[1] == g for t in chosen) == ref_w[g]
    return c, pairs, ref_w


def sentence_counts(ref, hyp) -> tuple[dict, list]:
    c, pairs, w = score_sentence_detail(ref, hyp)
    row = {k: 0 for k in COUNT_KEYS}
    row.update(words_vi=w["việt"], words_fx=w["ngoại"], words=w["việt"] + w["ngoại"])
    for k in "SDI":
        row[f"{k}_vi"], row[f"{k}_fx"] = c[f"{k}_việt"], c[f"{k}_ngoại"]
        row[k] = row[f"{k}_vi"] + row[f"{k}_fx"]
    row["errors"] = row["S"] + row["D"] + row["I"]
    row["err_vi"] = row["S_vi"] + row["D_vi"] + row["I_vi"]
    row["err_fx"] = row["S_fx"] + row["D_fx"] + row["I_fx"]
    detailed = []
    for r, h, g in pairs:
        cat = classify_substitution(r, h)
        row[cat] += 1
        diff = compare_syllables(r, h) if g == "việt" else None
        if g == "việt":
            if diff is None:
                row["syl_bad_hyp"] += int(not is_valid_syllable(h))
                # ref Việt không hợp lệ (từ mượn không gán nhãn ngoại) và h hợp lệ: không vào thống kê âm tiết
            else:
                row["syl_pairs"] += 1
                for part in diff:
                    row[f"syl_{part}"] += 1
                if not diff:  # chỉ khác chữ viết âm đầu cùng âm (k/c, gh/g, ngh/ng)
                    row["syl_spelling"] += 1
                elif len(diff) == 1:
                    row[f"syl_only_{next(iter(diff))}"] += 1
                else:
                    row["syl_multi"] += 1
                # Ngược lại không bắt buộc: nghẹn/ngèn chỉ sai thanh sau chuẩn hoá ngh→ng nhưng chữ khác nên là sub_other.
                assert cat != "sub_tone" or diff == {"tone"}, (r, h, diff, cat)
        detailed.append((r, h, g, cat, "" if diff is None else "+".join(sorted(diff))))
    hyp_words = [t[0] for t in hyp]
    row["hyp_words"] = len(hyp_words)
    row["glued"] = len(sticky_words(" ".join(hyp_words)))
    return row, detailed


def rate(tot: dict, metric: str) -> float:
    num, den, scale = METRICS[metric]
    return scale * tot[num] / tot[den] if tot[den] else float("nan")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples, tags, alt, hyps, systems = sct.load_hypotheses()
    names = {col: name for col, name, _ in systems}
    videos = sorted({s["source"] for s in samples})
    refs = [sct.ref_tokens(n, s["corrected_text"], tags, alt) for n, s in enumerate(samples, 1)]
    keys = [f"{s['split']}:{s['index']}" for s in samples]
    vid_idx = np.array([videos.index(s["source"]) for s in samples])
    cols_needed = sorted({c for *_, cols in ROWS for c in cols})
    missing = [c for c in cols_needed if c not in names]
    if missing:
        raise SystemExit(f"Thiếu cột giả thuyết: {missing}")

    count_cols = list(COUNT_KEYS) + ["err_vi", "err_fx"]
    per_video, per_sentence, confusions, totals = {}, [], [], {}
    for col in cols_needed:
        mat = np.zeros((len(videos), len(count_cols)))
        conf = Counter()
        for i, (ref, key) in enumerate(zip(refs, keys)):
            row, detailed = sentence_counts(ref, sct.hyp_tokens(hyps[key][col], alt))
            mat[vid_idx[i]] += [row[k] for k in count_cols]
            per_sentence.append({"system": col, "key": key, "video": samples[i]["source"], **row})
            conf.update(detailed)
        per_video[col] = mat
        totals[col] = dict(zip(count_cols, mat.sum(0).tolist()))
        confusions += [(col, *k, n) for k, n in conf.most_common()]

    # Ràng buộc chính: cộng về đúng bảng chính.
    published = json.loads(RESULTS.read_text(encoding="utf-8"))["systems"]
    check = {}
    for col in cols_needed:
        t, p = totals[col], published[col]
        d = {m: abs(rate(t, m) - p[m]) for m in ("wer", "wer_vi", "wer_fx")}
        sub_sum = sum(t[k] for k in SUB_TYPES)
        assert sub_sum == t["S"], col
        check[col] = max(d.values())
        if check[col] > 1e-9:
            raise SystemExit(f"{col}: phân rã lệch bảng chính {d}")

    rng_idx = np.random.default_rng(args.seed).integers(0, len(videos), size=(args.n_boot, len(videos)))
    ci_idx = {k: count_cols.index(k) for k in count_cols}

    def boot_rates(cols: list[str], metric: str) -> np.ndarray:
        num, den, scale = METRICS[metric]
        vals = []
        for c in cols:
            s = per_video[c][rng_idx].sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                vals.append(scale * s[:, ci_idx[num]] / s[:, ci_idx[den]])
        return np.mean(vals, axis=0)

    summary_rows = {}
    for key, label, section, cols in ROWS:
        per_seed = {m: [rate(totals[c], m) for c in cols] for m in METRICS}
        entry = {"label": label, "section": section, "columns": cols, "system_names": [names[c] for c in cols],
                 "n_seeds": len(cols),
                 "mean": {m: float(np.mean(v)) for m, v in per_seed.items()},
                 "sd": {m: (float(np.std(v, ddof=1)) if len(v) > 1 else None) for m, v in per_seed.items()},
                 "per_seed": per_seed,
                 "counts_mean": {k: float(np.mean([totals[c][k] for c in cols])) for k in count_cols},
                 "ci95": {}}
        for m in METRICS:
            b = boot_rates(cols, m)
            entry["ci95"][m] = [float(np.nanquantile(b, 0.025)), float(np.nanquantile(b, 0.975))]
        summary_rows[key] = entry

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"n_sentences": len(samples), "n_videos": len(videos), "n_boot": args.n_boot, "seed": args.seed,
            "results_checked": RESULTS.as_posix(), "max_abs_diff_vs_results": max(check.values()),
            "rules": "docs/notes/rq3_phan_ra_loi_2026-10-16.md mục 1",
            "ci_note": "bootstrap khối video; hàng nhiều seed: CI của trung bình seed (cùng chỉ số lấy mẫu), "
                       "không gồm dao động seed"}
    (out / "summary.json").write_text(json.dumps(
        {"meta": meta, "rows": summary_rows, "systems": {c: {"name": names[c], "counts": totals[c],
                                                             "rates": {m: rate(totals[c], m) for m in METRICS}}
                                                         for c in cols_needed}},
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "label", "section", "n_seeds"]
                   + [f"{m}_{x}" for m in METRICS for x in ("mean", "sd", "ci_low", "ci_high")])
        for key, e in summary_rows.items():
            w.writerow([key, e["label"], e["section"], e["n_seeds"]]
                       + [v for m in METRICS for v in (e["mean"][m], e["sd"][m], *e["ci95"][m])])
    with (out / "per_sentence.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(per_sentence[0]))
        w.writeheader()
        w.writerows(per_sentence)
    with (out / "confusions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "ref_word", "hyp_word", "ref_group", "category", "syllable_diff", "count"])
        w.writerows(confusions)
    md = render_markdown(summary_rows, totals, confusions, meta)
    (out / "table_3_8.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Khớp bảng chính: lệch tuyệt đối lớn nhất {meta['max_abs_diff_vs_results']:.2e} trên {len(check)} hệ thống")
    print(f"Ghi vào {out}")


def _f(x: float, nd: int = 2, scale: float = 100) -> str:
    return f"{scale * x:.{nd}f}".replace(".", ",")


def _cell(e: dict, m: str, nd: int = 2, scale: float = 100) -> str:
    s = _f(e["mean"][m], nd, scale)
    return s if e["sd"][m] is None else f"{s} ± {_f(e['sd'][m], nd, scale)}"


def render_markdown(rows: dict, totals: dict, confusions: list, meta: dict) -> str:
    L = ["# Bảng 3.8 — Phân rã lỗi clean-test (203 câu, 29 video)", "",
         "Số là % trên số từ ref (cùng mẫu số với WER, nên S + D + I = WER và thanh + dấu phụ + khác = S), trừ cột ghi "
         "khác. 40 ms: trung bình ± sd 3 seed. Quy tắc: `docs/notes/rq3_phan_ra_loi_2026-10-16.md` mục 1.", ""]
    for sec, title in SECTIONS.items():
        rs = [e for e in rows.values() if e["section"] == sec]
        L += [f"### 3.8a — {title}", "",
              "| Hệ thống | WER | S | D | I | S: thanh | S: dấu phụ | S: khác | WER từ Việt | WER từ ngoại |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for e in rs:
            L.append(f"| {e['label']} | " + " | ".join(_cell(e, m) for m in
                     ("wer", "S", "D", "I", "sub_tone", "sub_diacritic", "sub_other", "wer_vi", "wer_fx")) + " |")
        L += ["", f"### 3.8b — {title}: thay thế từ Việt theo thành phần âm tiết, từ dính", "",
              "| Hệ thống | Cặp hợp lệ (số) | Sai thanh | Sai âm đầu | Sai vần | Chỉ thanh | Chỉ vần | Chỉ âm đầu "
              "| ≥ 2 thành phần | h không phải âm tiết (% S Việt) | Từ dính / 1.000 từ |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in rs:
            n = e["counts_mean"]["syl_pairs"]
            L.append(f"| {e['label']} | {n:.0f} | " + " | ".join(_cell(e, m, 1) for m in (
                "syl_tone_share", "syl_onset_share", "syl_rhyme_share", "syl_only_tone_share",
                "syl_only_rhyme_share", "syl_only_onset_share", "syl_multi_share", "syl_bad_hyp_share"))
                + f" | {_cell(e, 'glued_per_1k', 2, 1)} |")
        L += ["", f"### 3.8c — {title}: CI 95% (bootstrap khối video, {meta['n_boot']} lần, seed {meta['seed']})", "",
              "| Hệ thống | WER | S | D | I | S: thanh | WER từ ngoại | Sai thanh (% cặp) | Từ dính / 1.000 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for e in rs:
            def ci(m, scale=100, nd=2):
                lo, hi = e["ci95"][m]
                return f"[{_f(lo, nd, scale)}; {_f(hi, nd, scale)}]"
            L.append(f"| {e['label']} | " + " | ".join(
                [ci(m) for m in ("wer", "S", "D", "I", "sub_tone", "wer_fx")]
                + [ci("syl_tone_share", nd=1), ci("glued_per_1k", 1, 2)]) + " |")
        L.append("")
    L += ["Ghi chú: \"Cặp hợp lệ\" = thay thế có từ ref nhóm Việt và cả hai phía là âm tiết Việt hợp lệ (số trung bình "
          "seed); các cột \"Sai …\" là % trên số cặp đó, đa nhãn (một cặp có thể sai nhiều thành phần); \"Chỉ …\" và "
          "\"≥ 2\" loại trừ, cộng = 100%. CI của hàng 3 seed chỉ phản ánh dao động mẫu test.", "",
          "## Top 10 cặp nhầm lẫn (ref → hyp: số lần) — seed 42 / 1 seed", ""]
    by_sys = defaultdict(list)
    for col, r, h, g, cat, diff, n in confusions:
        by_sys[col].append((r, h, n, diff or cat))
    for key, e in rows.items():
        col = e["columns"][0]
        L.append(f"- **{e['label']} ({e['section']})**: " + ", ".join(
            f"{r}→{h} {n} [{d}]" for r, h, n, d in by_sys[col][:10]))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
