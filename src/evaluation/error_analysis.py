"""Phân tích lỗi trên clean-test (RQ3): S/D/I theo mô hình, tách lỗi đặc thù tiếng Việt,
nhóm câu có ghi chú tên riêng/tiếng Anh, lỗi theo độ dài câu, cặp nhầm lẫn, câu rỗng/ảo giác.

    python -m src.evaluation.error_analysis \
        --predictions reports/results/final_compare_2026-10-08/predictions.tsv --name final_compare_2026-10-08

Quy tắc phân loại cố định TRƯỚC khi xem kết quả mô hình nào (tránh chỉnh quy tắc theo mô hình
có lợi), áp y hệt cho mọi cột `hyp_*`:
- Tham chiếu: `corrected_text` nếu đã hiệu đính, không thì `pseudo_label` (nhãn giả Zipformer)
  — giống `eval_clean_test`. Ref/hyp đi qua `normalize_text` như `compute_wer`. Cột `ref` trong
  predictions.tsv bị bỏ qua: nó chụp lúc chạy kernel, manifest có thể đã hiệu đính thêm.
- Căn chỉnh: Levenshtein mức từ của jiwer (cùng thuật toán với `compute_wer`) → tổng S+D+I và
  WER khớp `compute_wer`. Trong một đoạn `substitute`, jiwer ghép 1-1 theo vị trí.
- Mỗi cặp thay thế (r, h) xếp vào đúng một loại, theo thứ tự:
  (a) `tone`: bỏ 5 dấu thanh (huyền, sắc, ngã, hỏi, nặng) thì r == h;
  (b) `diacritic`: không thuộc (a), bỏ MỌI dấu (kể cả mũ/trăng/móc, đ→d) thì r == h
      — tức sai dấu phụ/nguyên âm (ă â ê ô ơ ư đ), có thể kèm sai thanh;
  (c) `other`: còn lại.
  Tỉ lệ = số lỗi / số từ tham chiếu, nên tone + diacritic + other + D + I = WER (phân rã cộng).
- Ảo giác/cắt cụt: một chuỗi chèn (hoặc xoá) liên tiếp dài ≥ `--run_len` từ (mặc định 5,
  chọn tay, không có chuẩn công bố — xem nó là ngưỡng mô tả, đổi được). Câu rỗng: hyp rỗng sau
  chuẩn hoá.
- Nhóm ghi chú: `notes` chứa "tên riêng" hoặc "tiếng anh" (không phân biệt hoa/thường).
- Độ dài câu: số từ tham chiếu, bin cố định độ rộng 20 (<40, 40-59, ≥60) — chọn theo phân bố
  độ dài ref (20-81 từ), không theo lỗi.
- CI 95%: bootstrap theo khối video (cùng cách lấy mẫu với `bootstrap.py`, seed 42) cho WER và
  từng tỉ lệ thành phần. Không phải kiểm định cặp giữa mô hình.

Đầu ra (mặc định reports/results/error_analysis/<name>/): `summary.json`, `summary.csv`
(mỗi mô hình một dòng), `per_sentence.csv`, `confusions.csv`, `summary.md` (bảng in ra).
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

from src.evaluation.bootstrap import _resample
from src.evaluation.text_normalize import normalize_text

MANIFEST = "data/processed/clean_test_manifest.json"
OUT_ROOT = "reports/results/error_analysis"
# Dấu thanh tiếng Việt dạng combining (NFD): huyền, sắc, ngã, hỏi, nặng.
TONE_MARKS = {"̀", "́", "̃", "̉", "̣"}
NOTE_TAGS = ("tên riêng", "tiếng anh")
LENGTH_BINS = [(0, 39, "<40"), (40, 59, "40-59"), (60, 10**9, ">=60")]
CATEGORIES = ["sub_tone", "sub_diacritic", "sub_other", "deletions", "insertions"]


def strip_tone(word: str) -> str:
    return unicodedata.normalize("NFC", "".join(
        ch for ch in unicodedata.normalize("NFD", word) if ch not in TONE_MARKS))


def strip_all_marks(word: str) -> str:
    base = "".join(ch for ch in unicodedata.normalize("NFD", word) if not unicodedata.combining(ch))
    return base.replace("đ", "d").replace("Đ", "D")


def classify_substitution(ref_word: str, hyp_word: str) -> str:
    if strip_tone(ref_word) == strip_tone(hyp_word):
        return "sub_tone"
    if strip_all_marks(ref_word) == strip_all_marks(hyp_word):
        return "sub_diacritic"
    return "sub_other"


def _max_runs(chunks, kind: str) -> list[int]:
    side = "hyp" if kind == "insert" else "ref"
    return [getattr(c, f"{side}_end_idx") - getattr(c, f"{side}_start_idx") for c in chunks if c.type == kind]


def analyze_sentence(ref: str, hyp: str, run_len: int) -> dict:
    """ref/hyp đã chuẩn hoá; ref khác rỗng."""
    out = jiwer.process_words(ref, hyp)
    r_words, h_words, chunks = out.references[0], out.hypotheses[0], out.alignments[0]
    counts = Counter({k: 0 for k in CATEGORIES})
    pairs, deleted, inserted = [], [], []
    for c in chunks:
        if c.type == "substitute":
            for rw, hw in zip(r_words[c.ref_start_idx:c.ref_end_idx], h_words[c.hyp_start_idx:c.hyp_end_idx]):
                counts[classify_substitution(rw, hw)] += 1
                pairs.append((rw, hw))
        elif c.type == "delete":
            deleted += r_words[c.ref_start_idx:c.ref_end_idx]
        elif c.type == "insert":
            inserted += h_words[c.hyp_start_idx:c.hyp_end_idx]
    counts["deletions"], counts["insertions"] = len(deleted), len(inserted)
    assert counts["deletions"] == out.deletions and counts["insertions"] == out.insertions
    assert sum(counts[k] for k in CATEGORIES[:3]) == out.substitutions
    ins_runs, del_runs = _max_runs(chunks, "insert"), _max_runs(chunks, "delete")
    return {
        "n_ref": len(r_words), "n_hyp": len(h_words), **counts,
        "errors": out.substitutions + out.deletions + out.insertions,
        "hyp_empty": len(h_words) == 0,
        "max_insert_run": max(ins_runs, default=0), "max_delete_run": max(del_runs, default=0),
        "long_insert_runs": sum(r >= run_len for r in ins_runs),
        "long_insert_words": sum(r for r in ins_runs if r >= run_len),
        "long_delete_runs": sum(r >= run_len for r in del_runs),
        "long_delete_words": sum(r for r in del_runs if r >= run_len),
        "_pairs": pairs, "_deleted": deleted, "_inserted": inserted,
    }


def load_references(manifest_path: str, ref_mode: str) -> dict:
    samples = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["samples"]
    refs = {}
    for s in samples:
        corrected = s.get("corrected_text", "").strip()
        if ref_mode == "corrected_only" and not corrected:
            continue
        use_corrected = bool(corrected) and ref_mode != "pseudo"
        notes = s.get("notes", "") or ""
        refs[(s["split"], int(s["index"]))] = {
            "ref": corrected if use_corrected else s["pseudo_label"],
            "ref_source": "corrected_text" if use_corrected else "pseudo_label",
            "video": s["source"], "duration_s": s["duration_s"], "notes": notes,
            "tagged": any(t in notes.lower() for t in NOTE_TAGS),
        }
    return refs


def load_predictions(path: str) -> tuple[list[str], dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    models = [c[len("hyp_"):] for c in rows[0] if c.startswith("hyp_")]
    return models, {(r["split"], int(r["index"])): r for r in rows}


def _length_bin(n: int) -> str:
    return next(label for lo, hi, label in LENGTH_BINS if lo <= n <= hi)


def _rates(sentences: list[dict]) -> dict:
    n_ref = sum(s["n_ref"] for s in sentences)
    tot = {k: sum(s[k] for s in sentences) for k in CATEGORIES + ["errors"]}
    return {"n_sentences": len(sentences), "n_ref_words": n_ref, "counts": tot,
            "rates": {k: (v / n_ref if n_ref else None) for k, v in tot.items()}}


def bootstrap_rates(sentences: list[dict], n_boot: int, seed: int, alpha: float = 0.05) -> dict:
    by_video = defaultdict(lambda: np.zeros(len(CATEGORIES) + 2))
    for s in sentences:
        by_video[s["video"]] += [s[k] for k in CATEGORIES] + [s["errors"], s["n_ref"]]
    mat = np.stack(list(by_video.values()))
    sums = mat[_resample(len(mat), n_boot, seed)].sum(1)  # (n_boot, n_cols)
    rates = sums[:, :-1] / sums[:, -1:]
    lo, hi = np.quantile(rates, [alpha / 2, 1 - alpha / 2], axis=0)
    return {k: [float(lo[i]), float(hi[i])] for i, k in enumerate(CATEGORIES + ["errors"])}


def analyze_model(model: str, preds: dict, refs: dict, run_len: int, top_k: int, n_boot: int, seed: int) -> tuple:
    sentences, pairs, deleted, inserted = [], Counter(), Counter(), Counter()
    skipped_empty_ref = 0
    for key, meta in refs.items():
        ref = normalize_text(meta["ref"])
        if not ref:
            skipped_empty_ref += 1
            continue
        res = analyze_sentence(ref, normalize_text(preds[key][f"hyp_{model}"]), run_len)
        pairs.update(res.pop("_pairs"))
        deleted.update(res.pop("_deleted"))
        inserted.update(res.pop("_inserted"))
        sentences.append({"model": model, "split": key[0], "index": key[1], "video": meta["video"],
                          "ref_source": meta["ref_source"], "tagged": meta["tagged"],
                          "length_bin": _length_bin(res["n_ref"]), **res})
    summary = _rates(sentences)
    summary["ci95"] = bootstrap_rates(sentences, n_boot, seed)
    summary["n_videos"] = len({s["video"] for s in sentences})
    summary["skipped_empty_ref"] = skipped_empty_ref
    summary["hyp_empty_sentences"] = sum(s["hyp_empty"] for s in sentences)
    for k in ("long_insert_runs", "long_insert_words", "long_delete_runs", "long_delete_words"):
        summary[k] = sum(s[k] for s in sentences)
    summary["sentences_with_long_insert"] = sum(s["long_insert_runs"] > 0 for s in sentences)
    summary["sentences_with_long_delete"] = sum(s["long_delete_runs"] > 0 for s in sentences)
    tagged = [s for s in sentences if s["tagged"]]
    summary["tagged_group"] = _rates(tagged)
    summary["untagged_group"] = _rates([s for s in sentences if not s["tagged"]])
    summary["by_length"] = {label: _rates([s for s in sentences if s["length_bin"] == label])
                            for _, _, label in LENGTH_BINS}
    summary["top_confusions"] = [[r, h, c] for (r, h), c in pairs.most_common(top_k)]
    summary["top_deleted"] = deleted.most_common(top_k)
    summary["top_inserted"] = inserted.most_common(top_k)
    return summary, sentences, pairs


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.2f}"


def render_markdown(header: dict, results: dict, top_k_md: int) -> str:
    lines = [f"# Phân tích lỗi clean-test — {header['name']}", "",
             f"Tham chiếu: {header['ref_counts']} (ref_mode={header['ref_mode']}). "
             f"Ngưỡng chuỗi chèn/xoá dài: ≥{header['run_len']} từ. CI95 bootstrap theo video, "
             f"{header['n_boot']} mẫu.", "",
             "## Phân rã WER (% số từ tham chiếu; số tuyệt đối trong ngoặc; CI95)", "",
             "| Mô hình | WER | S-thanh | S-dấu phụ | S-khác | D | I |", "|---|---|---|---|---|---|---|"]
    for m, s in results.items():
        cells = []
        for k in ["errors"] + CATEGORIES:
            lo, hi = s["ci95"][k]
            cells.append(f"{_pct(s['rates'][k])} ({s['counts'][k]}) [{_pct(lo)}, {_pct(hi)}]")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", f"Số từ tham chiếu: {next(iter(results.values()))['n_ref_words']}; "
              f"số câu: {next(iter(results.values()))['n_sentences']}; "
              f"số video: {next(iter(results.values()))['n_videos']}.", "",
              "## Câu rỗng / chuỗi chèn-xoá dài", "",
              "| Mô hình | hyp rỗng | câu có chèn dài | từ chèn dài | câu có xoá dài | từ xoá dài |",
              "|---|---|---|---|---|---|"]
    for m, s in results.items():
        lines.append(f"| {m} | {s['hyp_empty_sentences']} | {s['sentences_with_long_insert']} | "
                     f"{s['long_insert_words']} | {s['sentences_with_long_delete']} | {s['long_delete_words']} |")
    lines += ["", "## Nhóm ghi chú tên riêng / tiếng Anh", "",
              "| Mô hình | WER nhóm có ghi chú (câu, từ) | WER nhóm còn lại (câu, từ) |", "|---|---|---|"]
    for m, s in results.items():
        t, u = s["tagged_group"], s["untagged_group"]
        lines.append(f"| {m} | {_pct(t['rates']['errors'])} ({t['n_sentences']}, {t['n_ref_words']}) | "
                     f"{_pct(u['rates']['errors'])} ({u['n_sentences']}, {u['n_ref_words']}) |")
    labels = [label for _, _, label in LENGTH_BINS]
    lines += ["", "## WER theo độ dài câu (số từ ref)", "",
              "| Mô hình | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for m, s in results.items():
        lines.append(f"| {m} | " + " | ".join(
            f"{_pct(s['by_length'][b]['rates']['errors'])} ({s['by_length'][b]['n_sentences']} câu)"
            for b in labels) + " |")
    lines += ["", f"## Top {top_k_md} cặp nhầm lẫn (ref → hyp: số lần)", ""]
    for m, s in results.items():
        pairs = ", ".join(f"{r}→{h}: {c}" for r, h, c in s["top_confusions"][:top_k_md])
        lines.append(f"- **{m}**: {pairs}")
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--predictions", required=True, help="predictions.tsv (cột split, index, hyp_<mô hình>)")
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--name", help="tên thư mục đầu ra (mặc định: tên thư mục chứa predictions)")
    p.add_argument("--out_dir", help=f"ghi đè thư mục đầu ra (mặc định {OUT_ROOT}/<name>)")
    p.add_argument("--ref_mode", choices=["auto", "pseudo", "corrected_only"], default="auto",
                   help="auto: corrected_text nếu có, không thì pseudo_label; pseudo: luôn nhãn giả "
                        "(đối chiếu số cũ); corrected_only: chỉ câu đã hiệu đính")
    p.add_argument("--models", nargs="*", help="tập con mô hình (mặc định mọi cột hyp_*)")
    p.add_argument("--run_len", type=int, default=5)
    p.add_argument("--top_k", type=int, default=30)
    p.add_argument("--n_boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    models, preds = load_predictions(args.predictions)
    models = args.models or models
    refs = load_references(args.manifest, args.ref_mode)
    missing = [k for k in refs if k not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} câu trong manifest không có trong predictions, vd. {missing[:3]}")
    if not refs:
        raise SystemExit("Không còn câu nào với ref_mode này")
    name = args.name or Path(args.predictions).parent.name
    out_dir = Path(args.out_dir or f"{OUT_ROOT}/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_counts = dict(Counter(m["ref_source"] for m in refs.values()))
    results, all_sentences, all_pairs = {}, [], []
    for m in models:
        summary, sentences, pairs = analyze_model(m, preds, refs, args.run_len, args.top_k, args.n_boot, args.seed)
        results[m] = summary
        all_sentences += sentences
        all_pairs += [(m, r, h, c) for (r, h), c in pairs.most_common()]

    header = {"name": name, "predictions": args.predictions, "manifest": args.manifest, "ref_mode": args.ref_mode,
              "ref_counts": ref_counts, "n_tagged": sum(m["tagged"] for m in refs.values()),
              "run_len": args.run_len, "n_boot": args.n_boot, "seed": args.seed,
              "length_bins": [label for _, _, label in LENGTH_BINS]}
    (out_dir / "summary.json").write_text(json.dumps({"header": header, "models": results}, ensure_ascii=False,
                                                     indent=1), encoding="utf-8")
    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_sentences", "n_videos", "n_ref_words"]
                   + [f"{k}_{x}" for k in ["errors"] + CATEGORIES for x in ("count", "rate", "ci_low", "ci_high")]
                   + ["hyp_empty_sentences", "sentences_with_long_insert", "long_insert_words",
                      "sentences_with_long_delete", "long_delete_words", "tagged_wer", "tagged_n_sentences"])
        for m, s in results.items():
            w.writerow([m, s["n_sentences"], s["n_videos"], s["n_ref_words"]]
                       + [v for k in ["errors"] + CATEGORIES
                          for v in (s["counts"][k], s["rates"][k], *s["ci95"][k])]
                       + [s["hyp_empty_sentences"], s["sentences_with_long_insert"], s["long_insert_words"],
                          s["sentences_with_long_delete"], s["long_delete_words"],
                          s["tagged_group"]["rates"]["errors"], s["tagged_group"]["n_sentences"]])
    with open(out_dir / "per_sentence.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_sentences[0]))
        w.writeheader()
        w.writerows(all_sentences)
    with open(out_dir / "confusions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "ref_word", "hyp_word", "count", "category"])
        w.writerows([(m, r, h, c, classify_substitution(r, h)) for m, r, h, c in all_pairs])
    md = render_markdown(header, results, top_k_md=10)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Ghi vào {out_dir}")


if __name__ == "__main__":
    main()
