"""Chấm test GigaSpeech 2 tiếng Việt (phương án B, 2026-10-19) — quy tắc `docs/notes/so_sanh_hien_hanh_2026-10-19.md`
mục B.2, khoá khi người dùng xem kết quả lần đầu (19/10).

Dùng lại nguyên hàm chấm của `score_clean_test.py` (normalize_text + canon_tone + alt_spellings + bỏ từ đệm, đếm lỗi
Việt/ngoại); khác duy nhất: không có bảng gán nhóm tay → từ ref là âm tiết Việt hợp lệ thì "việt", còn lại "ngoại".
CI 95%: bootstrap khối theo 30 nhóm id của test.tsv (thư mục `test/<nhóm>/`); 40 ms 3 seed: bootstrap hai tầng như
clean-test. Hệ thống nào chưa có output thì bỏ qua — chạy lại khi đủ.

    python -m src.evaluation.score_gs2
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.evaluation.bootstrap import two_level_bootstrap
from src.evaluation.score_clean_test import hyp_tokens, load_alt, ref_tokens, score_sentence
from src.evaluation.text_normalize import normalize_text
from src.evaluation.vi_syllable import sticky_words

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

TSV = Path("data/raw/gigaspeech2/data/vi/test.tsv")  # gitignore — dữ liệu gated, không commit
RES = Path("reports/results/gs2_eval_2026-10-19")
CLEAN = Path("reports/results/clean_test_scored_2026-10-16_chung/results.json")
OUT = Path("reports/results/gs2_scored_2026-10-19")
ENC = {"conformer_relpos": "Conformer relPE", "mamba": "B1 Mamba", "conextbimamba": "ConExtBiMamba",
       "conformer": "Conformer không PE (ablation)"}


def load_systems() -> dict[str, tuple[str, str, dict]]:
    """{cột: (tên, nhóm, {id: hyp})}; cột trùng tên cột clean-test để ghép bảng trong/ngoài kênh."""
    out = {}

    def from_greedy(path: Path, rename):
        if path.exists():
            for m, hyps in json.loads(path.read_text(encoding="utf-8")).items():
                col, name, group = rename(m)
                out[col] = (name, group, {k.split("|", 1)[1]: v for k, v in hyps.items() if k.startswith("gs2_test|")})

    def r10(m):
        return f"v2a10_{m}", f"{ENC[m]} 10 ms ep37", "train từ đầu 10 ms"

    def r40(m):
        if m == "parakeet_ft":
            return "v2a10_parakeet_ft", "Parakeet-CTC-0.6B-vi fine-tune (nhãn giả)", "pre-train"
        enc, seed = m.rsplit("_seed", 1)
        suffix = "" if seed == "42" else f"_seed{seed}"
        return f"v2a_{enc}{suffix}_greedy", f"{ENC[enc]} 40 ms ep37 seed {seed} · greedy", "train từ đầu 40 ms"

    def rpre(m):
        return {"phowhisper": ("raw_phowhisper", "PhoWhisper-small zero-shot", "pre-train"),
                "parakeet_fp32": ("raw_parakeet_fp32", "Parakeet-CTC-0.6B-vi zero-shot", "pre-train")}[m]

    from_greedy(RES / "10ms/greedy.json", r10)
    from_greedy(RES / "40ms/greedy.json", r40)
    for sub in ("phowhisper", "parakeet_zs"):
        from_greedy(RES / sub / "greedy.json", rpre)
    lm = RES / "40ms/gs2_hyps_lm.tsv"
    if lm.exists():
        with lm.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for c in rows[0]:
            if c.endswith("_lm"):
                enc, seed = c[len("v2a_"):-len("_lm")].rsplit("_seed", 1)
                col = f"v2a_{enc}{'' if seed == '42' else f'_seed{seed}'}_lm"
                out[col] = (f"{ENC[enc]} 40 ms ep37 seed {seed} · beam+LM", "40 ms + giải mã", {r["key"]: r[c] for r in rows})
    return out


def main() -> None:
    refs = dict(line.rstrip("\n").split("\t", 1) for line in TSV.read_text(encoding="utf-8").splitlines())
    keys = list(refs)
    groups = sorted({k.split("-")[0] for k in keys}, key=int)
    gi = {g: i for i, g in enumerate(groups)}
    alt = load_alt()
    ref_tok = {k: ref_tokens(-1, refs[k], {}, alt) for k in keys}
    systems = load_systems()

    results, per_sentence = {}, []
    for col, (name, group, hyps) in systems.items():
        missing = [k for k in keys if k not in hyps]
        if missing:
            print(f"BỎ {col}: thiếu {len(missing)} câu", flush=True)
            continue
        ev = np.zeros(len(groups))
        wv = np.zeros(len(groups))
        tot = defaultdict(int)
        glued = 0
        for k in keys:
            e, w = score_sentence(ref_tok[k], hyp_tokens(hyps[k], alt))
            g = gi[k.split("-")[0]]
            ev[g] += e["việt"] + e["ngoại"]
            wv[g] += w["việt"] + w["ngoại"]
            for x in ("việt", "ngoại"):
                tot[f"e_{x}"] += e[x]
                tot[f"w_{x}"] += w[x]
            glued += len(sticky_words(normalize_text(hyps[k])))
            per_sentence.append([col, k, e["việt"], e["ngoại"], w["việt"], w["ngoại"]])
        idx = np.random.default_rng(42).integers(0, len(groups), size=(10_000, len(groups)))
        boot = ev[idx].sum(1) / wv[idx].sum(1)
        n_words = tot["w_việt"] + tot["w_ngoại"]
        results[col] = {"name": name, "group": group, "wer": ev.sum() / wv.sum(),
                        "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                        "wer_vi": tot["e_việt"] / tot["w_việt"], "wer_fx": tot["e_ngoại"] / max(tot["w_ngoại"], 1),
                        "glued_per_1000": 1000 * glued / n_words, "_ev": ev, "_wv": wv}

    # 40 ms 3 seed: cặp encoder (hai tầng) và so với pre-train — cùng tiêu chí clean-test.
    paired = []
    for mode in ("greedy", "lm"):
        cols = {m: [c for c in (f"v2a_{m}_{mode}", f"v2a_{m}_seed43_{mode}", f"v2a_{m}_seed44_{mode}") if c in results]
                for m in ("mamba", "conextbimamba", "conformer_relpos")}
        if min(map(len, cols.values())) < 3:
            continue
        wv = results[cols["mamba"][0]]["_wv"]
        others = [(a, b) for a, b in (("mamba", "conextbimamba"), ("mamba", "conformer_relpos"),
                                      ("conextbimamba", "conformer_relpos"))]
        others += [(a, p) for a in cols for p in ("raw_parakeet_fp32", "raw_phowhisper", "v2a10_parakeet_ft") if p in results]
        for a, b in others:
            eb = np.stack([results[c]["_ev"] for c in cols[b]]) if b in cols else results[b]["_ev"][None]
            r = two_level_bootstrap(np.stack([results[c]["_ev"] for c in cols[a]]), eb, wv)
            paired.append({"mode": mode, "a": a, "b": b, **{k: r[k] for k in ("wer_a_mean", "wer_a_sd", "wer_b_mean",
                                                                                "diff", "ci_low", "ci_high")}})

    # 10 ms (1 seed): bootstrap cặp theo nhóm, cùng mẫu lấy lại như clean-test.
    idx = np.random.default_rng(42).integers(0, len(groups), size=(10_000, len(groups)))
    for a, b in (("v2a10_mamba", "v2a10_conformer_relpos"), ("v2a10_conextbimamba", "v2a10_conformer_relpos"),
                 ("v2a10_mamba", "v2a10_conextbimamba"), ("v2a10_conformer_relpos", "v2a10_conformer")):
        if a in results and b in results:
            ra, rb = results[a], results[b]
            d = (ra["_ev"][idx].sum(1) - rb["_ev"][idx].sum(1)) / ra["_wv"][idx].sum(1)
            paired.append({"mode": "10ms", "a": a, "b": b, "wer_a_mean": ra["wer"], "wer_a_sd": None,
                           "wer_b_mean": rb["wer"], "diff": ra["wer"] - rb["wer"],
                           "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975))})

    clean = json.loads(CLEAN.read_text(encoding="utf-8"))["systems"] if CLEAN.exists() else {}
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"meta": {"n_sentences": len(keys), "n_groups": len(groups), "rules": "so_sanh_hien_hanh_2026-10-19.md B.2"},
           "systems": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()},
           "paired_40ms_3seed": paired}
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    with (OUT / "per_sentence.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["system", "key", "err_vi", "err_fx", "words_vi", "words_fx"])
        w.writerows(per_sentence)

    print(f"GigaSpeech 2 vi test: {len(keys)} câu, {len(groups)} nhóm, từ ref (sau bỏ từ đệm) "
          f"{int(sum(len(t) for t in ref_tok.values()))}")
    print(f"{'hệ thống':52s} {'GS2':>7s} {'CI 95%':>17s} {'Việt':>6s} {'ngoại':>6s} {'dính‰':>6s} {'clean':>7s} {'Δ':>7s}")
    for col, r in sorted(results.items(), key=lambda kv: (kv[1]["group"], kv[1]["wer"])):
        c = clean.get(col, {}).get("wer")
        print(f"{r['name'][:52]:52s} {r['wer']:7.4f} [{r['ci'][0]:.4f};{r['ci'][1]:.4f}] {r['wer_vi']:6.3f} "
              f"{r['wer_fx']:6.3f} {r['glued_per_1000']:6.1f} {c if c is None else f'{c:7.4f}'} "
              f"{'' if c is None else f'{r['wer'] - c:+7.4f}'}")
    for p in paired:
        verdict = "thấp hơn" if p["ci_high"] < 0 else "cao hơn" if p["ci_low"] > 0 else "không phân biệt được"
        print(f"[{p['mode']}] {p['a']} vs {p['b']}: {p['wer_a_mean']:.4f} vs {p['wer_b_mean']:.4f}  Δ {p['diff']:+.4f} "
              f"[{p['ci_low']:+.4f}; {p['ci_high']:+.4f}]  {verdict}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
