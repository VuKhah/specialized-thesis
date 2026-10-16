"""Chấm WER trên clean-test đã hiệu đính, tách nhóm từ Việt / từ ngoại (thêm 2026-10-15).

Quy tắc chốt trước khi xem điểm (người dùng duyệt 15/10; từ đệm/lặp từ người dùng giao AI chọn theo chuẩn):
- Cách viết thay thế: `configs/scoring/alt_spellings.tsv`, đổi về dạng chuẩn ở cả ref và hyp (như ánh xạ GLM của
  NIST SCTK; Hub-5NE 1997 eval plan cũng cho phép dạng thay thế liệt kê sẵn).
- Từ đệm phi từ vựng (ờ, ơ, ừm…; + chỗ người dùng gán `từ đệm` trong bảng từ): bỏ khỏi ref và hyp. Hub-5NE 1997:
  khớp hoặc bị xoá đều không tính lỗi; bộ chuẩn hoá của Whisper (Radford 2022) bỏ hẳn ở cả hai phía. Từ có nghĩa
  (ừ = "vâng", à/á/ạ trợ từ, ồ) vẫn chấm như từ thường — Hub5 cũng chấm "uh-huh", "oh" như từ.
- Lặp từ: chấm như từ thường (người nói thật sự nói hai lần; không chuẩn nào bỏ).
- Từ vấp dở (fragment, vd. *in* trước *independence* #153): tuỳ chọn — bị xoá không tính lỗi (Hub-5NE 1997).
- Nhóm từ: ref theo `clean_test_word_tags.tsv` (tiếng anh / tên riêng / viết tắt → "ngoại"); từ chèn thêm ở hyp
  xếp nhóm theo dạng chữ (âm tiết Việt hợp lệ hoặc có dấu Việt → Việt). Mỗi lỗi thuộc đúng 1 nhóm → lỗi hai nhóm cộng lại = lỗi
  chung; mẫu số mỗi nhóm = số từ ref của nhóm đó.

    python -m src.evaluation.score_clean_test
"""

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import jiwer
import numpy as np

from src.data.correct_clean_test import OURS_VISIBLE_FROM
from src.evaluation.bootstrap import two_level_bootstrap
from src.evaluation.text_normalize import normalize_text
from src.evaluation.vi_syllable import is_valid_syllable, is_vi_marked

MANIFEST = Path("data/processed/clean_test_manifest.json")
TAGS = Path("data/processed/clean_test_word_tags.tsv")
ALT = Path("configs/scoring/alt_spellings.tsv")
PRED_0810 = Path("reports/results/final_compare_2026-10-08/predictions.tsv")
# Giải mã greedy + beam/LM của 3 run 40 ms seed 42 (15/10; tái lập 203/203 bằng `src.evaluation.lm_decode` 16/10).
HYP_40MS = Path("reports/results/decode_v2a_s4_2026-10-15/cleantest_hyps.tsv")
# Seed 43/44 (L3): `lm_decode decode` trên log-prob của kernel `dump_seeds_40ms`; cột v2a_<mô hình>_seed<N>_{greedy,lm}.
HYP_SEEDS = Path("reports/results/decode_v2a_s4_seeds/cleantest_hyps.tsv")
SEED_MODELS = [("conformer_relpos", "Conformer relPE"), ("mamba", "B1 Mamba"), ("conextbimamba", "ConExtBiMamba")]
SEEDS = (43, 44)
# Kernel `scripts/kaggle/dump_greedy_10ms/` (16/10): 4 checkpoint ep37 10 ms + Parakeet fine-tune, khoá "clean_test|split:index".
HYP_10MS = Path("reports/results/dump_greedy_10ms_2026-10-16/greedy.json")
OUT = Path("reports/results/clean_test_scored_2026-10-16")

FILLERS = {"ờ", "ơ", "ừm", "ưm", "ầm", "hừm", "hmm", "hm", "uh", "um"}
FRAGMENTS = {(153, 9)}  # (số thứ tự câu 1-based, vị trí từ trong ref đã chuẩn hoá) — người dùng xác nhận 15/10
FOREIGN = {"tiếng anh", "tên riêng", "viết tắt"}

SYSTEMS = [  # (cột, tên hiển thị, nhóm so sánh)
    ("pseudo_label", "Nhãn giả Zipformer (tham khảo)", "nhãn"),
    ("raw_conformer", "Conformer v1 10 ms ep28", "train từ đầu v1"),
    ("raw_mamba", "B1 Mamba v1 10 ms ep28", "train từ đầu v1"),
    ("raw_conextbimamba", "ConExtBiMamba v1 10 ms ep28", "train từ đầu v1"),
    ("v2a10_conformer_relpos", "Conformer relPE 10 ms ep37", "train từ đầu 10 ms"),
    ("v2a10_mamba", "B1 Mamba 10 ms ep37", "train từ đầu 10 ms"),
    ("v2a10_conextbimamba", "ConExtBiMamba 10 ms ep37", "train từ đầu 10 ms"),
    ("v2a10_conformer", "Conformer không PE 10 ms ep37 (ablation)", "train từ đầu 10 ms"),
    ("v2a_conformer_relpos_greedy", "Conformer relPE 40 ms ep37 · greedy", "train từ đầu 40 ms"),
    ("v2a_mamba_greedy", "B1 Mamba 40 ms ep37 · greedy", "train từ đầu 40 ms"),
    ("v2a_conextbimamba_greedy", "ConExtBiMamba 40 ms ep37 · greedy", "train từ đầu 40 ms"),
    ("v2a_conformer_relpos_lm", "Conformer relPE 40 ms ep37 · beam+LM", "40 ms + giải mã"),
    ("v2a_mamba_lm", "B1 Mamba 40 ms ep37 · beam+LM", "40 ms + giải mã"),
    ("v2a_conextbimamba_lm", "ConExtBiMamba 40 ms ep37 · beam+LM", "40 ms + giải mã"),
    ("raw_parakeet_fp32", "Parakeet-CTC-0.6B-vi zero-shot", "pre-train"),
    ("raw_phowhisper", "PhoWhisper-small zero-shot", "pre-train"),
    ("v2a10_parakeet_ft", "Parakeet-CTC-0.6B-vi fine-tune (nhãn giả) best.pt", "pre-train"),
]


def load_alt() -> list[tuple[tuple[str, ...], str]]:
    pairs = []
    for line in ALT.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        canon, variants = line.split("#")[0].rstrip().split("\t")[:2]
        for v in variants.split("|"):
            pairs.append((tuple(v.split()), canon.strip()))
    return sorted(pairs, key=lambda p: -len(p[0]))  # khớp dài nhất trước


_TONES = "̣́̀̉̃"  # sắc, huyền, hỏi, ngã, nặng (dạng tổ hợp NFD)
_TONE_OPEN = re.compile(r"(?<!q)(oa|oe|uy)$")


def canon_tone(word: str) -> str:
    """Thống nhất vị trí dấu thanh ở vần mở oa/oe/uy (hóa/hoá, khỏe/khoẻ, thủy/thuỷ) về kiểu đặt ở nguyên âm sau.

    Hai kiểu là cùng một từ, cùng cách đọc (kiểu cũ đặt dấu ở o/u, kiểu mới ở a/e/y) — chấm khác nhau là đo định dạng
    chữ viết, không đo nhận dạng. Người dùng chốt 16/10 (quyết định #11), trước khi có kết quả seed 43/44; đo trên
    output đã có: ảnh hưởng ≤ 4 từ/hệ thống. Chỉ áp khi chấm clean-test, **không** đổi `normalize_text` (dùng lúc
    train). Bỏ qua sau `q` (qua/que/quy: `qu` là âm đầu, dấu luôn ở nguyên âm sau) và vần đóng (hoàn, khuyết: không có
    hai cách viết)."""
    d = unicodedata.normalize("NFD", word)
    tones = [c for c in d if c in _TONES]
    if len(tones) != 1:
        return word
    base = "".join(c for c in d if c not in _TONES)
    if not _TONE_OPEN.search(unicodedata.normalize("NFC", base)):
        return word
    return unicodedata.normalize("NFC", base + tones[0])


def canonicalize(tokens: list[tuple[str, str, bool]], alt) -> list[tuple[str, str, bool]]:
    """tokens = (từ, nhóm, tuỳ chọn). Gộp dạng nhiều chữ về dạng chuẩn; nhóm "ngoại" nếu có phần nào ngoại.
    Vị trí dấu thanh thống nhất trước (`canon_tone`) để bảng cách viết thay thế chỉ cần một dạng."""
    tokens = [(canon_tone(t[0]), *t[1:]) for t in tokens]
    out, i = [], 0
    while i < len(tokens):
        for var, canon in alt:
            n = len(var)
            if tuple(t[0] for t in tokens[i:i + n]) == var:
                grp = "ngoại" if any(t[1] == "ngoại" for t in tokens[i:i + n]) or not is_valid_syllable(canon) else "việt"
                out.append((canon, grp, False))
                i += n
                break
        else:
            out.append(tokens[i])
            i += 1
    return out


def ref_tokens(n: int, text: str, tags: dict, alt, drop_fillers: bool = True):
    toks = []
    for i, w in enumerate(normalize_text(text).split()):
        label = tags.get(f"{w}@#{n}:{i}") or tags.get(w)
        if drop_fillers and (w in FILLERS or label == "từ đệm"):
            continue
        grp = "ngoại" if label in FOREIGN or (label is None and not is_valid_syllable(w)) else "việt"
        toks.append((w, grp, (n, i) in FRAGMENTS))
    return canonicalize(toks, alt)


def hyp_tokens(text: str, alt, drop_fillers: bool = True):
    # Từ dính (nghỉơi) không phải âm tiết hợp lệ nhưng có dấu Việt → nhóm Việt, không đổ lỗi sang nhóm ngoại.
    toks = [(w, "việt" if is_valid_syllable(w) or is_vi_marked(w) else "ngoại", False)
            for w in normalize_text(text).split() if not (drop_fillers and w in FILLERS)]
    return canonicalize(toks, alt)


def count_errors(ref, hyp) -> dict:
    e = {"việt": 0, "ngoại": 0}
    if not ref:
        for w in hyp:
            e[w[1]] += 1
        return e
    if not hyp:
        for w in ref:
            e[w[1]] += 1
        return e
    out = jiwer.process_words(" ".join(t[0] for t in ref), " ".join(t[0] for t in hyp))
    chunks = out.alignments[0]
    for k, ch in enumerate(chunks):
        if ch.type in ("substitute", "delete"):
            for t in ref[ch.ref_start_idx:ch.ref_end_idx]:
                e[t[1]] += 1
        elif ch.type == "insert":
            # Một từ ngoại bị phiên thành nhiều âm tiết (marketing → ma két tinh) = 1 thay thế + chèn kề bên → chèn
            # tính cho nhóm của từ ref bị thay thế liền kề; chèn đứng riêng thì xếp theo dạng chữ của từ chèn.
            near = [c for c in (chunks[k - 1] if k else None, chunks[k + 1] if k + 1 < len(chunks) else None)
                    if c is not None and c.type == "substitute"]
            for t in hyp[ch.hyp_start_idx:ch.hyp_end_idx]:
                if near:
                    c = near[0]
                    e[ref[c.ref_end_idx - 1 if c is chunks[k - 1] else c.ref_start_idx][1]] += 1
                else:
                    e[t[1]] += 1
    return e


def score_sentence(ref, hyp) -> tuple[dict, dict]:
    """Từ vấp dở là tuỳ chọn: thử bỏ khỏi ref, lấy phương án ít lỗi hơn (= bị xoá không tính lỗi)."""
    best = count_errors(ref, hyp)
    if any(t[2] for t in ref):
        ref2 = [t for t in ref if not t[2]]
        alt_e = count_errors(ref2, hyp)
        if sum(alt_e.values()) < sum(best.values()):
            return alt_e, {g: sum(t[1] == g for t in ref2) for g in ("việt", "ngoại")}
    return best, {g: sum(t[1] == g for t in ref) for g in ("việt", "ngoại")}


def block_bootstrap(err: np.ndarray, words: np.ndarray, n_boot: int = 10_000, seed: int = 42):
    """Khối = video (như src/evaluation/bootstrap.py, Liu & Peng arXiv:1912.09508)."""
    idx = np.random.default_rng(seed).integers(0, len(err), size=(n_boot, len(err)))
    return err[idx].sum(1) / words[idx].sum(1)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    samples = json.loads(MANIFEST.read_text(encoding="utf-8"))["samples"]
    with TAGS.open(encoding="utf-8-sig") as f:
        tags = {r["key"]: r["label"] for r in csv.DictReader(f, delimiter="\t") if r["label"]}
    alt = load_alt()
    hyps = {f"{s['split']}:{s['index']}": {"pseudo_label": s["pseudo_label"]} for s in samples}
    with PRED_0810.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            hyps[f"{r['split']}:{r['index']}"].update({k: v for k, v in r.items() if k.startswith("raw_")})
    with HYP_40MS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            hyps[r["key"]].update({k: v for k, v in r.items() if k != "key"})
    systems = list(SYSTEMS)
    if HYP_SEEDS.exists():
        with HYP_SEEDS.open(encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="	"):
                hyps[r["key"]].update({k: v for k, v in r.items() if k != "key"})
        for mode, label, group in (("greedy", "greedy", "train từ đầu 40 ms"), ("lm", "beam+LM", "40 ms + giải mã")):
            systems += [(f"v2a_{m}_seed{sd}_{mode}", f"{name} 40 ms ep37 seed {sd} · {label}", group)
                        for m, name in SEED_MODELS for sd in SEEDS]
    for model, greedy in json.loads(HYP_10MS.read_text(encoding="utf-8")).items():
        for key, hyp in greedy.items():
            set_name, sent = key.split("|", 1)
            if set_name == "clean_test":
                hyps[sent][f"v2a10_{model}"] = hyp

    videos = sorted({s["source"] for s in samples})
    pre_mark = [not s.get("corrected_at") or s["corrected_at"] < OURS_VISIBLE_FROM for s in samples]
    results, per_sent = {}, []
    for col, name, group in systems:
        rows = {}
        for drop in (True, False):
            E = np.zeros((len(samples), 2)); W = np.zeros((len(samples), 2))
            for n, s in enumerate(samples, 1):
                key = f"{s['split']}:{s['index']}"
                ref = ref_tokens(n, s["corrected_text"], tags, alt, drop)
                e, w = score_sentence(ref, hyp_tokens(hyps[key][col], alt, drop))
                E[n - 1] = [e["việt"], e["ngoại"]]; W[n - 1] = [w["việt"], w["ngoại"]]
                if drop:
                    per_sent.append({"system": col, "key": key, "video": s["source"], "err_vi": e["việt"],
                                     "err_fx": e["ngoại"], "words_vi": w["việt"], "words_fx": w["ngoại"]})
            rows[drop] = (E, W)
        E, W = rows[True]
        vid = np.array([videos.index(s["source"]) for s in samples])
        ev = np.array([E[vid == v].sum(0) for v in range(len(videos))]); wv = np.array([W[vid == v].sum(0) for v in range(len(videos))])
        boot = block_bootstrap(ev.sum(1), wv.sum(1))
        pm = np.array(pre_mark)
        E0, W0 = rows[False]
        results[col] = {
            "name": name, "group": group,
            "wer": E.sum() / W.sum(), "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "wer_vi": E[:, 0].sum() / W[:, 0].sum(), "wer_fx": E[:, 1].sum() / W[:, 1].sum(),
            "wer_pre_mark": E[pm].sum() / W[pm].sum(), "wer_no_filler_rule": E0.sum() / W0.sum(),
            "words": W.sum(0).tolist(), "n_pre_mark": int(pm.sum()), "_ev": ev.sum(1), "_wv": wv.sum(1),
        }

    # So cặp trong cùng nhóm (khối video, cùng mẫu lấy lại).
    pairs = [("v2a_mamba_greedy", "v2a_conextbimamba_greedy"), ("v2a_mamba_greedy", "v2a_conformer_relpos_greedy"),
             ("v2a_conextbimamba_greedy", "v2a_conformer_relpos_greedy"), ("v2a_mamba_lm", "v2a_conextbimamba_lm"),
             ("v2a_mamba_lm", "v2a_conformer_relpos_lm"), ("v2a_conextbimamba_lm", "v2a_conformer_relpos_lm"),
             ("raw_mamba", "raw_conextbimamba"), ("raw_mamba", "raw_conformer"), ("raw_conextbimamba", "raw_conformer"),
             ("raw_parakeet_fp32", "raw_phowhisper"),
             ("v2a10_mamba", "v2a10_conformer_relpos"), ("v2a10_conextbimamba", "v2a10_conformer_relpos"),
             ("v2a10_mamba", "v2a10_conextbimamba"), ("v2a10_conformer_relpos", "v2a10_conformer"),
             ("v2a10_mamba", "v2a10_conformer"), ("v2a10_conextbimamba", "v2a10_conformer"),
             ("v2a10_parakeet_ft", "raw_parakeet_fp32")]
    idx = np.random.default_rng(42).integers(0, len(videos), size=(10_000, len(videos)))
    paired = []
    for a, b in pairs:
        ra, rb = results[a], results[b]
        d = (ra["_ev"][idx].sum(1) - rb["_ev"][idx].sum(1)) / ra["_wv"][idx].sum(1)
        paired.append({"a": a, "b": b, "diff": ra["wer"] - rb["wer"],
                       "ci": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]})

    # 40 ms nhiều seed: bootstrap hai tầng video × seed (note gap 9.6 điều 4); seed 42 là cột không hậu tố.
    paired_seeds = []
    for mode in ("greedy", "lm"):
        cols = {m: [c for c in [f"v2a_{m}_{mode}"] + [f"v2a_{m}_seed{sd}_{mode}" for sd in SEEDS] if c in results]
                for m, _ in SEED_MODELS}
        if min(len(v) for v in cols.values()) < 2:
            continue
        for a, b in [("mamba", "conextbimamba"), ("mamba", "conformer_relpos"), ("conextbimamba", "conformer_relpos")]:
            r = two_level_bootstrap(np.stack([results[c]["_ev"] for c in cols[a]]),
                                    np.stack([results[c]["_ev"] for c in cols[b]]), results[cols[a][0]]["_wv"])
            paired_seeds.append({"mode": mode, "a": a, "b": b, "seeds_a": cols[a], "seeds_b": cols[b], **r})

    OUT.mkdir(parents=True, exist_ok=True)
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}
    meta = {"n_sentences": len(samples), "n_videos": len(videos), "fillers": sorted(FILLERS),
            "fragments": sorted(FRAGMENTS), "alt_spellings": str(ALT), "ours_visible_from": OURS_VISIBLE_FROM}
    (OUT / "results.json").write_text(json.dumps({"meta": meta, "systems": clean, "paired": paired, "paired_seeds": paired_seeds}, ensure_ascii=False,
                                                 indent=2, default=float), encoding="utf-8")
    with (OUT / "per_sentence.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(per_sent[0]), delimiter="\t"); w.writeheader(); w.writerows(per_sent)

    r0 = next(iter(results.values()))
    print(f"{len(samples)} câu · {len(videos)} video · từ ref: Việt {r0['words'][0]:.0f} · ngoại {r0['words'][1]:.0f} "
          f"· {r0['n_pre_mark']} câu hiệu đính trước mốc {OURS_VISIBLE_FROM}")
    print(f"{'Hệ thống':42} {'WER':>6} {'CI 95%':>15} {'Việt':>6} {'ngoại':>6} {'trước mốc':>9} {'không bỏ đệm':>12}")
    for col, *_ in systems:
        r = results[col]
        print(f"{r['name']:42} {r['wer']:6.4f} [{r['ci'][0]:.4f};{r['ci'][1]:.4f}] {r['wer_vi']:6.4f} {r['wer_fx']:6.4f}"
              f" {r['wer_pre_mark']:9.4f} {r['wer_no_filler_rule']:12.4f}")
    print("So cặp (A − B, CI 95% khối video):")
    for p in paired:
        print(f"  {p['a']} − {p['b']}: {p['diff']:+.4f} [{p['ci'][0]:+.4f}; {p['ci'][1]:+.4f}]")

    if paired_seeds:
        print("40 ms nhiều seed — bootstrap hai tầng video × seed (A − B, CI 95%; trung bình ± sd từng mô hình):")
        for p in paired_seeds:
            print(f"  [{p['mode']}] {p['a']} {p['wer_a_mean']:.4f}±{p['wer_a_sd']:.4f} − {p['b']} "
                  f"{p['wer_b_mean']:.4f}±{p['wer_b_sd']:.4f}: {p['diff']:+.4f} [{p['ci_low']:+.4f}; {p['ci_high']:+.4f}]"
                  f" · cặp seed cùng dấu {p['pairs_same_sign']}/{len(p['pairs'])}")

if __name__ == "__main__":
    main()
