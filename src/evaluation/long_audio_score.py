"""Chấm WER khi encoder nhận nguyên cửa sổ audio dài (nối các đoạn gốc liên tiếp) so với giải mã từng đoạn gốc.

    python -m src.evaluation.long_audio_score [--dir reports/results/long_audio_2026-10-27]

Quy tắc chốt trước khi chạy: `docs/notes/ket_qua/audio_dai_nguyen_chuoi_quy_tac_2026-10-27.md` (mục 3: thống kê, cách
đọc). Đầu vào `<dir>/hyps/<model>_s<seed>.jsonl` (mỗi dòng một cửa sổ, L ∈ seg/L1/L5/all, greedy); tham chiếu = nhãn giả
nối theo `keys` từ `data/processed/longform/chains.json`. Ghi `<dir>/results.json`; hình: `scripts/fig_audio_dai.py`.
Dữ liệu về dần → file/seed/cửa sổ thiếu được bỏ qua và liệt kê trong output, không dừng.
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

MODELS = ("b1_mamba_40ms", "conextbimamba_40ms", "conformer_relpe_40ms")
SEEDS = (42, 43, 44)
LONG = ("L1", "L5", "all")
LEVELS = ("seg",) + LONG
STATUSES = ("ok", "skip_mem", "error")
CHAINS = Path("data/processed/longform/chains.json")
DEFAULT_DIR = Path("reports/results/long_audio_2026-10-27")
RULE = "docs/notes/ket_qua/audio_dai_nguyen_chuoi_quy_tac_2026-10-27.md"
N_BOOT, BOOT_SEED = 10_000, 42
UNSURE = "không phân biệt được"


def load_runs(hyp_dir: Path) -> tuple[dict, list[str], dict]:
    """{(model, seed): {L: {wid: bản ghi}}}, wid = (tên chuỗi, keys) — cùng cửa sổ ở mọi seed/encoder."""
    runs, missing, bad = {}, [], {}
    for m in MODELS:
        for s in SEEDS:
            p = hyp_dir / f"{m}_s{s}.jsonl"
            if not p.exists():
                missing.append(p.name)
                continue
            run, n_bad = {L: {} for L in LEVELS}, 0
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    # Chạy lại một cửa sổ (nối thêm vào file) → bản ghi sau thay bản trước.
                    run[r["L"]][(r["chain"], tuple(r["keys"]))] = r
                except (json.JSONDecodeError, KeyError, TypeError):
                    n_bad += 1  # dòng cuối đang ghi dở khi file còn được bổ sung
            if n_bad:
                bad[p.name] = n_bad
            runs[(m, s)] = run
    return runs, missing, bad


def score_runs(runs: dict, text: dict, alt) -> dict:
    """Chấm mọi cửa sổ ok: {(model, seed): {L: {wid: (lỗi, số từ ref)}}}; cách chấm của P1 sản phẩm."""
    from src.demo.longform import _chain_counts
    unknown = {k for run in runs.values() for recs in run.values() for wid in recs for k in wid[1] if k not in text}
    if unknown:
        raise SystemExit(f"{len(unknown)} key không có trong chains.json (vd. {sorted(unknown)[:3]}) → sai bản chains.json?")
    return {ms: {L: {wid: _chain_counts(r.get("hyp") or "", [text[k] for k in wid[1]], alt)
                     for wid, r in recs.items() if r.get("status") == "ok"}
                 for L, recs in run.items()}
            for ms, run in runs.items()}


def valid_windows(runs: dict, scored: dict, m: str, seeds: list[int], L: str) -> list:
    """Cửa sổ L ok ở mọi seed có mặt và mọi đoạn gốc của nó cũng ok ở seed đó → hai vế cùng tập đoạn, seed ghép cặp."""
    return sorted(wid for wid in runs[(m, seeds[0])][L] if all(
        wid in scored[(m, s)][L] and all((wid[0], (k,)) in scored[(m, s)]["seg"] for k in wid[1]) for s in seeds))


def per_video(runs: dict, scored: dict, m: str, seeds: list[int], L: str, wids: list):
    """Lỗi vế L và vế seg theo [seed, video], số từ ref theo video; đếm cửa sổ có số từ hai vế lệch nhau."""
    video = {wid: runs[(m, seeds[0])][L][wid]["video"] for wid in wids}
    videos = sorted(set(video.values()))
    vi = {v: j for j, v in enumerate(videos)}
    e_long, e_seg = np.zeros((len(seeds), len(videos))), np.zeros((len(seeds), len(videos)))
    words, mismatch = np.zeros(len(videos)), 0
    for wid in wids:
        j = vi[video[wid]]
        for i, s in enumerate(seeds):
            e, n = scored[(m, s)][L][wid]
            segs = [scored[(m, s)]["seg"][(wid[0], (k,))] for k in wid[1]]
            e_long[i, j] += e
            e_seg[i, j] += sum(x[0] for x in segs)
            if i == 0:
                words[j] += n
                mismatch += n != sum(x[1] for x in segs)
    return videos, e_long, e_seg, words, mismatch


def delta_ci(e_long: np.ndarray, e_seg: np.ndarray, words: np.ndarray) -> list[float]:
    """Rút lại video; seed cố định và ghép cặp (cùng checkpoint hai vế) → Δ từng seed trên cùng mẫu video rồi lấy
    trung bình seed. Không rút seed như `two_level_bootstrap`: câu hỏi chính là tác dụng của L với chính mô hình đó."""
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(words), size=(N_BOOT, len(words)))
    d = ((e_long - e_seg)[:, idx].sum(2) / words[idx].sum(1)).mean(0)
    return np.quantile(d, [0.025, 0.975]).tolist()


def verdict(ci: list[float], pos: str, neg: str) -> str:
    return pos if ci[0] > 0 else neg if ci[1] < 0 else UNSURE


def _stats(vals) -> dict | None:
    a = np.array([v for v in vals if v is not None], float)
    return {"mean": float(a.mean()), "max": float(a.max())} if len(a) else None


def perf_summary(runs: dict, m: str, seeds: list[int]) -> tuple[dict, list, dict]:
    """Thời gian, RAM đỉnh theo L (mọi seed, cửa sổ ok) — chỉ mô tả; điểm từng cửa sổ dài lấy seed đầu cho hình."""
    out = {}
    for L in LEVELS:
        rs = [r for s in seeds for r in runs[(m, s)][L].values() if r.get("status") == "ok"]
        if rs:
            out[L] = {"n": len(rs), "duration_s": _stats(r.get("duration_s") for r in rs),
                      "sec": _stats(r.get("sec") for r in rs), "peak_mb": _stats(r.get("peak_mb") for r in rs),
                      "rtf": _stats(r["sec"] / r["duration_s"] for r in rs if r.get("sec") and r.get("duration_s"))}
    first = runs[(m, seeds[0])]
    points = [[L, r.get("duration_s"), r.get("sec"), r.get("peak_mb")] for L in LONG for r in first[L].values()
              if r.get("status") == "ok"]
    all_recs = [r for s in seeds for L in LEVELS for r in runs[(m, s)][L].values()]
    ok_dur = [r["duration_s"] for r in all_recs if r.get("status") == "ok" and r.get("duration_s")]
    skip_dur = [r["duration_s"] for r in all_recs if r.get("status") == "skip_mem" and r.get("duration_s")]
    limit = {"max_ok_duration_s": max(ok_dur, default=None), "min_skip_mem_duration_s": min(skip_dur, default=None)}
    return out, points, limit


def pct(x: float) -> str:
    return f"{x * 100:+.2f}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="thư mục có hyps/; results.json ghi vào đây")
    ap.add_argument("--chains", type=Path, default=CHAINS)
    a = ap.parse_args()
    from src.evaluation.bootstrap import two_level_bootstrap
    from src.evaluation.score_clean_test import load_alt

    chains = json.loads(a.chains.read_text(encoding="utf-8"))
    text = {k: t for c in chains for k, t in zip(c["keys"], c["texts"])}
    runs, missing, bad = load_runs(a.dir / "hyps")
    if not runs:
        raise SystemExit(f"Không có file hyp nào trong {a.dir / 'hyps'}")
    scored = score_runs(runs, text, load_alt())
    notes = [f"thiếu file: {', '.join(missing)}"] if missing else []
    notes += [f"{f}: bỏ {n} dòng hỏng" for f, n in bad.items()]

    # Cửa sổ "kỳ vọng" ở mỗi L = hợp mọi file; thiếu so với đó = file đang ghi dở hoặc chạy lỗi giữa chừng.
    union = {L: set().union(*(run[L] for run in runs.values())) for L in LEVELS}
    res = {"rule": RULE, "dir": str(a.dir).replace("\\", "/"), "decode": "greedy CTC", "reference": "nhãn giả val_unseen",
           "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
           "data": {"n_chains": len(chains), "n_videos": len({c["video"] for c in chains}),
                    "hours": sum(c["duration_s"] for c in chains) / 3600,
                    "n_windows_expected": {L: len(union[L]) for L in LEVELS}},
           "models": {}, "pairs": {}, "notes": notes}
    valid = {}
    print(f"{'encoder':22} {'L':>4} {'seed':>4} {'cửa sổ':>6} {'video':>5} {'giờ':>5} {'WER seg':>8} {'WER L':>8} "
          f"{'ΔWER (đ%)':>10} {'CI 95%':>17}  đọc · Δ từng seed")
    for m in MODELS:
        seeds = [s for s in SEEDS if (m, s) in runs]
        if not seeds:
            notes.append(f"{m}: không có seed nào → bỏ")
            continue
        status = {}
        for s in seeds:
            status[s] = {}
            for L in LEVELS:
                recs = runs[(m, s)][L]
                cnt = {st: sum(r.get("status") == st for r in recs.values()) for st in STATUSES}
                cnt["other"] = len(recs) - sum(cnt.values())
                cnt["missing"] = len(union[L] - set(recs))
                status[s][L] = cnt
                if cnt["missing"]:
                    notes.append(f"{m} s{s} {L}: thiếu {cnt['missing']}/{len(union[L])} cửa sổ (file chưa đủ?)")
        perf, points, limit = perf_summary(runs, m, seeds)
        md = {"seeds": seeds, "status": status, "delta": {}, "perf": perf, "perf_points": points, "limit": limit}
        for L in LONG:
            wids = valid_windows(runs, scored, m, seeds, L)
            valid[(m, L)] = wids
            if not wids:
                notes.append(f"{m} {L}: không có cửa sổ hợp lệ")
                continue
            videos, e_long, e_seg, words, mismatch = per_video(runs, scored, m, seeds, L, wids)
            if mismatch:
                notes.append(f"{m} {L}: {mismatch} cửa sổ có số từ ref vế L ≠ tổng vế seg")
            wl, ws = e_long.sum(1) / words.sum(), e_seg.sum(1) / words.sum()
            ci = delta_ci(e_long, e_seg, words)
            first = runs[(m, seeds[0])][L]
            md["delta"][L] = {
                "n_windows": len(wids), "n_segments": sum(len(w[1]) for w in wids),
                "n_chains": len({w[0] for w in wids}), "n_videos": len(videos), "words": int(words.sum()),
                "hours": sum(first[w].get("duration_s") or 0 for w in wids) / 3600,
                "wer_seg_seeds": ws.tolist(), "wer_long_seeds": wl.tolist(), "delta_seeds": (wl - ws).tolist(),
                "wer_seg": float(ws.mean()), "wer_long": float(wl.mean()), "delta": float((wl - ws).mean()),
                "ci": ci, "verdict": verdict(ci, "tăng", "giảm"),
                "per_video": {"videos": videos, "words": words.tolist(), "err_long": e_long.tolist(),
                              "err_seg": e_seg.tolist()}}
            r = md["delta"][L]
            print(f"{m:22} {L:>4} {len(seeds):>4} {r['n_windows']:>6} {r['n_videos']:>5} {r['hours']:>5.2f} "
                  f"{r['wer_seg'] * 100:>8.2f} {r['wer_long'] * 100:>8.2f} {pct(r['delta']):>10} "
                  f"[{pct(ci[0])}; {pct(ci[1])}]  {r['verdict']} · {' '.join(pct(x) for x in r['delta_seeds'])}")
        res["models"][m] = md

    print("\nSo encoder: ΔWER(a) − ΔWER(b), chỉ cửa sổ cả hai đều chạy được; bootstrap hai tầng video × seed")
    for L in LONG:
        res["pairs"][L] = []
        for ma, mb in combinations(MODELS, 2):
            common = sorted(set(valid.get((ma, L), [])) & set(valid.get((mb, L), [])))
            if not common:
                continue
            sa, sb = res["models"][ma]["seeds"], res["models"][mb]["seeds"]
            va, al, as_, w = per_video(runs, scored, ma, sa, L, common)[:4]
            vb, bl, bs = per_video(runs, scored, mb, sb, L, common)[:3]
            assert va == vb  # cùng cửa sổ → cùng video, cùng ref
            t = two_level_bootstrap(al - as_, bl - bs, w, N_BOOT, BOOT_SEED)
            ci = [t["ci_low"], t["ci_high"]]
            p = {"a": ma, "b": mb, "n_windows": len(common), "n_videos": len(va), "words": int(w.sum()),
                 "delta_a": t["wer_a_mean"], "delta_b": t["wer_b_mean"], "delta_a_seeds": t["wer_a_seeds"],
                 "delta_b_seeds": t["wer_b_seeds"], "diff": t["diff"], "ci": ci, "verdict": verdict(ci, "khác", "khác"),
                 "pairs_same_sign": t["pairs_same_sign"], "n_pairs": len(t["pairs"])}
            res["pairs"][L].append(p)
            print(f"  {L:>3} {ma} − {mb}: {pct(p['diff'])} [{pct(ci[0])}; {pct(ci[1])}]  {p['verdict']}"
                  f" ({p['n_windows']} cửa sổ, {p['n_videos']} video)")

    print("\nThời gian / RAM đỉnh theo L (mô tả; mọi seed, cửa sổ ok)")
    for m, md in res["models"].items():
        for L, p in md["perf"].items():
            f = lambda k, nd=1: f"{p[k]['mean']:.{nd}f}/{p[k]['max']:.{nd}f}" if p[k] else "–"  # noqa: E731
            print(f"  {m:22} {L:>4} n={p['n']:<5} dài(s) {f('duration_s')} · giây {f('sec', 2)} · "
                  f"RAM(MB) {f('peak_mb', 0)} · RTF {f('rtf', 3)}  (tb/max)")
        print(f"  {m:22} skip_mem: {sum(c[L]['skip_mem'] for c in md['status'].values() for L in LEVELS)} · "
              f"error: {sum(c[L]['error'] for c in md['status'].values() for L in LEVELS)}")
    if notes:
        print("\nGhi chú:\n  " + "\n  ".join(notes))
    out = a.dir / "results.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
