"""Bảng so sánh + cổng giữ/bỏ sau train thử nhóm train từ đầu (2026-10-04).

Đọc thư mục `<root>/<experiment_name>/` do train.py ghi (`metrics.jsonl`,
`eval_<tập>_epoch<k>.jsonl`), xuất `trial_report.md` + `trial_report.json`:

1. Chất lượng — WER/CER val và val_unseen kèm CI95 bootstrap theo video, hiệu
   WER so với mô hình mốc (Conformer-12M) theo cặp.
2. Học và ổn định — đường WER theo epoch, mức giảm WER ở epoch cuối, khoảng
   cách val_unseen − val, train/val loss, step bị bỏ, grad norm, câu rỗng,
   cơ cấu lỗi S/D/I (xoá nhiều + câu rỗng = dấu hiệu sụp về blank của CTC).
3. Tài nguyên — tham số, s/step, VRAM, ước GPU-giờ train full, RTF eval (proxy).
4. Cổng loại cứng G0-G3 tự đánh giá theo ngưỡng ở `GATES`; tiêu chí cân nhắc
   (chất lượng, Pareto) chỉ báo số — người dùng/GVHD quyết.

Ngưỡng trong `GATES` là đề xuất của dự án (docs/notes/lineup_preparation.md mục
Train thử), không lấy từ tài liệu; đổi ngưỡng thì sửa ở đó và ở đây.

    python -m src.evaluation.trial_report <thư mục chứa các experiment> [--reference conformer_ctc_baseline]
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import jiwer

from src.data.vietsuperspeech_dataset import manifest_rows
from src.evaluation.bootstrap import block_bootstrap_wer, load_records, paired_block_bootstrap
from src.evaluation.text_normalize import normalize_text

FULL_TRAIN_MANIFESTS = ["train_shard0", "train_shard1", "train_shard2", "train_shard3"]
FULL_EPOCHS = 30
EVAL_SETS = ["val", "val_unseen"]
GATES = {
    "usable_max_ref_wer": 0.95,  # G0: WER val_unseen của mốc phải dưới mức này, không thì chưa kết luận
    "max_empty_hyp_frac": 0.05,  # G1: câu giải mã rỗng ở epoch cuối (val_unseen) — CTC sụp về blank
    "max_skipped_frac": 0.01,  # G2: step bị bỏ (loss không hữu hạn + AMP) / tổng step
    "max_full_gpu_hours": 30.0,  # G3: ước train full 30 epoch, mỗi mô hình
    "max_vram_gib": 14.0,  # G3: VRAM đỉnh mỗi T4 (16 GB)
}


def _eval_files(exp: Path) -> dict[str, dict[int, Path]]:
    out: dict[str, dict[int, Path]] = {}
    for p in exp.glob("eval_*_epoch*.jsonl"):
        m = re.fullmatch(r"eval_(.+)_epoch(\d+)\.jsonl", p.name)
        if m:
            out.setdefault(m.group(1), {})[int(m.group(2))] = p
    return out


def error_profile(records: list[dict]) -> dict:
    refs = [normalize_text(r["ref"]) for r in records]
    hyps = [normalize_text(r["hyp"]) for r in records]
    keep = [i for i, r in enumerate(refs) if r]
    refs, hyps = [refs[i] for i in keep], [hyps[i] for i in keep]
    w = jiwer.process_words(refs, hyps)
    n_ref = w.substitutions + w.deletions + w.hits
    return {
        "cer": jiwer.cer(refs, hyps),
        "sub_rate": w.substitutions / n_ref, "del_rate": w.deletions / n_ref, "ins_rate": w.insertions / n_ref,
        "empty_hyp_frac": sum(not h for h in hyps) / len(hyps),
    }


def _durations(name: str) -> dict[tuple[str, int], float]:
    try:
        return {(r["split"], r["index"]): r["duration"] for r in manifest_rows([name])}
    except FileNotFoundError:
        return {}


def analyze_experiment(exp: Path, full_steps_per_epoch: int | None) -> dict:
    metrics = [json.loads(l) for l in (exp / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    last = metrics[-1]
    files = _eval_files(exp)
    res = {"name": exp.name, "epochs_done": last["epoch"] + 1, "metrics": metrics, "eval": {}}
    for name in EVAL_SETS:
        if name not in files:
            continue
        epoch = max(files[name])
        records = load_records(str(files[name][epoch]))
        ci = block_bootstrap_wer(records)
        curve = [m.get(f"wer_{name}") for m in metrics]
        prev = curve[-2] if len(curve) >= 2 else None
        durs = _durations(name)
        audio_s = sum(durs.get((r["split"], r["index"]), 0.0) for r in records)
        eval_s = last.get(f"eval_seconds_{name}")
        res["eval"][name] = {
            "epoch": epoch, "file": str(files[name][epoch]), **ci, **error_profile(records),
            "wer_curve": curve, "loss": last.get(f"loss_{name}"),
            "last_epoch_rel_gain": (prev - curve[-1]) / prev if prev else None,
            # Proxy: batch 8/GPU × số GPU, gồm cả đọc dữ liệu — không phải phép đo RQ2.
            "rtf_eval_proxy": eval_s / audio_s if eval_s and audio_s else None,
        }
    steps = sum(m["steps"] for m in metrics)
    skipped = sum(m["nonfinite_loss_steps"] + m["amp_skipped_steps"] for m in metrics)
    # Epoch có step bỏ (loss không hữu hạn) không chạy backward/optimizer → s/step thấp giả: train thử B1
    # 2026-10-05 epoch cuối bỏ 708/711 step, ghi 0,114 thay vì ~0,47 s/step. Lấy epoch sạch cuối cùng;
    # không có thì epoch ít step bỏ nhất.
    clean = [m for m in metrics if m["nonfinite_loss_steps"] == 0]
    timing = clean[-1] if clean else min(metrics, key=lambda m: m["nonfinite_loss_steps"] / max(m["steps"], 1))
    s_step = timing["s_per_step"]
    eval_s = sum(last.get(f"eval_seconds_{n}", 0.0) for n in EVAL_SETS)
    res.update({
        "encoder_params": last.get("encoder_params"),
        "train_loss": last["train_loss"], "train_loss_curve": [m["train_loss"] for m in metrics],
        "skipped_steps": skipped, "total_steps": steps, "skipped_frac": skipped / max(steps, 1),
        "grad_norm_mean": last["grad_norm_mean"], "grad_norm_max": max(m["grad_norm_max"] for m in metrics),
        "s_per_step": s_step, "s_per_step_epoch": timing["epoch"], "peak_vram_gib": max((m["peak_vram_gib"] or 0.0) for m in metrics) or None,
        "full_gpu_hours": (s_step * full_steps_per_epoch + eval_s) * FULL_EPOCHS / 3600
        if full_steps_per_epoch else None,
    })
    if "val" in res["eval"] and "val_unseen" in res["eval"]:
        res["unseen_gap"] = res["eval"]["val_unseen"]["wer"] - res["eval"]["val"]["wer"]
    return res


def apply_gates(res: dict, ref_usable: bool | None) -> dict:
    wer_curve = res["eval"].get("val", {}).get("wer_curve") or []
    clean = [w for w in wer_curve if w is not None]
    finite_loss = all(math.isfinite(x) for x in res["train_loss_curve"])
    g2 = (finite_loss and res["skipped_frac"] <= GATES["max_skipped_frac"]
          and len(clean) >= 2 and clean[-1] < clean[0])
    vram = res["peak_vram_gib"]
    g3 = (res["full_gpu_hours"] is not None and res["full_gpu_hours"] <= GATES["max_full_gpu_hours"]
          and (vram is None or vram <= GATES["max_vram_gib"]))
    empty = res["eval"].get("val_unseen", {}).get("empty_hyp_frac")
    g1 = None if empty is None else empty <= GATES["max_empty_hyp_frac"]
    return {"G0_usable": ref_usable, "G1_no_collapse": g1, "G2_stable": g2, "G3_resources": g3}


def build_report(root: Path, reference: str) -> dict:
    try:
        n_full = len(manifest_rows(FULL_TRAIN_MANIFESTS))
    except FileNotFoundError:
        n_full = None
    exps = sorted(p for p in root.iterdir() if (p / "metrics.jsonl").exists())
    results = {}
    for exp in exps:
        world = json.loads((exp / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]).get("world_size", 2)
        batch = json.loads((exp / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]).get("global_batch", 16)
        full_steps = math.ceil(math.ceil(n_full / world) / (batch // world)) if n_full else None
        results[exp.name] = analyze_experiment(exp, full_steps)

    ref = results.get(reference)
    ref_usable = None
    if ref and "val_unseen" in ref["eval"]:
        ref_usable = ref["eval"]["val_unseen"]["wer"] < GATES["usable_max_ref_wer"]
    for name, r in results.items():
        r["gates"] = apply_gates(r, ref_usable)
        r["vs_reference"] = {}
        if ref and name != reference:
            for s in EVAL_SETS:
                if s in r["eval"] and s in ref["eval"]:
                    try:
                        r["vs_reference"][s] = paired_block_bootstrap(load_records(r["eval"][s]["file"]),
                                                                      load_records(ref["eval"][s]["file"]))
                    except ValueError as e:
                        r["vs_reference"][s] = {"error": str(e)}
    # Pareto theo (WER val_unseen, ước GPU-giờ train full): mô hình bị trội khi có
    # mô hình khác không tệ hơn ở cả hai và tốt hơn ở ít nhất một.
    pts = {n: (r["eval"].get("val_unseen", {}).get("wer"), r["full_gpu_hours"]) for n, r in results.items()}
    for n, (w, h) in pts.items():
        results[n]["pareto_dominated_by"] = [
            m for m, (w2, h2) in pts.items()
            if m != n and None not in (w, h, w2, h2) and w2 <= w and h2 <= h and (w2 < w or h2 < h)
        ]
    return {"reference": reference, "gates": GATES, "n_full_train": n_full, "models": results}


def _f(x, fmt="{:.4f}"):
    return "–" if x is None else fmt.format(x)


def _ci(e):
    return f"{e['wer']:.4f} [{e['ci_low']:.4f}, {e['ci_high']:.4f}]" if e else "–"


def to_markdown(rep: dict) -> str:
    models, ref = rep["models"], rep["reference"]
    names = sorted(models, key=lambda n: (n != ref, n))
    L = [f"# Báo cáo train thử — mốc: `{ref}`", ""]
    L += ["## 1. Chất lượng (epoch cuối, CI95 bootstrap theo khối video)", "",
          "| Mô hình | Epoch | WER val | WER val_unseen | CER val_unseen | Δ WER val so với mốc [CI95] "
          "| Δ WER val_unseen so với mốc [CI95] | P(tốt hơn mốc, val_unseen) |",
          "|---|---|---|---|---|---|---|---|"]

    def delta(n, s):
        d = models[n]["vs_reference"].get(s)
        if d and "error" not in d:
            return f"{d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
        return "(mốc)" if n == ref else "–"

    for n in names:
        r = models[n]
        v, u = r["eval"].get("val"), r["eval"].get("val_unseen")
        d = r["vs_reference"].get("val_unseen")
        p = _f(d.get("p_a_better"), "{:.2f}") if d and "error" not in d else "–"
        L.append(f"| {n} | {r['epochs_done']} | {_ci(v)} | {_ci(u)} | {_f(u and u['cer'])} | "
                 f"{delta(n, 'val')} | {delta(n, 'val_unseen')} | {p} |")
    L += ["", "## 2. Học và ổn định", "",
          "| Mô hình | WER val_unseen theo epoch | Giảm WER epoch cuối | val_unseen − val | Train loss | Val loss | Step bị bỏ | Grad norm tb / max | Câu rỗng | S / D / I (val_unseen) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        r = models[n]
        u = r["eval"].get("val_unseen") or {}
        curve = " → ".join(_f(w, "{:.3f}") for w in u.get("wer_curve", []))
        sdi = (f"{u['sub_rate']:.3f} / {u['del_rate']:.3f} / {u['ins_rate']:.3f}" if u else "–")
        L.append(f"| {n} | {curve or '–'} | {_f(u.get('last_epoch_rel_gain'), '{:.1%}')} | "
                 f"{_f(r.get('unseen_gap'), '{:+.4f}')} | {r['train_loss']:.3f} | {_f(u.get('loss'), '{:.3f}')} | "
                 f"{r['skipped_steps']}/{r['total_steps']} | {r['grad_norm_mean']:.2f} / {r['grad_norm_max']:.2f} | "
                 f"{_f(u.get('empty_hyp_frac'), '{:.1%}')} | {sdi} |")
    L += ["", "## 3. Tài nguyên", "",
          "| Mô hình | Tham số encoder | s/step | VRAM đỉnh (GiB) | Ước GPU-giờ train full 30 epoch | RTF eval (proxy) | Bị trội Pareto bởi |",
          "|---|---|---|---|---|---|---|"]
    for n in names:
        r = models[n]
        u = r["eval"].get("val_unseen") or {}
        L.append(f"| {n} | {_f(r['encoder_params'], '{:,}')} | {r['s_per_step']:.3f} | "
                 f"{_f(r['peak_vram_gib'], '{:.2f}')} | {_f(r['full_gpu_hours'], '{:.1f}')} | "
                 f"{_f(u.get('rtf_eval_proxy'), '{:.4f}')} | {', '.join(r['pareto_dominated_by']) or 'không'} |")
    g = rep["gates"]
    L += ["", "## 4. Cổng loại cứng (tự đánh giá) — quyết định giữ/bỏ: người dùng/GVHD", "",
          f"G0 lần chạy dùng được (WER val_unseen mốc < {g['usable_max_ref_wer']}) · "
          f"G1 không sụp về blank (câu rỗng ≤ {g['max_empty_hyp_frac']:.0%}) · "
          f"G2 ổn định (loss hữu hạn, step bỏ ≤ {g['max_skipped_frac']:.0%}, WER val giảm) · "
          f"G3 tài nguyên (≤ {g['max_full_gpu_hours']:g} GPU-giờ, VRAM ≤ {g['max_vram_gib']:g} GiB)", "",
          "| Mô hình | G0 | G1 | G2 | G3 |", "|---|---|---|---|---|"]
    sym = {True: "✓", False: "✗", None: "?"}
    for n in names:
        gt = models[n]["gates"]
        L.append(f"| {n} | {sym[gt['G0_usable']]} | {sym[gt['G1_no_collapse']]} | {sym[gt['G2_stable']]} "
                 f"| {sym[gt['G3_resources']]} |")
    L += ["", "Ghi chú: 1 seed, 5 epoch trên 1/4 dữ liệu — thứ hạng có thể đảo khi train full. "
          "RTF eval là proxy (batch, 2 GPU, gồm đọc dữ liệu), không phải phép đo RQ2."]
    return "\n".join(L) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", help="Thư mục chứa các <experiment_name>/ (ckpt_dir của train.py)")
    parser.add_argument("--reference", default="conformer_ctc_baseline")
    parser.add_argument("--out", default=None, help="Thư mục ghi trial_report.{md,json} (mặc định = root)")
    args = parser.parse_args()
    root = Path(args.root)
    rep = build_report(root, args.reference)
    out = Path(args.out or root)
    out.mkdir(parents=True, exist_ok=True)
    md = to_markdown(rep)
    (out / "trial_report.md").write_text(md, encoding="utf-8")
    (out / "trial_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
