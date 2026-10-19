"""So nhóm train từ đầu (40 ms, 3 seed) với mô hình pre-train trên clean-test — phương án A, người dùng chốt
2026-10-19 (trả lời GVHD 17/10 "so với mấy model hiện thời thế nào?").

Chỉ dùng lại số đã chấm ở lần chấm chung (`per_sentence.tsv`), không chấm lại. CI: bootstrap hai tầng video × seed
(`two_level_bootstrap`, cùng hàm với tiêu chí note gap 9.6); bên pre-train là 1 "seed" (giải mã tất định). Hai nhóm
khác điều kiện (dữ liệu, cỡ mô hình, nhãn) — đây là so sánh mô tả, không phải so encoder.

    python -m src.evaluation.compare_pretrained
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.evaluation.bootstrap import two_level_bootstrap

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SCORED = Path("reports/results/clean_test_scored_2026-10-16_chung")
OUT = Path("reports/results/compare_pretrained_2026-10-19")
OURS = {"Conformer relPE": "conformer_relpos", "B1 Mamba": "mamba", "ConExtBiMamba": "conextbimamba"}
PRETRAINED = {"Parakeet-CTC-0.6B zero-shot": "raw_parakeet_fp32", "PhoWhisper-small zero-shot": "raw_phowhisper",
              "Parakeet-CTC-0.6B fine-tune": "v2a10_parakeet_ft"}


def load_per_video() -> tuple[dict[str, np.ndarray], np.ndarray]:
    """{hệ thống: số lỗi theo video}, số từ ref theo video — video sắp như `score_clean_test` (sorted source)."""
    err, words = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    with open(SCORED / "per_sentence.tsv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            err[r["system"]][r["video"]] += int(r["err_vi"]) + int(r["err_fx"])
            words[r["system"]][r["video"]] += int(r["words_vi"]) + int(r["words_fx"])
    videos = sorted(words["pseudo_label"])
    w = np.array([words["pseudo_label"][v] for v in videos])
    return {s: np.array([e[v] for v in videos]) for s, e in err.items()}, w


def main() -> None:
    err, words = load_per_video()
    official = json.loads((SCORED / "results.json").read_text(encoding="utf-8"))
    # Tự kiểm: tái lập đúng một cặp đã công bố (cùng hàm, 3 seed mỗi bên) trước khi tính cặp mới.
    ref = next(p for p in official["paired_seeds"] if p["mode"] == "greedy" and p["a"] == "mamba" and p["b"] == "conformer_relpos")
    chk = two_level_bootstrap(np.stack([err[c] for c in ref["seeds_a"]]), np.stack([err[c] for c in ref["seeds_b"]]), words)
    assert abs(chk["ci_low"] - ref["ci_low"]) < 1e-12 and abs(chk["ci_high"] - ref["ci_high"]) < 1e-12, (chk, ref)

    rows = []
    for mode, label in (("greedy", "greedy"), ("lm", "beam+LM")):
        for name, m in OURS.items():
            cols = [f"v2a_{m}_{mode}", f"v2a_{m}_seed43_{mode}", f"v2a_{m}_seed44_{mode}"]
            for pname, p in PRETRAINED.items():
                r = two_level_bootstrap(np.stack([err[c] for c in cols]), err[p][None], words)
                verdict = ("thấp hơn" if r["ci_high"] < 0 else "cao hơn" if r["ci_low"] > 0 else "không phân biệt được")
                rows.append({"ours": f"{name} 40 ms · {label}", "pretrained": pname,
                             "wer_ours_mean": r["wer_a_mean"], "wer_ours_sd": r["wer_a_sd"],
                             "wer_pretrained": r["wer_b_mean"], "diff": r["diff"],
                             "ci_low": r["ci_low"], "ci_high": r["ci_high"], "verdict": verdict})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "pairs.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"tự kiểm: tái lập CI cặp B1–relPE greedy [{ref['ci_low']:.4f}; {ref['ci_high']:.4f}] — khớp")
    for r in rows:
        print(f"{r['ours']:34s} vs {r['pretrained']:28s} {r['wer_ours_mean']:.4f}±{r['wer_ours_sd']:.4f} "
              f"vs {r['wer_pretrained']:.4f}  Δ {r['diff']:+.4f} [{r['ci_low']:+.4f}; {r['ci_high']:+.4f}]  "
              f"{r['verdict']}")
    print(f"→ {OUT / 'pairs.csv'}")


if __name__ == "__main__":
    main()
