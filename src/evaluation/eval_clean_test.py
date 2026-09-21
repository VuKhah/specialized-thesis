"""Đo WER của một checkpoint trên clean-test (`data/processed/clean_test_manifest.json`).

Chạy từ gốc repo, cùng một lệnh cho cả hai thí nghiệm:
    python -m src.evaluation.eval_clean_test --config configs/model_conformer.yaml
    python -m src.evaluation.eval_clean_test --config configs/model_mamba.yaml

Kết quả ghi ra `reports/results/<experiment_name>_clean_test.json` (WER, S/D/I,
và ref/hyp từng câu để phân tích lỗi RQ3).

Cạm bẫy khi đọc kết quả:
- Reference = `corrected_text` nếu đã hiệu đính, không thì quay về `pseudo_label`
  (do Zipformer sinh, không phải người). Số câu thuộc mỗi loại được in ra và
  lưu vào file kết quả — chỉ báo cáo WER "clean-test" khi phần lớn là
  `corrected_text`.
- clean-test lấy từ `validation`, mà `best.pt` cũng chọn theo WER `validation`
  → số này lạc quan hơn WER trên dữ liệu chưa từng dùng để chọn model.
- batch_size mặc định 1: Mamba chưa mask padding nên kết quả sẽ phụ thuộc cách
  ghép batch. `train.py` eval theo batch nên số ở đây có thể lệch nhẹ số đó.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.vietsuperspeech_dataset import VietSuperSpeechDataset, collate_fn
from src.evaluation.wer import compute_wer_report
from src.models.ctc_model import CTCASRModel
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.train import CHECKPOINT_ROOT, build_encoder

RESULTS_DIR = Path("reports/results")


def normalize(text: str) -> str:
    # Transcript gốc và tokenizer đều viết HOA; hiệu đính tay có thể lệch hoa/thường
    # hoặc thừa khoảng trắng — không nên tính là lỗi nhận dạng.
    return re.sub(r"\s+", " ", text).strip().upper()


def load_references(samples: list[dict]) -> tuple[list[str], list[str]]:
    refs, sources = [], []
    for s in samples:
        corrected = s["corrected_text"].strip()
        if corrected:
            refs.append(normalize(corrected))
            sources.append("corrected_text")
        else:
            refs.append(normalize(s["pseudo_label"]))
            sources.append("pseudo_label")
    return refs, sources


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Mặc định: checkpoints/<experiment_name>/best.pt")
    parser.add_argument("--manifest", default="data/processed/clean_test_manifest.json")
    parser.add_argument("--tokenizer_model", default="configs/tokenizer.model")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--allow_random_init",
        action="store_true",
        help="Cho phép chạy khi chưa có checkpoint (trọng số ngẫu nhiên) — chỉ để kiểm tra script, WER vô nghĩa",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    experiment_name = cfg["experiment_name"]
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    samples = manifest["samples"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BPETokenizer(args.tokenizer_model)
    model = CTCASRModel(encoder=build_encoder(cfg), vocab_size=tokenizer.vocab_size).to(device)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_ROOT / experiment_name / "best.pt"
    ckpt_epoch = None
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        ckpt_epoch = state["epoch"]
        print(f"[{experiment_name}] checkpoint: {ckpt_path} (epoch {ckpt_epoch})")
    elif args.allow_random_init:
        print(f"[{experiment_name}] CẢNH BÁO: không có {ckpt_path} — dùng trọng số ngẫu nhiên, WER vô nghĩa.")
    else:
        sys.exit(f"Không thấy checkpoint {ckpt_path}. Train trước, hoặc truyền --checkpoint / --allow_random_init.")
    model.eval()

    references, ref_sources = load_references(samples)
    n_corrected = ref_sources.count("corrected_text")
    print(f"Reference: {n_corrected}/{len(samples)} câu là corrected_text, còn lại là pseudo_label")

    # Manifest lưu `index` trong split gốc; audio tải lẻ nếu chưa có trong cache (chỉ ~250 file).
    dataset = VietSuperSpeechDataset(split=manifest["split"], tokenizer=None)
    loader = DataLoader(
        Subset(dataset, [s["index"] for s in samples]),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    hypotheses = []
    for batch in loader:
        pred_ids = model.greedy_decode(batch["waveform"].to(device), batch["waveform_lengths"].to(device))
        hypotheses.extend(normalize(tokenizer.decode(ids)) for ids in pred_ids)

    report = compute_wer_report(references, hypotheses)
    n_empty = sum(1 for h in hypotheses if not h)
    print(
        f"WER={report['wer']:.4f}  S={report['substitutions']} D={report['deletions']} "
        f"I={report['insertions']} hits={report['hits']}  (hyp rỗng: {n_empty}/{len(hypotheses)})"
    )

    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{experiment_name}_clean_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment_name": experiment_name,
        "checkpoint": str(ckpt_path) if ckpt_epoch is not None else None,
        "checkpoint_epoch": ckpt_epoch,
        "random_init": ckpt_epoch is None,
        "n": len(samples),
        "n_corrected_text": n_corrected,
        "batch_size": args.batch_size,
        "metrics": report,
        "note": "clean-test lấy từ validation, best.pt cũng chọn theo validation (xem docstring eval_clean_test.py)",
        "samples": [
            {
                "index": s["index"],
                "source": s["source"],
                "duration_s": s["duration_s"],
                "ref_source": rs,
                "ref": r,
                "hyp": h,
            }
            for s, rs, r, h in zip(samples, ref_sources, references, hypotheses)
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Đã ghi {out_path}")


if __name__ == "__main__":
    main()
