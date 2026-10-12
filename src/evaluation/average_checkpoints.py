"""Trung bình trọng số nhiều checkpoint `epochs/epochNN.pt` rồi đo WER trên val / val_unseen / clean-test
(P1 kế hoạch 2026-10-12: 3 mô hình v1, epoch 20-29; về sau: các epoch pha giảm của mọi run v2).

Chạy từ gốc repo:
    # trung bình + eval (mặc định cả 3 tập), cột TSV tên `conformer_avg20-29`
    python -m src.evaluation.average_checkpoints --config configs/model_conformer.yaml \
        --ckpt_dir D:/Model/kltn_checkpoints/conformer --epoch_range 20 29 \
        --name conformer_avg20-29 --out_dir reports/results/avg_v1
    # chỉ trung bình, ghi file (vd. để upload Kaggle) — không cần --config
    python -m src.evaluation.average_checkpoints --ckpt_dir <dir> --last 8 --save_avg avg_last8.pt --sets
    # eval một checkpoint có sẵn (best.pt, file trung bình đã ghi) cùng đường đo để so
    python -m src.evaluation.average_checkpoints --config ... --checkpoint <file.pt> --name conformer_best ...

Đầu ra trong --out_dir — chạy nhiều lần (nhiều mô hình/checkpoint) vào CÙNG thư mục thì gộp lại:
- `results.json`: {"runs": {name: {checkpoint, averaged_epochs, sets: {set: WER/S/D/I/CER, n, ...}}}}
- `predictions.tsv`: clean-test, CÙNG định dạng `reports/results/final_compare_2026-10-08/predictions.tsv`
  (split, index, duration_s, ref_source, ref, hyp_<name>..., raw_<name>...) → chấm lại sau khi hiệu đính
  chỉ cần ghép cột hyp với ref mới theo (split, index).
- `eval_<set>_<name>.jsonl`: split, index, video, ref, hyp — đầu vào của `src.evaluation.bootstrap`.

BatchNorm (Conformer và ConExtBiMamba có BN trong conv module của torchaudio; B1 không có): trung bình
CẢ `running_mean`/`running_var` như tham số, KHÔNG tính lại thống kê trên dữ liệu. Đây là cách của
ESPnet (`espnet2/main_funcs/average_nbest_models.py`: cộng mọi khóa của state_dict rồi chia n, khóa kiểu
int như `num_batches_tracked` chỉ cộng dồn) và icefall (`icefall/checkpoint.py::average_checkpoints`:
số thực chia n, số nguyên chia nguyên); ConMamba dùng lại recipe ESPnet (`avg_checkpoints: 10`). Tính lại
BN trên dữ liệu là bước của SWA (Izmailov 2018, `torch.optim.swa_utils.update_bn`), không phải của
checkpoint averaging trong ASR. `num_batches_tracked` lấy của checkpoint cuối: chỉ dùng khi BN
`momentum=None`, torchaudio dùng momentum mặc định 0,1 nên giá trị này không ảnh hưởng eval.

Chọn epoch nào phải chốt TRƯỚC khi xem kết quả và áp Y HỆT cho mọi mô hình cùng nhóm (P1: 10 checkpoint
epoch 20-29); không chọn N theo mô hình, không dùng clean-test để chọn gì cả (chỉ val). Muốn báo thêm N khác
thì báo đủ mọi N cho mọi mô hình.
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


def average_state_dicts(paths: list[Path]) -> dict:
    avg, last = None, None
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        if avg is None:
            avg = {k: v.clone().double() if v.is_floating_point() else v.clone() for k, v in sd.items()}
        else:
            if sd.keys() != avg.keys():
                raise ValueError(f"{p} khác tập tham số với checkpoint đầu")
            for k, v in sd.items():
                if v.is_floating_point():
                    avg[k] += v.double()
        last = sd
    n = len(paths)
    return {k: (v / n).to(last[k].dtype) if v.is_floating_point() else last[k].clone() for k, v in avg.items()}


def select_epochs(ep_dir: Path, last: int | None = None, epoch_range: list[int] | None = None,
                  epochs: list[int] | None = None) -> list[int]:
    available = sorted(int(p.stem.removeprefix("epoch")) for p in ep_dir.glob("epoch*.pt"))
    if epochs:
        chosen = sorted(epochs)
    elif epoch_range:
        chosen = list(range(epoch_range[0], epoch_range[1] + 1))
    elif last:
        chosen = available[-last:]
    else:
        raise SystemExit("cần --last, --epoch_range hoặc --epochs (hoặc --checkpoint)")
    missing = [e for e in chosen if e not in available]
    if missing:
        raise SystemExit(f"thiếu epoch {missing} trong {ep_dir}")
    if last and len(chosen) < last:
        raise SystemExit(f"chỉ có {len(chosen)} epoch trong {ep_dir}, cần {last}")
    return chosen


# --------------------------------------------------------------------------- eval

def _prefetch_audio(rel_paths: list[str]) -> None:
    """Tải song song audio còn thiếu trong AUDIO_CACHE_DIR — để `__getitem__` không tải lẻ từng file
    (~1-4 s/file). Kaggle: tar của 5 dataset đã có gần đủ, phần này chỉ là lưới an toàn."""
    from huggingface_hub import hf_hub_download

    from src.data.vietsuperspeech_dataset import AUDIO_CACHE_DIR, HF_DATASET_ID, HF_REVISION

    missing = [r for r in rel_paths if not (Path(AUDIO_CACHE_DIR) / r).exists()]
    if not missing:
        return
    print(f"tải {len(missing)} file audio còn thiếu vào {AUDIO_CACHE_DIR}", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda r: hf_hub_download(HF_DATASET_ID, r, repo_type="dataset", revision=HF_REVISION,
                                                local_dir=AUDIO_CACHE_DIR), missing))


def _score(refs: list[str], hyps: list[str]) -> dict:
    import jiwer

    from src.evaluation.wer import compute_wer_report

    report = compute_wer_report(refs, hyps)
    report["cer"] = jiwer.cer(refs, hyps)
    report["n"] = len(refs)
    report["n_hyp_empty"] = sum(1 for h in hyps if not h)
    return report


def eval_manifest_set(model, name: str, tokenizer, device, amp: bool, batch_size: int, limit: int) -> tuple[dict, list]:
    """val / val_unseen: đúng `train.evaluate` (theo batch, cùng thứ tự manifest) để số so được với
    `wer_val` ghi lúc train — lệch nhẹ vẫn có thể do ghép batch khác lúc DDP (padding qua conv)."""
    from torch.utils.data import DataLoader

    from src.data.vietsuperspeech_dataset import VietSuperSpeechDataset, collate_fn, manifest_rows
    from src.training.train import evaluate

    rows = manifest_rows([name])
    rows = rows[:limit] if limit else rows
    _prefetch_audio([r["audio"] for r in rows])
    loader = DataLoader(VietSuperSpeechDataset(tokenizer=tokenizer, items=[(r["split"], r["index"]) for r in rows]),
                        batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2,
                        pin_memory=device.type == "cuda")
    t0 = time.time()
    _, loss, records = evaluate(model, loader, rows, tokenizer, device, amp, name=name, log_every=100)
    # `evaluate` trả model về train() khi xong (để vòng train chạy tiếp) — eval tiếp mà quên dòng này là
    # chạy với dropout + SpecAugment.
    model.eval()
    from src.evaluation.text_normalize import normalize_text
    for r in records:
        r["ref"], r["hyp"] = normalize_text(r["ref"]), normalize_text(r["hyp"])
    metrics = _score([r["ref"] for r in records], [r["hyp"] for r in records])
    metrics.update({"ctc_loss": loss, "batch_size": batch_size, "limit": limit or None,
                    "eval_seconds": round(time.time() - t0, 1)})
    return metrics, records


def clean_test_samples(manifest_path: str) -> list[dict]:
    from src.data.vietsuperspeech_dataset import load_excluded
    from src.evaluation.text_normalize import normalize_text

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    excluded = load_excluded()
    out = []
    for s in manifest["samples"]:
        if (s["split"], s["index"]) in excluded:
            continue
        corrected = s["corrected_text"].strip()
        # Như final_compare: ref = corrected_text nếu đã hiệu đính, không thì pseudo_label.
        out.append({**s, "ref_source": "corrected_text" if corrected else "pseudo_label",
                    "ref": normalize_text(corrected or s["pseudo_label"])})
    return out


@torch.no_grad()
def eval_clean_test(model, samples: list[dict], tokenizer, device, amp: bool) -> tuple[dict, list, list]:
    """Batch 1 từng câu, autocast fp16 trên GPU — cùng cách đo với kernel final_compare (`load_ours`)."""
    from src.data.vietsuperspeech_dataset import VietSuperSpeechDataset, video_of
    from src.evaluation.text_normalize import normalize_text
    from src.models.ctc_model import greedy_collapse

    _prefetch_audio([s["audio"] for s in samples])
    ds = VietSuperSpeechDataset(tokenizer=None, items=[(s["split"], s["index"]) for s in samples])
    raw_hyps, records = [], []
    t0 = time.time()
    for i, s in enumerate(samples):
        item = ds[i]
        if item["text"] != s["pseudo_label"]:
            # index manifest lệch revision HF → ref sai câu; dừng hẳn thay vì ra WER sai.
            raise SystemExit(f"pseudo_label lệch text HF ở {s['split']}:{s['index']}")
        x = item["waveform"][None].to(device)
        lengths = torch.tensor([x.shape[1]], device=device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            log_probs, out_lengths = model(x, lengths)
        raw = tokenizer.decode(greedy_collapse(log_probs.float(), out_lengths)[0])
        raw_hyps.append(raw)
        records.append({"split": s["split"], "index": s["index"], "video": video_of(s["audio"]),
                        "ref": s["ref"], "hyp": normalize_text(raw)})
    metrics = _score([r["ref"] for r in records], [r["hyp"] for r in records])
    metrics.update({"batch_size": 1, "eval_seconds": round(time.time() - t0, 1),
                    "n_corrected_text": sum(1 for s in samples if s["ref_source"] == "corrected_text")})
    return metrics, records, raw_hyps


def merge_predictions_tsv(path: Path, samples: list[dict], name: str, hyps: list[str], raw_hyps: list[str]) -> None:
    """Thêm/thay cột hyp_<name>, raw_<name> vào predictions.tsv (định dạng final_compare); file có sẵn
    phải cùng danh sách câu, nếu không thì dừng — ghép nhầm câu là chấm sai mà không thấy."""
    clean = lambda t: t.replace("\t", " ").replace("\n", " ")  # noqa: E731
    base = ["split", "index", "duration_s", "ref_source", "ref"]
    hyp_cols: dict[str, list[str]] = {}
    raw_cols: dict[str, list[str]] = {}
    if path.exists():
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"))
        if [(r["split"], int(r["index"])) for r in rows] != [(s["split"], s["index"]) for s in samples]:
            raise SystemExit(f"{path} có danh sách câu khác — dùng --out_dir khác")
        for col in rows[0]:
            if col.startswith("hyp_"):
                hyp_cols[col[4:]] = [r[col] for r in rows]
            elif col.startswith("raw_"):
                raw_cols[col[4:]] = [r[col] for r in rows]
    hyp_cols[name], raw_cols[name] = [clean(h) for h in hyps], [clean(h) for h in raw_hyps]
    names = list(hyp_cols)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(base + [f"hyp_{n}" for n in names] + [f"raw_{n}" for n in names])
        for i, s in enumerate(samples):
            w.writerow([s["split"], s["index"], f"{s['duration_s']:.2f}", s["ref_source"], s["ref"]]
                       + [hyp_cols[n][i] for n in names] + [raw_cols[n][i] for n in names])


# --------------------------------------------------------------------------- main

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt_dir", default=None, help="Thư mục thí nghiệm (chứa epochs/)")
    ap.add_argument("--last", type=int, default=None, help="Trung bình N epoch cuối có trong epochs/")
    ap.add_argument("--epoch_range", type=int, nargs=2, default=None, metavar=("FROM", "TO"),
                    help="Trung bình epoch FROM..TO (gồm cả hai đầu)")
    ap.add_argument("--epochs", type=int, nargs="+", default=None, help="Hoặc liệt kê epoch cụ thể")
    ap.add_argument("--checkpoint", default=None, help="Không trung bình: eval file này (best.pt, file avg đã ghi)")
    ap.add_argument("--save_avg", default=None, help="Ghi trọng số trung bình ra file này")
    ap.add_argument("--config", default=None, help="yaml của mô hình (bắt buộc khi eval)")
    ap.add_argument("--sets", nargs="*", default=["clean_test", "val", "val_unseen"],
                    help="Tập đo: clean_test, val, val_unseen; để trống = không eval")
    ap.add_argument("--name", default=None, help="Tên run trong results.json / cột TSV")
    ap.add_argument("--out_dir", default="reports/results/avg_eval")
    ap.add_argument("--limit_eval", type=int, default=0, help="Chỉ N câu đầu của val/val_unseen (thử CPU)")
    ap.add_argument("--batch_size", type=int, default=8, help="val/val_unseen; = batch mỗi GPU lúc train v1")
    ap.add_argument("--manifest", default="data/processed/clean_test_manifest.json")
    ap.add_argument("--tokenizer_model", default="configs/tokenizer.model")
    args = ap.parse_args()

    if args.checkpoint:
        ckpt_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt_state["model"]
        source = {"checkpoint": str(args.checkpoint), "checkpoint_epoch": ckpt_state.get("epoch"),
                  "averaged_epochs": ckpt_state.get("averaged_epochs")}
        default_name = Path(args.checkpoint).stem
    else:
        if not args.ckpt_dir:
            raise SystemExit("cần --ckpt_dir hoặc --checkpoint")
        ep_dir = Path(args.ckpt_dir) / "epochs"
        chosen = select_epochs(ep_dir, args.last, args.epoch_range, args.epochs)
        t0 = time.time()
        state_dict = average_state_dicts([ep_dir / f"epoch{e:02d}.pt" for e in chosen])
        source = {"checkpoint": str(ep_dir), "averaged_epochs": chosen}
        default_name = f"avg{chosen[0]}-{chosen[-1]}" if chosen == list(range(chosen[0], chosen[-1] + 1)) \
            else "avg" + "_".join(map(str, chosen))
        print(f"trung bình {len(chosen)} checkpoint {chosen} ({time.time() - t0:.0f} s)", flush=True)
        if args.save_avg:
            Path(args.save_avg).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": state_dict, "averaged_epochs": chosen, "source": str(ep_dir)}, args.save_avg)
            print(f"đã ghi {args.save_avg}", flush=True)
    if not args.sets:
        return
    if not args.config:
        raise SystemExit("eval cần --config")

    import yaml

    from src.models import mamba_encoder
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    from src.training.train import build_model

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    name = args.name or f"{cfg['experiment_name']}_{default_name}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not mamba_encoder.MAMBA_SSM_AVAILABLE and cfg["encoder"]["type"] in ("mamba", "conextbimamba"):
        raise SystemExit("mô hình có khối Mamba cần mamba_ssm (CUDA) — chạy trên Kaggle")
    amp = bool(cfg["training"].get("amp")) and device.type == "cuda"
    tokenizer = BPETokenizer(args.tokenizer_model)
    model = build_model(cfg, tokenizer.vocab_size)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = {**source, "config": args.config, "device": str(device),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "amp": amp,
           "decoding": "CTC greedy, không LM", "sets": {}}
    for set_name in args.sets:
        print(f"[{name}] eval {set_name}", flush=True)
        if set_name == "clean_test":
            samples = clean_test_samples(args.manifest)
            metrics, records, raw_hyps = eval_clean_test(model, samples, tokenizer, device, amp)
            merge_predictions_tsv(out_dir / "predictions.tsv", samples, name, [r["hyp"] for r in records], raw_hyps)
        elif set_name in ("val", "val_unseen"):
            metrics, records = eval_manifest_set(model, set_name, tokenizer, device, amp, args.batch_size,
                                                 args.limit_eval)
        else:
            raise SystemExit(f"tập không hỗ trợ: {set_name}")
        with open(out_dir / f"eval_{set_name}_{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        run["sets"][set_name] = metrics
        print(f"[{name}] {set_name}: WER={metrics['wer']:.4f} CER={metrics['cer']:.4f} n={metrics['n']} "
              f"S={metrics['substitutions']} D={metrics['deletions']} I={metrics['insertions']} "
              f"({metrics['eval_seconds']} s)", flush=True)

        # Ghi sau MỖI tập: kernel chết giữa chừng vẫn còn số của tập trước. Không an toàn khi hai tiến
        # trình cùng ghi một out_dir (predictions.tsv) — kernel cho mỗi tiến trình một thư mục rồi gộp.
        res_path = out_dir / "results.json"
        results = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {"runs": {}}
        results["runs"][name] = run
        tmp = res_path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, res_path)


if __name__ == "__main__":
    main()
