"""Trung bình trọng số nhiều checkpoint `epochs/epochNN.pt` (P1 + mọi lần train v2, kế hoạch 2026-10-12).

    python -m src.evaluation.average_checkpoints --ckpt_dir checkpoints/v2_mamba --last 5 --out avg_last5.pt

Như ESPnet (`average_nbest_models`) / ConMamba (`avg_checkpoints: 10`): tham số và buffer số thực lấy trung
bình cộng; buffer số nguyên (BatchNorm `num_batches_tracked`) lấy của checkpoint cuối. Thống kê BatchNorm
được trung bình luôn (không tính lại trên dữ liệu như SWA gốc — Izmailov 2018) — cùng cách các toolkit trên.

Chọn epoch nào phải chốt TRƯỚC khi xem kết quả (không chọn theo clean-test): mặc định N epoch cuối.
"""

import argparse
import json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="Thư mục thí nghiệm (chứa epochs/)")
    ap.add_argument("--last", type=int, default=None, help="Trung bình N epoch cuối có trong epochs/")
    ap.add_argument("--epochs", type=int, nargs="+", default=None, help="Hoặc liệt kê epoch cụ thể")
    ap.add_argument("--out", required=True, help="Tên file ghi vào ckpt_dir (vd. avg_last5.pt)")
    args = ap.parse_args()

    ep_dir = Path(args.ckpt_dir) / "epochs"
    available = sorted(int(p.stem.removeprefix("epoch")) for p in ep_dir.glob("epoch*.pt"))
    if args.epochs:
        chosen = args.epochs
    elif args.last:
        chosen = available[-args.last:]
    else:
        raise SystemExit("cần --last hoặc --epochs")
    missing = [e for e in chosen if e not in available]
    if missing:
        raise SystemExit(f"thiếu epoch {missing} trong {ep_dir}")
    paths = [ep_dir / f"epoch{e:02d}.pt" for e in chosen]
    out = Path(args.ckpt_dir) / args.out
    torch.save({"model": average_state_dicts(paths), "averaged_epochs": chosen}, out)
    print(json.dumps({"out": str(out), "averaged_epochs": chosen}, ensure_ascii=False))


if __name__ == "__main__":
    main()
