"""Công cụ tra số tham số — dùng để tinh chỉnh config sao cho Mamba-CTC và
Conformer-CTC có số tham số khớp nhau (matched-parameter), theo đúng yêu cầu
so sánh công bằng trong đề cương (Chương 2).

Đọc trực tiếp `configs/model_{conformer,mamba}.yaml` và dựng encoder bằng
`build_encoder` của `train.py` — cùng đường với lúc train thật, nên số ở đây
đúng bằng dòng "encoder params" mà `train.py` in ra. Chỉ đếm encoder (front-end
không có tham số học; CTC head giống hệt nhau khi hai d_model bằng nhau).

Chạy từ gốc repo:
    python -m src.models.param_count
    python -m src.models.param_count --mamba configs/model_mamba.yaml
"""

import argparse
import sys

import yaml

from src.models.mamba_encoder import MAMBA_SSM_AVAILABLE
from src.training.train import build_encoder

TARGET_DIFF_PCT = 5.0


def count_from_yaml(path: str) -> tuple[str, int]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = {k: v for k, v in cfg["encoder"].items() if k != "type"}
    n = build_encoder(cfg).num_parameters()
    print(f"{cfg['experiment_name']:24s} {n:>12,d} tham số  {enc_cfg}")
    return cfg["encoder"]["type"], n


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--conformer", default="configs/model_conformer.yaml")
    parser.add_argument("--mamba", default="configs/model_mamba.yaml")
    args = parser.parse_args()

    _, n_conformer = count_from_yaml(args.conformer)

    if not MAMBA_SSM_AVAILABLE:
        print("Mamba: mamba-ssm chưa cài (cần CUDA) — chạy trên Kaggle, xem notebooks/00_setup_environment.ipynb.")
        return

    _, n_mamba = count_from_yaml(args.mamba)
    diff_pct = abs(n_mamba - n_conformer) / n_conformer * 100
    status = "ĐẠT" if diff_pct < TARGET_DIFF_PCT else "CHƯA đạt"
    print(f"Chênh lệch: {diff_pct:.2f}%  (mục tiêu < {TARGET_DIFF_PCT:.0f}%) -> {status}")


if __name__ == "__main__":
    main()
