"""Công cụ tra số tham số — dùng để tinh chỉnh config sao cho Mamba-CTC và
Conformer-CTC có số tham số khớp nhau (matched-parameter), theo đúng yêu cầu
so sánh công bằng trong đề cương (Chương 2).

Chạy: python -m src.models.param_count
"""

from src.models.conformer_encoder import ConformerEncoder

try:
    from src.models.mamba_encoder import MambaEncoder, MAMBA_SSM_AVAILABLE
except ImportError:
    MAMBA_SSM_AVAILABLE = False


def report(name: str, module) -> int:
    n = sum(p.numel() for p in module.parameters())
    print(f"{name:20s} {n:>12,d} tham số")
    return n


if __name__ == "__main__":
    conformer = ConformerEncoder()
    n_conformer = report("ConformerEncoder", conformer)

    if MAMBA_SSM_AVAILABLE:
        mamba = MambaEncoder()
        n_mamba = report("MambaEncoder", mamba)
        diff_pct = abs(n_mamba - n_conformer) / n_conformer * 100
        print(f"Chênh lệch: {diff_pct:.1f}%  (mục tiêu: < 5% — chỉnh d_model/n_layers nếu lệch nhiều)")
    else:
        print("MambaEncoder: mamba-ssm chưa cài — chạy notebooks/00_setup_environment.ipynb trước.")
