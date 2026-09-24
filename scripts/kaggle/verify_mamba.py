"""Kernel Kaggle (GPU T4) xác nhận nhánh Mamba — thứ máy local không chạy được
vì thiếu CUDA. Được đẩy lên bằng Kaggle CLI, không chạy trên máy local:

    kaggle kernels push -p scripts/kaggle -t 7200   (cần kernel-metadata.json local, gitignore)
    kaggle kernels status <username>/verify-mamba-asr
    kaggle kernels output <username>/verify-mamba-asr -p <thư mục>

Các bước: clone repo từ GitHub (nên phải push code trước) → build wheel
`causal-conv1d` + `mamba-ssm` v2.3.1 (`--no-build-isolation`, lý do ở
docs/notes/mamba_ssm_install_log.md) và lưu vào /kaggle/working/wheels để lần
sau cài lại không phải build → `param_count` → chạy thử vài step train + 1 lần
eval cho cả hai encoder, đo thời gian/step và VRAM đỉnh.

Mỗi bước chạy trong tiến trình Python con: import torch rồi mới cài extension
CUDA trong cùng tiến trình từng gây lỗi khó hiểu (log Lần 2).
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_URL = "https://github.com/VuKhah/specialized-thesis"
REPO_DIR = Path("/tmp/specialized-thesis")  # ngoài /kaggle/working: không muốn repo + audio thành output
WHEEL_DIR = Path("/kaggle/working/wheels")
CAUSAL_CONV1D = "git+https://github.com/Dao-AILab/causal-conv1d@v1.5.4"
MAMBA_SSM = "git+https://github.com/state-spaces/mamba@v2.3.1"
SMOKE_STEPS = 5


def run(cmd, cwd=None, check=True, env=None):
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if check and result.returncode != 0:
        sys.exit(f"LỖI: lệnh trên trả về {result.returncode}")
    return result.returncode


SMOKE_CODE = textwrap.dedent(
    f"""
    import sys, time, yaml, torch
    from torch.utils.data import DataLoader, Subset
    from src.data.vietsuperspeech_dataset import VietSuperSpeechDataset, collate_fn
    from src.models.ctc_model import CTCASRModel
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    from src.training.train import build_encoder, evaluate

    sys.stdout.reconfigure(encoding="utf-8")
    tok = BPETokenizer("configs/tokenizer.model")
    train_ds = VietSuperSpeechDataset(split="train", tokenizer=tok)
    eval_ds = Subset(VietSuperSpeechDataset(split="validation", tokenizer=tok), range(16))

    for cfg_path in ["configs/model_conformer.yaml", "configs/model_mamba.yaml"]:
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        bs = cfg["training"]["batch_size"]
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        encoder = build_encoder(cfg)
        model = CTCASRModel(encoder=encoder, vocab_size=tok.vocab_size).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
        loader = DataLoader(Subset(train_ds, range(bs * {SMOKE_STEPS})), batch_size=bs, collate_fn=collate_fn)
        print(f"\\n=== {{cfg['experiment_name']}}: {{encoder.num_parameters():,}} tham số encoder, batch {{bs}} ===", flush=True)
        model.train()
        for step, batch in enumerate(loader):
            batch = {{k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)}}
            torch.cuda.synchronize(); t0 = time.time()
            loss = model.compute_loss(batch["waveform"], batch["waveform_lengths"], batch["targets"], batch["target_lengths"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            opt.step(); torch.cuda.synchronize()
            print(f"step {{step}}: loss={{loss.item():.3f}}  {{time.time() - t0:.2f}} s  "
                  f"max_len={{int(batch['waveform_lengths'].max()) / 16000:.1f}} s audio", flush=True)
        print(f"VRAM đỉnh (train): {{torch.cuda.max_memory_allocated() / 2**30:.2f}} GiB", flush=True)
        wer = evaluate(model, DataLoader(eval_ds, batch_size=bs, collate_fn=collate_fn), tok, "cuda")
        print(f"eval 16 câu validation: WER={{wer:.3f}} (vô nghĩa sau {SMOKE_STEPS} step, chỉ để chạy qua đường decode)", flush=True)
        del model, opt, encoder
    print("\\nSMOKE TEST XONG", flush=True)
    """
)


def main():
    run(["nvidia-smi"], check=False)
    run([sys.executable, "-c", "import torch, sys; print(sys.version); print('torch', torch.__version__, 'cuda', torch.version.cuda)"])

    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    run(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)

    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run(pip + ["packaging", "ninja", "einops", "librosa", "soundfile", "sentencepiece", "datasets", "jiwer", "pyyaml", "tensorboard"])

    WHEEL_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MAX_JOBS": "4"}  # build song song quá nhiều job dễ hết RAM của kernel
    for spec in [CAUSAL_CONV1D, MAMBA_SSM]:
        run([sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "-w", str(WHEEL_DIR), spec], env=env)
    run(pip + ["--no-deps"] + sorted(str(p) for p in WHEEL_DIR.glob("*.whl")))
    run([sys.executable, "-c", "from mamba_ssm import Mamba; import causal_conv1d; print('import mamba_ssm OK')"])

    run([sys.executable, "-m", "src.models.param_count"], cwd=REPO_DIR)
    run([sys.executable, "-c", SMOKE_CODE], cwd=REPO_DIR)


if __name__ == "__main__":
    main()
