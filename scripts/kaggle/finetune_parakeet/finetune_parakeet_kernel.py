"""Kernel Kaggle **GPU** (2x T4) — fine-tune Parakeet-CTC-0.6B-vi (`src/training/finetune_parakeet.py`).

Đừng push tay — `launch.py` điền các hằng dưới đây (MODE, COMMIT, MAIN_ARGS) và push:
    python scripts/kaggle/finetune_parakeet/launch.py --mode trial --push
    python scripts/kaggle/finetune_parakeet/launch.py --mode main --account B --extra --lr 1e-5 --total_steps 6000 --push

MODE "trial" (~1,5 h, chốt cách chạy lần chính — số đo, không kết luận WER):
  1. tokenizer .nemo phủ nhãn train tới đâu (tỉ lệ token unk sau `normalize_text`);
  2. giải mã greedy của script khớp `model.decoding` của NeMo trên vài câu;
  3. probe 1 GPU trên batch dài nhất: fp32 và fp16 — s/step, VRAM đỉnh, loss hữu hạn (06/10 fp16 tràn khi suy luận);
  4. hai lần chạy ngắn DDP fp32, ~20 phút mỗi lần, LR 1e-5 (linto_stt_fr_fastconformer) và 5e-5, eval 400 câu
     val + 400 val_unseen trước/sau — để thấy LR nào không phá mô hình và đo s/step thật.
MODE "main": một lần trong trần ~7 GPU-h (chốt 2026-10-11), tham số từ launcher sau khi xem kết quả trial.

Output `/kaggle/working`: `trial/` hoặc `parakeet_ft/` (metrics.jsonl, eval_*.jsonl, best.pt, last.pt),
`logs/`, `kernel_summary.json`.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# --- launch.py điền các hằng này ---
MODE = "trial"
COMMIT = ""
SESSION_MINUTES = 420  # trần GPU-giờ của lần chính (chốt 2026-10-11) — tính cả setup
MAIN_ARGS: list[str] = []
# ------------------------------------

REPO_URL = "https://github.com/VuKhah/specialized-thesis"
REPO_DIR = Path("/tmp/specialized-thesis")
AUDIO_DIR = "/tmp/audio_cache"
OUT = Path("/kaggle/working")
NEMO_REPO, NEMO_FILE = "nvidia/parakeet-ctc-0.6b-Vietnamese", "parakeet-ctc-0.6b-vi.nemo"
NEMO_REVISION = "b0493142b49458810324e3db8be9e8e07b4ebc17"  # = zero_shot.py (sha HF 2026-10-02)
TAIL_MARGIN = 20  # phút: eval cuối + lưu 2 file trọng số ~2,4 GB
T0 = time.time()

CHECK_CODE = r"""
import sys, json, torch
sys.stdout.reconfigure(encoding="utf-8")
from collections import Counter
import nemo.collections.asr as nemo_asr
from src.data.vietsuperspeech_dataset import VietSuperSpeechDataset, collate_fn, manifest_rows
from src.evaluation.text_normalize import normalize_text
from src.models.ctc_model import greedy_collapse
from src.training.finetune_parakeet import TRAIN_MANIFESTS, forward
from datasets import load_dataset
m = nemo_asr.models.ASRModel.restore_from(sys.argv[1], map_location="cuda").eval()
tok = m.tokenizer
unk = tok.tokenizer.unk_id() if hasattr(tok, "tokenizer") else None
rows = manifest_rows(TRAIN_MANIFESTS)
hf = {s: load_dataset("thanhnew2001/VietSuperSpeech", split=s, revision="cbf624ae9b30e1c2793a27e95b262115c69601f3")
      for s in {r["split"] for r in rows}}
n_tok = n_unk = n_sent_unk = 0
unk_chars = Counter()
for r in rows:
    text = normalize_text(hf[r["split"]][r["index"]]["text"])
    ids = tok.text_to_ids(text)
    k = sum(i == unk for i in ids)
    n_tok += len(ids); n_unk += k; n_sent_unk += k > 0
    if k:
        unk_chars.update(c for c in set(text) if tok.text_to_ids(c) == [unk] or unk in tok.text_to_ids(c))
blank = m.decoder.num_classes_with_blank - 1
val = manifest_rows(["val_unseen"])[:16]
ds = VietSuperSpeechDataset(items=[(r["split"], r["index"]) for r in val])
agree = 0
with torch.no_grad():
    for i in range(len(ds)):
        b = collate_fn([ds[i]])
        lp, el = forward(m, m, b["waveform"].cuda(), b["waveform_lengths"].cuda(), False)
        ours = tok.ids_to_text(greedy_collapse(lp, el, blank)[0])
        ref = m.decoding.ctc_decoder_predictions_tensor(lp, decoder_lengths=el, return_hypotheses=False)
        ref = ref[0] if isinstance(ref, tuple) else ref
        ref = ref[0].text if hasattr(ref[0], "text") else str(ref[0])
        agree += ours == ref
        if ours != ref and i < 3:
            print("KHÁC:", repr(ours[:80]), "|", repr(ref[:80]))
out = {"vocab_size": tok.vocab_size, "blank": blank, "unk_id": unk, "n_tokens": n_tok, "unk_token_ratio": n_unk / max(n_tok, 1),
       "sentences_with_unk": n_sent_unk, "n_sentences": len(rows), "unk_chars_top": unk_chars.most_common(20),
       "greedy_agree": f"{agree}/{len(ds)}", "n_params": sum(p.numel() for p in m.parameters())}
print("CHECK " + json.dumps(out, ensure_ascii=False))
"""


def bootstrap() -> None:
    subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO_DIR)], check=True)
    if COMMIT:
        subprocess.run(["git", "checkout", "-q", COMMIT], cwd=REPO_DIR, check=True)
    sys.path.insert(0, str(REPO_DIR))


def install_nemo(run) -> None:
    """Như `zero_shot.py`: ghim torch đang có (NeMo kéo torch mới thì lệch CUDA của Kaggle); thử bản card (2.6)
    trước — 2.6.2 đã chạy được trên Kaggle 2026-10-06/08."""
    import torch
    constraints = Path("/tmp/nemo_constraints.txt")
    constraints.write_text(f"torch=={torch.__version__}\n", encoding="utf-8")
    for req in ["nemo_toolkit[asr]==2.6.2", "nemo_toolkit[asr]"]:
        code, _ = run([sys.executable, "-m", "pip", "install", "-q", req, "-c", str(constraints)], check=False)
        if code == 0:
            break
    else:
        sys.exit("LỖI: cài nemo_toolkit[asr] thất bại")
    run([sys.executable, "-c", "import nemo, torch; print('nemo', nemo.__version__, 'torch', torch.__version__)"])


def main() -> None:
    from scripts.kaggle.train_trial.train_trial import run, stamp
    import torch

    print(f"{stamp()} FINE-TUNE PARAKEET — MODE {MODE} | torch {torch.__version__} | "
          f"{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}", flush=True)
    if torch.cuda.device_count() != 2:
        sys.exit("LỖI: cần 2 GPU (T4 x2)")
    run(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)
    run([sys.executable, "-m", "pip", "install", "-q", "librosa", "soundfile", "datasets", "jiwer", "pyyaml",
         "tensorboard", "einops"])
    install_nemo(run)
    from huggingface_hub import hf_hub_download
    nemo_path = hf_hub_download(NEMO_REPO, NEMO_FILE, revision=NEMO_REVISION)
    run([sys.executable, "-m", "src.data.extract_audio", "--input", "/kaggle/input", "--dest", AUDIO_DIR], cwd=REPO_DIR)
    run([sys.executable, "-m", "src.data.vietsuperspeech_dataset"], cwd=REPO_DIR)
    env = {**os.environ, "AUDIO_CACHE_DIR": AUDIO_DIR, "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}
    summary = {"mode": MODE, "setup_minutes": (time.time() - T0) / 60, "steps": {}}
    mod = ["-m", "src.training.finetune_parakeet", "--nemo_path", nemo_path]
    torchrun = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2"]

    def step(name, cmd, **kw):
        t = time.time()
        code, out = run(cmd, cwd=REPO_DIR, env=env, check=False, log_name=name, **kw)
        tagged = [json.loads(l.split(" ", 1)[1]) for l in out.splitlines() if l.startswith(("CHECK ", "PROBE "))]
        summary["steps"][name] = {"returncode": code, "minutes": (time.time() - t) / 60, "result": tagged}
        (OUT / "kernel_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return code

    if MODE == "trial":
        step("check", [sys.executable, "-c", CHECK_CODE, nemo_path])
        for amp in (False, True):
            for bs in (8, 4):  # 8 OOM thì thử 4 (+ tích lũy gradient ở lần chính)
                code = step(f"probe_{'fp16' if amp else 'fp32'}_b{bs}",
                            [sys.executable, *mod, "--out_dir", "/tmp/probe", "--probe", "--per_gpu_batch", str(bs),
                             *(["--amp"] if amp else [])])
                if code == 0:
                    break
        for lr in ("1e-5", "5e-5"):
            step(f"short_lr{lr}", [*torchrun, *mod, "--out_dir", str(OUT / "trial" / f"lr{lr}"), "--lr", lr,
                                   "--warmup_steps", "100", "--max_minutes", "20", "--limit_eval", "400",
                                   "--per_gpu_batch", "4", "--global_batch", "16"])
    else:
        budget = SESSION_MINUTES - (time.time() - T0) / 60 - TAIL_MARGIN
        step("main", [*torchrun, *mod, "--out_dir", str(OUT / "parakeet_ft"), "--max_minutes", f"{budget:.0f}",
                      *MAIN_ARGS])
    summary["kernel_minutes"] = (time.time() - T0) / 60
    (OUT / "kernel_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    bootstrap()
    from scripts.kaggle.train_trial.train_trial import LOG_DIR, Tee, start_gpu_monitor, stamp
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log = open(LOG_DIR / f"finetune_parakeet_{MODE}.log", "a", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.stdout, _log), Tee(sys.stderr, _log)
    monitor = start_gpu_monitor()
    try:
        main()
    finally:
        if monitor:
            monitor.terminate()
        print(f"{stamp()} kết thúc kernel", flush=True)
