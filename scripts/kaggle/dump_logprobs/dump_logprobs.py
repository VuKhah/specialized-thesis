"""Kernel Kaggle **GPU** (2x T4) — lưu log-prob CTC của 3 mô hình 40 ms (v2a-s4, ep37) để thử giải mã beam +
từ điển + LM trên CPU local (2026-10-15, người dùng duyệt: "nếu cần thiết hãy đẩy lên kaggle").

Vì sao: chẩn đoán 15/10 — lỗi "từ dính" do BPE-1000 + greedy CTC; giải mã beam + từ điển + LM sửa được mà không train
lại, nhưng Mamba/ConExt cần CUDA nên không chạy local. Kernel chỉ chạy bước front-end → encoder → head (code đúng
commit đã train), phần giải mã làm ở local.

Câu: 958 câu val_unseen (đúng tập tune/dev của thử nghiệm giải mã local, chia theo video) + 203 câu clean-test, lấy
từ `sets.json` trong dataset `<A>/kltn-v2a-s4-ep37` (cùng 3 checkpoint `<model>_ep37.pt`, chỉ trọng số).
Đo giống `average_checkpoints.eval_clean_test`: batch 1, autocast fp16 trên GPU (front-end và log_softmax fp32).

    kaggle kernels push -p scripts/kaggle/dump_logprobs -t 3600

Output /kaggle/working: `lp_<model>_<set>.npz` (khoá "split:index" → log-prob fp16 [T', 1001]), `greedy.json`
(chuỗi greedy từng câu để đối chiếu với eval khi train), `summary.json`.
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

COMMIT = "0090bd75535c980542885aa680a50789b9991630"  # commit đã train 3 run v2a-s4 (session_summary.json)
REPO_URL = "https://github.com/VuKhah/specialized-thesis"
REPO_DIR = Path("/tmp/specialized-thesis")
AUDIO_DIR = Path("/tmp/audio_cache")
OUT = Path("/kaggle/working")
HF_DATASET_ID = "thanhnew2001/VietSuperSpeech"
HF_REVISION = "cbf624ae9b30e1c2793a27e95b262115c69601f3"
MODELS = {"mamba": "configs/v2a/model_mamba_sub4.yaml",
          "conextbimamba": "configs/v2a/model_conextbimamba_sub4.yaml",
          "conformer_relpos": "configs/v2a/model_conformer_relpos_sub4.yaml"}
QUEUES = [["mamba"], ["conextbimamba", "conformer_relpos"]]  # mỗi phần tử = một GPU; Mamba fp32-scan chậm nhất
T0 = time.time()


def find_input(name: str) -> Path:
    hits = sorted(Path("/kaggle/input").glob(f"**/kltn-v2a-s4-ep37/**/{name}")) or \
        sorted(Path("/kaggle/input").glob(f"*/*/kltn-v2a-s4-ep37/{name}")) or \
        sorted(Path("/kaggle/input").glob(f"*/{name}"))
    if not hits:
        sys.exit(f"LỖI: không thấy {name} trong dataset kltn-v2a-s4-ep37")
    return hits[0]


def dump(model_name: str, gpu: int, sets: dict, results: dict) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    import yaml

    from src.models.ctc_model import greedy_collapse
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    from src.training.train import build_model

    device = torch.device(f"cuda:{gpu}")
    cfg = yaml.safe_load((REPO_DIR / MODELS[model_name]).read_text(encoding="utf-8"))
    tok = BPETokenizer(str(REPO_DIR / "configs/tokenizer.model"))
    model = build_model(cfg, tok.vocab_size)
    state = torch.load(find_input(f"{model_name}_ep37.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    greedy = {}
    for set_name, rows in sets.items():
        arrays = {}
        t = time.time()
        for r in rows:
            wav, sr = sf.read(str(AUDIO_DIR / r["audio"]), dtype="float32")
            assert sr == 16000, r["audio"]
            wav = torch.from_numpy(wav if wav.ndim == 1 else wav.mean(1))[None].to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                lp, lens = model(wav, torch.tensor([wav.shape[1]], device=device))
            lp = lp.float()
            key = f"{r['split']}:{r['index']}"
            arrays[key] = lp[0, : int(lens[0])].cpu().numpy().astype(np.float16)
            greedy[f"{set_name}|{key}"] = tok.decode(greedy_collapse(lp, lens)[0])
        np.savez(OUT / f"lp_{model_name}_{set_name}.npz", **arrays)
        print(f"[{model_name}] {set_name}: {len(rows)} câu, {time.time() - t:.0f} s", flush=True)
    results[model_name] = {"greedy": greedy, "wer_val_unseen_ckpt": state.get("wer_val_unseen"),
                           "epoch": state.get("epoch")}


def main():
    subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run(["git", "checkout", "-q", COMMIT], cwd=REPO_DIR, check=True)
    sys.path.insert(0, str(REPO_DIR))
    from scripts.kaggle.train_trial.train_trial import check_env, find_wheels, run

    check_env()
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run(pip + ["einops", "librosa", "soundfile", "sentencepiece", "pyyaml", "tensorboard", "jiwer"])
    wheels = find_wheels()
    if not wheels:
        sys.exit("LỖI: không thấy wheel mamba — gắn dataset mamba-wheels-v2")
    run(pip + ["--no-deps"] + wheels)

    sets = json.loads(find_input("sets.json").read_text(encoding="utf-8"))
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download
    rels = sorted({r["audio"] for rows in sets.values() for r in rows})
    t = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda rel: hf_hub_download(HF_DATASET_ID, rel, repo_type="dataset", revision=HF_REVISION,
                                                  local_dir=AUDIO_DIR), rels))
    print(f"Tải {len(rels)} audio ({time.time() - t:.0f} s)", flush=True)

    import os
    os.chdir(REPO_DIR)
    results, errors = {}, {}

    def worker(gpu, names):
        for n in names:
            try:
                dump(n, gpu, sets, results)
            except Exception as e:  # noqa: BLE001 — một mô hình lỗi không làm mất mô hình khác
                import traceback
                errors[n] = traceback.format_exc()
                print(errors[n], flush=True)

    threads = [threading.Thread(target=worker, args=(g, q)) for g, q in enumerate(QUEUES)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    (OUT / "greedy.json").write_text(json.dumps({k: v["greedy"] for k, v in results.items()}, ensure_ascii=False),
                                     encoding="utf-8")
    summary = {"commit": COMMIT, "minutes": round((time.time() - T0) / 60, 1), "errors": list(errors),
               "models": {k: {kk: vv for kk, vv in v.items() if kk != "greedy"} for k, v in results.items()},
               "n": {k: len(v) for k, v in sets.items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if errors:
        sys.exit(f"LỖI: {list(errors)}")


if __name__ == "__main__":
    main()
