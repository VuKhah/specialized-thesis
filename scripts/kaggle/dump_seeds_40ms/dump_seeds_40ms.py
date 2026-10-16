"""Kernel Kaggle **GPU** (2x T4) — log-prob + greedy clean-test cho 6 checkpoint ep37 seed 43/44 nhóm 40 ms (L3,
quyết định #4 chốt 16/10). Cùng cách đo với `dump_logprobs/` (seed 42, 15/10): batch 1, autocast fp16 (front-end và
log_softmax fp32), code đúng commit đã train.

Khác `dump_logprobs/`: mô hình lấy từ tên file `<model>_seed<N>_ep37.pt` trong dataset `<A>/kltn-v2a-s4-seeds-ep37`
(dựng bằng `scripts/kaggle/train_full/collect_seeds.py`); log-prob chỉ lưu cho clean-test (tham số giải mã đã chốt
trên nửa tune 15/10 nên không cần log-prob dev); thêm 100 câu val_unseen chỉ để đối chiếu greedy với eval lúc train.

    kaggle kernels push -p scripts/kaggle/dump_seeds_40ms -t 3600

Output /kaggle/working: `lp_<model>_seed<N>_clean_test.npz` (khoá "split:index" → log-prob fp16 [T', 1001]),
`greedy.json` ({"<model>_seed<N>": {"<tập>|split:index": hyp}}), `summary.json`.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

COMMIT = "0090bd75535c980542885aa680a50789b9991630"  # commit train cả 9 run 40 ms (seed 42/43/44)
REPO_URL = "https://github.com/VuKhah/specialized-thesis"
REPO_DIR = Path("/tmp/specialized-thesis")
AUDIO_DIR = Path("/tmp/audio_cache")
OUT = Path("/kaggle/working")
DATASET = "kltn-v2a-s4-seeds-ep37"
HF_DATASET_ID = "thanhnew2001/VietSuperSpeech"
HF_REVISION = "cbf624ae9b30e1c2793a27e95b262115c69601f3"
CONFIGS = {"mamba": "configs/v2a/model_mamba_sub4.yaml",
           "conextbimamba": "configs/v2a/model_conextbimamba_sub4.yaml",
           "conformer_relpos": "configs/v2a/model_conformer_relpos_sub4.yaml"}
NPZ_SETS = {"clean_test"}
T0 = time.time()


def dump(name: str, ckpt: Path, gpu: int, sets: dict, results: dict) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    import yaml

    from src.models.ctc_model import greedy_collapse
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    from src.training.train import build_model

    device = torch.device(f"cuda:{gpu}")
    cfg = yaml.safe_load((REPO_DIR / CONFIGS[name.rsplit("_seed", 1)[0]]).read_text(encoding="utf-8"))
    tok = BPETokenizer(str(REPO_DIR / "configs/tokenizer.model"))
    model = build_model(cfg, tok.vocab_size)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    greedy = {}
    for set_name, rows in sets.items():
        arrays, t = {}, time.time()
        for r in rows:
            wav, sr = sf.read(str(AUDIO_DIR / r["audio"]), dtype="float32")
            assert sr == 16000, r["audio"]
            wav = torch.from_numpy(wav if wav.ndim == 1 else wav.mean(1))[None].to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                lp, lens = model(wav, torch.tensor([wav.shape[1]], device=device))
            lp = lp.float()
            key = f"{r['split']}:{r['index']}"
            if set_name in NPZ_SETS:
                arrays[key] = lp[0, : int(lens[0])].cpu().numpy().astype(np.float16)
            greedy[f"{set_name}|{key}"] = tok.decode(greedy_collapse(lp, lens)[0])
        if arrays:
            np.savez(OUT / f"lp_{name}_{set_name}.npz", **arrays)
        print(f"[{name}] {set_name}: {len(rows)} câu, {time.time() - t:.0f} s", flush=True)
    results[name] = {"greedy": greedy, "epoch": state.get("epoch"),
                     "encoder_params": sum(p.numel() for p in model.encoder.parameters())}


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

    ckpts = {m.group(1): p for p in sorted(Path("/kaggle/input").rglob(f"{DATASET}/**/*_ep37.pt"))
             if (m := re.fullmatch(r"(.+_seed\d+)_ep37\.pt", p.name))}
    if not ckpts:
        sys.exit(f"LỖI: không thấy checkpoint *_seed<N>_ep37.pt trong dataset {DATASET}")
    sets_path = next(Path("/kaggle/input").rglob(f"{DATASET}/**/sets.json"))
    sets = json.loads(sets_path.read_text(encoding="utf-8"))
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download
    rels = sorted({r["audio"] for rows in sets.values() for r in rows})
    t = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda rel: hf_hub_download(HF_DATASET_ID, rel, repo_type="dataset", revision=HF_REVISION,
                                                  local_dir=AUDIO_DIR), rels))
    print(f"Tải {len(rels)} audio ({time.time() - t:.0f} s); checkpoint: {sorted(ckpts)}", flush=True)

    os.chdir(REPO_DIR)
    results, errors = {}, {}
    names = sorted(ckpts, key=lambda n: not n.startswith("mamba"))  # Mamba fp32-scan chậm nhất → chia đều 2 GPU
    queues = [names[0::2], names[1::2]]

    def worker(gpu, todo):
        for n in todo:
            try:
                dump(n, ckpts[n], gpu, sets, results)
            except Exception:  # noqa: BLE001 — một mô hình lỗi không làm mất mô hình khác
                import traceback
                errors[n] = traceback.format_exc()
                print(errors[n], flush=True)

    threads = [threading.Thread(target=worker, args=(g, q)) for g, q in enumerate(queues)]
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
