"""Kernel Kaggle **GPU** (2x T4) — chấm greedy clean-test cho 4 checkpoint ep37 nhánh 10 ms (v2a) + Parakeet fine-tune
`best.pt` (2026-10-16, việc L2 của `docs/notes/review_fable_2026-10-15.md`, người dùng chốt quyết định #3, #7).

Vì sao: bảng 10 ms đang chấm bằng ep28 v1 trong khi quy tắc 9.4 điều 9 chốt ep37; Conformer relPE 10 ms chưa chấm;
Parakeet fine-tune chưa có số clean-test. Mamba/ConExt cần CUDA nên không chạy local.

Khác `dump_logprobs/` (40 ms): không lưu log-prob — 10 ms có T' gấp 4 (~0,5 GB/mô hình chỉ riêng clean-test) và bảng
phụ beam + LM chỉ làm cho 40 ms (review 2.4). Chỉ xuất chuỗi greedy.

Câu: 203 câu clean-test + 100 câu val_unseen đầu của `val_unseen_subset` (đối chiếu với `eval_val_unseen_epoch37.jsonl`
lúc train / `eval_val_unseen_step*.jsonl` của Parakeet — kiểm nạp đúng checkpoint, đúng code). Lấy từ `sets.json` trong
dataset `<A>/kltn-v2a-10ms-ep37` (cùng 4 checkpoint `<model>_ep37.pt`, chỉ trọng số).
Đo như `dump_logprobs.py`: batch 1, autocast fp16 trên GPU (front-end và log_softmax fp32). Parakeet fp32 toàn bộ
(fp16 tràn ở 16/203 câu, 06/10).

    kaggle kernels push -p scripts/kaggle/dump_greedy_10ms -t 3600

Output /kaggle/working: `greedy.json` ({mô hình: {"<tập>|split:index": hyp}}), `summary.json`.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

COMMIT = "edd7dd40d8057cf931d259038092a2c0d0d5c64e"  # commit đã train 4 run v2a 10 ms (session_summary.json)
REPO_URL = "https://github.com/VuKhah/specialized-thesis"
REPO_DIR = Path("/tmp/specialized-thesis")
AUDIO_DIR = Path("/tmp/audio_cache")
OUT = Path("/kaggle/working")
HF_DATASET_ID = "thanhnew2001/VietSuperSpeech"
HF_REVISION = "cbf624ae9b30e1c2793a27e95b262115c69601f3"
NEMO_REPO, NEMO_FILE = "nvidia/parakeet-ctc-0.6b-Vietnamese", "parakeet-ctc-0.6b-vi.nemo"
NEMO_REVISION = "b0493142b49458810324e3db8be9e8e07b4ebc17"  # = zero_shot.py / finetune_parakeet_kernel.py
MODELS = {"mamba": "configs/v2a/model_mamba.yaml",
          "conextbimamba": "configs/v2a/model_conextbimamba.yaml",
          "conformer_relpos": "configs/v2a/model_conformer_relpos.yaml",
          "conformer": "configs/v2a/model_conformer.yaml"}
QUEUES = [["mamba", "conformer"], ["conextbimamba", "conformer_relpos"]]  # mỗi phần tử = một GPU
T0 = time.time()

PARAKEET_CODE = r"""
import json, sys, time
import soundfile as sf, torch
sys.stdout.reconfigure(encoding="utf-8")
from src.models.ctc_model import greedy_collapse
from src.training.finetune_parakeet import forward, load_parakeet
nemo_path, ckpt, sets_path, audio_dir, out_path = sys.argv[1:6]
m = load_parakeet(nemo_path, "cuda").eval()
state = torch.load(ckpt, map_location="cpu", weights_only=False)
m.load_state_dict(state["model"])
blank = m.decoder.num_classes_with_blank - 1
sets = json.load(open(sets_path, encoding="utf-8"))
greedy = {}
with torch.no_grad():
    for set_name, rows in sets.items():
        t = time.time()
        for r in rows:
            wav, sr = sf.read(f"{audio_dir}/{r['audio']}", dtype="float32")
            assert sr == 16000, r["audio"]
            wav = torch.from_numpy(wav if wav.ndim == 1 else wav.mean(1))[None].cuda()
            lp, el = forward(m, m, wav, torch.tensor([wav.shape[1]], device="cuda"), False)
            ids = greedy_collapse(lp, el, blank)[0]
            greedy[f"{set_name}|{r['split']}:{r['index']}"] = m.tokenizer.ids_to_text(ids) if ids else ""
        print(f"[parakeet_ft] {set_name}: {len(rows)} câu, {time.time() - t:.0f} s", flush=True)
meta = {k: v for k, v in state.items() if k != "model"}
json.dump({"greedy": greedy, "meta": meta}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
print("PARAKEET " + json.dumps(meta, ensure_ascii=False), flush=True)
"""


def find_input(pattern: str) -> Path:
    hits = sorted(Path("/kaggle/input").rglob(pattern))
    if not hits:
        sys.exit(f"LỖI: không thấy {pattern} trong /kaggle/input")
    return hits[0]


def dump(model_name: str, gpu: int, sets: dict, results: dict) -> None:
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
    state = torch.load(find_input(f"kltn-v2a-10ms-ep37/**/{model_name}_ep37.pt"), map_location="cpu",
                       weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    greedy = {}
    for set_name, rows in sets.items():
        t = time.time()
        for r in rows:
            wav, sr = sf.read(str(AUDIO_DIR / r["audio"]), dtype="float32")
            assert sr == 16000, r["audio"]
            wav = torch.from_numpy(wav if wav.ndim == 1 else wav.mean(1))[None].to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                lp, lens = model(wav, torch.tensor([wav.shape[1]], device=device))
            greedy[f"{set_name}|{r['split']}:{r['index']}"] = tok.decode(greedy_collapse(lp.float(), lens)[0])
        print(f"[{model_name}] {set_name}: {len(rows)} câu, {time.time() - t:.0f} s", flush=True)
    results[model_name] = {"greedy": greedy, "epoch": state.get("epoch"),
                           "encoder_params": sum(p.numel() for p in model.encoder.parameters())}


def main():
    subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run(["git", "checkout", "-q", COMMIT], cwd=REPO_DIR, check=True)
    sys.path.insert(0, str(REPO_DIR))
    from scripts.kaggle.finetune_parakeet.finetune_parakeet_kernel import install_nemo
    from scripts.kaggle.train_trial.train_trial import check_env, find_wheels, run

    check_env()
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run(pip + ["einops", "librosa", "soundfile", "sentencepiece", "pyyaml", "tensorboard", "jiwer"])
    wheels = find_wheels()
    if not wheels:
        sys.exit("LỖI: không thấy wheel mamba — gắn dataset mamba-wheels-v2")
    run(pip + ["--no-deps"] + wheels)

    sets_path = find_input("kltn-v2a-10ms-ep37/**/sets.json")
    sets = json.loads(sets_path.read_text(encoding="utf-8"))
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download
    rels = sorted({r["audio"] for rows in sets.values() for r in rows})
    t = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda rel: hf_hub_download(HF_DATASET_ID, rel, repo_type="dataset", revision=HF_REVISION,
                                                  local_dir=AUDIO_DIR), rels))
    print(f"Tải {len(rels)} audio ({time.time() - t:.0f} s)", flush=True)

    os.chdir(REPO_DIR)
    results, errors = {}, {}

    def worker(gpu, names):
        for n in names:
            try:
                dump(n, gpu, sets, results)
            except Exception:  # noqa: BLE001 — một mô hình lỗi không làm mất mô hình khác
                import traceback
                errors[n] = traceback.format_exc()
                print(errors[n], flush=True)

    threads = [threading.Thread(target=worker, args=(g, q)) for g, q in enumerate(QUEUES)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    # Ghi ngay phần nhóm train từ đầu: NeMo cài sau có thể hỏng môi trường, không được làm mất kết quả trên.
    (OUT / "greedy.json").write_text(json.dumps({k: v["greedy"] for k, v in results.items()}, ensure_ascii=False),
                                     encoding="utf-8")

    parakeet_meta = None
    try:
        install_nemo(run)
        nemo_path = hf_hub_download(NEMO_REPO, NEMO_FILE, revision=NEMO_REVISION)
        ckpt = find_input("finetune-parakeet-main/**/best.pt")
        out_path = Path("/tmp/parakeet_ft.json")
        code, _ = run([sys.executable, "-c", PARAKEET_CODE, nemo_path, str(ckpt), str(sets_path), str(AUDIO_DIR),
                       str(out_path)], cwd=REPO_DIR, check=False,
                      env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if code != 0:
            raise RuntimeError(f"Parakeet returncode {code}")
        pk = json.loads(out_path.read_text(encoding="utf-8"))
        results["parakeet_ft"] = {"greedy": pk["greedy"]}
        parakeet_meta = pk["meta"]
    except Exception:  # noqa: BLE001
        import traceback
        errors["parakeet_ft"] = traceback.format_exc()
        print(errors["parakeet_ft"], flush=True)

    (OUT / "greedy.json").write_text(json.dumps({k: v["greedy"] for k, v in results.items()}, ensure_ascii=False),
                                     encoding="utf-8")
    summary = {"commit": COMMIT, "minutes": round((time.time() - T0) / 60, 1), "errors": list(errors),
               "models": {k: {kk: vv for kk, vv in v.items() if kk != "greedy"} for k, v in results.items()},
               "parakeet_ft_ckpt": parakeet_meta, "n": {k: len(v) for k, v in sets.items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if errors:
        sys.exit(f"LỖI: {list(errors)}")


if __name__ == "__main__":
    main()
