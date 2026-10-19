"""Kernel Kaggle **GPU** (2x T4) — giải mã test GigaSpeech 2 tiếng Việt (4.422 câu, 11,02 h, người gán nhãn) cho mọi
hệ thống bảng chính — phương án B, người dùng chốt 2026-10-19 (`docs/notes/so_sanh_hien_hanh_2026-10-19.md`).
Chỉ eval, không train, không chọn gì trên tập này.

Một file cho 3 chế độ (kernel script chỉ upload 1 file; `launch_gs2.py` thay `MODE` và sinh metadata từng tài khoản):
- `40ms` (A): 9 checkpoint ep37 nhánh 40 ms (3 encoder × seed 42/43/44) + Parakeet fine-tune `best.pt`.
  Lưu top-32 log-prob mỗi khung để giải mã beam + LM trên CPU local (`lm_decode` dùng `beam_size_token=20`; kiểm
  19/10 trên 609 câu clean-test: top-32 cho kết quả greedy và beam+LM trùng 100% bản đầy đủ, dung lượng 6,4%).
- `40ms_dev` (A): như `40ms` nhưng trên **dev** GigaSpeech 2 (6.612 câu, 10,2 h) — top-32 để dò tham số giải mã / LM
  (người dùng 19/10: tối ưu giải mã, chỉ chỉnh trên dev); không Parakeet FT, không clean-test (đã kiểm ở `40ms`).
- `10ms` (A, kernel thứ hai chạy song song — B tạo dataset bị 403 ngày 19/10): 4 checkpoint ep37 nhánh 10 ms từ dataset
  `kltn-v2a-10ms-ep37`, greedy (như bảng 10 ms clean-test).
- `phowhisper` (C): PhoWhisper-small zero-shot (autocast fp16), một bản trên mỗi GPU, chia đôi câu.
- `parakeet_zs` (D): Parakeet-CTC-0.6B zero-shot (fp32).
  Hai chế độ pre-train dùng chính `load_phowhisper` / `load_parakeet` của `scripts/kaggle/zero_shot/zero_shot.py`
  (cùng revision mô hình, cùng giải mã như lần chấm clean-test). Tách 2 tài khoản để chạy song song (thêm D 19/10).

Mọi chế độ chạy thêm 203 câu clean-test: greedy phải trùng giả thuyết đã chấm 16/10 → bằng chứng nạp đúng checkpoint/
code (đối chiếu ở local, không chấm ở đây). Đo như các kernel dump trước: batch 1, autocast fp16 (front-end, khối
Mamba, log_softmax fp32), code đúng commit đã train.

Output /kaggle/working: `greedy.json` ({hệ thống: {"<tập>|<khoá>": hyp}}), `tk_<hệ thống>_gs2_test.npz` (chế độ 40ms:
`keys`, `lengths`, `idx` uint16 [khung, 32], `val` fp16 [khung, 32]), `summary.json`.
"""

import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

MODE = "__MODE__"
COMMITS = {"40ms": "0090bd75535c980542885aa680a50789b9991630",  # commit train 9 run 40 ms
           "40ms_dev": "0090bd75535c980542885aa680a50789b9991630",
           "10ms": "edd7dd40d8057cf931d259038092a2c0d0d5c64e",  # commit train 4 run 10 ms
           "phowhisper": "ecb3d2d538e43708378128831f4eec6a18e2f2b2",  # zero_shot.py như lần chấm clean-test
           "parakeet_zs": "ecb3d2d538e43708378128831f4eec6a18e2f2b2"}
PRETRAIN = {"phowhisper": ("phowhisper", "load_phowhisper", True, True),  # tên, loader, autocast, chia 2 GPU
            "parakeet_zs": ("parakeet_fp32", "load_parakeet", False, False)}
CONFIGS = {"40ms": {"mamba": "configs/v2a/model_mamba_sub4.yaml",
                    "conextbimamba": "configs/v2a/model_conextbimamba_sub4.yaml",
                    "conformer_relpos": "configs/v2a/model_conformer_relpos_sub4.yaml"},
           "10ms": {"mamba": "configs/v2a/model_mamba.yaml", "conextbimamba": "configs/v2a/model_conextbimamba.yaml",
                    "conformer_relpos": "configs/v2a/model_conformer_relpos.yaml",
                    "conformer": "configs/v2a/model_conformer.yaml"}}
SPLIT = "dev" if MODE.endswith("_dev") else "test"
GS2_SLUG = f"gs2-vi-{SPLIT}"
SET = f"gs2_{SPLIT}"
REPO_URL = "https://github.com/VuKhah/specialized-thesis"
REPO_DIR = Path("/tmp/specialized-thesis")
GS2_DIR = Path("/tmp/gs2")
CT_DIR = Path("/tmp/audio_cache")
OUT = Path("/kaggle/working")
HF_DATASET_ID = "thanhnew2001/VietSuperSpeech"
HF_REVISION = "cbf624ae9b30e1c2793a27e95b262115c69601f3"
NEMO_REPO, NEMO_FILE = "nvidia/parakeet-ctc-0.6b-Vietnamese", "parakeet-ctc-0.6b-vi.nemo"
NEMO_REVISION = "b0493142b49458810324e3db8be9e8e07b4ebc17"
TOPK = 32
T0 = time.time()

PARAKEET_FT_CODE = r"""
import json, sys, time
import soundfile as sf, torch
sys.stdout.reconfigure(encoding="utf-8")
from src.models.ctc_model import greedy_collapse
from src.training.finetune_parakeet import forward, load_parakeet
nemo_path, ckpt, sets_path, out_path = sys.argv[1:5]
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
            wav, sr = sf.read(r["path"], dtype="float32")
            assert sr == 16000, r["path"]
            wav = torch.from_numpy(wav if wav.ndim == 1 else wav.mean(1))[None].cuda()
            lp, el = forward(m, m, wav, torch.tensor([wav.shape[1]], device="cuda"), False)
            ids = greedy_collapse(lp, el, blank)[0]
            greedy[f"{set_name}|{r['key']}"] = m.tokenizer.ids_to_text(ids) if ids else ""
        print(f"[parakeet_ft] {set_name}: {len(rows)} câu, {time.time() - t:.0f} s", flush=True)
meta = {k: v for k, v in state.items() if k != "model"}
json.dump({"greedy": greedy, "meta": meta}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
"""


def find_inputs(pattern: str) -> list[Path]:
    # Dataset gắn ở /kaggle/input/<slug>/ hoặc /kaggle/input/datasets/<user>/<slug>/ — glob giới hạn độ sâu, không
    # rglob cả /kaggle/input (sự cố 2026-10-05: duyệt hàng chục nghìn wav, im lặng vài phút).
    return sorted({p for pre in ("", "*/", "*/*/", "*/*/*/") for p in Path("/kaggle/input").glob(pre + pattern)})


def find_input(pattern: str) -> Path:
    hits = find_inputs(pattern)
    if not hits:
        sys.exit(f"LỖI: không thấy {pattern} trong /kaggle/input")
    return hits[0]


def prepare_sets() -> dict:
    """{"gs2_test": [{key, path}], "clean_test": [{key, path}]} — audio GS2 giải nén từ dataset, clean-test tải từ HF."""
    tsv = find_input(f"{GS2_SLUG}/{SPLIT}.tsv")
    tar = tsv.parent / f"{SPLIT}.tar.gz"
    t = time.time()
    # Kaggle tự giải nén archive khi tạo dataset (19/10: `test.tar.gz` → `test/test/<nhóm>/<id>.wav`); còn tar thì tự
    # giải nén vào /tmp.
    if tar.exists():
        with tarfile.open(tar) as f:
            f.extractall(GS2_DIR, filter="data")
    first = tsv.read_text(encoding="utf-8").split("\t", 1)[0]
    roots = [GS2_DIR / SPLIT, tsv.parent / SPLIT / SPLIT, tsv.parent / SPLIT]
    root = next((r for r in roots if (r / first.split("-")[0] / f"{first}.wav").exists()), None)
    if root is None:
        sys.exit(f"LỖI: không thấy {first}.wav dưới {[str(r) for r in roots]}")
    gs2 = []
    for line in tsv.read_text(encoding="utf-8").splitlines():
        key = line.split("\t", 1)[0]
        path = root / key.split("-")[0] / f"{key}.wav"
        if not path.exists():
            sys.exit(f"LỖI: thiếu {path}")
        gs2.append({"key": key, "path": str(path)})
    print(f"GS2 {SPLIT}: {len(gs2)} câu ({time.time() - t:.0f} s)", flush=True)
    if MODE == "40ms_dev":
        return {SET: gs2}

    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download
    samples = json.loads((REPO_DIR / "data/processed/clean_test_manifest.json").read_text(encoding="utf-8"))["samples"]
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda s: hf_hub_download(HF_DATASET_ID, s["audio"], repo_type="dataset", revision=HF_REVISION,
                                                local_dir=CT_DIR), samples))
    ct = [{"key": f"{s['split']}:{s['index']}", "path": str(CT_DIR / s["audio"])} for s in samples]
    return {SET: gs2, "clean_test": ct}


def dump_ours(name: str, cfg_path: str, ckpt: Path, gpu: int, sets: dict, results: dict, save_topk: bool) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    import yaml

    from src.models.ctc_model import greedy_collapse
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    from src.training.train import build_model

    device = torch.device(f"cuda:{gpu}")
    cfg = yaml.safe_load((REPO_DIR / cfg_path).read_text(encoding="utf-8"))
    tok = BPETokenizer(str(REPO_DIR / "configs/tokenizer.model"))
    model = build_model(cfg, tok.vocab_size)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    greedy, seconds = {}, {}
    for set_name, rows in sets.items():
        keys, lengths, idx_parts, val_parts, t = [], [], [], [], time.time()
        for r in rows:
            wav, sr = sf.read(r["path"], dtype="float32")
            assert sr == 16000, r["path"]
            wav = torch.from_numpy(wav if wav.ndim == 1 else wav.mean(1))[None].to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                lp, lens = model(wav, torch.tensor([wav.shape[1]], device=device))
            lp = lp.float()
            greedy[f"{set_name}|{r['key']}"] = tok.decode(greedy_collapse(lp, lens)[0])
            if save_topk and set_name.startswith("gs2_"):
                # Làm tròn fp16 trước rồi mới lấy top-k — đúng thứ tự như npz đầy đủ của các kernel dump trước.
                lp16 = lp[0, : int(lens[0])].half()
                val, idx = lp16.float().topk(TOPK, dim=-1)
                keys.append(r["key"])
                lengths.append(lp16.shape[0])
                idx_parts.append(idx.to(torch.int32).cpu().numpy().astype(np.uint16))
                val_parts.append(val.half().cpu().numpy())
        seconds[set_name] = round(time.time() - t, 1)
        if keys:
            np.savez(OUT / f"tk_{name}_{set_name}.npz", keys=np.array(keys), lengths=np.array(lengths),
                     idx=np.concatenate(idx_parts), val=np.concatenate(val_parts))
        print(f"[{name}] {set_name}: {len(rows)} câu, {seconds[set_name]:.0f} s", flush=True)
    results[name] = {"greedy": greedy, "epoch": state.get("epoch"), "seconds": seconds,
                     "encoder_params": sum(p.numel() for p in model.encoder.parameters())}


def run_queue(jobs: list, worker_fn) -> dict:
    """Hàng đợi động cho 2 GPU (Mamba fp32-scan chậm nhất → chia tĩnh lệch); lỗi một hệ không làm mất hệ khác."""
    errors, lock = {}, threading.Lock()

    def worker(gpu):
        while True:
            with lock:
                if not jobs:
                    return
                job = jobs.pop(0)
            try:
                worker_fn(job, gpu)
            except Exception:  # noqa: BLE001
                import traceback
                errors[job[0]] = traceback.format_exc()
                print(errors[job[0]], flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return errors


def write_outputs(results: dict, errors: dict, extra: dict) -> None:
    (OUT / "greedy.json").write_text(json.dumps({k: v["greedy"] for k, v in results.items()}, ensure_ascii=False),
                                     encoding="utf-8")
    summary = {"mode": MODE, "commit": COMMITS[MODE], "minutes": round((time.time() - T0) / 60, 1),
               "errors": list(errors), **extra,
               "models": {k: {kk: vv for kk, vv in v.items() if kk != "greedy"} for k, v in results.items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def parakeet_ft(sets: dict, run) -> dict:
    from huggingface_hub import hf_hub_download

    from scripts.kaggle.finetune_parakeet.finetune_parakeet_kernel import install_nemo
    install_nemo(run)
    nemo_path = hf_hub_download(NEMO_REPO, NEMO_FILE, revision=NEMO_REVISION)
    ckpt = find_input("finetune-parakeet-main/**/best.pt")
    sets_path, out_path = Path("/tmp/sets.json"), Path("/tmp/parakeet_ft.json")
    sets_path.write_text(json.dumps(sets, ensure_ascii=False), encoding="utf-8")
    t = time.time()
    code, _ = run([sys.executable, "-c", PARAKEET_FT_CODE, nemo_path, str(ckpt), str(sets_path), str(out_path)],
                  cwd=REPO_DIR, check=False, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if code != 0:
        raise RuntimeError(f"Parakeet returncode {code}")
    pk = json.loads(out_path.read_text(encoding="utf-8"))
    return {"greedy": pk["greedy"], "ckpt": pk["meta"], "seconds_total": round(time.time() - t, 1)}


def main_ours(sets: dict, run) -> None:
    import re
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run(pip + ["einops", "librosa", "soundfile", "sentencepiece", "pyyaml", "tensorboard", "jiwer"])
    from scripts.kaggle.train_trial.train_trial import find_wheels
    wheels = find_wheels()
    if not wheels:
        sys.exit("LỖI: không thấy wheel mamba — gắn dataset mamba-wheels-v2")
    run(pip + ["--no-deps"] + wheels)

    if MODE.startswith("40ms"):
        ckpts = {f"{p.name[:-len('_ep37.pt')]}_seed42": p for p in find_inputs("kltn-v2a-s4-ep37/**/*_ep37.pt")}
        ckpts.update({m.group(1): p for p in find_inputs("kltn-v2a-s4-seeds-ep37/**/*_ep37.pt")
                      if (m := re.fullmatch(r"(.+_seed\d+)_ep37\.pt", p.name))})
    else:
        ckpts = {p.name[:-len("_ep37.pt")]: p for p in find_inputs("kltn-v2a-10ms-ep37/**/*_ep37.pt")}
    expected = 9 if MODE.startswith("40ms") else 4
    if len(ckpts) != expected:
        sys.exit(f"LỖI: cần {expected} checkpoint, thấy {sorted(ckpts)}")
    print(f"checkpoint: {sorted(ckpts)}", flush=True)

    os.chdir(REPO_DIR)
    results = {}
    jobs = sorted(ckpts.items(), key=lambda kv: (not kv[0].startswith(("mamba", "conextbimamba")), kv[0]))
    errors = run_queue(jobs, lambda job, gpu: dump_ours(job[0], CONFIGS[MODE.removesuffix("_dev")][re.sub(r"_seed\d+$", "", job[0])],
                                                        job[1], gpu, sets, results, MODE.startswith("40ms")))
    write_outputs(results, errors, {})  # ghi ngay: NeMo cài sau có thể làm hỏng môi trường
    extra = {}
    if MODE == "40ms":
        try:
            pk = parakeet_ft(sets, run)
            results["parakeet_ft"] = {"greedy": pk["greedy"], "seconds_total": pk["seconds_total"]}
            extra["parakeet_ft_ckpt"] = pk["ckpt"]
        except Exception:  # noqa: BLE001
            import traceback
            errors["parakeet_ft"] = traceback.format_exc()
            print(errors["parakeet_ft"], flush=True)
    write_outputs(results, errors, extra)
    if errors:
        sys.exit(f"LỖI: {list(errors)}")


def main_pretrain(sets: dict, run) -> None:
    import soundfile as sf
    run([sys.executable, "-m", "pip", "install", "-q", "jiwer", "soundfile", "datasets"])
    from scripts.kaggle.zero_shot import zero_shot as zs

    results, errors = {}, {}

    def transcribe_all(name, transcribe, rows, out):
        for r in rows:
            wav, sr = sf.read(r["path"], dtype="float32")
            assert sr == 16000, r["path"]
            out[r["set"] + "|" + r["key"]] = transcribe(wav if wav.ndim == 1 else wav.mean(1))

    flat = [{**r, "set": s} for s, rows in sets.items() for r in rows]
    for name, loader_name, amp, split_gpus in [PRETRAIN[MODE]]:
        loader, t = getattr(zs, loader_name), time.time()
        try:
            spec = zs.MODEL_SPECS["parakeet" if name.startswith("parakeet") else name]
            gpus = [0, 1] if split_gpus else [0]
            parts = [flat[i::len(gpus)] for i in range(len(gpus))]
            outs = [{} for _ in gpus]
            models = [loader(spec, f"cuda:{g}", amp) for g in gpus]
            threads = [threading.Thread(target=transcribe_all, args=(name, m[1], p, o))
                       for m, p, o in zip(models, parts, outs)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            greedy = {k: v for o in outs for k, v in o.items()}
            if len(greedy) != len(flat):
                raise RuntimeError(f"{name}: {len(greedy)}/{len(flat)} câu")
            meta = {k: v for k, v in models[0][2].items() if isinstance(v, (str, int, float))}
            results[name] = {"greedy": greedy, "seconds_total": round(time.time() - t, 1), "meta": meta,
                             "revision": spec["revision"], "amp": amp}
            print(f"[{name}] {len(flat)} câu, {time.time() - t:.0f} s", flush=True)
            del models
        except Exception:  # noqa: BLE001
            import traceback
            errors[name] = traceback.format_exc()
            print(errors[name], flush=True)
        write_outputs(results, errors, {})
    if errors:
        sys.exit(f"LỖI: {list(errors)}")


def main() -> None:
    assert MODE in COMMITS, MODE
    subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run(["git", "checkout", "-q", COMMITS[MODE]], cwd=REPO_DIR, check=True)
    sys.path.insert(0, str(REPO_DIR))
    from scripts.kaggle.train_trial.train_trial import check_env, run

    check_env()
    sets = prepare_sets()
    os.chdir(REPO_DIR)
    (main_pretrain if MODE in PRETRAIN else main_ours)(sets, run)


if __name__ == "__main__":
    main()
