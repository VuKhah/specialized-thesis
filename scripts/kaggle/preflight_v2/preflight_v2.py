"""Kernel Kaggle **GPU** (2x T4) — preflight công thức v2 (2026-10-12) trước P2/P3.

Kiểm những gì local không kiểm được (khối Mamba CUDA, DDP, VRAM T4, tốc độ) cho 7 config `configs/v2/`:
  K1 dựng mô hình CUDA thật: tham số, bất biến padding ở mức encoder, InterCTC đúng lớp
  Đ2 VRAM đỉnh trường hợp xấu nhất: 1 GPU, batch 8 × 16,5 s (15 s × speed 1,1), AMP, forward + backward
     (Conformer relpos tạo ma trận vị trí [8, 4, T, 2T-1] — rủi ro OOM chính)
  K3 pipeline tí hon DDP mỗi config (train → eval → best.pt → metrics.jsonl có train_loss_interctc)
  Đ4 s/step + VRAM trên đủ 4 shard, DDP, mỗi config SPEED_MINUTES phút
  K5 dừng giữa epoch + resume với lịch WSD (decay_epochs > 0) — v2 ConExtBiMamba
Kết quả: `/kaggle/working/preflight_v2_summary.json`, log ở `logs/`. Ước ~75-90 phút GPU.

    kaggle kernels push -p scripts/kaggle/preflight_v2 -t 7200   (KAGGLE_CONFIG_DIR của tài khoản B)
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

COMMIT = "1384e70"  # ghim: code v2 đã test CPU
REPO_URL = "https://github.com/VuKhah/specialized-thesis"
REPO_DIR = Path("/tmp/specialized-thesis")
AUDIO_DIR = "/tmp/audio_cache"
OUT = Path("/kaggle/working")
CONFIGS = {n: f"configs/v2/model_{n}.yaml" for n in
           ["mamba", "conextbimamba", "conformer_relpos", "conformer_nope",
            "mamba_sub4", "conextbimamba_sub4", "conformer_relpos_sub4"]}
SPEED_MINUTES = 6
SUMMARY: dict = {"checks": {}, "measures": {}}

MODEL_CHECK = r'''
import sys, json, torch, yaml
sys.stdout.reconfigure(encoding="utf-8")
from src.training.train import build_model
name, path = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(path, encoding="utf-8"))
dev = torch.device("cuda", 0)
torch.manual_seed(0)
m = build_model(cfg, 1001).to(dev)
res = {"encoder_params": m.encoder.num_parameters(), "total_params": sum(p.numel() for p in m.parameters()),
       "interctc_layer": m.interctc["layer"] if m.interctc else None}
# K1: bất biến padding ở mức encoder (cùng đặc trưng), eval
m.eval()
with torch.no_grad():
    w = torch.randn(3, 16000 * 4, device=dev); wl = torch.tensor([64000, 40000, 23000], device=dev)
    f, fl = m.feature_extractor(w, wl)
    a, _ = m.forward_features(f, fl); b, lb = m.forward_features(f[2:, : int(fl[2])], fl[2:])
    res["padding_diff"] = (a[2, : int(lb[0])] - b[0, : int(lb[0])]).abs().max().item()
# Đ2: VRAM xấu nhất (1 GPU = nửa batch toàn cục 16), AMP như train
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
torch.cuda.reset_peak_memory_stats(dev)
B, S = 8, int(16000 * 16.5)
w = torch.randn(B, S, device=dev); wl = torch.full((B,), S, device=dev)
tg = torch.randint(1, 1001, (B, 120), device=dev); tl = torch.full((B,), 120, device=dev)
scaler = torch.amp.GradScaler("cuda")
for _ in range(2):
    with torch.autocast("cuda", dtype=torch.float16):
        lp, ol, inter = m(w, wl, return_inter=True)
    loss = m.total_ctc_loss(lp, ol, inter, tg, tl)[0]
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
torch.cuda.synchronize(dev)
res["worst_case_peak_vram_gib"] = torch.cuda.max_memory_allocated(dev) / 2**30
res["worst_case_loss_finite"] = bool(torch.isfinite(loss).item())
print("MODEL " + json.dumps(res))
'''


def bootstrap() -> None:
    subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run(["git", "checkout", "-q", COMMIT], cwd=REPO_DIR, check=True)
    sys.path.insert(0, str(REPO_DIR))


def last_progress(out: str) -> dict:
    pat = re.compile(r"step (\d+)/(\d+) loss tb ([\d.na]+) \| ([\d.]+) s/step.*bỏ (\d+)\+(\d+)(?:, VRAM đỉnh ([\d.]+) GiB)?")
    hits = pat.findall(out)
    if not hits:
        return {}
    s, total, loss, sps, nf, amp, vram = hits[-1]
    return {"step": int(s), "loss": loss, "s_per_step": float(sps), "nonfinite": int(nf), "amp_skipped": int(amp),
            "peak_vram_gib": float(vram) if vram else None}


def record(section: str, key: str, value) -> None:
    SUMMARY[section][key] = value
    (OUT / "preflight_v2_summary.json").write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== {section}.{key}: {json.dumps(value, ensure_ascii=False)}", flush=True)


def main() -> None:
    from scripts.kaggle.train_trial.train_trial import check_env, find_wheels, run

    check_env()
    _, head = run(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)
    SUMMARY["commit"] = head.strip()
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run(pip + ["einops", "librosa", "soundfile", "sentencepiece", "datasets", "jiwer", "pyyaml", "tensorboard"])
    wheels = find_wheels()
    if not wheels:
        sys.exit("LỖI: không thấy wheel mamba trong /kaggle/input — kiểm tra dataset mamba-wheels-v2")
    run(pip + ["--no-deps"] + wheels)

    # K1 + Đ2 trước khi giải nén audio: lỗi CUDA/OOM thì dừng sớm, đỡ tốn GPU.
    failed = []
    for name, cfg in CONFIGS.items():
        code, out = run([sys.executable, "-c", MODEL_CHECK, name, cfg], cwd=REPO_DIR, check=False,
                        log_name=f"model_{name}")
        hits = [json.loads(l[6:]) for l in out.splitlines() if l.startswith("MODEL ")]
        record("checks", f"K1_D2_{name}", hits[0] if hits else f"LỖI returncode {code} — xem logs/model_{name}.log")
        if not hits or hits[0]["padding_diff"] > 1e-3 or not hits[0]["worst_case_loss_finite"]:
            failed.append(name)
    record("checks", "K1_D2_failed", failed)

    run([sys.executable, "-m", "src.data.extract_audio", "--input", "/kaggle/input", "--dest", AUDIO_DIR],
        cwd=REPO_DIR)
    run([sys.executable, "-m", "src.data.vietsuperspeech_dataset"], cwd=REPO_DIR)
    env = {**os.environ, "AUDIO_CACHE_DIR": AUDIO_DIR, "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}

    def train(config, ckpt, minutes, extra=(), log_name=None):
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "-m", "src.training.train",
               "--config", config, "--max_minutes", str(minutes), "--ckpt_dir", str(ckpt),
               "--log_dir", str(Path(ckpt) / "runs"), "--log_every_steps", "25", *extra]
        return run(cmd, cwd=REPO_DIR, env=env, check=False, log_name=log_name)

    for name, cfg in CONFIGS.items():
        if name in failed:
            continue
        code, out = train(cfg, Path("/tmp/pipeline"), 10,
                          ("--epochs", "2", "--decay_epochs", "1", "--limit_train", "64", "--limit_eval", "64",
                           "--log_every_steps", "1", "--no_resume"), log_name=f"k3_{name}")
        mpath = Path("/tmp/pipeline") / f"v2{'s4' if name.endswith('sub4') else ''}_{name.removesuffix('_sub4')}" \
            / "metrics.jsonl"
        lines = mpath.read_text(encoding="utf-8").splitlines() if mpath.exists() else []
        ok = code == 0 and len(lines) == 2 and "train_loss_interctc" in lines[-1] and "WER=" in out
        record("checks", f"K3_pipeline_{name}", "OK" if ok else f"LỖI (code {code}, {len(lines)} dòng metrics)")

    for name, cfg in CONFIGS.items():
        if name in failed:
            continue
        _, out = train(cfg, Path(f"/tmp/speed_{name}"), SPEED_MINUTES, ("--no_resume",), log_name=f"d4_{name}")
        record("measures", f"D4_speed_{name}", last_progress(out))

    # K5: dừng giữa epoch rồi resume, lịch WSD (pha giảm nằm ở epoch 1 của 2).
    ck = Path("/tmp/resume")
    extra = ("--epochs", "2", "--decay_epochs", "1", "--limit_train", "1600", "--limit_eval", "32")
    _, out1 = train(CONFIGS["conextbimamba"], ck, 1.5, extra + ("--no_resume",), log_name="k5_part1")
    _, out2 = train(CONFIGS["conextbimamba"], ck, 15, extra, log_name="k5_part2")
    m = re.search(r"resume từ .*?: epoch (\d+) step (\d+)", out2)
    mfile = ck / "v2_conextbimamba" / "metrics.jsonl"
    rows = [json.loads(l) for l in mfile.read_text(encoding="utf-8").splitlines()] if mfile.exists() else []
    record("checks", "K5_resume_wsd", {"resumed_at": m.groups() if m else None, "epochs_logged": len(rows),
                                       "lr_end_epoch": [r.get("lr") for r in rows]})


if __name__ == "__main__":
    bootstrap()
    from scripts.kaggle.train_trial.train_trial import LOG_DIR, Tee, start_gpu_monitor, stamp
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log = open(LOG_DIR / "preflight_v2.log", "a", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.stdout, _log), Tee(sys.stderr, _log)
    monitor = start_gpu_monitor()
    try:
        main()
    finally:
        if monitor:
            monitor.terminate()
        print(f"{stamp()} kết thúc kernel", flush=True)
