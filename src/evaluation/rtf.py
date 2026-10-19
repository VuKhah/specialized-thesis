"""Đo RQ2 cuối: độ trễ, RTF, VRAM đỉnh, số tham số theo độ dài audio ghép nhân tạo (batch 1, 1 GPU).

Thay bản cũ (bucket 3-30 s theo đề cương — dữ liệu thật chỉ 10-15 s; RQ2 đổi cách hỏi 2026-10-03,
`docs/notes/du_lieu/dataset_discrepancy.md`) và gom phần đo của pilot (`scripts/kaggle/rq2_pilot/`, ghi chú
`docs/notes/ket_qua/rq2_pilot_2026-10-05.md`) thành module. Khác pilot:
- Dựng mô hình đúng code train (`build_model` + yaml, kể cả subsampling 4× thật của nhánh 40 ms) thay khối
  subsampling dựng tạm; nạp được checkpoint thật (`--checkpoints`), mặc định trọng số ngẫu nhiên seed 0 —
  chi phí không phụ thuộc giá trị trọng số (không có rẽ nhánh theo dữ liệu trong encoder).
- Làm ấm **đúng từng shape** trước khi đo (cuDNN dựng kế hoạch conv cho mỗi shape mới — lượt đầu Conformer/ConExt
  tốn ~1,1 s cố định, `docs/notes/ket_qua/subsample_speed_2026-10-08.md` mục 4).
- Số lần đo thích ứng (≥ `min_runs`), ghi trung vị + IQR + min/max; OOM là một kết quả (ghi dòng `oom`, chia đôi
  khoảng để tìm độ dài tối đa, các mức dài hơn ghi `skipped`).
- InterCTC tắt mặc định (`--interctc off`): phạm vi "A + 40 ms" (2026-10-12) train không InterCTC. Lưu ý: trong
  `CTCASRModel.forward_features`, khi `interctc` bật thì hook lớp giữa (CTC head + softmax + Linear
  self-conditioning) chạy **cả lúc suy luận** — đo checkpoint có InterCTC phải dùng `--interctc yaml`.

Chính sách số học = lúc train/eval (D8, 2026-10-05): autocast fp16, front-end fp32 (tự tắt autocast), khối Mamba
fp32 (`mamba_fp32` trong code mô hình). Độ trễ = front-end + (subsampling) + encoder + CTC head + log-softmax;
giải mã tham lam đo riêng (`decode_*`), không cộng vào độ trễ.

    python -m src.evaluation.rtf --models conformer,conformer_relpos --lengths 5,10 --out /tmp/rq2 \
        --audio concat --allow_missing          # local CPU
    python -m src.evaluation.rtf --models final --out /kaggle/working/rq2_final   # Kaggle (kernel rq2_final)
"""

import argparse
import csv
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

SAMPLE_RATE = 16000

# ĐỀ XUẤT (chờ người dùng duyệt): dày quanh điểm giao của pilot — 60-120 s (10 ms: ConExt 60 s, B1 120 s) và
# 240-480 s (40 ms); thêm 5-30 s (độ dài câu train 10-15 s) và đuôi tới 1 giờ (độ dốc log-log, Conformer bậc hai).
DEFAULT_LENGTHS_S = [5, 10, 15, 20, 30, 45, 60, 75, 90, 105, 120, 150, 180, 240, 300, 360, 420, 480, 600, 900,
                     1200, 1800, 3600]

# Đội hình đo cuối (người dùng duyệt 2026-10-12). Encoder của B1/ConExt ở configs/ và configs/v2/ giống hệt;
# InterCTC trong yaml v2 bị tắt bởi --interctc off.
FINAL_LINEUP = {
    "mamba": "configs/model_mamba.yaml",
    "conextbimamba": "configs/model_conextbimamba.yaml",
    "conformer": "configs/model_conformer.yaml",
    "conformer_relpos": "configs/v2/model_conformer_relpos.yaml",
    "mamba_sub4": "configs/v2/model_mamba_sub4.yaml",
    "conextbimamba_sub4": "configs/v2/model_conextbimamba_sub4.yaml",
    "conformer_relpos_sub4": "configs/v2/model_conformer_relpos_sub4.yaml",
}


class OOM(Exception):
    pass


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class CTCRunner:
    """Mô hình train từ đầu (Mamba B1 / ConExtBiMamba / Conformer v1 / relPE, 10 ms hoặc 40 ms)."""

    def __init__(self, name: str, config: str, device: torch.device, vocab_size: int, amp: bool,
                 checkpoint: str | None = None, interctc: str = "off", seed: int = 0):
        from src.training.train import build_model

        cfg = yaml.safe_load(open(config, encoding="utf-8"))
        if interctc == "off":
            cfg["interctc"] = None
        torch.manual_seed(seed)
        model = build_model(cfg, vocab_size)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"] if "model" in state else state)
        self.model = model.to(device).eval()
        self.device, self.amp = device, amp
        sub = model.subsampling
        n_sub = sum(p.numel() for p in sub.parameters()) if sub is not None else 0
        n_enc = sum(p.numel() for p in model.encoder.parameters())
        self.info = {
            "model": name, "family": "scratch", "config": config, "checkpoint": checkpoint,
            "weights": "checkpoint" if checkpoint else f"random(seed={seed})",
            "frame_ms": 40 if sub is not None else 10, "interctc": bool(model.interctc),
            "encoder_type": cfg["encoder"]["type"],
            "params_encoder": n_enc, "params_subsampling": n_sub,
            "params_total": sum(p.numel() for p in model.parameters()),
            "precision": "autocast fp16, front-end fp32, khối Mamba fp32" if amp else "fp32",
        }

    @torch.no_grad()
    def run_once(self, wav: torch.Tensor, lengths: torch.Tensor) -> dict:
        from src.models.ctc_model import greedy_collapse

        m, dev = self.model, self.device
        _sync(dev)
        t0 = time.perf_counter()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=self.amp):
            feats, feat_lengths = m.feature_extractor(wav, lengths)
            _sync(dev)
            t1 = time.perf_counter()
            log_probs, out_lengths = m.forward_features(feats, feat_lengths)
            _sync(dev)
            t2 = time.perf_counter()
        greedy_collapse(log_probs, out_lengths)
        t3 = time.perf_counter()
        return {"latency_s": t2 - t0, "frontend_s": t1 - t0, "model_s": t2 - t1, "decode_s": t3 - t2,
                "frames_frontend": int(feat_lengths[0]), "frames_out": int(out_lengths[0]), "_out": log_probs}


class ParakeetRunner:
    """Nhóm pre-train: pipeline riêng của NeMo (preprocessor fp32 → encoder FastConformer 8× → CTC), chỉ dùng
    chung audio ghép + cách đo. fp32 mặc định: fp16 tràn số trên vài câu (Plan.md nhật ký, zero-shot 2026-10)."""

    def __init__(self, nemo_path: str, device: torch.device, precision: str = "fp32"):
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location=device).eval()
        self.device, self.amp = device, precision == "fp16"
        self.info = {
            "model": f"parakeet_{precision}", "family": "pretrained", "config": nemo_path, "checkpoint": nemo_path,
            "weights": "pretrained", "frame_ms": 80, "interctc": False, "encoder_type": "fastconformer",
            "params_encoder": sum(p.numel() for p in self.model.encoder.parameters()), "params_subsampling": None,
            "params_total": sum(p.numel() for p in self.model.parameters()),
            "precision": "autocast fp16 (preprocessor fp32)" if self.amp else "fp32",
        }

    @torch.no_grad()
    def run_once(self, wav: torch.Tensor, lengths: torch.Tensor) -> dict:
        from src.models.ctc_model import greedy_collapse

        m, dev = self.model, self.device
        _sync(dev)
        t0 = time.perf_counter()
        feats, feat_lengths = m.preprocessor(input_signal=wav, length=lengths)
        _sync(dev)
        t1 = time.perf_counter()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=self.amp):
            log_probs, enc_len, _ = m(processed_signal=feats, processed_signal_length=feat_lengths)
        _sync(dev)
        t2 = time.perf_counter()
        greedy_collapse(log_probs, enc_len)
        t3 = time.perf_counter()
        return {"latency_s": t2 - t0, "frontend_s": t1 - t0, "model_s": t2 - t1, "decode_s": t3 - t2,
                "frames_frontend": int(feat_lengths[0]), "frames_out": int(enc_len[0]), "_out": log_probs}


def _stats(xs: list[float]) -> dict:
    a = np.asarray(xs)
    return {"median": float(np.median(a)), "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
            "min": float(a.min()), "max": float(a.max())}


def measure_length(runner, wav_np: np.ndarray, min_runs: int = 3, max_runs: int = 30, target_s: float = 20.0,
                   extra_warmup_below_s: float = 5.0, fake_oom_s: float = 0.0) -> dict:
    """Một mức độ dài: làm ấm ở đúng shape này (1 lượt; thêm 1 lượt nếu lượt đầu < extra_warmup_below_s), rồi đo
    n = clamp(ceil(target_s / t_lượt_đầu), min_runs, max_runs) lượt. VRAM đỉnh lấy trong các lượt đo."""
    dev = runner.device
    seconds = len(wav_np) / SAMPLE_RATE
    if fake_oom_s and seconds >= fake_oom_s:  # chỉ để test nhánh OOM trên CPU
        raise OOM()
    oom = False
    try:
        if dev.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        static = torch.cuda.memory_allocated(dev) / 2**30 if dev.type == "cuda" else None
        wav = torch.from_numpy(wav_np)[None].to(dev)
        lengths = torch.tensor([wav.size(1)], device=dev)
        first = runner.run_once(wav, lengths)
        finite = bool(torch.isfinite(first.pop("_out")).all())
        n_warmup = 1
        if first["latency_s"] < extra_warmup_below_s:
            runner.run_once(wav, lengths)
            n_warmup = 2
        n_runs = int(min(max_runs, max(min_runs, math.ceil(target_s / max(first["latency_s"], 1e-6)))))
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        runs = []
        for i in range(n_runs):
            r = runner.run_once(wav, lengths)
            r.pop("_out")
            runs.append(r)
            if r["latency_s"] > 30:  # watchdog kernel kill sau 10 phút im lặng (Conformer 3.600 s ~157 s/lượt)
                print(f"    {seconds:.0f} s: lượt {i + 1}/{n_runs} {r['latency_s']:.1f} s", flush=True)
        peak = torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else None
        peak_reserved = torch.cuda.max_memory_reserved(dev) / 2**30 if dev.type == "cuda" else None
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not _is_oom(e):
            raise
        oom = True  # dọn bộ nhớ ngoài khối except: traceback còn giữ tham chiếu tensor lớn
    if oom:
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        raise OOM()
    lat = _stats([r["latency_s"] for r in runs])
    row = {"seconds": seconds, "frames_frontend": runs[0]["frames_frontend"], "frames_out": runs[0]["frames_out"],
           "n_warmup": n_warmup, "n_runs": n_runs, "first_run_s": first["latency_s"], "output_finite": finite,
           **{f"latency_{k}_s": v for k, v in lat.items()},
           "rtf_median": lat["median"] / seconds,
           "frontend_median_s": _stats([r["frontend_s"] for r in runs])["median"],
           "model_median_s": _stats([r["model_s"] for r in runs])["median"],
           "decode_median_s": _stats([r["decode_s"] for r in runs])["median"],
           "static_vram_gib": static, "peak_vram_gib": peak, "peak_reserved_gib": peak_reserved}
    del wav, lengths
    return row


def sweep(runner, long_wav: np.ndarray, lengths_s: list[float], record, audio_label: str, bisect_steps: int = 4,
          max_run_s: float = 300.0, **measure_kw) -> dict:
    """Quét các mức tăng dần. OOM → ghi dòng oom, chia đôi khoảng (mức đạt cuối, mức OOM) `bisect_steps` lần,
    các mức dài hơn ghi skipped (VRAM tăng đơn điệu theo độ dài). Lượt đo > max_run_s → bỏ các mức dài hơn."""
    tag = {k: runner.info[k] for k in ("model", "family", "frame_ms", "weights", "interctc")} | {"audio": audio_label}
    last_ok, fail, slow = None, None, None
    t_model = time.time()

    def one(s: float, bisect: bool = False) -> bool:
        nonlocal slow
        try:
            r = measure_length(runner, long_wav[: int(round(s * SAMPLE_RATE))], **measure_kw)
        except OOM:
            record({**tag, "kind": "measure", "seconds": s, "oom": True, "bisect": bisect})
            print(f"  {runner.info['model']:>22} {s:>7.1f} s: OOM", flush=True)
            return False
        record({**tag, "kind": "measure", **r, "oom": False, "bisect": bisect})
        vram = "-" if r["peak_vram_gib"] is None else f"{r['peak_vram_gib']:.2f}"
        print(f"  {runner.info['model']:>22} {s:>7.1f} s ({r['frames_out']:>6} khung ra): "
              f"{r['latency_median_s'] * 1000:10.1f} ms [IQR {r['latency_p25_s'] * 1000:.1f}-"
              f"{r['latency_p75_s'] * 1000:.1f}] ×{r['n_runs']}, RTF {r['rtf_median']:.5f}, VRAM {vram} GiB", flush=True)
        if r["latency_median_s"] > max_run_s:
            slow = s
        return True

    for s in lengths_s:
        if s * SAMPLE_RATE > len(long_wav):
            record({**tag, "kind": "skipped", "seconds": s, "reason": "audio ghép ngắn hơn mức này"})
            continue
        if fail is not None or slow is not None:
            reason = f"dài hơn mức OOM {fail} s" if fail is not None else f"lượt đo ở {slow} s > {max_run_s} s"
            record({**tag, "kind": "skipped", "seconds": s, "reason": reason})
            continue
        if one(s):
            last_ok = s
        else:
            fail = s
    limit = {**tag, "kind": "limit", "max_ok_s": last_ok, "oom_s": None, "wall_s": None}
    if fail is not None:
        lo, hi = (last_ok or 0.0), fail
        for _ in range(bisect_steps):
            mid = round((lo + hi) / 2, 1)
            if one(mid, bisect=True):
                lo = mid
            else:
                hi = mid
        limit.update(max_ok_s=lo, oom_s=hi)
    limit["wall_s"] = time.time() - t_model
    record(limit)
    return limit


CSV_FIELDS = ["model", "family", "frame_ms", "weights", "interctc", "audio", "seconds", "oom", "bisect",
              "frames_out", "latency_median_s", "latency_p25_s", "latency_p75_s", "latency_min_s", "latency_max_s",
              "rtf_median", "frontend_median_s", "model_median_s", "decode_median_s", "peak_vram_gib",
              "static_vram_gib", "n_warmup", "n_runs", "first_run_s", "output_finite"]


def write_csv(jsonl: Path, out_csv: Path) -> None:
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            if r.get("kind") == "measure":
                w.writerow(r)


def _env(device: torch.device, args) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    env = {"python": platform.python_version(), "torch": torch.__version__, "commit": commit, "device": str(device),
           "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
           "cudnn_benchmark": torch.backends.cudnn.benchmark, "amp": device.type == "cuda" and not args.no_amp,
           "args": vars(args)}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        env.update(gpu=p.name, gpu_mem_gib=p.total_memory / 2**30,
                   sdp_flash=torch.backends.cuda.flash_sdp_enabled(),
                   sdp_mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                   sdp_math=torch.backends.cuda.math_sdp_enabled())
    else:
        env.update(cpu_threads=torch.get_num_threads())
    return env


def _parse_pairs(s: str) -> dict[str, str]:
    return dict(p.split("=", 1) for p in s.split(",") if p) if s else {}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="final",
                    help="'final' (FINAL_LINEUP) hoặc danh sách tên trong FINAL_LINEUP / tên=đường_dẫn_yaml")
    ap.add_argument("--checkpoints", default="", help="tên=đường_dẫn.pt,...; thiếu = trọng số ngẫu nhiên seed 0")
    ap.add_argument("--interctc", choices=["off", "yaml"], default="off")
    ap.add_argument("--lengths", default=",".join(map(str, DEFAULT_LENGTHS_S)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio", choices=["concat", "noise"], default="concat")
    ap.add_argument("--noise_check_lengths", default="",
                    help="đo thêm nhiễu Gauss ở các mức này (kiểm chi phí không phụ thuộc nội dung audio)")
    ap.add_argument("--input_dir", default=os.environ.get("RQ2_INPUT", "/kaggle/input"))
    ap.add_argument("--allow_missing", action="store_true", help="chỉ local: bỏ đoạn thiếu file khi ghép")
    ap.add_argument("--tokenizer_model", default="configs/tokenizer.model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--min_runs", type=int, default=3)
    ap.add_argument("--max_runs", type=int, default=30)
    ap.add_argument("--target_s", type=float, default=20.0, help="tổng thời gian đo mong muốn mỗi mức")
    ap.add_argument("--max_run_s", type=float, default=300.0)
    ap.add_argument("--bisect_steps", type=int, default=4)
    ap.add_argument("--mamba_ref", action="store_true", help="chỉ test CPU: khối Mamba thuần PyTorch (mamba_ref)")
    ap.add_argument("--fake_oom_s", type=float, default=0.0, help="chỉ test: coi mức ≥ giá trị này là OOM")
    ap.add_argument("--parakeet_nemo", default="", help="đường dẫn .nemo → đo thêm Parakeet (cần nemo_toolkit)")
    ap.add_argument("--parakeet_precision", choices=["fp32", "fp16"], default="fp32")
    args = ap.parse_args()

    from src.data.concat_audio import build_concat_audio, find_audio_roots
    from src.tokenizer.bpe_tokenizer import BPETokenizer

    if args.mamba_ref:
        from src.models.mamba_ref import use_reference_mamba
        use_reference_mamba()
    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lengths = sorted(float(x) for x in args.lengths.split(","))
    noise_lengths = [float(x) for x in args.noise_check_lengths.split(",") if x]
    if args.models == "final":
        models = dict(FINAL_LINEUP)
    elif args.models == "none":  # chỉ đo Parakeet
        models = {}
    else:
        models = {p.split("=", 1)[0]: (p.split("=", 1)[1] if "=" in p else FINAL_LINEUP[p])
                  for p in args.models.split(",") if p}
    checkpoints = _parse_pairs(args.checkpoints)
    env = _env(device, args)
    (out / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")
    print("môi trường:", json.dumps({k: v for k, v in env.items() if k != "args"}, ensure_ascii=False), flush=True)

    rng = np.random.default_rng(0)
    if args.audio == "concat":
        t0 = time.time()
        long_wav, src = build_concat_audio(max(lengths), find_audio_roots(args.input_dir),
                                           allow_missing=args.allow_missing)
        print(f"audio ghép: {src['total_s']:.0f} s từ {src['n_videos']} video, {src['n_segments']} đoạn "
              f"({time.time() - t0:.0f} s)", flush=True)
    else:
        long_wav = (0.1 * rng.standard_normal(int(max(lengths) * SAMPLE_RATE))).astype(np.float32)
        src = {"total_s": len(long_wav) / SAMPLE_RATE, "kind": "noise", "std": 0.1, "seed": 0}
    (out / "audio_source.json").write_text(json.dumps(src, ensure_ascii=False, indent=1), encoding="utf-8")
    noise_wav = (0.1 * rng.standard_normal(int(max(noise_lengths or [1]) * SAMPLE_RATE))).astype(np.float32)

    jsonl = out / "results.jsonl"
    f = open(jsonl, "a", encoding="utf-8")

    def record(row: dict) -> None:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()

    measure_kw = dict(min_runs=args.min_runs, max_runs=args.max_runs, target_s=args.target_s,
                      fake_oom_s=args.fake_oom_s)
    runners = [("ctc", name, cfg) for name, cfg in models.items()]
    if args.parakeet_nemo:
        runners.append(("parakeet", "parakeet", args.parakeet_nemo))
    vocab = BPETokenizer(args.tokenizer_model).vocab_size if models else None
    for kind, name, cfg in runners:
        if kind == "ctc":
            runner = CTCRunner(name, cfg, device, vocab, amp, checkpoints.get(name), args.interctc)
        else:
            runner = ParakeetRunner(cfg, device, args.parakeet_precision)
        record({**runner.info, "kind": "model"})
        print(f"=== {name}: {json.dumps(runner.info, ensure_ascii=False)}", flush=True)
        sweep(runner, long_wav, lengths, record, args.audio, args.bisect_steps, args.max_run_s, **measure_kw)
        if noise_lengths:
            sweep(runner, noise_wav, noise_lengths, record, "noise_check", 0, args.max_run_s, **measure_kw)
        del runner
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    f.close()
    write_csv(jsonl, out / "results.csv")
    print(f"XONG → {jsonl}, {out / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
