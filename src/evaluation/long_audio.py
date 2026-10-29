"""WER khi encoder nhận cả cửa sổ audio dài (không cắt) — phần chạy mô hình: ghi giả thuyết greedy từng cửa sổ.

    python -m src.evaluation.long_audio run --model <b1_mamba_40ms|conextbimamba_40ms|conformer_relpe_40ms> --seed 42
        [--L seg L1 L5 all] [--limit-chains N] [--out-dir D] [--threads 1]

Quy tắc chốt trước (dữ liệu, cửa sổ, giới hạn bộ nhớ Conformer, cách chấm):
`docs/notes/ket_qua/audio_dai_nguyen_chuoi_quy_tac_2026-10-27.md`. Chấm điểm ở bước riêng, đọc
`<out-dir>/hyps/<model>_s<seed>.jsonl` (mỗi dòng một cửa sổ; dòng `error` có thể lặp khi chạy lại — lấy dòng cuối).
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.demo.longform import CHAINS, CKPT_DIR, MODELS, SAMPLE_RATE, log_probs, read_audio

LEVELS = {"seg": None, "L1": 60.0, "L5": 300.0, "all": float("inf")}  # trần thời lượng cửa sổ (s)
FRAMES_PER_S = 25  # đầu ra encoder 40 ms
CONFORMER_MAX_FRAMES = 9000  # attention O(T²): 6 phút là trần chạy được trên máy 16 GB (quy tắc mục 2)
OUT = Path("reports/results/long_audio_2026-10-27")
DONE = ("ok", "skip_mem")


def windows(chain: dict, L: str, durations: list[float] | None = None) -> list[list[int]]:
    """Chỉ số đoạn (trong `chain`) của từng cửa sổ. Chỉ gom đoạn nguyên → tham chiếu ghép khớp đúng; đoạn đầu tiên
    luôn vào cửa sổ dù dài hơn trần (không xảy ra với đoạn 10–15 s). `durations` = số mẫu / 16000 của audio đã đọc."""
    if durations is None:
        durations = [len(read_audio(p)) / SAMPLE_RATE for p in chain["paths"]]
    if L == "seg":
        return [[i] for i in range(len(durations))]
    cap, out, cur, total = LEVELS[L], [], [], 0.0
    for i, d in enumerate(durations):
        if cur and total + d > cap:
            out.append(cur)
            cur, total = [], 0.0
        cur.append(i)
        total += d
    return out + [cur] if cur else out


def load_checkpoint(name: str, seed: int):
    """= `longform.load_model`, chọn checkpoint theo seed (tên file của `MODELS` là seed 42)."""
    from src.models import mamba_encoder
    from src.models.mamba_ref import use_reference_mamba
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    from src.training.train import build_model
    if not mamba_encoder.MAMBA_SSM_AVAILABLE:
        use_reference_mamba()
    cfg_path, fname = MODELS[name]
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    model = build_model(cfg, BPETokenizer("configs/tokenizer.model").vocab_size)
    model.load_state_dict(torch.load(CKPT_DIR / fname.replace("seed42", f"seed{seed}"), map_location="cpu")["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), device


class PeakRSS:
    """RSS đỉnh của tiến trình (MB) trong khối `with`, lấy mẫu mỗi 50 ms ở luồng phụ — torch nhả GIL khi tính nên
    luồng phụ vẫn chạy; đỉnh nhọn ngắn hơn 50 ms có thể bị sót. Không có psutil → `mb` = None."""

    def __init__(self, interval: float = 0.05):
        try:
            import psutil
            self.proc = psutil.Process()
        except ImportError:
            self.proc = None
        self.interval, self.mb = interval, None

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            self._peak = max(self._peak, self.proc.memory_info().rss)

    def __enter__(self):
        if self.proc is not None:
            self._peak, self._stop = self.proc.memory_info().rss, threading.Event()
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self.proc is not None:
            self._stop.set()
            self._thread.join()
            self.mb = round(max(self._peak, self.proc.memory_info().rss) / 2**20, 1)
        return False


def _done(path: Path) -> set[tuple[str, str, int]]:
    """Cửa sổ đã xong (ok/skip_mem) → chạy tiếp không ghi trùng; dòng `error` được chạy lại."""
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # dòng cụt do tiến trình bị ngắt giữa lúc ghi
                continue
            if r.get("status") in DONE:
                done.add((r["chain"], r["L"], r["window"]))
    return done


def run_window(model, device: str, tok, wav: np.ndarray, conformer: bool) -> dict:
    from src.evaluation.lm_decode import greedy
    est = len(wav) / SAMPLE_RATE * FRAMES_PER_S
    if conformer and est > CONFORMER_MAX_FRAMES:
        return {"frames": round(est), "status": "skip_mem", "hyp": "", "sec": None, "peak_mb": None}
    try:
        with PeakRSS() as mem:
            t0 = time.perf_counter()
            lp = log_probs(model, device, wav)
            hyp = greedy(lp, tok)
            sec = time.perf_counter() - t0
        return {"frames": int(lp.shape[0]), "status": "ok", "hyp": hyp, "sec": round(sec, 3), "peak_mb": mem.mb}
    except Exception as e:  # MemoryError, lỗi cấp phát của torch (RuntimeError)… → ghi lại, chạy cửa sổ sau
        return {"frames": round(est), "status": "error", "hyp": "", "sec": None, "peak_mb": mem.mb,
                "error": f"{type(e).__name__}: {e}"[:500]}


def cmd_run(a) -> None:
    from src.evaluation.lm_decode import _Tokens
    torch.set_num_threads(a.threads)
    chains = json.loads((CHAINS / "chains.json").read_text(encoding="utf-8"))
    if a.limit_chains:
        chains = chains[: a.limit_chains]
    out = Path(a.out_dir) / "hyps" / f"{a.model}_s{a.seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done(out)
    model, device = load_checkpoint(a.model, a.seed)
    tok, conformer = _Tokens(), a.model.startswith("conformer")
    log_probs(model, device, np.zeros(12 * SAMPLE_RATE, np.float32))  # làm ấm: lần gọi đầu chậm hơn hẳn
    if out.exists() and out.stat().st_size and not out.read_bytes().endswith(b"\n"):
        with out.open("a", encoding="utf-8") as f:  # dòng cụt cuối file: xuống dòng để dòng mới không dính vào
            f.write("\n")
    t_all, n_new = time.perf_counter(), 0
    with out.open("a", encoding="utf-8") as f:
        for k, c in enumerate(chains, 1):
            wavs = [read_audio(p) for p in c["paths"]]
            durs = [len(x) / SAMPLE_RATE for x in wavs]
            todo = [(L, w, idx) for L in a.L for w, idx in enumerate(windows(c, L, durs))
                    if (c["name"], L, w) not in done]
            for L, w, idx in todo:
                wav = np.concatenate([wavs[i] for i in idx])
                r = {"chain": c["name"], "video": c["video"], "L": L, "window": w, "keys": [c["keys"][i] for i in idx],
                     "duration_s": round(len(wav) / SAMPLE_RATE, 3)} | run_window(model, device, tok, wav, conformer)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                n_new += 1
                print(f"[{k}/{len(chains)}] {c['name'][:44]} {L}#{w}: {r['duration_s']:.0f} s, {r['frames']} khung, "
                      f"{r['status']}" + (f", {r['sec']:.1f} s, {r['peak_mb']} MB" if r["status"] == "ok" else "")
                      + (f" — {r['error']}" if r["status"] == "error" else ""), flush=True)
    print(f"{n_new} cửa sổ mới trong {time.perf_counter() - t_all:.0f} s → {out}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", choices=list(MODELS), required=True)
    r.add_argument("--seed", type=int, choices=[42, 43, 44], required=True)
    r.add_argument("--L", nargs="+", choices=list(LEVELS), default=list(LEVELS))
    r.add_argument("--limit-chains", type=int, default=0, help="chỉ N chuỗi đầu (chạy thử)")
    r.add_argument("--out-dir", default=str(OUT))
    r.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    {"run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
