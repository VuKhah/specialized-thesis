"""Demo Gradio (Chương 4): chạy cùng một audio qua các mô hình train từ đầu, so transcript cạnh nhau.

    python -m src.demo.app                     # http://127.0.0.1:7860
    python -m src.demo.app --share             # link công khai tạm thời của Gradio

Mô hình nào có `checkpoints/<experiment_name>/best.pt` thì nạp (đổi bằng `--checkpoint tên=đường_dẫn`).
Máy không có `mamba-ssm` (Windows/CPU): khối Mamba chạy bằng `src/models/mamba_ref.py` (selective_scan_ref thuần
PyTorch, kiểm 2026-10-06: 37/40 câu val_unseen trùng tuyệt đối với eval CUDA trên Kaggle) — đúng nhưng chậm.

Cạm bẫy khi trình bày:
- Transcript chuẩn của clean-test hiện là **nhãn giả** (Zipformer sinh) cho tới khi hiệu đính xong — WER câu
  ở đây là so với nhãn giả, giao diện ghi rõ nguồn.
- Thời gian đo là front-end + encoder + CTC head + giải mã greedy, batch 1, trên thiết bị đang chạy demo; số
  RQ2 chính thức đo bằng `src/evaluation/rtf.py` trên T4, không lấy từ đây.
- Mô hình chỉ train trên đoạn 10-15 s; audio dài hơn nhiều là ngoài phân bố (cảnh báo, vẫn chạy).
"""

import argparse
import html
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import jiwer
import soundfile as sf
import torch
import torchaudio
import yaml

from src.data.vietsuperspeech_dataset import AUDIO_CACHE_DIR, load_excluded
from src.evaluation.text_normalize import normalize_text
from src.models import mamba_encoder
from src.models.mamba_ref import use_reference_mamba
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.train import CHECKPOINT_ROOT, build_model

SAMPLE_RATE = 16000
TRAIN_MAX_SECONDS = 15.0
MODEL_CONFIGS = {
    "Conformer-CTC": "configs/model_conformer.yaml",
    "Mamba B1-CTC": "configs/model_mamba.yaml",
    "ConExtBiMamba-CTC": "configs/model_conextbimamba.yaml",
}
MAMBA_TYPES = {"mamba", "conextbimamba"}
CLEAN_TEST_MANIFEST = "data/processed/clean_test_manifest.json"
TRAIN_FULL_RESULTS = Path("reports/results/train_full")


@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module | None
    device: str
    status: str
    info: dict


def _train_metrics(experiment_name: str, epoch: int | None) -> dict:
    # Phiên sau chép nối metrics của phiên trước → lấy file của phiên số lớn nhất.
    files = sorted(TRAIN_FULL_RESULTS.glob(f"*/s*/checkpoints/{experiment_name}/metrics.jsonl"),
                   key=lambda p: int(p.parts[-4][1:]))
    if not files or epoch is None:
        return {}
    rows = [json.loads(line) for line in files[-1].read_text(encoding="utf-8").splitlines() if line.strip()]
    return next((r for r in rows if r.get("epoch") == epoch), {})


def load_models(tokenizer: BPETokenizer, overrides: dict[str, str]) -> dict[str, LoadedModel]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference_mamba = not mamba_encoder.MAMBA_SSM_AVAILABLE
    use_reference_mamba()
    loaded = {}
    for name, cfg_path in MODEL_CONFIGS.items():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        exp = cfg["experiment_name"]
        ckpt = Path(overrides.get(exp, CHECKPOINT_ROOT / exp / "best.pt"))
        info = {"experiment_name": exp, "checkpoint": str(ckpt), "encoder": cfg["encoder"]["type"]}
        if not ckpt.exists():
            loaded[name] = LoadedModel(name, None, device, f"chưa có checkpoint ({ckpt})", info)
            continue
        try:
            model = build_model(cfg, tokenizer.vocab_size)
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state["model"])
            model.to(device).eval()
        except Exception as e:  # demo không được sập vì một mô hình
            loaded[name] = LoadedModel(name, None, device, f"lỗi nạp: {e}", info)
            continue
        info.update(
            epoch=state.get("epoch"),
            encoder_params=sum(p.numel() for p in model.encoder.parameters()),
            best_wer_val=state.get("best_wer"),
            metrics=_train_metrics(exp, state.get("epoch")),
        )
        status = "đã nạp"
        if cfg["encoder"]["type"] in MAMBA_TYPES and reference_mamba:
            status += " · khối Mamba thuần PyTorch (CPU, chậm)"
        loaded[name] = LoadedModel(name, model, device, status, info)
        print(f"[demo] {name}: {ckpt} (epoch {state.get('epoch')}, {device})", flush=True)
    return loaded


def load_clean_test() -> list[dict]:
    with open(CLEAN_TEST_MANIFEST, encoding="utf-8") as f:
        samples = json.load(f)["samples"]
    excluded = load_excluded()
    return [s for s in samples if (s["split"], s["index"]) not in excluded]


def read_audio(path: str) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav).mean(dim=1)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


@torch.no_grad()
def transcribe(lm: LoadedModel, tokenizer: BPETokenizer, wav: torch.Tensor) -> tuple[str, float]:
    x = wav.unsqueeze(0).to(lm.device)
    lengths = torch.tensor([wav.numel()], device=lm.device)
    if lm.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ids = lm.model.greedy_decode(x, lengths)[0]
    if lm.device == "cuda":
        torch.cuda.synchronize()
    return tokenizer.decode(ids), time.perf_counter() - t0


def diff_html(ref: str, hyp: str) -> tuple[str, dict]:
    """Tô hyp theo căn chỉnh jiwer (cùng `normalize_text` với WER chính thức)."""
    ref_n, hyp_n = normalize_text(ref), normalize_text(hyp)
    if not ref_n:
        return html.escape(hyp_n), {}
    out = jiwer.process_words(ref_n, hyp_n)
    rw, hw = ref_n.split(), hyp_n.split()
    parts = []
    for ch in out.alignments[0]:
        r, h = rw[ch.ref_start_idx:ch.ref_end_idx], hw[ch.hyp_start_idx:ch.hyp_end_idx]
        if ch.type == "equal":
            parts += [html.escape(w) for w in h]
        elif ch.type == "substitute":
            parts += [f'<span class="sub" title="chuẩn: {html.escape(a)}">{html.escape(b)}'
                      f'<sup>{html.escape(a)}</sup></span>' for a, b in zip(r, h)]
        elif ch.type == "delete":
            parts += [f'<span class="del" title="bị bỏ sót">{html.escape(a)}</span>' for a in r]
        elif ch.type == "insert":
            parts += [f'<span class="ins" title="bị chèn">{html.escape(b)}</span>' for b in h]
    stats = {"wer": out.wer, "S": out.substitutions, "D": out.deletions, "I": out.insertions}
    return " ".join(parts), stats


def result_card(name: str, lm: LoadedModel, ref: str, hyp: str | None, seconds: float | None, duration: float) -> str:
    if lm.model is None:
        return f'<div class="card off"><div class="head"><b>{name}</b><span>{html.escape(lm.status)}</span></div></div>'
    body, st = diff_html(ref, hyp)
    rtf = seconds / duration if duration else 0.0
    meta = f"{seconds * 1000:.0f} ms · RTF {rtf:.3f} · {lm.device}"
    wer = (f'<span class="wer">WER câu {st["wer"] * 100:.1f}%</span>'
           f'<span class="sdi">S {st["S"]} · D {st["D"]} · I {st["I"]}</span>') if st else ""
    return (f'<div class="card"><div class="head"><b>{name}</b>{wer}</div>'
            f'<div class="text">{body or "<i>(rỗng)</i>"}</div><div class="meta">{meta}</div></div>')


LEGEND = ('<div class="legend"><span class="sub">thay thế<sup>từ chuẩn</sup></span>'
          '<span class="del">bị bỏ sót</span><span class="ins">bị chèn</span></div>')

CSS = """
.card{border:1px solid var(--border-color-primary);border-radius:10px;padding:12px 14px;margin:8px 0}
.card.off{opacity:.6}
.card .head{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;margin-bottom:6px}
.card .wer{font-weight:600}
.card .sdi,.card .meta,.card.off .head span{color:var(--body-text-color-subdued);font-size:.9em}
.card .text,.ref{line-height:1.9;font-size:1.05em}
.ref{padding:10px 14px;border-radius:10px;background:var(--background-fill-secondary)}
.sub{background:#fde68a55;border-bottom:2px solid #d97706;border-radius:3px;padding:0 2px}
.sub sup{color:#b45309;font-size:.7em;margin-left:2px}
.del{background:#fecaca55;color:#b91c1c;text-decoration:line-through;border-radius:3px;padding:0 2px}
.ins{background:#bfdbfe55;border-bottom:2px dashed #2563eb;border-radius:3px;padding:0 2px}
.legend{display:flex;gap:14px;font-size:.9em;margin:4px 0}
.warn{color:#b45309}
"""


def build_app(models: dict[str, LoadedModel], tokenizer: BPETokenizer, clean_test: list[dict]) -> gr.Blocks:
    choices = [(f"#{i + 1} · {s['duration_s']:.1f} s · {Path(s['source']).stem[:60]}", i)
               for i, s in enumerate(clean_test)]

    def run_all(wav: torch.Tensor, ref: str) -> str:
        duration = wav.numel() / SAMPLE_RATE
        cards = []
        for name, lm in models.items():
            hyp, sec = transcribe(lm, tokenizer, wav) if lm.model is not None else (None, None)
            cards.append(result_card(name, lm, ref, hyp, sec, duration))
        return LEGEND * bool(ref.strip()) + "".join(cards)

    def on_clean_test(i: int):
        s = clean_test[i]
        path = str(Path(AUDIO_CACHE_DIR) / s["audio"])
        ref = s["corrected_text"].strip() or s["pseudo_label"]
        src = "bản hiệu đính tay" if s["corrected_text"].strip() else "nhãn giả (Zipformer) — chưa hiệu đính"
        ref_box = f'<div class="ref"><b>Transcript chuẩn</b> <small>({src})</small><br>{html.escape(normalize_text(ref))}</div>'
        return path, ref_box, run_all(read_audio(path), ref)

    def on_upload(path: str | None, ref: str):
        if not path:
            return "<i>Chưa có audio.</i>"
        wav = read_audio(path)
        duration = wav.numel() / SAMPLE_RATE
        warn = (f'<p class="warn">Audio {duration:.1f} s — dài hơn đoạn train (10-15 s), kết quả ngoài phân bố.</p>'
                if duration > TRAIN_MAX_SECONDS * 1.5 else "")
        return warn + run_all(wav, ref or "")

    def info_table() -> list[list]:
        rows = []
        for name, lm in models.items():
            inf, m = lm.info, lm.info.get("metrics", {})
            fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "—"  # noqa: E731
            rows.append([name, inf["encoder"], f"{inf['encoder_params']:,}".replace(",", ".") if "encoder_params" in inf else "—",
                         inf.get("epoch", "—"), fmt(m.get("wer_val")), fmt(m.get("wer_val_unseen")), lm.status])
        return rows

    with gr.Blocks(title="ASR tiếng Việt hội thoại · Mamba vs Conformer") as app:
        gr.Markdown("## ASR tiếng Việt hội thoại · encoder Mamba vs Conformer\n"
                    "Cùng front-end log-mel, tokenizer BPE 1000, CTC head, giải mã greedy không LM — chỉ encoder khác nhau.")
        with gr.Tab("Clean-test"):
            with gr.Row():
                pick = gr.Dropdown(choices=choices, value=0, label=f"Chọn câu ({len(clean_test)} câu, video giữ riêng)", scale=3)
                audio_ct = gr.Audio(label="Audio", type="filepath", interactive=False, scale=2)
            ref_ct = gr.HTML()
            out_ct = gr.HTML()
            pick.change(on_clean_test, pick, [audio_ct, ref_ct, out_ct])
            app.load(on_clean_test, pick, [audio_ct, ref_ct, out_ct])
        with gr.Tab("Audio của bạn"):
            with gr.Row():
                audio_in = gr.Audio(label="Tải file hoặc ghi âm (tự chuyển 16 kHz mono)",
                                    sources=["upload", "microphone"], type="filepath")
                ref_in = gr.Textbox(label="Transcript chuẩn (tuỳ chọn, để tính WER)", lines=4)
            btn = gr.Button("Nhận dạng", variant="primary")
            out_in = gr.HTML()
            btn.click(on_upload, [audio_in, ref_in], out_in)
        with gr.Tab("Thông tin mô hình"):
            gr.Dataframe(value=info_table(), interactive=False,
                         headers=["Mô hình", "Encoder", "Tham số encoder", "Epoch (best.pt)",
                                  "WER val", "WER val_unseen", "Trạng thái"])
            gr.Markdown("WER val / val_unseen: so với nhãn giả, giải mã greedy, lấy từ `metrics.jsonl` của epoch best.pt. "
                        "Dữ liệu train: VietSuperSpeech (4 shard, đã bỏ `excluded.tsv` và video giữ riêng).")
    return app


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream.encoding != "utf-8":
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", default=[], metavar="EXPERIMENT=PATH",
                    help="Đổi checkpoint, vd. conformer_ctc_baseline=D:/kltn_checkpoints/conformer/best.pt")
    ap.add_argument("--tokenizer_model", default="configs/tokenizer.model")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    tokenizer = BPETokenizer(args.tokenizer_model)
    models = load_models(tokenizer, dict(kv.split("=", 1) for kv in args.checkpoint))
    if not any(lm.model is not None for lm in models.values()):
        sys.exit("Không nạp được mô hình nào — cần ít nhất một checkpoints/<experiment_name>/best.pt")
    build_app(models, tokenizer, load_clean_test()).launch(server_port=args.port, share=args.share, css=CSS)


if __name__ == "__main__":
    main()
