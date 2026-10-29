"""Demo Gradio (Chương 4): chạy cùng một audio qua các mô hình train từ đầu, so transcript cạnh nhau.

    python -m src.demo.app                     # http://127.0.0.1:7860
    python -m src.demo.app --share             # link công khai tạm thời của Gradio

Nạp 7 checkpoint ep37 của bảng chính (cập nhật 2026-10-16): 4 encoder cấu hình 10 ms + 3 encoder cấu hình 40 ms
(mặc định seed 42, đổi bằng `--seed 43|44`) từ `--ckpt_dir` (mặc định `checkpoints/final/`, tên file như bản sao lưu
Drive `checkpoints_kltn_2026-10-16`). Đổi một file bằng `--checkpoint tên_file=đường_dẫn`.
Máy không có `mamba-ssm` (Windows/CPU): khối Mamba chạy bằng `src/models/mamba_ref.py` (selective_scan_ref thuần
PyTorch, kiểm 2026-10-06: 37/40 câu val_unseen trùng tuyệt đối với eval CUDA trên Kaggle) — đúng nhưng chậm.

Cạm bẫy khi trình bày:
- WER câu ở đây chỉ qua `normalize_text` (không áp luật từ đệm / cách viết thay thế của lần chấm chính thức
  `score_clean_test`) — để minh hoạ, không thay số trong bảng.
- Thời gian đo là front-end + encoder + CTC head + giải mã greedy, batch 1, trên thiết bị đang chạy demo; số
  RQ2 chính thức đo bằng `src/evaluation/rtf.py` trên T4, không lấy từ đây.
- Mô hình chỉ train trên đoạn 10-15 s; audio dài hơn nhiều là ngoài phân bố (cảnh báo, vẫn chạy).
"""

import argparse
import csv
import hashlib
import html
import json
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import jiwer
import numpy as np
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
# (tên hiển thị, config, file checkpoint trong --ckpt_dir; {seed} điền theo --seed)
MODEL_SPECS = [
    ("B1 Mamba · 40 ms", "configs/v2a/model_mamba_sub4.yaml", "b1_mamba_40ms_seed{seed}_ep37.pt"),
    ("ConExtBiMamba · 40 ms", "configs/v2a/model_conextbimamba_sub4.yaml", "conextbimamba_40ms_seed{seed}_ep37.pt"),
    ("Conformer relPE · 40 ms", "configs/v2a/model_conformer_relpos_sub4.yaml", "conformer_relpe_40ms_seed{seed}_ep37.pt"),
    ("B1 Mamba · 10 ms", "configs/v2a/model_mamba.yaml", "b1_mamba_10ms_ep37.pt"),
    ("ConExtBiMamba · 10 ms", "configs/v2a/model_conextbimamba.yaml", "conextbimamba_10ms_ep37.pt"),
    ("Conformer relPE · 10 ms", "configs/v2a/model_conformer_relpos.yaml", "conformer_relpe_10ms_ep37.pt"),
    ("Conformer không PE · 10 ms (ablation)", "configs/v2a/model_conformer.yaml", "conformer_khong_pe_10ms_ep37_ablation.pt"),
]
MAMBA_TYPES = {"mamba", "conextbimamba"}
CLEAN_TEST_MANIFEST = "data/processed/clean_test_manifest.json"


@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module | None
    device: str
    status: str
    info: dict


def load_models(tokenizer: BPETokenizer, overrides: dict[str, str], ckpt_dir: Path, seed: int) -> dict[str, LoadedModel]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference_mamba = not mamba_encoder.MAMBA_SSM_AVAILABLE
    use_reference_mamba()
    loaded = {}
    for name, cfg_path, fname in MODEL_SPECS:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        exp = cfg["experiment_name"]
        fname = fname.format(seed=seed)
        ckpt = Path(overrides.get(fname, ckpt_dir / fname))
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
            metrics={k: state.get(k) for k in ("wer_val", "wer_val_unseen")},
            sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),  # đối chiếu SHA256SUMS.txt của bản sao lưu
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
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
    except sf.LibsndfileError:  # webm/m4a từ trình duyệt hoặc điện thoại: soundfile không đọc, nhờ ffmpeg
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
                             capture_output=True, check=True).stdout
        return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())
    wav = torch.from_numpy(wav).mean(dim=1)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


GREEDY, BEAM_LM = "greedy (không LM)", "beam + LM 3-gram"


class LMDecoder:
    """Beam + từ điển + LM 3-gram của Bảng 3.4 (`src.evaluation.lm_decode`, tham số chốt 15/10, không chỉnh ở demo).
    Nạp lần đầu khi cần: LM ~45 MB, dựng decoder vài giây.

    Có tiêu đề tập (tuỳ chọn) → cơ chế (iii) của bổ sung A (`src.evaluation.bias_title`): từ không phải âm tiết tiếng
    Việt trong tiêu đề được thêm vào từ điển và cộng b (chọn trên nửa tune val_unseen, đọc từ `grid.json`). Tiêu đề
    trống → đúng decoder gốc."""

    def __init__(self):
        self._parts = None
        self._biased = None

    def available(self) -> bool:
        from src.evaluation.lm_decode import LM_DIR
        return (LM_DIR / "lm3.pkl").exists() and (LM_DIR / "lexicon.txt").exists()

    @staticmethod
    def title_bonus() -> float | None:
        from src.evaluation import bias_title
        try:
            return bias_title.chosen_b()
        except (OSError, KeyError, ValueError):
            return None

    @staticmethod
    def title_words(title: str) -> list[str]:
        from src.evaluation import bias_title
        from src.evaluation.score_clean_test import load_alt
        return bias_title.context_words(title, load_alt()) if title and title.strip() else []

    def __call__(self, log_probs, title: str = "") -> str:
        from src.evaluation import lm_decode as L
        if self._parts is None:
            tok = L._Tokens(); knlm = L.load_lm(L.LM_DIR / "lm3.pkl")
            self._parts = (tok, knlm, L.make_decoder(L.LM_DIR / "lexicon.txt", knlm, tok))
        tok, knlm, dec = self._parts
        words, b = self.title_words(title), self.title_bonus()
        if words and b is not None:
            if self._biased is None:
                self._biased = L.BiasedDecoders(knlm, tok, bonus=b)
            dec = self._biased(words)
        # Cùng luật với `lm_decode.decode`: beam ngắn hơn FALLBACK × greedy → greedy đã tách từ dính.
        return L.lm_hyp(dec, knlm, tok, log_probs)[1]


LM_DECODER = LMDecoder()


@torch.no_grad()
def transcribe(lm: LoadedModel, tokenizer: BPETokenizer, wav: torch.Tensor, mode: str = GREEDY,
               title: str = "") -> tuple[str, float]:
    x = wav.unsqueeze(0).to(lm.device)
    lengths = torch.tensor([wav.numel()], device=lm.device)
    if lm.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    if mode == BEAM_LM:
        log_probs, out_lengths = lm.model(x, lengths)
        text = LM_DECODER(log_probs[0, : int(out_lengths[0])].float().cpu().numpy(), title)
    else:
        text = tokenizer.decode(lm.model.greedy_decode(x, lengths)[0])
    if lm.device == "cuda":
        torch.cuda.synchronize()
    return text, time.perf_counter() - t0


def diff_html(ref: str, hyp: str, show_errors: bool = True) -> tuple[str, dict]:
    """Căn chỉnh jiwer (cùng `normalize_text` với WER chính thức). show_errors=False: chữ sạch, chỉ trả số lỗi."""
    ref_n, hyp_n = normalize_text(ref), normalize_text(hyp)
    if not ref_n:
        return html.escape(hyp_n), {}
    out = jiwer.process_words(ref_n, hyp_n)
    stats = {"wer": out.wer, "S": out.substitutions, "D": out.deletions, "I": out.insertions, "N": len(ref_n.split())}
    if not show_errors:
        return html.escape(hyp_n), stats
    rw, hw = ref_n.split(), hyp_n.split()
    parts = []
    for ch in out.alignments[0]:
        r, h = rw[ch.ref_start_idx:ch.ref_end_idx], hw[ch.hyp_start_idx:ch.hyp_end_idx]
        if ch.type == "equal":
            parts += [html.escape(w) for w in h]
        elif ch.type == "substitute":  # từ chuẩn chỉ hiện khi rê chuột — in kèm làm dòng chữ rối gấp đôi
            parts += [f'<span class="sub" title="chuẩn: {html.escape(a)}">{html.escape(b)}</span>' for a, b in zip(r, h)]
        elif ch.type == "delete":
            parts += [f'<span class="del" title="bị bỏ sót: {html.escape(a)}">{html.escape(a)}</span>' for a in r]
        elif ch.type == "insert":
            parts += [f'<span class="ins" title="bị chèn">{html.escape(b)}</span>' for b in h]
    return " ".join(parts), stats


def result_card(name: str, lm: LoadedModel, ref: str, hyp: str | None, seconds: float | None, duration: float,
                mode: str = GREEDY, show_errors: bool = False) -> str:
    if lm.model is None:
        return f'<div class="card off"><div class="head"><b>{name}</b><span>{html.escape(lm.status)}</span></div></div>'
    body, st = diff_html(ref, hyp, show_errors)
    rtf = seconds / duration if duration else 0.0
    meta = f"{mode} · {seconds * 1000:.0f} ms · RTF {rtf:.3f} · {lm.device}"
    wer = (f'<span class="wer">WER câu {st["wer"] * 100:.1f}%</span>'
           f'<span class="sdi">thay {st["S"]} · thiếu {st["D"]} · thừa {st["I"]} / {st["N"]} từ</span>') if st else ""
    return (f'<div class="card"><div class="head"><b>{name}</b>{wer}</div>'
            f'<div class="text">{body or "<i>(rỗng)</i>"}</div><div class="meta">{meta}</div></div>')


LEGEND = ('<div class="legend"><span class="sub">từ bị thay (rê chuột xem từ chuẩn)</span>'
          '<span class="del">từ bị bỏ sót</span><span class="ins">từ bị chèn</span></div>')

CSS = """
.card{border:1px solid var(--border-color-primary);border-radius:10px;padding:12px 14px;margin:8px 0}
.card.off{opacity:.6}
.card .head{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:-12px -14px 10px;padding:10px 14px;
  background:var(--background-fill-secondary);border-bottom:1px solid var(--border-color-primary);
  border-left:4px solid var(--color-accent);border-radius:10px 10px 0 0}
.card .head b{font-size:1.15em}
.card .wer{font-weight:700;font-size:1.15em;padding:2px 12px;border-radius:999px;border:2px solid var(--color-accent);
  color:var(--body-text-color);background:var(--background-fill-primary)}
.card .sdi,.card .meta,.card.off .head span{color:var(--body-text-color-subdued);font-size:.9em}
.card .text,.ref{line-height:1.9;font-size:1.05em}
.ref{padding:10px 14px;border-radius:10px;background:var(--background-fill-secondary)}
.sub{text-decoration:underline wavy #d97706;text-decoration-thickness:1px;text-underline-offset:4px;cursor:help}
.del{color:var(--body-text-color-subdued);text-decoration:line-through;font-size:.92em;cursor:help}
.ins{text-decoration:underline dotted #2563eb;text-underline-offset:4px;cursor:help}
.legend{display:flex;gap:14px;font-size:.9em;margin:4px 0;flex-wrap:wrap}
.warn{color:#b45309}
.audio-info,.bench{color:var(--body-text-color-subdued);font-size:.92em;margin:4px 0}
.bench b{color:var(--body-text-color)}
.more-title{margin:14px 0 2px;font-weight:600;color:var(--body-text-color-subdued)}
"""

SCORED = Path("reports/results/clean_test_scored_2026-10-16_chung")
LM40 = [f"v2a_{m}_lm" for m in ("mamba", "conextbimamba", "conformer_relpos")]


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%".replace(".", ",")


def benchmarks() -> tuple[str, dict]:
    """Khung tham chiếu (WER clean-test chính thức) + độ khó từng câu — đọc từ lần chấm chung, không gõ tay số."""
    try:
        res = json.loads((SCORED / "results.json").read_text(encoding="utf-8"))["systems"]
        lm40 = [res[c]["wer"] for c in LM40]
        line = (f'<p class="bench">Tham chiếu trên clean-test (203 câu): nhãn giả Zipformer <b>{_pct(res["pseudo_label"]["wer"])}</b>'
                f' · Parakeet 0,6B fine-tune <b>{_pct(res["v2a10_parakeet_ft"]["wer"])}</b>'
                f' · 3 mô hình 40 ms seed 42 + LM <b>{_pct(min(lm40))}–{_pct(max(lm40))}</b>. WER 10% ≈ cứ 10 từ sai 1.</p>')
        per = {}
        with (SCORED / "per_sentence.tsv").open(encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["system"] in LM40:
                    e, w = per.get(r["key"], (0.0, 0.0))
                    per[r["key"]] = (e + float(r["err_vi"]) + float(r["err_fx"]), w + float(r["words_vi"]) + float(r["words_fx"]))
        wer = {k: e / w for k, (e, w) in per.items() if w}
        vals = list(wer.values())
        rank = {k: sum(v > x for x in vals) / len(vals) for k, v in wer.items()}  # tỉ lệ câu dễ hơn câu này
        return line, rank
    except (OSError, KeyError, ValueError):
        return "", {}


def build_app(models: dict[str, LoadedModel], tokenizer: BPETokenizer, clean_test: list[dict]) -> gr.Blocks:
    choices = [(f"#{i + 1} · {s['duration_s']:.1f} s · {Path(s['source']).stem[:60]}", i)
               for i, s in enumerate(clean_test)]
    main_names = [n for n in models if "40 ms" in n]  # cấu hình tốt nhất hiện mặc định; 10 ms + ablation ở "Xem thêm"
    more_names = [n for n in models if n not in main_names]
    bench, difficulty = benchmarks()

    def decode(state: dict, names: list[str]) -> None:
        """Giải mã các mô hình chưa có trong state (giữ kết quả để bật/tắt hiển thị không phải chạy lại)."""
        todo = [n for n in names if n not in state["results"] and models[n].model is not None]
        if not todo:
            return
        wav = read_audio(state["path"])
        for n in todo:
            state["results"][n] = transcribe(models[n], tokenizer, wav, state["mode"], state.get("title", ""))

    def title_note(state: dict) -> str:
        words, b = LM_DECODER.title_words(state.get("title", "")), LM_DECODER.title_bonus()
        if not words or state["mode"] != BEAM_LM or b is None:
            return ""
        return (f'<p class="audio-info">Ngữ cảnh tiêu đề (bổ sung A, cơ chế (iii)): thêm vào từ điển và cộng thưởng '
                f'b = {b:g} cho <b>{html.escape(" ".join(words))}</b></p>')

    def render(state: dict | None, show_errors: bool, show_more: bool) -> str:
        if not state:
            return "<i>Chưa có audio — bấm micro để thu hoặc kéo file vào.</i>"
        decode(state, main_names + (more_names if show_more else []))
        ref, dur, mode = state["ref"], state["duration"], state["mode"]

        def cards(names):
            return "".join(result_card(n, models[n], ref, *state["results"].get(n, (None, None)), dur, mode, show_errors)
                           for n in names)

        out = (state.get("prefix", "") + title_note(state) + bench + (LEGEND if show_errors and ref.strip() else "")
               + cards(main_names))
        if show_more:
            out += '<div class="more-title">Cấu hình 10 ms và ablation</div>' + cards(more_names)
        return out

    def on_clean_test(i: int, mode: str, title: str, show_errors: bool, show_more: bool):
        s = clean_test[i]
        path = str(Path(AUDIO_CACHE_DIR) / s["audio"])
        ref = s["corrected_text"].strip() or s["pseudo_label"]
        key = f"{s['split']}:{s['index']}"
        hard = (f" · câu này khó hơn {difficulty[key] * 100:.0f}% số câu clean-test (theo WER của 3 mô hình 40 ms + LM)"
                if key in difficulty else "")
        ref_box = (f'<div class="ref"><b>Transcript chuẩn</b> <small>(bản hiệu đính tay{hard})</small><br>'
                   f'{html.escape(normalize_text(ref))}</div>')
        state = {"path": path, "ref": ref, "mode": mode, "results": {}, "duration": sf.info(path).duration,
                 "title": title or ""}
        return path, ref_box, render(state, show_errors, show_more), state

    def on_random(i: int) -> int:
        return random.choice([j for j in range(len(clean_test)) if j != i])

    def video_title(i: int) -> str:
        return Path(clean_test[i]["source"]).stem  # tên file video = nguồn danh sách của bổ sung A

    def on_upload(path: str | None, ref: str, mode: str, title: str, show_errors: bool, show_more: bool):
        if not path:
            return render(None, show_errors, show_more), None
        try:
            wav = read_audio(path)
        except Exception as e:  # định dạng lạ (webm/m4a không có bộ giải mã) → báo, không sập
            return f'<p class="warn">Không đọc được audio ({html.escape(Path(path).suffix)}): {html.escape(str(e))}</p>', None
        duration = wav.numel() / SAMPLE_RATE
        peak = float(wav.abs().max()) if wav.numel() else 0.0
        rms_db = 20 * torch.log10(wav.pow(2).mean().sqrt() + 1e-9).item()
        notes = []
        if duration < 1.0:
            notes.append("quá ngắn (< 1 s)")
        elif duration > TRAIN_MAX_SECONDS * 1.5:
            notes.append("dài hơn đoạn train (10-15 s) — kết quả ngoài phân bố")
        if rms_db < -45:
            notes.append("rất nhỏ — nói gần micro hơn")
        # opus/mp3 vượt đỉnh 1,0 khi giải mã dù không vỡ → xét tỉ lệ mẫu chạm trần thay vì chỉ đỉnh
        if wav.numel() and (wav.abs() >= 0.99).float().mean().item() > 1e-3:
            notes.append("bị vỡ tiếng (clipping) — nói nhỏ lại hoặc lùi xa micro")
        info = (f'<p class="audio-info">Audio {duration:.1f} s · mức trung bình {rms_db:.0f} dBFS · đỉnh {peak:.2f}'
                + (f' · <span class="warn">{"; ".join(notes)}</span>' if notes else "") + "</p>")
        state = {"path": path, "ref": ref or "", "mode": mode, "results": {}, "duration": duration, "prefix": info,
                 "title": title or ""}
        return render(state, show_errors, show_more), state

    long_decoders: dict = {}

    def on_longform(path: str | None, name: str, context: str):
        from src.demo import longform as LF
        if not path or name not in models:
            return "<i>Chưa có audio.</i>", None
        context = context or ""
        if context not in long_decoders:  # nạp LM + dựng decoder mất vài giây → giữ theo ngữ cảnh
            if len(long_decoders) >= 4:
                long_decoders.pop(next(iter(long_decoders)))
            long_decoders[context] = LF.Decoder(context)
        lm = models[name]
        res = LF.transcribe(read_audio(path).numpy(), lm.model, lm.device, long_decoders[context])
        res["model"] = name
        out = Path(tempfile.mkdtemp(prefix="longform_"))
        LF.write_outputs(res, Path(path), out)
        t = res["timing_s"]
        head = (f'<p class="audio-info">{res["duration_s"] / 60:.1f} phút · {res["n_segments"]} đoạn · encoder '
                f'{t["encoder"]:.0f} s + giải mã {t["decode"]:.0f} s · RTF {t["rtf"]:.3f} ({lm.device})'
                + (f' · ngữ cảnh: {html.escape(" ".join(res["context_words"][:20]))}' if res["context_words"] else "")
                + "</p>")
        stem = Path(path).stem
        return head + LF.transcript_spans(res, inline_color=True), [str(out / f"{stem}.{e}")
                                                                     for e in ("srt", "txt", "json", "html")]

    def info_table() -> list[list]:
        rows = []
        for name, lm in models.items():
            inf, m = lm.info, lm.info.get("metrics", {})
            fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "—"  # noqa: E731
            rows.append([name, inf["encoder"], f"{inf['encoder_params']:,}".replace(",", ".") if "encoder_params" in inf else "—",
                         inf.get("epoch", "—"), fmt(m.get("wer_val")), fmt(m.get("wer_val_unseen")), lm.status,
                         inf["checkpoint"], inf.get("sha256", "—")[:12]])
        return rows

    with gr.Blocks(title="ASR tiếng Việt hội thoại · Mamba vs Conformer") as app:
        gr.Markdown("## ASR tiếng Việt hội thoại · encoder Mamba vs Conformer\n"
                    "Cùng front-end log-mel, tokenizer BPE 1000, CTC head — chỉ encoder khác nhau. Checkpoint ep37; "
                    "40 ms là cấu hình subsampling 4×, 10 ms không subsampling.")
        with gr.Row():
            modes = [BEAM_LM, GREEDY] if LM_DECODER.available() else [GREEDY]
            mode = gr.Radio(modes, value=modes[0], label="Giải mã", scale=3,
                            info="beam + LM: beam 10, từ điển + LM 3-gram học từ nhãn giả train, lmw 0,8 / ws 1,0 (Bảng 3.4). "
                                 "Đổi chế độ rồi chọn câu khác / bấm nhận dạng lại.")
            show_errors = gr.Checkbox(value=False, label="Hiện lỗi", info="đánh dấu từ sai so với transcript chuẩn", scale=1)
            show_more = gr.Checkbox(value=False, label="Xem thêm 10 ms + ablation", info="chạy thêm 4 mô hình (chậm hơn)", scale=1)
        b_title = LM_DECODER.title_bonus() if LM_DECODER.available() else None
        title = gr.Textbox(
            label="Tiêu đề tập (tuỳ chọn, chỉ dùng với beam + LM)", lines=1, visible=b_title is not None,
            info=f"Tên riêng/từ tiếng Anh trong tiêu đề được thêm vào từ điển và cộng thưởng b = {b_title:g} (bổ sung A); "
                 "để trống = như Bảng 3.4. Nhập rồi chọn câu khác / bấm nhận dạng lại." if b_title is not None else None)
        toggles = [show_errors, show_more]
        with gr.Tab("Clean-test"):
            with gr.Row():
                pick = gr.Dropdown(choices=choices, value=0, label=f"Chọn câu ({len(clean_test)} câu, video giữ riêng)", scale=3)
                rand = gr.Button("Câu ngẫu nhiên", scale=1)
                use_name = gr.Button("Lấy tên file video làm tiêu đề", scale=1, visible=b_title is not None)
                audio_ct = gr.Audio(label="Audio", type="filepath", interactive=False, scale=2)
            ref_ct = gr.HTML()
            out_ct = gr.HTML()
            st_ct = gr.State(None)
            pick.change(on_clean_test, [pick, mode, title, *toggles], [audio_ct, ref_ct, out_ct, st_ct])
            app.load(on_clean_test, [pick, mode, title, *toggles], [audio_ct, ref_ct, out_ct, st_ct])
            rand.click(on_random, pick, pick)
            use_name.click(video_title, pick, title).then(on_clean_test, [pick, mode, title, *toggles],
                                                          [audio_ct, ref_ct, out_ct, st_ct])
        with gr.Tab("Thu âm"):
            gr.Markdown(
                "**Cách thử:** bấm micro → **chờ khoảng 1 giây** → nói 5-15 s giọng trò chuyện tự nhiên → bấm dừng; "
                "kết quả hiện sau khoảng 10 s. Nên ở phòng yên tĩnh, nói gần micro.\n\n"
                "**Điểm yếu đã biết** (Chương 3, RQ3): tên riêng, tên sản phẩm, từ tiếng Anh, từ địa phương ít gặp; "
                "nhầm âm đầu gần nhau (m/b, x/s, gi/d); từ đầu câu nếu nói ngay khi vừa bấm thu. "
                "Mô hình chỉ học ~165 giờ audio YouTube hội thoại với nhãn tự động.")
            with gr.Row():
                # Giữ định dạng gốc của trình duyệt (webm): format="wav" làm Gradio ghi wav int32 mà trình phát trong
                # trình duyệt không giải mã được → khung audio trống sau khi dừng (tái hiện bằng Chrome 16/10).
                # Đổi sang 16 kHz mono làm ở read_audio (ffmpeg cho webm).
                mic_in = gr.Audio(label="Thu âm", sources=["microphone"], type="filepath")
                ref_mic = gr.Textbox(label="Câu bạn vừa nói (tuỳ chọn, để tính WER)", lines=4)
            btn_mic = gr.Button("Nhận dạng lại", variant="secondary")
            out_mic = gr.HTML()
            st_mic = gr.State(None)
            # change (giá trị đã lưu xong trên server) thay vì stop_recording (có thể chạy trước khi bản thu tải lên)
            mic_in.change(on_upload, [mic_in, ref_mic, mode, title, *toggles], [out_mic, st_mic])
            btn_mic.click(on_upload, [mic_in, ref_mic, mode, title, *toggles], [out_mic, st_mic])
        with gr.Tab("Tải file audio"):
            gr.Markdown("Kéo file **wav/flac/mp3/ogg** vào ô: tự nhận dạng khi tải xong (tự chuyển 16 kHz mono). "
                        "Hợp nhất là đoạn hội thoại 5-15 s.")
            with gr.Row():
                file_in = gr.Audio(label="File audio", sources=["upload"], type="filepath")
                ref_file = gr.Textbox(label="Transcript chuẩn (tuỳ chọn, để tính WER)", lines=4)
            btn_file = gr.Button("Nhận dạng lại", variant="secondary")
            out_file = gr.HTML()
            st_file = gr.State(None)
            file_in.upload(on_upload, [file_in, ref_file, mode, title, *toggles], [out_file, st_file])
            btn_file.click(on_upload, [file_in, ref_file, mode, title, *toggles], [out_file, st_file])
        with gr.Tab("Podcast dài → phụ đề"):
            gr.Markdown(
                "Audio dài bất kỳ (vd. một tập podcast 10-60 phút) → tự cắt ở chỗ lặng thành đoạn 10-15 s (đúng độ dài lúc "
                "train) → beam + LM → **phụ đề SRT**, văn bản có mốc thời gian, JSON từng từ, trang HTML nghe-đọc. "
                "Nền đỏ đậm = mô hình kém chắc chắn ở từ đó (độ tin cậy theo entropy). Dán tiêu đề + mô tả tập để "
                "tên riêng/từ tiếng Anh trong đó được ưu tiên (bổ sung A). Thiết kế + số đo: "
                "`docs/notes/ket_qua/san_pham_podcast_quy_tac_2026-10-27.md`.")
            with gr.Row():
                long_in = gr.Audio(label="Audio dài", sources=["upload"], type="filepath", scale=2)
                with gr.Column(scale=2):
                    long_model = gr.Dropdown([n for n in main_names if models[n].model is not None],
                                             value=next((n for n in main_names if "Conformer" in n), None),
                                             label="Mô hình", info="CPU: Conformer relPE nhanh nhất; Mamba trên CPU dùng "
                                                                   "bản tham chiếu thuần PyTorch (chậm)")
                    long_ctx = gr.Textbox(label="Tiêu đề + mô tả tập (tuỳ chọn)", lines=4)
            btn_long = gr.Button("Phiên âm", variant="primary")
            long_files = gr.File(label="Tải về (SRT · TXT · JSON · HTML)", file_count="multiple")
            long_out = gr.HTML()
            btn_long.click(on_longform, [long_in, long_model, long_ctx], [long_out, long_files])
        # Bật/tắt hiển thị: vẽ lại từ kết quả đã có (mở "Xem thêm" chỉ chạy thêm mô hình chưa chạy).
        def rerender(state, show_errors, show_more):
            return render(state, show_errors, show_more), state

        for st, out in ((st_ct, out_ct), (st_mic, out_mic), (st_file, out_file)):
            for tg in toggles:
                tg.change(rerender, [st, *toggles], [out, st])
        with gr.Tab("Thông tin mô hình"):
            gr.Dataframe(value=info_table(), interactive=False,
                         headers=["Mô hình", "Encoder", "Tham số encoder", "Epoch",
                                  "WER val", "WER val_unseen", "Trạng thái", "Checkpoint", "sha256 (12 ký tự đầu)"])
            gr.Markdown("WER val / val_unseen: so với nhãn giả, giải mã greedy, ghi trong checkpoint ep37 lúc train. "
                        "Dữ liệu train: VietSuperSpeech (4 shard, đã bỏ `excluded.tsv` và video giữ riêng).")
    return app


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream.encoding != "utf-8":
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", default=[], metavar="FILE=PATH",
                    help="Đổi một checkpoint, vd. b1_mamba_10ms_ep37.pt=D:/khac/b1.pt")
    ap.add_argument("--ckpt_dir", type=Path, default=CHECKPOINT_ROOT / "final")
    ap.add_argument("--seed", type=int, default=42, choices=(42, 43, 44), help="seed của 3 mô hình 40 ms")
    ap.add_argument("--tokenizer_model", default="configs/tokenizer.model")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    tokenizer = BPETokenizer(args.tokenizer_model)
    models = load_models(tokenizer, dict(kv.split("=", 1) for kv in args.checkpoint), args.ckpt_dir, args.seed)
    if not any(lm.model is not None for lm in models.values()):
        sys.exit(f"Không nạp được mô hình nào — kiểm {args.ckpt_dir}")
    build_app(models, tokenizer, load_clean_test()).launch(server_port=args.port, share=args.share, css=CSS)


if __name__ == "__main__":
    main()
