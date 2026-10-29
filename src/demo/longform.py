"""Sản phẩm (Chương 4): phiên âm podcast/audio dài → phụ đề SRT, văn bản, JSON từng từ (mốc thời gian + độ tin cậy) và
trang HTML nghe-đọc (bấm từ để nghe, từ đáng ngờ tô màu).

    python -m src.demo.longform transcribe <audio> --out <thư mục> [--context "tiêu đề + mô tả tập"] [--model ...]
    python -m src.demo.longform chains     # audio dài thử: nối đoạn liên tiếp val_unseen (nhãn giả) → data/processed/
    python -m src.demo.longform p1         # cắt tự động vs đoạn gốc (WER, bootstrap theo video) + P3 độ tin cậy
    python -m src.demo.longform p2         # tốc độ CPU, 3 encoder 40 ms

Thiết kế và quy tắc đo chốt trước: `docs/notes/ket_qua/san_pham_podcast_quy_tac_2026-10-27.md`. Mô hình chỉ train trên đoạn
10–15 s → audio dài được cắt tại chỗ lặng thành đoạn 10–15 s rồi giải mã từng đoạn bằng đúng decoder beam + LM đã khoá.
"""

import argparse
import csv
import html
import json
import re
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

SAMPLE_RATE = 16000
CUT_MIN_S, CUT_MAX_S = 10.0, 15.0  # phân bố độ dài đoạn lúc train
SMOOTH_S = 0.2
MODELS = {  # tên → (config, checkpoint trong checkpoints/final/)
    "conformer_relpe_40ms": ("configs/v2a/model_conformer_relpos_sub4.yaml", "conformer_relpe_40ms_seed42_ep37.pt"),
    "b1_mamba_40ms": ("configs/v2a/model_mamba_sub4.yaml", "b1_mamba_40ms_seed42_ep37.pt"),
    "conextbimamba_40ms": ("configs/v2a/model_conextbimamba_sub4.yaml", "conextbimamba_40ms_seed42_ep37.pt"),
}
DEFAULT_MODEL = "conformer_relpe_40ms"  # chạy CPU được; 40 ms ba encoder không phân biệt được về WER
CKPT_DIR = Path("checkpoints/final")
CHAINS = Path("data/processed/longform")  # audio nối + nhãn giả: dữ liệu, không track
OUT = Path("reports/results/longform_2026-10-27")
CUE_MAX_S, CUE_MAX_CHARS, CUE_GAP_S = 6.0, 84, 0.8
WORD_MAX_S = 0.6


# ---------------------------------------------------------------- audio, cắt đoạn

def read_audio(path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav).mean(dim=1)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.numpy()


def cut_points(wav: np.ndarray) -> list[int]:
    """Điểm cắt (mẫu): khung 10 ms có năng lượng trung bình trượt thấp nhất trong [10; 15] s sau điểm cắt trước."""
    hop = SAMPLE_RATE // 100
    n = len(wav) // hop
    rms = np.sqrt(np.mean(wav[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    k = int(SMOOTH_S * 100)
    energy = np.convolve(rms, np.ones(k) / k, mode="same")
    cuts, last = [0], 0
    while n - last > CUT_MAX_S * 100:
        lo, hi = last + int(CUT_MIN_S * 100), last + int(CUT_MAX_S * 100)
        last = lo + int(np.argmin(energy[lo:hi]))
        cuts.append(last * hop)
    return cuts + [len(wav)]


# ---------------------------------------------------------------- mô hình

def load_model(name: str, ckpt_dir: Path = CKPT_DIR):
    from src.models import mamba_encoder
    from src.models.mamba_ref import use_reference_mamba
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    from src.training.train import build_model
    if not mamba_encoder.MAMBA_SSM_AVAILABLE:
        use_reference_mamba()
    cfg_path, fname = MODELS[name]
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    model = build_model(cfg, BPETokenizer("configs/tokenizer.model").vocab_size)
    model.load_state_dict(torch.load(ckpt_dir / fname, map_location="cpu")["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), device


@torch.no_grad()
def log_probs(model, device: str, seg: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(seg)[None].to(device)
    lp, n = model(x, torch.tensor([len(seg)], device=device))
    return lp[0, : int(n[0])].float().cpu().numpy()


# ---------------------------------------------------------------- giải mã + mốc thời gian + độ tin cậy

class Decoder:
    """Đúng decoder của Bảng 3.4 (`lm_decode`), thêm mốc thời gian từng từ và độ tin cậy; có ngữ cảnh → từ ngữ cảnh
    (bộ lọc A2) được thêm từ điển + thưởng b (bổ sung A)."""

    def __init__(self, context: str = ""):
        from src.evaluation import bias_title
        from src.evaluation import lm_decode as L
        from src.evaluation.score_clean_test import load_alt
        self.L = L
        self.tok, self.knlm = L._Tokens(), L.load_lm(L.LM_DIR / "lm3.pkl")
        self.words = bias_title.context_words(context, load_alt()) if context.strip() else []
        self.bonus = bias_title.chosen_b("iv") if self.words else 0.0
        self.dec = (L.BiasedDecoders(self.knlm, self.tok, bonus=self.bonus)(self.words) if self.words
                    else L.make_decoder(L.LM_DIR / "lexicon.txt", self.knlm, self.tok))
        self.sil = len(self.tok.tokens) - 1

    def __call__(self, lp: np.ndarray) -> list[dict]:
        """[{w, f0, f1, conf, conf_maxprob}] — f0/f1 là chỉ số khung trong đoạn."""
        L, tok = self.L, self.tok
        p = np.exp(lp.astype(np.float64))
        conf = 1.0 + (p * lp).sum(-1) / np.log(lp.shape[1])  # 1 − H/ln V
        pmax = p.max(-1)
        e = np.full((lp.shape[0], lp.shape[1] + 1), -1e4, dtype=np.float32)
        e[:, :-1] = lp
        hyp = self.dec(torch.from_numpy(e)[None])[0][0]
        greedy_ids, greedy_t = _collapse(lp.argmax(-1))
        g_text = tok.ids_to_text(greedy_ids)
        if len(hyp.words) < L.FALLBACK * len(g_text.split()):  # cùng luật `lm_hyp`
            groups = _word_groups(tok, greedy_ids, greedy_t)
            words = _sticky_align([tok.ids_to_text(ids) for ids, _ in groups],
                                  L.split_sticky(g_text, self.knlm).split(), groups)
        else:
            ids = [int(i) for i in hyp.tokens if int(i) != self.sil]
            # Đường beam của flashlight có T + 2 vị trí (SIL giả ở đầu và cuối) → khung thật = timestep − 1.
            ts = [min(max(int(t) - 1, 0), lp.shape[0] - 1) for i, t in zip(hyp.tokens, hyp.timesteps)
                  if int(i) != self.sil]
            groups = _word_groups(tok, ids, ts)
            if len(groups) != len(hyp.words):  # không gặp khi lexicon tách theo cùng tokenizer; giữ an toàn
                groups = [groups[min(k * len(groups) // max(len(hyp.words), 1), len(groups) - 1)]
                          for k in range(len(hyp.words))]
            words = [(w, frames) for w, (_, frames) in zip(hyp.words, groups)]
        return [{"w": w.lower(), "f0": fr[0], "f1": fr[-1] + 1, "conf": float(conf[fr].min()),
                 "conf_maxprob": float(pmax[fr].min())} for w, fr in words]


def _collapse(path) -> tuple[list[int], list[int]]:
    ids, ts, prev = [], [], -1
    for t, i in enumerate(path):
        if i != prev and i != 0:
            ids.append(int(i))
            ts.append(t)
        prev = i
    return ids, ts


def _word_groups(tok, ids, ts) -> list[tuple[list[int], list[int]]]:
    """Gom token thành từ: mảnh bắt đầu "▁" mở từ mới (từ điển tách theo cùng tokenizer SentencePiece)."""
    groups = []
    for i, t in zip(ids, ts):
        if not groups or tok.tokens[i].startswith("▁"):
            groups.append(([], []))
        groups[-1][0].append(i)
        groups[-1][1].append(t)
    return groups


def _sticky_align(g_words, out_words, groups):
    """`split_sticky` chỉ tách một từ greedy thành 2–3 từ (có thể thêm phụ âm đầu) → khớp tuần tự, từ tách chung khung."""
    res, j = [], 0
    for k, gw in enumerate(g_words):
        if j < len(out_words) and out_words[j] == gw.lower():
            n = 1
        else:
            nxt = g_words[k + 1].lower() if k + 1 < len(g_words) else None
            n = next((m for m in (2, 3) if j + m >= len(out_words) or out_words[j + m] == nxt), 2)
        res += [(w, groups[k][1]) for w in out_words[j:j + n]]
        j += n
    return res


# ---------------------------------------------------------------- toàn bộ audio

def transcribe(wav: np.ndarray, model, device: str, decoder: Decoder) -> dict:
    t0 = time.perf_counter()
    cuts = cut_points(wav)
    t_cut = time.perf_counter() - t0
    segments, t_enc, t_dec = [], 0.0, 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        t1 = time.perf_counter()
        lp = log_probs(model, device, wav[a:b])
        t2 = time.perf_counter()
        words = decoder(lp)
        t3 = time.perf_counter()
        t_enc, t_dec = t_enc + t2 - t1, t_dec + t3 - t2
        frame = (b - a) / SAMPLE_RATE / lp.shape[0]
        for w in words:
            w["start"] = round(a / SAMPLE_RATE + w.pop("f0") * frame, 3)
            w["end"] = round(a / SAMPLE_RATE + w.pop("f1") * frame, 3)
        # CTC phát mỗi âm tiết ở ~1 khung → hết từ = đầu từ sau, tối đa WORD_MAX_S (âm tiết tiếng Việt ~0,2–0,3 s).
        for w, nxt in zip(words, [x["start"] for x in words[1:]] + [b / SAMPLE_RATE]):
            w["end"] = round(max(w["end"], min(nxt, w["start"] + WORD_MAX_S)), 3)
        segments.append({"start": a / SAMPLE_RATE, "end": b / SAMPLE_RATE, "text": " ".join(w["w"] for w in words),
                         "words": words})
    dur = len(wav) / SAMPLE_RATE
    total = t_cut + t_enc + t_dec
    return {"duration_s": dur, "n_segments": len(segments), "context_words": decoder.words, "bonus": decoder.bonus,
            "timing_s": {"cut": t_cut, "encoder": t_enc, "decode": t_dec, "total": total, "rtf": total / dur},
            "segments": segments}


def _ts(s: float, sep: str = ",") -> str:
    ms = int(round(s * 1000))
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"


def cues(result: dict) -> list[tuple[float, float, str]]:
    out, cur = [], []
    for w in (w for s in result["segments"] for w in s["words"]):
        if cur and (w["end"] - cur[0]["start"] > CUE_MAX_S or w["start"] - cur[-1]["end"] > CUE_GAP_S
                    or len(" ".join(x["w"] for x in cur + [w])) > CUE_MAX_CHARS):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    res = []
    for c in out:
        text = " ".join(x["w"] for x in c)
        if len(text) > CUE_MAX_CHARS // 2:  # hai dòng, ngắt ở khoảng trắng gần giữa nhất
            mid = min((i for i, ch in enumerate(text) if ch == " "), key=lambda i: abs(i - len(text) / 2))
            text = text[:mid] + "\n" + text[mid + 1:]
        res.append((c[0]["start"], c[-1]["end"], text))
    return res


def write_outputs(result: dict, audio: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stem = audio.stem
    (out / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / f"{stem}.srt").write_text("".join(f"{k}\n{_ts(a)} --> {_ts(b)}\n{t}\n\n" for k, (a, b, t) in
                                             enumerate(cues(result), 1)), encoding="utf-8")
    (out / f"{stem}.txt").write_text("".join(f"[{_ts(s['start'], '.')[:8]}] {s['text']}\n" for s in result["segments"]),
                                     encoding="utf-8")
    (out / f"{stem}.html").write_text(html_page(result, audio, out), encoding="utf-8")


def transcript_spans(result: dict, inline_color: bool = False) -> str:
    """Đoạn văn theo đoạn cắt, mỗi từ một span (mốc thời gian + độ tin cậy). inline_color: màu viết thẳng vào style
    (Gradio không nạp CSS/JS của trang)."""
    spans = []
    for s in result["segments"]:
        spans.append(f'<p><a class="t" data-t="{s["start"]:.2f}">[{_ts(s["start"], ".")[:8]}]</a> ')
        for w in s["words"]:
            # Độ tin cậy thấp → nền đậm hơn (thang liên tục, không ngưỡng — quy tắc mục 1).
            alpha = max(0.0, min(1.0, (1 - w["conf"]) * 1.6))
            style = (f"background:rgba(220,60,40,{alpha * .45:.2f});border-radius:3px" if inline_color
                     else f"--a:{alpha:.2f}")
            spans.append(f'<span class="w" data-t="{w["start"]:.2f}" data-e="{w["end"]:.2f}" '
                         f'style="{style}" title="{_ts(w["start"], ".")[:8]} · tin cậy {w["conf"]:.2f}">'
                         f'{html.escape(w["w"])}</span> ')
        spans.append("</p>")
    return "".join(spans)


def html_page(result: dict, audio: Path, out: Path) -> str:
    try:
        src = Path(audio).resolve().relative_to(out.resolve()).as_posix()
    except ValueError:
        src = Path(audio).resolve().as_uri()
    spans = [transcript_spans(result)]
    tm = result["timing_s"]
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(Path(audio).stem)}</title><style>
:root{{--bg:#fff;--fg:#1f2328;--mute:#656d76;--hi:#fff3b0;--low:220,60,40}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d1117;--fg:#e6edf3;--mute:#9198a1;--hi:#5a4a00;--low:255,110,90}}}}
body{{background:var(--bg);color:var(--fg);font:17px/1.7 system-ui,sans-serif;max-width:860px;margin:0 auto;padding:0 16px 80px}}
audio{{position:sticky;top:0;width:100%;padding:12px 0;background:var(--bg)}}
.w{{cursor:pointer;border-radius:3px;background:rgba(var(--low),calc(var(--a)*.45))}}
.w.on{{outline:2px solid #e3b341;background:var(--hi)}} .t{{color:var(--mute);cursor:pointer;font-size:13px}}
small{{color:var(--mute)}}</style></head><body>
<h1 style="font-size:20px">{html.escape(Path(audio).stem)}</h1>
<small>{result['duration_s'] / 60:.1f} phút · {result['n_segments']} đoạn · RTF {tm['rtf']:.2f} · nền đỏ đậm = mô hình kém chắc chắn
{(' · ngữ cảnh: ' + html.escape(' '.join(result['context_words'][:30]))) if result['context_words'] else ''}</small>
<audio id="a" controls src="{html.escape(src)}"></audio>
{''.join(spans)}
<script>
const a=document.getElementById('a'),ws=[...document.querySelectorAll('.w')];let cur=null;
document.body.addEventListener('click',e=>{{const t=e.target.dataset.t;if(t!==undefined){{a.currentTime=+t;a.play()}}}});
a.addEventListener('timeupdate',()=>{{const t=a.currentTime;let lo=0,hi=ws.length-1,k=-1;
while(lo<=hi){{const m=(lo+hi)>>1;if(+ws[m].dataset.t<=t){{k=m;lo=m+1}}else hi=m-1}}
const w=k>=0?ws[k]:null;if(w!==cur){{cur&&cur.classList.remove('on');w&&w.classList.add('on');cur=w}}}});
</script></body></html>"""


def cmd_transcribe(a) -> None:
    model, device = load_model(a.model)
    wav = read_audio(a.audio)
    context = a.context + ("\n" + Path(a.context_file).read_text(encoding="utf-8") if a.context_file else "")
    res = transcribe(wav, model, device, Decoder(context))
    res["model"] = a.model
    write_outputs(res, Path(a.audio), Path(a.out))
    t = res["timing_s"]
    print(f"{res['duration_s'] / 60:.1f} phút, {res['n_segments']} đoạn · cắt {t['cut']:.1f} s · encoder {t['encoder']:.1f} s"
          f" · giải mã {t['decode']:.1f} s · RTF {t['rtf']:.3f} ({device}) → {a.out}")


# ---------------------------------------------------------------- audio dài thử + đo

def build_chains(min_s: float = 60.0) -> list[dict]:
    """Chuỗi = các đoạn val_unseen có audio local, số `_seg` liền nhau, cùng video; nhãn giả từ meta giải mã 15/10."""
    from src.data.vietsuperspeech_dataset import AUDIO_CACHE_DIR
    from src.evaluation.bias_title import TUNE_META, VAL_UNSEEN, video_of
    text = {f"{m['split']}:{m['index']}": m["text"] for m in json.loads(TUNE_META.read_text(encoding="utf-8"))["meta"]}
    by_video = {}
    with VAL_UNSEEN.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            k, p = f"{r['split']}:{r['index']}", Path(AUDIO_CACHE_DIR) / r["audio"]
            if k in text and p.exists():
                seg = int(re.search(r"_seg(\d+)\.wav$", r["audio"]).group(1))
                by_video.setdefault(video_of(r["audio"]), []).append((seg, k, p, float(r["duration"])))
    chains = []
    for v, rows in sorted(by_video.items()):
        rows.sort()
        run = [rows[0]]
        for r in rows[1:] + [None]:
            if r is not None and r[0] == run[-1][0] + 1:
                run.append(r)
                continue
            if sum(x[3] for x in run) >= min_s:
                chains.append({"video": v, "segs": [x[0] for x in run], "keys": [x[1] for x in run],
                               "paths": [str(x[2]) for x in run], "texts": [text[x[1]] for x in run]})
            if r is not None:
                run = [r]
    return chains


def cmd_chains(_a=None) -> None:
    CHAINS.mkdir(parents=True, exist_ok=True)
    chains = build_chains()
    for c in chains:
        c["name"] = f"{c['video'][:40]}_seg{c['segs'][0]:03d}-{c['segs'][-1]:03d}"
        sf.write(CHAINS / f"{c['name']}.wav", np.concatenate([read_audio(p) for p in c["paths"]]), SAMPLE_RATE)
        c["duration_s"] = sf.info(str(CHAINS / f"{c['name']}.wav")).duration
    (CHAINS / "chains.json").write_text(json.dumps(chains, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(chains)} chuỗi, {len({c['video'] for c in chains})} video, {sum(c['duration_s'] for c in chains) / 3600:.2f} h;"
          f" dài nhất {max(c['duration_s'] for c in chains) / 60:.1f}′ → {CHAINS}")


def _chain_counts(hyp: str, texts: list[str], alt) -> tuple[int, int]:
    """(lỗi, số từ ref) trên cả chuỗi, chấm như nửa tune của bổ sung A."""
    from src.evaluation.score_clean_test import hyp_tokens, ref_tokens
    ref = [t[0] for x in texts for t in ref_tokens(0, x, {}, alt, True)]
    h = [t[0] for t in hyp_tokens(hyp, alt, True)]
    o = jiwer.process_words(" ".join(ref), " ".join(h) or "<empty>")
    return o.substitutions + o.deletions + o.insertions, len(ref)


def _auroc(score: np.ndarray, pos: np.ndarray) -> float:
    """AUROC qua hạng (Mann–Whitney), hạng hoà lấy trung bình."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = pos.sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(pos) - n1)))


def _word_labels(words: list[dict], texts: list[str]) -> list[tuple[float, float, bool]]:
    """P3: (conf, conf_maxprob, sai?) mỗi từ hyp; căn bằng jiwer sau `normalize_text`, sai = thay hoặc chèn."""
    from src.evaluation.text_normalize import normalize_text
    hw = [(normalize_text(w["w"]), w) for w in words]
    hw = [(t, w) for t, w in hw if t and len(t.split()) == 1]
    ref = normalize_text(" ".join(texts)).split()
    o = jiwer.process_words(" ".join(ref), " ".join(t for t, _ in hw))
    wrong = np.zeros(len(hw), bool)
    for ch in o.alignments[0]:
        if ch.type in ("substitute", "insert"):
            wrong[ch.hyp_start_idx:ch.hyp_end_idx] = True
    return [(w["conf"], w["conf_maxprob"], bool(x)) for (_, w), x in zip(hw, wrong)]


def cmd_p1(_a=None) -> None:
    """P1 (+ P3 trên cùng output): mô hình mặc định, không ngữ cảnh; sản phẩm trên chuỗi nối vs giải mã từng đoạn gốc."""
    from src.evaluation.score_clean_test import load_alt
    chains, alt = json.loads((CHAINS / "chains.json").read_text(encoding="utf-8")), load_alt()
    model, device = load_model(DEFAULT_MODEL)
    dec = Decoder("")
    rows, words = [], {}
    for k, c in enumerate(chains):
        res = transcribe(read_audio(CHAINS / f"{c['name']}.wav"), model, device, dec)
        prod = " ".join(s["text"] for s in res["segments"])
        orig = " ".join(" ".join(w["w"] for w in dec(log_probs(model, device, read_audio(p)))) for p in c["paths"])
        (e_p, n), (e_o, _) = _chain_counts(prod, c["texts"], alt), _chain_counts(orig, c["texts"], alt)
        rows.append({"chain": c["name"], "video": c["video"], "duration_s": c["duration_s"], "n_orig": len(c["paths"]),
                     "n_auto": res["n_segments"], "words": n, "err_auto": e_p, "err_orig": e_o})
        words.setdefault(c["video"], []).extend(_word_labels([w for s in res["segments"] for w in s["words"]], c["texts"]))
        print(f"[{k + 1}/{len(chains)}] {c['name'][:50]}: WER tự cắt {e_p / n:.4f} · đoạn gốc {e_o / n:.4f}", flush=True)
    videos = sorted(words)
    per = {v: np.array([[r["err_auto"], r["err_orig"], r["words"]] for r in rows if r["video"] == v]).sum(0) for v in videos}
    arr = np.array([per[v] for v in videos], float)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(videos), size=(10_000, len(videos)))
    s = arr[idx].sum(1)
    d = (s[:, 0] - s[:, 1]) / s[:, 2]
    tot = arr.sum(0)
    p1 = {"wer_auto": tot[0] / tot[2], "wer_orig": tot[1] / tot[2], "diff": (tot[0] - tot[1]) / tot[2],
          "ci": np.quantile(d, [0.025, 0.975]).tolist(), "n_chains": len(rows), "n_videos": len(videos),
          "hours": sum(r["duration_s"] for r in rows) / 3600, "words": int(tot[2])}
    lab = {v: np.array(words[v]) for v in videos}
    allw = np.concatenate([lab[v] for v in videos])
    au = lambda a, col: _auroc(1 - a[:, col], a[:, 2].astype(bool))
    boot = []
    for ix in idx[:2000]:  # AUROC tốn hơn: 2.000 lần rút video
        a = np.concatenate([lab[videos[i]] for i in ix])
        if 0 < a[:, 2].sum() < len(a):
            boot.append(au(a, 0) - au(a, 1))
    p3 = {"n_words": len(allw), "wrong_rate": float(allw[:, 2].mean()), "auroc_entropy": au(allw, 0),
          "auroc_maxprob": au(allw, 1), "diff": au(allw, 0) - au(allw, 1),
          "ci": np.quantile(boot, [0.025, 0.975]).tolist(), "n_boot": len(boot)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p1_p3.json").write_text(json.dumps({"p1": p1, "p3": p3, "chains": rows}, ensure_ascii=False, indent=1,
                                               default=float), encoding="utf-8")
    print(f"P1: {p1['n_chains']} chuỗi / {p1['n_videos']} video / {p1['hours']:.2f} h · WER tự cắt {p1['wer_auto']:.4f} · "
          f"đoạn gốc {p1['wer_orig']:.4f} · hiệu {p1['diff']:+.4f} [{p1['ci'][0]:+.4f}; {p1['ci'][1]:+.4f}]")
    print(f"P3: {p3['n_words']} từ, sai {p3['wrong_rate']:.3f} · AUROC entropy {p3['auroc_entropy']:.3f} · max-prob "
          f"{p3['auroc_maxprob']:.3f} · hiệu {p3['diff']:+.3f} [{p3['ci'][0]:+.3f}; {p3['ci'][1]:+.3f}]")


def cmd_p2(_a=None) -> None:
    """P2: chuỗi ≥ 5 phút đầu tiên (thứ tự chains.json), 3 encoder 40 ms, CPU; làm ấm 1 đoạn trước khi đo."""
    import platform
    c = next(c for c in json.loads((CHAINS / "chains.json").read_text(encoding="utf-8")) if c["duration_s"] >= 300)
    wav = read_audio(CHAINS / f"{c['name']}.wav")
    dec = Decoder("")
    res = {"chain": c["name"], "duration_s": len(wav) / SAMPLE_RATE, "cpu": platform.processor(),
           "torch_threads": torch.get_num_threads(), "models": {}}
    for name in MODELS:
        model, device = load_model(name)
        log_probs(model, device, wav[: 12 * SAMPLE_RATE])  # làm ấm
        r = transcribe(wav, model, device, dec)
        res["models"][name] = r["timing_s"] | {"device": device}
        print(f"{name}: {r['timing_s']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p2_speed.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("transcribe")
    t.add_argument("audio")
    t.add_argument("--out", default="reports/results/longform_2026-10-27/demo")
    t.add_argument("--context", default="", help="tiêu đề + mô tả tập (tuỳ chọn)")
    t.add_argument("--context-file", help="file văn bản tiêu đề + mô tả tập")
    t.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL)
    for c in ("chains", "p1", "p2"):
        sub.add_parser(c)
    a = ap.parse_args()
    {"transcribe": cmd_transcribe, "chains": cmd_chains, "p1": cmd_p1, "p2": cmd_p2}[a.cmd](a)


if __name__ == "__main__":
    main()
