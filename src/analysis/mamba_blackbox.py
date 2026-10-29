"""Bổ sung B — mở hộp đen Mamba: bước chọn lọc Δ quanh ranh giới từ / khung lặng (M1, M2) và khoảng cách hiệu dụng
của attention ẩn Mamba so với attention Conformer (M3). Mô tả, không kiểm định nhân quả.

Quy tắc chốt trước: `docs/notes/ket_qua/bo_sung_AB_quy_tac_2026-10-27.md` mục B (B1–B5). Kết quả + nhận xét:
`docs/notes/ket_qua/bo_sung_B_ket_qua_2026-10-27.md`. Chạy từ gốc repo, CPU fp32, khối Mamba bằng `mamba_ref.py`:

    python -m src.analysis.mamba_blackbox select    # B3: 30 câu val_unseen (seed 42) → utterances.tsv; tải audio còn thiếu
    python -m src.analysis.mamba_blackbox check     # hook không đổi log-prob; attention dựng lại khớp đầu ra khối
    python -m src.analysis.mamba_blackbox run       # B4: M1, M2, M3 → per_sentence.csv, frames.csv, m1_m2.csv, m3.csv
    python -m src.analysis.mamba_blackbox figures   # B5: 3 hình → reports/figures/mamba_blackbox/

Mọi đại lượng lấy bằng forward hook chỉ đọc (`x_proj` của khối Mamba, `self_attn` của Conformer) rồi tính lại bên ngoài
— không sửa lớp nào của mô hình.
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml

from src.data.vietsuperspeech_dataset import AUDIO_CACHE_DIR, HF_DATASET_ID, HF_REVISION, manifest_rows, video_of
from src.features.log_mel import LogMelFeatureExtractor
from src.models.conformer_relpos_encoder import ConformerRelPosEncoder, rel_shift
from src.models.mamba_encoder import MambaEncoder
from src.models.mamba_ref import MambaRef, use_reference_mamba
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.train import CHECKPOINT_ROOT, build_model

OUT = Path("reports/results/mamba_blackbox_2026-10-27")
FIG = Path("reports/figures/mamba_blackbox")
CKPT_DIR = CHECKPOINT_ROOT / "final"
TOKENIZER = "configs/tokenizer.model"
CLEAN_TEST_MANIFEST = "data/processed/clean_test_manifest.json"
N_UTT, SEED, N_BOOT = 30, 42, 1000
BLANK = BPETokenizer.BLANK_ID
BLANK_P = 0.95
N_CH_10MS = 128  # kênh lấy mẫu cho attention ẩn ở 10 ms (B4); 40 ms dùng đủ 512 kênh
WORD = "▁"


class Spec(NamedTuple):
    encoder: str
    res: str
    config: str
    ckpt: str
    sec: float  # độ dài một khung đầu ra encoder


MODELS = {
    "b1_40": Spec("B1", "40 ms", "configs/v2a/model_mamba_sub4.yaml", "b1_mamba_40ms_seed42_ep37.pt", 0.04),
    "conext_40": Spec("ConExtBiMamba", "40 ms", "configs/v2a/model_conextbimamba_sub4.yaml",
                      "conextbimamba_40ms_seed42_ep37.pt", 0.04),
    "conformer_40": Spec("Conformer PE", "40 ms", "configs/v2a/model_conformer_relpos_sub4.yaml",
                         "conformer_relpe_40ms_seed42_ep37.pt", 0.04),
    "b1_10": Spec("B1", "10 ms", "configs/v2a/model_mamba.yaml", "b1_mamba_10ms_ep37.pt", 0.01),
    "conext_10": Spec("ConExtBiMamba", "10 ms", "configs/v2a/model_conextbimamba.yaml", "conextbimamba_10ms_ep37.pt", 0.01),
    "conformer_10": Spec("Conformer PE", "10 ms", "configs/v2a/model_conformer_relpos.yaml",
                         "conformer_relpe_10ms_ep37.pt", 0.01),
}


# ---------------------------------------------------------------- dữ liệu

def cmd_select(_args) -> None:
    """B3: rút 30 câu từ toàn bộ val_unseen (sau excluded.tsv) bằng random.Random(42).sample, giữ thứ tự rút;
    câu minh hoạ = dòng đầu. Audio chưa có trong cache thì tải lẻ từ HF (đúng revision pin)."""
    from huggingface_hub import hf_hub_download

    rows = manifest_rows(["val_unseen"])
    picked = random.Random(SEED).sample(rows, N_UTT)
    with open(CLEAN_TEST_MANIFEST, encoding="utf-8") as f:
        clean = {(s["split"], int(s["index"])) for s in json.load(f)["samples"]}
    assert not any((r["split"], r["index"]) in clean for r in picked), "lẫn câu clean-test"
    for r in picked:
        if not (Path(AUDIO_CACHE_DIR) / r["audio"]).exists():
            hf_hub_download(HF_DATASET_ID, r["audio"], repo_type="dataset", revision=HF_REVISION,
                            local_dir=AUDIO_CACHE_DIR)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "utterances.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["order", "split", "index", "video", "duration", "audio"])
        for i, r in enumerate(picked, 1):
            w.writerow([i, r["split"], r["index"], video_of(r["audio"]), r["duration"], r["audio"]])
    print(f"{len(picked)} câu từ {len(rows)} câu val_unseen, {len({video_of(r['audio']) for r in picked})} video → "
          f"{OUT / 'utterances.tsv'}")


def read_utterances() -> list[dict]:
    with open(OUT / "utterances.tsv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_wav(audio: str) -> torch.Tensor:
    wav, sr = sf.read(Path(AUDIO_CACHE_DIR) / audio, dtype="float32")
    if sr != 16000 or wav.ndim != 1:
        raise ValueError(f"{audio}: {sr} Hz, shape {wav.shape}")
    return torch.from_numpy(wav)


# ---------------------------------------------------------------- mô hình + hook

def load_model(key: str, tok: BPETokenizer) -> torch.nn.Module:
    use_reference_mamba()
    spec = MODELS[key]
    with open(spec.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg, tok.vocab_size)
    model.load_state_dict(torch.load(CKPT_DIR / spec.ckpt, map_location="cpu")["model"])
    return model.eval()


def mamba_blocks(enc) -> list[tuple[int, str, MambaRef]]:
    """[(lớp, chiều, khối)], lớp đếm từ 1 như layer_hook của encoder."""
    if isinstance(enc, MambaEncoder):
        return ([(i, "xuoi", b) for i, b in enumerate(enc.layers, 1)]
                + [(i, "nguoc", b) for i, b in enumerate(enc.layers_bwd, 1)])
    return [(i, d, layer.mamba_fwd if d == "xuoi" else layer.mamba_bwd)
            for i, layer in enumerate(enc.layers, 1) for d in ("xuoi", "nguoc")]


def is_conformer(model) -> bool:
    return isinstance(model.encoder, ConformerRelPosEncoder)


class Tracer:
    """Forward hook chỉ ghi lại (args, output), không trả gì → PyTorch giữ nguyên đầu ra của module."""

    def __init__(self, model):
        self.rec = {}
        if is_conformer(model):
            targets = [((i, "hai_chieu"), layer.self_attn) for i, layer in enumerate(model.encoder.layers, 1)]
        else:
            targets = [((i, d), blk.x_proj) for i, d, blk in mamba_blocks(model.encoder)]
        self.handles = [m.register_forward_hook(self._hook(key)) for key, m in targets]

    def _hook(self, key):
        def hook(_module, args, output):
            self.rec[key] = (args, output)
        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


@torch.inference_mode()
def forward_log_probs(model, wav: torch.Tensor) -> torch.Tensor:
    lp, _ = model(wav[None], torch.tensor([wav.numel()]))
    return lp[0]


def ssm_tensors(blk: MambaRef, args, out):
    """Từ hook `x_proj`: u = đầu vào S6 (sau conv1d + SiLU) [L, D]; Δ sau softplus [L, D]; B, C [L, N]; A [D, N]."""
    dt, b, c = torch.split(out, [blk.dt_rank, blk.d_state, blk.d_state], dim=-1)
    delta = F.softplus(dt @ blk.dt_proj.weight.T + blk.dt_proj.bias)  # đúng biểu thức trong MambaRef.forward
    return args[0][0], delta[0], b[0], c[0], -torch.exp(blk.A_log.float())


# ---------------------------------------------------------------- nhãn khung (B4)

def word_initial(tok: BPETokenizer) -> torch.Tensor:
    """[V]: token (id đã dịch +1 như BPETokenizer, 0 = blank) có mảnh bắt đầu bằng '▁'."""
    return torch.tensor([False] + [tok.sp.id_to_piece(i).startswith(WORD) for i in range(tok.sp.get_piece_size())])


def frame_labels(lp: torch.Tensor, initial: torch.Tensor) -> dict:
    """Căn chỉnh greedy CTC của chính mô hình. Khung "phát" token = khung đầu của một đoạn argmax giống nhau, khác blank
    (đúng chỗ greedy_collapse giữ token)."""
    ids = lp.argmax(-1)
    new_run = torch.ones_like(ids, dtype=torch.bool)
    new_run[1:] = ids[1:] != ids[:-1]
    start = new_run & (ids != BLANK) & initial[ids]
    boundary = start.clone()
    boundary[1:] |= start[:-1]
    boundary[:-1] |= start[1:]
    silent = (lp[:, BLANK].exp() > BLANK_P) & ~boundary
    return {"ids": ids, "start": start, "boundary": boundary, "silent": silent, "inside": ~boundary & ~silent}


def words_at_starts(lab: dict, tok: BPETokenizer) -> list[tuple[int, str]]:
    """[(khung bắt đầu từ, từ)] — gom các token phát ra từ một khung bắt đầu từ tới trước khung bắt đầu kế."""
    out, prev = [], None
    for t, i in enumerate(lab["ids"].tolist()):
        if i != prev and i != BLANK:
            piece = tok.sp.id_to_piece(i - 1)
            if lab["start"][t]:
                out.append([t, piece[1:]])
            elif out:
                out[-1][1] += piece
        prev = i
    return [(t, w) for t, w in out]


def ratio(x: torch.Tensor, num: torch.Tensor, den: torch.Tensor) -> float:
    if not num.any() or not den.any():
        return float("nan")
    return float(x[num].mean() / x[den].mean())


# ---------------------------------------------------------------- attention ẩn (Ali và cs. 2025) + attention Conformer

def hidden_attention(delta, b, c, a, chans, u=None, want_matrix=False, group=256, chunk=16) -> dict:
    """Attention ẩn của một khối S6, từng kênh d trong `chans` (Ali, Zimerman, Wolf, ACL 2025, mục 3.1, pt. 14 và 16;
    rời rạc hoá pt. 7 Ā = exp(ΔA), B̄ = ΔB — trùng selective_scan_ref):
        α_d(t, s) = Σ_n C_t[n] · exp(A[d, n] · (S_t − S_s)) · Δ_s[d] · B_s[n],  s ≤ t,  S_t = Σ_{k ≤ t} Δ_k[d]
    Như bài: trên đầu vào S6 (sau conv1d + SiLU), không gồm D·u và cổng SiLU(z). y = α·u tái tạo đúng đầu ra scan
    (kiểm ở `check`).

    Trả dict: dist [G] = trung bình theo t của Σ_s w(t, s)·|t − s| (khung), w = |α| chuẩn hoá theo hàng (B4);
    y [G, L] nếu có u [L, D]; mat [L, L] = trung bình theo kênh của w nếu want_matrix.

    Chia trục thời gian thành khối c khung. Với t thuộc khối i, s thuộc khối j < i:
        exp(A(S_t − S_s)) = exp(A(S_t − S_đầu(i))) · exp(A(S_đầu(i) − S_cuối(j))) · exp(A(S_cuối(j) − S_s)),
    S tăng (Δ > 0) nên cả ba số mũ ≤ 0 → không tràn số; phần ngoài khối chéo thành nhân ma trận hạng N, exp chỉ còn
    O(G·N·L + G·N·(L/c)²). Khối chéo tính thẳng.
    """
    length, n_state = b.shape
    n_chunk = -(-length // chunk)
    pad = n_chunk * chunk - length
    bb, cc = F.pad(b, (0, 0, 0, pad)), F.pad(c, (0, 0, 0, pad))
    rel = torch.arange(chunk, dtype=torch.float32)  # t − đầu khối
    tri = torch.ones(chunk, chunk).tril()
    dists, ys = [], []
    mat = torch.zeros(length + pad, length + pad) if want_matrix else None
    for g0 in range(0, len(chans), group):
        ch = chans[g0:g0 + group]
        n_ch = len(ch)
        dl = F.pad(delta[:, ch].T, (0, pad))  # Δ = 0 ở phần đệm → B̄ = 0, không đóng góp
        a_ch = a[ch]  # [G, N]
        s_cum = torch.cumsum(dl.double(), dim=1).view(n_ch, n_chunk, chunk)  # float64: S lớn, cần hiệu nhỏ chính xác
        first, last = s_cum[..., :1], s_cum[..., -1:]
        db = (dl[:, :, None] * bb[None]).view(n_ch, n_chunk, chunk, n_state)  # Δ_s·B_s
        q = cc.view(1, n_chunk, chunk, n_state) * torch.exp(a_ch[:, None, None] * (s_cum - first).float()[..., None])
        k = db * torch.exp(a_ch[:, None, None] * (last - s_cum).float()[..., None])
        uu = F.pad(u[:, ch], (0, 0, 0, pad)).T if u is not None else None  # [G, L']
        dist = torch.empty(n_ch, n_chunk, chunk)
        y = torch.empty(n_ch, n_chunk, chunk) if u is not None else None
        for i in range(n_chunk):
            i0, i1 = i * chunk, (i + 1) * chunk
            diff = (s_cum[:, i, :, None] - s_cum[:, i, None, :]).float().clamp(min=0)  # s > t bị che bằng tri
            near = (torch.exp(a_ch[:, :, None, None] * diff[:, None]) * cc[i0:i1].T[None, :, :, None]
                    * db[:, i].transpose(1, 2)[:, :, None, :]).sum(1) * tri  # [G, c, c]
            if y is not None:
                y[:, i] = (near * uu[:, None, i0:i1]).sum(-1)
            near = near.abs()
            z = near.sum(-1)
            lag = rel * z - near @ rel  # Σ_s |α|·(t − s), phần trong khối
            if i:
                m = torch.exp(a_ch[:, None] * (first[:, i] - last[:, :i, 0]).float()[..., None])  # [G, i, N]
                past = torch.bmm(q[:, i], (k[:, :i] * m[:, :, None]).reshape(n_ch, i0, n_state).transpose(1, 2))
                if y is not None:
                    y[:, i] += torch.bmm(past, uu[:, :i0, None])[..., 0]
                past = past.abs_()
                zp = past.sum(-1)
                lag += rel * zp + past @ (i0 - torch.arange(i0, dtype=torch.float32))  # t − s = (t − i0) + (i0 − s)
                z = z + zp
            z = z.clamp_min(1e-30)
            dist[:, i] = lag / z
            if mat is not None:
                mat[i0:i1, i0:i1] += (near / z[..., None]).sum(0)
                if i:
                    mat[i0:i1, :i0] += (past / z[..., None]).sum(0)
        dists.append(dist.reshape(n_ch, -1)[:, :length].mean(1))
        if y is not None:
            ys.append(y.reshape(n_ch, -1)[:, :length])
    return {"dist": torch.cat(dists), "y": torch.cat(ys) if ys else None,
            "mat": mat[:length, :length] / len(chans) if mat is not None else None}


def conformer_attention(attn, x, pos_emb, pad_mask):
    """Trọng số sau softmax [B, H, T, T] của RelPositionMultiHeadAttention, tính tường minh đúng công thức của nó
    (SDPA bên trong không trả trọng số), kèm đầu ra dựng lại để đối chiếu."""
    b, t, _ = x.shape
    q = attn.linear_q(x).view(b, t, attn.h, attn.d_k)
    k = attn.linear_k(x).view(b, t, attn.h, attn.d_k).transpose(1, 2)
    v = attn.linear_v(x).view(b, t, attn.h, attn.d_k).transpose(1, 2)
    p = attn.linear_pos(pos_emb).view(-1, attn.h, attn.d_k).transpose(0, 1)
    bd = rel_shift(torch.matmul((q + attn.pos_bias_v).transpose(1, 2), p.transpose(-2, -1)))
    ac = torch.matmul((q + attn.pos_bias_u).transpose(1, 2), k.transpose(-2, -1))
    scores = ((ac + bd) / math.sqrt(attn.d_k)).masked_fill(pad_mask[:, None, None, :], float("-inf"))
    w = scores.softmax(-1)
    return w, attn.linear_out((w @ v).transpose(1, 2).reshape(b, t, -1))


def lag_mean(w: torch.Tensor) -> torch.Tensor:
    """w [..., T, T] (hàng đã chuẩn hoá) → [...] trung bình theo t của Σ_s w(t, s)·|t − s| (khung)."""
    t = w.shape[-1]
    pos = torch.arange(t, dtype=w.dtype)
    return (w * (pos[:, None] - pos[None, :]).abs()).sum(-1).mean(-1)


def channel_subset() -> torch.Tensor:
    """Cùng một tập kênh (seed 42) cho mọi khối 10 ms — kênh của các khối khác nhau vốn không liên quan nhau."""
    return torch.randperm(512, generator=torch.Generator().manual_seed(SEED))[:N_CH_10MS].sort().values


# ---------------------------------------------------------------- đo một câu

@torch.inference_mode()
def analyze_sentence(model, spec: Spec, rec: dict, lab: dict, chans10: torch.Tensor) -> list[dict]:
    rows = []
    if is_conformer(model):
        for (layer, direction), (args, _out) in sorted(rec.items()):
            w, _ = conformer_attention(model.encoder.layers[layer - 1].self_attn, *args)
            rows.append({"layer": layer, "direction": direction, "M3_s": float(lag_mean(w[0]).mean()) * spec.sec})
        return rows
    all_ch = torch.arange(512)
    for layer, direction, blk in mamba_blocks(model.encoder):
        args, out = rec[(layer, direction)]
        u, delta, b, c, a = ssm_tensors(blk, args, out)
        dbar = delta.mean(1)  # Δ_t: trung bình theo kênh
        if direction == "nguoc":
            dbar = dbar.flip(0)  # khối ngược quét câu đã đảo (batch 1, không padding) → về trục thời gian gốc
        chans = all_ch if spec.res == "40 ms" else chans10
        dist = hidden_attention(delta, b, c, a, chans)["dist"]
        row = {"layer": layer, "direction": direction,
               "M1": ratio(dbar, lab["boundary"], lab["inside"]), "M2": ratio(dbar, lab["silent"], lab["inside"]),
               "M3_s": float(dist.mean()) * spec.sec,
               "d_boundary": float(dbar[lab["boundary"]].mean()) if lab["boundary"].any() else float("nan"),
               "d_inside": float(dbar[lab["inside"]].mean()) if lab["inside"].any() else float("nan"),
               "d_silent": float(dbar[lab["silent"]].mean()) if lab["silent"].any() else float("nan")}
        if spec.res == "40 ms":  # sai số do lấy mẫu kênh: cùng tập kênh của 10 ms, trên khối 40 ms tính đủ kênh
            row["M3_sub_s"] = float(dist[chans10].mean()) * spec.sec
        rows.append(row)
    return rows


# ---------------------------------------------------------------- lệnh

def cmd_check(_args) -> None:
    """Trên câu minh hoạ, từng mô hình: (1) log-prob có/không hook; (2) Mamba: α·u so với selective_scan_ref (kênh dùng
    cho M3); (3) Conformer: trọng số dựng lại → đầu ra MHSA so với đầu ra thật, tổng hàng = 1. Ghi checks.json."""
    tok = BPETokenizer(TOKENIZER)
    utt = read_utterances()[0]
    wav = load_wav(utt["audio"])
    chans10 = channel_subset()
    report = {"utterance": f"{utt['split']}:{utt['index']}", "duration_s": float(utt["duration"]),
              "n_channels_10ms": N_CH_10MS, "channels_10ms": chans10.tolist(), "models": {}}
    for key, spec in MODELS.items():
        model = load_model(key, tok)
        t0 = time.perf_counter()
        lp_plain = forward_log_probs(model, wav)
        t_fwd = time.perf_counter() - t0
        tracer = Tracer(model)
        lp_hook = forward_log_probs(model, wav)
        tracer.remove()
        r = {"frames": lp_plain.shape[0], "forward_s": round(t_fwd, 2),
             "hook_max_abs_diff": float((lp_hook - lp_plain).abs().max()), "hook_equal": bool(torch.equal(lp_hook, lp_plain))}
        with torch.inference_mode():
            if is_conformer(model):
                errs, rowsum = [], []
                for (layer, _d), (args, out) in sorted(tracer.rec.items()):
                    w, out2 = conformer_attention(model.encoder.layers[layer - 1].self_attn, *args)
                    errs.append(float((out2 - out).abs().max() / out.abs().max()))
                    rowsum.append(float((w.sum(-1) - 1).abs().max()))
                r.update(attn_rel_err_max=max(errs), attn_rowsum_err_max=max(rowsum))
            else:
                errs, t_ha = [], 0.0
                chans = torch.arange(512) if spec.res == "40 ms" else chans10
                for layer, direction, blk in mamba_blocks(model.encoder):
                    u, delta, b, c, a = ssm_tensors(blk, *tracer.rec[(layer, direction)])
                    t0 = time.perf_counter()
                    y = hidden_attention(delta, b, c, a, chans, u=u)["y"]
                    t_ha += time.perf_counter() - t0
                    ref = MambaRef._scan(u.T[None], delta.T[None], a, b[None], c[None])[0][chans]
                    errs.append(float((y - ref).abs().max() / ref.abs().max()))
                r.update(hidden_attn_rel_err_max=max(errs), hidden_attn_channels=len(chans),
                         hidden_attn_s_per_sentence=round(t_ha, 1))
        report["models"][key] = r
        print(key, r, flush=True)
    (OUT / "checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_run(_args) -> None:
    tok = BPETokenizer(TOKENIZER)
    initial = word_initial(tok)
    utts = read_utterances()
    chans10 = channel_subset()
    per, frames, timing = [], [], {}
    for key, spec in MODELS.items():
        t_model = time.perf_counter()
        model = load_model(key, tok)
        tracer = Tracer(model)
        for utt in utts:
            lp = forward_log_probs(model, load_wav(utt["audio"]))
            lab = frame_labels(lp, initial)
            frames.append({"model": key, "order": utt["order"], "frames": len(lp), "word_starts": int(lab["start"].sum()),
                           "boundary": int(lab["boundary"].sum()), "silent": int(lab["silent"].sum()),
                           "inside": int(lab["inside"].sum())})
            for row in analyze_sentence(model, spec, tracer.rec, lab, chans10):
                per.append({"model": key, "order": utt["order"], **row})
            tracer.rec.clear()
            print(f"{key} câu {utt['order']}/{len(utts)} · {time.perf_counter() - t_model:.0f} s", flush=True)
        tracer.remove()
        timing[key] = round(time.perf_counter() - t_model, 1)
    cols = ["model", "order", "layer", "direction", "M1", "M2", "M3_s", "M3_sub_s", "d_boundary", "d_inside", "d_silent"]
    write_csv(OUT / "per_sentence.csv", per, cols)
    write_csv(OUT / "frames.csv", frames, list(frames[0]))
    info = {"date": time.strftime("%Y-%m-%d %H:%M"), "torch": torch.__version__, "threads": torch.get_num_threads(),
            "n_utterances": len(utts), "n_channels_10ms": N_CH_10MS, "seconds_per_model": timing}
    (OUT / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    aggregate()


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def boot_ci(values: list[float]) -> tuple[float, float, float, int]:
    """Trung bình theo câu + CI 95% percentile, bootstrap theo câu 1.000 lần (B4); NaN (câu thiếu loại khung) bị bỏ."""
    v = np.array([x for x in values if np.isfinite(x)])
    if not len(v):
        return float("nan"), float("nan"), float("nan"), 0
    means = v[np.random.default_rng(SEED).integers(0, len(v), size=(N_BOOT, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(v)


def read_per_sentence() -> dict:
    groups = defaultdict(lambda: defaultdict(list))
    with open(OUT / "per_sentence.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            g = groups[(r["model"], int(r["layer"]), r["direction"])]
            for k in ("M1", "M2", "M3_s", "M3_sub_s", "d_boundary", "d_inside", "d_silent"):
                if r.get(k):
                    g[k].append(float(r[k]))
    return groups


def aggregate() -> None:
    groups = read_per_sentence()
    m12, m3 = [], []
    for (key, layer, direction), g in groups.items():
        spec = MODELS[key]
        base = {"model": key, "encoder": spec.encoder, "res": spec.res, "layer": layer, "direction": direction}
        mean, lo, hi, n = boot_ci(g["M3_s"])
        units = ("4 đầu" if direction == "hai_chieu" else f"{512 if spec.res == '40 ms' else N_CH_10MS} kênh")
        row3 = {**base, "units": units, "M3_s": mean, "M3_lo": lo, "M3_hi": hi, "n": n}
        if g["M3_sub_s"]:
            row3["M3_sub_s"] = float(np.mean(g["M3_sub_s"]))
        m3.append(row3)
        if direction == "hai_chieu":
            continue
        row = dict(base)
        for q in ("M1", "M2"):
            row[q], row[f"{q}_lo"], row[f"{q}_hi"], row[f"{q}_n"] = boot_ci(g[q])
        for k in ("d_boundary", "d_inside", "d_silent"):
            row[k] = float(np.nanmean(g[k]))
        m12.append(row)
    order = list(MODELS)
    key_fn = lambda r: (order.index(r["model"]), r["direction"] == "nguoc", r["layer"])  # noqa: E731
    write_csv(OUT / "m1_m2.csv", sorted(m12, key=key_fn),
              ["model", "encoder", "res", "layer", "direction", "M1", "M1_lo", "M1_hi", "M1_n", "M2", "M2_lo", "M2_hi",
               "M2_n", "d_boundary", "d_inside", "d_silent"])
    write_csv(OUT / "m3.csv", sorted(m3, key=key_fn),
              ["model", "encoder", "res", "layer", "direction", "units", "M3_s", "M3_lo", "M3_hi", "n", "M3_sub_s"])
    print(f"→ {OUT / 'm1_m2.csv'}, {OUT / 'm3.csv'}")


# ---------------------------------------------------------------- hình (B5)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # như scripts/rq2_final_tables.py: màu theo encoder
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
COLOR = {"B1": BLUE, "ConExtBiMamba": ORANGE, "Conformer PE": AQUA}
DIR_VI = {"xuoi": "xuôi", "nguoc": "ngược", "hai_chieu": "hai chiều"}


def vi_num(x: float, nd: int) -> str:
    return f"{x:.{nd}f}".replace(".", ",")



def style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7)


def frame_time(n: int, spec: Spec) -> np.ndarray:
    """Tâm khung (giây): 10 ms khung t ↔ t·10 ms (STFT center); 40 ms khung j phủ khung vào 4j…4j+6 → tâm 4j+3."""
    j = np.arange(n)
    return j * 0.01 if spec.res == "10 ms" else (4 * j + 3) * 0.01


def cmd_figures(_args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.lines
    import matplotlib.patches
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    import matplotlib.ticker

    comma = matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v:g}".replace(".", ","))  # dấu phẩy thập phân
    plt.rcParams["font.family"] = "Times New Roman"  # như các hình Ch3 (khớp font quyển)
    FIG.mkdir(parents=True, exist_ok=True)
    tok = BPETokenizer(TOKENIZER)
    initial = word_initial(tok)
    utt = read_utterances()[0]  # câu minh hoạ = câu đầu danh sách (B3)
    wav = load_wav(utt["audio"])

    # (a) log-mel + Δ_t khối đầu/giữa/cuối, B1 và ConExt 40 ms
    traces = {}
    for key in ("b1_40", "conext_40"):
        model = load_model(key, tok)
        tracer = Tracer(model)
        lp = forward_log_probs(model, wav)
        lab = frame_labels(lp, initial)
        n_layers = len(model.encoder.layers)
        layers = (1, n_layers // 2, n_layers)
        with torch.inference_mode():
            series = {}
            for layer, direction, blk in mamba_blocks(model.encoder):
                if layer in layers:
                    dbar = ssm_tensors(blk, *tracer.rec[(layer, direction)])[1].mean(1)
                    series[(layer, direction)] = (dbar.flip(0) if direction == "nguoc" else dbar).numpy()
            if key == "b1_40":
                mid = n_layers // 2
                hid = {}
                for direction in ("xuoi", "nguoc"):
                    blk = dict(((l, d), b) for l, d, b in mamba_blocks(model.encoder))[(mid, direction)]
                    u, delta, b, c, a = ssm_tensors(blk, *tracer.rec[(mid, direction)])
                    m = hidden_attention(delta, b, c, a, torch.arange(512), want_matrix=True)["mat"]
                    hid[direction] = m.flip(0).flip(1).numpy() if direction == "nguoc" else m.numpy()
                b1_mid, b1_layers = mid, n_layers
        traces[key] = (lab, layers, series, n_layers, words_at_starts(lab, tok))
        tracer.remove()
    model = load_model("conformer_40", tok)
    tracer = Tracer(model)
    forward_log_probs(model, wav)
    c_mid = len(model.encoder.layers) // 2
    with torch.inference_mode():
        w, _ = conformer_attention(model.encoder.layers[c_mid - 1].self_attn, *tracer.rec[(c_mid, "hai_chieu")][0])
    conf_mat, c_layers = w[0].mean(0).numpy(), len(model.encoder.layers)
    tracer.remove()

    with torch.inference_mode():  # log-mel trước CMVN, cùng tham số front-end chung
        logmel = torch.log(LogMelFeatureExtractor().mel(wav).clamp(min=1e-5)).numpy()
    dur = wav.numel() / 16000
    spec40 = MODELS["b1_40"]
    fig, axes = plt.subplots(7, 1, figsize=(7.0, 9.8), sharex=True,
                             gridspec_kw={"height_ratios": [1.3] + [1] * 6, "hspace": 0.6})
    axes[0].imshow(logmel, origin="lower", aspect="auto", cmap="magma", extent=(0, logmel.shape[1] * 0.01, 0, 80))
    axes[0].set_ylabel("kênh mel", fontsize=8, color=INK2)
    axes[0].set_title(f"Log-mel câu minh hoạ (dòng 1 của utterances.tsv, {utt['split']}:{utt['index']}, "
                      f"{vi_num(dur, 1)} s); chữ phía trên = từ theo căn chỉnh greedy của B1 40 ms", fontsize=8.5,
                      color=INK, loc="left", pad=34)
    axes[0].tick_params(colors=INK2, labelsize=7)
    lab_b1 = traces["b1_40"][0]
    x_b1 = frame_time(len(lab_b1["ids"]), spec40)
    for t, word in traces["b1_40"][4]:
        axes[0].text(x_b1[t], 1.02, word.lower(), transform=axes[0].get_xaxis_transform(), fontsize=5.5, rotation=70,
                     ha="left", va="bottom", color=INK2)
    for g, key in enumerate(("b1_40", "conext_40")):
        lab, layers, series, n_layers, _words = traces[key]
        name = MODELS[key].encoder
        x = frame_time(len(lab["ids"]), spec40)
        for j, layer in enumerate(layers):
            ax = axes[1 + 3 * g + j]
            ax.plot(x, series[(layer, "xuoi")], color=COLOR[name], lw=1.0)
            ax.plot(x, series[(layer, "nguoc")], color=COLOR[name], lw=1.0, ls="--")
            ylim = ax.get_ylim()  # theo Δ; nền và vạch vẽ sau không được kéo trục y
            ax.fill_between(x, 0, 1, where=lab["silent"].numpy(), step="mid", color="#d9d8d2", alpha=0.7, lw=0,
                            transform=ax.get_xaxis_transform(), zorder=0)
            for s in x[lab["start"].numpy()]:  # vạch theo căn chỉnh của chính mô hình này
                ax.axvline(s, color=INK2, lw=0.4, alpha=0.5, zorder=1)
            ax.set_ylim(ylim)
            pos = "đầu" if layer == 1 else ("cuối" if layer == n_layers else "giữa")
            ax.set_title(f"{name} 40 ms · lớp {layer}/{n_layers} ({pos})", fontsize=8.5, color=INK, loc="left")
            ax.set_ylabel("Δ (TB kênh)", fontsize=7.5, color=INK2)
            style(ax)
            ax.grid(False)
            ax.yaxis.set_major_formatter(comma)
    axes[-1].set_xlabel("thời gian (s)", fontsize=8, color=INK2)
    axes[-1].set_xlim(0, dur)
    axes[-1].xaxis.set_major_formatter(comma)
    handles = [matplotlib.lines.Line2D([], [], color=INK, lw=1.0, label="khối xuôi"),
               matplotlib.lines.Line2D([], [], color=INK, lw=1.0, ls="--", label="khối ngược"),
               matplotlib.lines.Line2D([], [], color=INK2, lw=0.6, label="khung bắt đầu từ (greedy CTC của chính mô hình)"),
               matplotlib.patches.Patch(color="#d9d8d2", label="khung lặng (P(blank) > 0,95)")]
    fig.subplots_adjust(bottom=0.075)
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.012))
    fig.savefig(FIG / "fig_a_delta_cau_minh_hoa.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # (b) attention Conformer + attention ẩn B1 ở lớp giữa, 40 ms
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), constrained_layout=True)
    span = conf_mat.shape[0] * spec40.sec
    ext = (0, span, span, 0)
    norm = LogNorm(vmin=1e-4, vmax=1)
    panels = [(conf_mat, f"Conformer PE 40 ms · lớp {c_mid}/{c_layers}\ntrung bình 4 đầu (softmax)"),
              (hid["xuoi"], f"B1 40 ms · lớp {b1_mid}/{b1_layers} xuôi\n|α| chuẩn hoá theo hàng, TB 512 kênh"),
              (hid["nguoc"], f"B1 40 ms · lớp {b1_mid}/{b1_layers} ngược\n|α| chuẩn hoá theo hàng, TB 512 kênh")]
    for ax, (m, title) in zip(axes, panels):
        im = ax.imshow(np.clip(m, 1e-6, None), cmap="viridis", norm=norm, extent=ext, interpolation="nearest")
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("khung được nhìn s (s)", fontsize=7.5, color=INK2)
        ax.tick_params(colors=INK2, labelsize=7)
        ax.xaxis.set_major_formatter(comma)
        ax.yaxis.set_major_formatter(comma)
    axes[0].set_ylabel("khung đầu ra t (s)", fontsize=7.5, color=INK2)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("trọng số (thang log)", fontsize=7.5, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    fig.savefig(FIG / "fig_b_attention_lop_giua_40ms.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # (c) M3 theo lớp, 6 mô hình
    with open(OUT / "m3.csv", encoding="utf-8", newline="") as f:
        m3 = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True, constrained_layout=True)
    for ax, res in zip(axes, ("40 ms", "10 ms")):
        for name, direction, ls, mk in (("B1", "xuoi", "-", "o"), ("B1", "nguoc", "--", "o"),
                                        ("ConExtBiMamba", "xuoi", "-", "s"), ("ConExtBiMamba", "nguoc", "--", "s"),
                                        ("Conformer PE", "hai_chieu", "-.", "^")):
            rows = sorted((r for r in m3 if r["encoder"] == name and r["res"] == res and r["direction"] == direction),
                          key=lambda r: int(r["layer"]))
            n_layers = len(rows)
            depth = np.array([int(r["layer"]) / n_layers for r in rows])
            y = np.array([float(r["M3_s"]) for r in rows])
            err = np.array([[y[i] - float(r["M3_lo"]), float(r["M3_hi"]) - y[i]] for i, r in enumerate(rows)]).T
            ax.errorbar(depth, y, yerr=err, color=COLOR[name], ls=ls, marker=mk, ms=3.5, lw=1.0, capsize=1.5,
                        mfc=COLOR[name] if direction != "nguoc" else "white", label=f"{name} {DIR_VI[direction]}")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(comma)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_formatter(comma)
        ax.set_title(f"Cấu hình {res}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("độ sâu tương đối (lớp / số lớp)", fontsize=8, color=INK2)
        style(ax)
    axes[0].set_ylabel("khoảng cách hiệu dụng M3 (s)", fontsize=8, color=INK2)
    axes[1].legend(fontsize=7, frameon=False, loc="best")
    fig.savefig(FIG / "fig_c_m3_theo_lop.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {FIG}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream.encoding != "utf-8":
            stream.reconfigure(encoding="utf-8")
    torch.set_num_threads(2)  # CPU dùng chung với việc A
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "check", "run", "aggregate", "figures"])
    args = ap.parse_args()
    {"select": cmd_select, "check": cmd_check, "run": cmd_run, "aggregate": lambda _a: aggregate(),
     "figures": cmd_figures}[args.cmd](args)


if __name__ == "__main__":
    main()
