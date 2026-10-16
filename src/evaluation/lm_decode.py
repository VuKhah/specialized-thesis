"""Giải mã CTC có từ điển + mô hình ngôn ngữ 3-gram cho bảng phụ RQ1 (Bảng 3.4) — đưa vào repo 2026-10-16 từ script
agent giải mã 15/10 (`D:/Model/kltn_decoding_2026-10-15/decoding/`: `ngram_lm.py`, `decode_lib.py`,
`cleantest_decode.py`), giữ nguyên thuật toán và tham số; chỉ bỏ phần dò tham số.

Vì sao tự viết LM: flashlight-text bản Windows không kèm KenLM và máy không có trình biên dịch C++ để build `kenlm`.
KN nội suy dạng backoff như ARPA; ghi được ARPA để dùng với KenLM nơi khác.

Thiết lập **chốt trên nửa tune val_unseen 15/10, không chỉnh lại theo seed/mô hình** (note gap 9.6 điều 7;
`glued_words_2026-10-15.md` mục 3): từ điển = mọi từ của văn bản LM; LM 3-gram KN; beam 10, beam_size_token 20,
beam_threshold 15, lm_weight 0,8, word_score 1,0; nếu câu beam ngắn hơn 0,7 × số từ greedy thì thay bằng greedy đã
tách từ dính bằng LM (`split_sticky`).

Văn bản LM: `train_transcripts_filtered.txt` trừ mọi câu trùng nguyên văn với validation, val_unseen, clean-test
(bản `train_transcripts.txt` cũ chứa 179/203 câu clean-test — CLAUDE.md). Dựng lại từ đầu khớp tuyệt đối `lm3c.pkl`
của 15/10 (kiểm 16/10: 1.384.233 n-gram, từ điển 15.391 từ).

    python -m src.evaluation.lm_decode build --text <lm_text_clean.txt>      # → checkpoints/lm/{lexicon.txt,lm3.pkl}
    python -m src.evaluation.lm_decode build                                  # dựng cả văn bản (cần HF, ~1 phút)
    python -m src.evaluation.lm_decode decode --npz relpos_seed43=<lp.npz> ... --out <hyps.tsv>

Đầu vào `decode`: npz do kernel dump ghi (khoá "split:index" → log-prob [T', 1001], cột 0 = blank). Đầu ra TSV: cột
`key`, rồi `<tên>_greedy`, `<tên>_lm` — cùng dạng `cleantest_hyps.tsv` mà `score_clean_test.py` đọc.
"""

import argparse
import collections
import csv
import json
import math
import pickle
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

from src.evaluation.vi_syllable import is_valid_syllable, is_vi_marked

LM_DIR = Path("checkpoints/lm")  # gitignore (checkpoints/*): LM ~45 MB
TOKENIZER = Path("configs/tokenizer.model")
BOS, EOS, UNK = "<s>", "</s>", "<unk>"
BLANK, SIL = "<blk>", "<sil>"
DECODE = {"beam": 10, "beam_size_token": 20, "beam_threshold": 15, "lm_weight": 0.8, "word_score": 1.0}
FALLBACK = 0.7


class KNLM:
    def __init__(self, order=3):
        self.order = order
        self.logp = {}  # tuple ngram -> log10 p (dạng ARPA)
        self.bow = {}  # tuple context -> log10 backoff

    def fit(self, sentences):
        N = self.order
        counts = [collections.Counter() for _ in range(N + 1)]
        for s in sentences:
            w = [BOS] + s + [EOS]
            for i in range(1, len(w)):
                for n in range(1, N + 1):
                    if i - n + 1 < 0:
                        break
                    counts[n][tuple(w[i - n + 1: i + 1])] += 1
        # KN: bậc cao nhất dùng count thật; bậc thấp hơn dùng continuation count (số kiểu từ đứng trước),
        # trừ n-gram bắt đầu bằng <s> (không có ngữ cảnh trái) giữ count thật.
        kn = [collections.Counter() for _ in range(N + 1)]
        kn[N] = counts[N]
        for n in range(N, 1, -1):
            for g in counts[n]:
                kn[n - 1][g[1:]] += 1
            for g, c in counts[n - 1].items():
                if g[0] == BOS:
                    kn[n - 1][g] = c
        self.vocab = {g[0] for g in kn[1]} | {UNK}
        D = {}  # discount mỗi bậc: Ney D = n1/(n1+2 n2)
        for n in range(1, N + 1):
            cc = collections.Counter(kn[n].values())
            n1, n2 = cc[1], cc[2]
            D[n] = n1 / (n1 + 2 * n2) if n1 + 2 * n2 > 0 else 0.5
        self.D = D
        tot1 = sum(kn[1].values())
        V = len(self.vocab)
        types1 = len(kn[1])
        p1 = {g: (max(c - D[1], 0) + D[1] * types1 / V) / tot1 for g, c in kn[1].items()}
        p1[(UNK,)] = D[1] * types1 / V / tot1
        p1.setdefault((BOS,), 1e-99)
        prob = {1: p1}
        for n in range(2, N + 1):
            ctx_tot, ctx_types = collections.Counter(), collections.Counter()
            for g, c in kn[n].items():
                ctx_tot[g[:-1]] += c
                ctx_types[g[:-1]] += 1
            pn = {}
            for g, c in kn[n].items():
                h = g[:-1]
                gamma = D[n] * ctx_types[h] / ctx_tot[h]
                pn[g] = max(c - D[n], 0) / ctx_tot[h] + gamma * self._p_lower(prob, g[1:])
            for h in ctx_tot:
                self.bow[h] = math.log10(D[n] * ctx_types[h] / ctx_tot[h])
            prob[n] = pn
        for n in prob:
            for g, p in prob[n].items():
                self.logp[g] = math.log10(p) if p > 0 else -99.0
        return self

    def _p_lower(self, prob, g):
        if len(g) == 1:
            return prob[1].get(g, prob[1][(UNK,)])
        if g in prob[len(g)]:
            return prob[len(g)][g]
        h = g[:-1]
        b = 10 ** self.bow[h] if h in self.bow else 1.0
        return b * self._p_lower(prob, g[1:])

    def score(self, ctx: tuple, w: str) -> float:
        """log10 p(w | ctx), ctx = tuple các từ trước (cắt về order-1)."""
        if w not in self.vocab:
            w = UNK
        ctx = ctx[-(self.order - 1):] if self.order > 1 else ()
        s = 0.0
        while True:
            lp = self.logp.get(ctx + (w,))
            if lp is not None:
                return s + lp
            if not ctx:
                return s + self.logp[(UNK,)]
            s += self.bow.get(ctx, 0.0)
            ctx = ctx[1:]


# Lớp KNLM ở module này phải đọc được pickle 15/10 (lớp `ngram_lm.KNLM`, cùng thuộc tính).
class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return KNLM if name == "KNLM" else super().find_class(module, name)


def load_lm(path: Path) -> KNLM:
    with open(path, "rb") as f:
        return _Unpickler(f).load()


class _Tokens:
    def __init__(self):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
        self.pieces = [self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())]
        # SIL = cột giả thêm vào emission (log-prob rất thấp) — vocab không có ký hiệu ngắt từ riêng
        self.tokens = [BLANK] + self.pieces + [SIL]

    def ids_to_text(self, ids):
        return self.sp.decode([i - 1 for i in ids if 0 < i <= len(self.pieces)])


def greedy(lp: np.ndarray, tok: _Tokens) -> str:
    out, prev = [], -1
    for t in lp.argmax(-1):
        if t != prev and t != 0:
            out.append(int(t))
        prev = t
    return tok.ids_to_text(out)


def make_decoder(lexicon: Path, knlm: KNLM, tok: _Tokens):
    from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState, ctc_decoder
    import torchaudio.models.decoder._ctc_decoder as _m

    class NgramLM(CTCDecoderLM):
        """Bọc KNLM cho flashlight: state → tuple ngữ cảnh; điểm log10 như KenLM trong flashlight."""

        def __init__(self, lm):
            super().__init__()
            self.lm, self.ctx, self.words, self.cache = lm, {}, None, {}

        def set_words(self, word_dict):
            self.words = [word_dict.get_entry(i) for i in range(word_dict.index_size())]

        def start(self, start_with_nothing=False):
            self.ctx = {}
            st = CTCDecoderLMState()
            self.ctx[st] = (BOS,)
            return st

        def _sc(self, ctx, w):
            v = self.cache.get((ctx, w))
            if v is None:
                v = self.cache[(ctx, w)] = self.lm.score(ctx, w)
                if len(self.cache) > 2_000_000:
                    self.cache.clear()
            return v

        def score(self, state, idx):
            out = state.child(idx)
            ctx, w = self.ctx[state], self.words[idx]
            if out not in self.ctx:
                self.ctx[out] = (ctx + (w,))[-(self.lm.order - 1):]
            return out, self._sc(ctx, w)

        def finish(self, state):
            return state, self._sc(self.ctx[state], EOS)

    lm = NgramLM(knlm)
    lm.set_words(_m._create_word_dict(_m._load_words(str(lexicon))))  # trie gọi lm.score lúc dựng
    return ctc_decoder(lexicon=str(lexicon), tokens=tok.tokens, lm=lm, nbest=1, beam_size=DECODE["beam"],
                       beam_threshold=DECODE["beam_threshold"], beam_size_token=DECODE["beam_size_token"],
                       lm_weight=DECODE["lm_weight"], word_score=DECODE["word_score"], blank_token=BLANK,
                       sil_token=SIL, unk_word=UNK, unk_score=float("-inf"))


def beam_words(dec, lp: np.ndarray) -> str:
    import torch
    e = np.full((lp.shape[0], lp.shape[1] + 1), -1e4, dtype=np.float32)
    e[:, :-1] = lp
    return " ".join(dec(torch.from_numpy(e)[None])[0][0].words)


ONSETS = ["", "B", "C", "CH", "D", "Đ", "G", "GH", "GI", "H", "K", "KH", "L", "M", "N", "NG", "NGH", "NH", "P", "PH",
          "QU", "R", "S", "T", "TH", "TR", "V", "X"]


def split_sticky(text: str, knlm: KNLM, max_parts: int = 3) -> str:
    """Từ có dấu nhưng không phải âm tiết hợp lệ → tách thành (âm tiết hợp lệ, [phụ âm đầu bị nuốt] + phần còn lại),
    chọn bằng LM trong ngữ cảnh 2 từ trước + 1 từ sau. Không tách được thì giữ nguyên."""
    words, out = text.split(), []
    for k, w in enumerate(words):
        if not (is_vi_marked(w) and not is_valid_syllable(w)):
            out.append(w)
            continue
        nxt = words[k + 1] if k + 1 < len(words) else EOS
        best, best_s = None, -1e9
        for cand in _splits(w, knlm.vocab, max_parts):
            ctx, s = tuple(([BOS] + out)[-(knlm.order - 1):]), 0.0
            for c in cand + [nxt]:
                s += knlm.score(ctx, c)
                ctx = (ctx + (c,))[-(knlm.order - 1):]
            if s > best_s:
                best, best_s = cand, s
        out.extend(best if best else [w])
    return " ".join(out)


def _splits(w, vocab, max_parts):
    if is_valid_syllable(w) and w in vocab:
        yield [w]
    if max_parts <= 1:
        return
    for i in range(1, len(w)):
        left, right = w[:i], w[i:]
        if not (is_valid_syllable(left) and left in vocab):
            continue
        for on in ONSETS:
            r = on + right
            if is_valid_syllable(r) and r in vocab:
                yield [left, r]
        if not is_valid_syllable(right):
            for rest in _splits(right, vocab, max_parts - 1):
                if len(rest) > 1:
                    yield [left] + rest


def build_lm_text(out: Path) -> None:
    """Văn bản LM sạch (nguyên văn `clean_lm_text.py` 15/10): filtered trừ câu trùng validation/val_unseen/clean-test."""
    from datasets import load_dataset

    from src.data.vietsuperspeech_dataset import HF_DATASET_ID, HF_REVISION, manifest_rows
    norm = lambda s: " ".join(unicodedata.normalize("NFC", s).upper().split())  # noqa: E731
    hf = {s: load_dataset(HF_DATASET_ID, split=s, revision=HF_REVISION) for s in ["train", "validation"]}
    black = {"validation": {norm(t) for t in hf["validation"]["text"]}}
    tr, va = hf["train"]["text"], hf["validation"]["text"]
    black["val_unseen"] = {norm((tr if r["split"] == "train" else va)[r["index"]]) for r in manifest_rows(["val_unseen"])}
    ct = json.loads(Path("data/processed/clean_test_manifest.json").read_text(encoding="utf-8"))["samples"]
    black["clean_test"] = {norm(s["pseudo_label"]) for s in ct} | {norm(s["corrected_text"]) for s in ct
                                                                   if s.get("corrected_text")}
    lines = [norm(x) for x in open("data/processed/train_transcripts_filtered.txt", encoding="utf-8") if x.strip()]
    allb = set().union(*black.values())
    keep = [x for x in lines if x not in allb]
    print(f"filtered {len(lines)} → sạch {len(keep)}", flush=True)
    out.write_text("\n".join(keep) + "\n", encoding="utf-8")


def build(text: Path | None) -> None:
    LM_DIR.mkdir(parents=True, exist_ok=True)
    if text is None:
        text = LM_DIR / "lm_text_clean.txt"
        build_lm_text(text)
    sents = [x.split() for x in open(text, encoding="utf-8") if x.strip()]
    t = time.time()
    lm = KNLM(3).fit(sents)
    with open(LM_DIR / "lm3.pkl", "wb") as f:
        pickle.dump(lm, f)
    tok = _Tokens()
    words = sorted({w for s in sents for w in s})
    with open(LM_DIR / "lexicon.txt", "w", encoding="utf-8") as f:
        for w in words:
            f.write(w + "\t" + " ".join(tok.sp.encode(w, out_type=str)) + "\n")
    print(f"LM {len(lm.logp)} n-gram · từ điển {len(words)} từ · {time.time() - t:.0f} s → {LM_DIR}", flush=True)


def decode(npz: dict[str, Path], out: Path) -> None:
    tok = _Tokens()
    knlm = load_lm(LM_DIR / "lm3.pkl")
    dec = make_decoder(LM_DIR / "lexicon.txt", knlm, tok)
    hyps = {}
    for name, path in npz.items():
        data = np.load(path)
        t = time.time()
        for k in data.files:
            lp = data[k].astype(np.float32)
            g = greedy(lp, tok)
            h = beam_words(dec, lp)
            if len(h.split()) < FALLBACK * len(g.split()):
                h = split_sticky(g, knlm)
            hyps.setdefault(k, {})[f"{name}_greedy"] = g
            hyps[k][f"{name}_lm"] = h
        print(f"{name}: {len(data.files)} câu, {time.time() - t:.0f} s", flush=True)
    cols = sorted({c for v in hyps.values() for c in v})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["key"] + cols)
        for k, v in hyps.items():
            w.writerow([k] + [v.get(c, "") for c in cols])
    print(f"ghi {len(hyps)} câu → {out}", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--text", type=Path, default=None, help="Văn bản LM đã lọc sẵn; bỏ trống thì dựng từ HF")
    d = sub.add_parser("decode")
    d.add_argument("--npz", nargs="+", required=True, help="tên=đường_dẫn.npz (tên thành tiền tố cột)")
    d.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.text)
    else:
        decode({x.split("=", 1)[0]: Path(x.split("=", 1)[1]) for x in a.npz}, a.out)


if __name__ == "__main__":
    main()
