"""Kernel Kaggle **CPU** (B, C, D — người dùng 19/10: việc CPU chia cho các tài khoản trừ A) — tối ưu giải mã beam + LM
cho nhánh 40 ms trên GigaSpeech 2 (`docs/notes/ket_qua/so_sanh_hien_hanh_2026-10-19.md` B.2 mục 4: chỉ chỉnh trên **dev**, test
chỉ để báo). Mỗi tài khoản một encoder (`ENCODER`); cấu hình chung cho cả 3 encoder chọn ở local từ `tune.json` của 3 kernel.

Môi trường: Python 3.12 riêng qua uv + torch/torchaudio 2.8 CPU + flashlight-text 0.0.7 (bản Linux kèm KenLM; image
Kaggle Py 3.13 không có wheel, build từ nguồn hỏng — kernel thử `lm_probe` 19/10). KenLM `lmplz` build từ nguồn.

STAGE `tune`:
1. Kiểm cổng LM: 200 câu test đầu, LM Vietcetera (`lm3.arpa`, xuất từ `lm3.pkl`) qua KenLM, cấu hình cố định → phải trùng
   giả thuyết beam+LM local (LM Python) — sai thì dừng (mọi so sánh sau dựa trên giả định hai đường cho cùng kết quả).
2. LM rộng: văn bản LM Vietcetera + văn bản train GigaSpeech 2 (`train_refined.tsv`, nhãn tự động), bỏ câu trùng nguyên
   văn với dev/test GigaSpeech 2; KenLM 3-gram (`lmplz`, KN sửa đổi — khác KN một discount của LM Python, ghi khi báo).
   Từ điển = mọi từ của văn bản LM rộng, viết bằng BPE như `lm_decode.build`.
3. Lưới {LM Vietcetera, LM rộng} × lm_weight × word_score trên `N_DEV` câu dev (chia đều 30 nhóm), chấm bằng đúng hàm
   chấm của clean-test (`score_clean_test`) → `tune.json`.
STAGE `final`: giải mã test 3 seed của encoder với cấu hình `FINAL` → `test_hyps.tsv` (chấm ở local bằng `score_gs2`).
"""

import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Kernel Kaggle chạy script như notebook (không có `__file__`) → phần trong venv chạy bản sao `decode_cpu.py` trong bundle,
# tham số truyền qua biến môi trường; `launch_cpu.py` thay 3 placeholder dưới đây ở bản đẩy lên.
STAGE = os.environ.get("GS2_STAGE", "__STAGE__")
ENCODER = os.environ.get("GS2_ENCODER", "__ENCODER__")
FINAL = json.loads(os.environ.get("GS2_FINAL", '__FINAL__').replace("__FINAL__", "null"))  # {"lm": vss|broad, "lmw": .., "ws": ..}
# Lưới mở rộng cho lần tune sau (19/10: tối ưu ws = 2,0 nằm ở mép lưới) — {"lmw": [..], "ws": [..]}, null = lưới mặc định.
GRID = json.loads(os.environ.get("GS2_GRID", '__GRID__').replace("__GRID__", "null"))
N_DEV = 2000
LMW = [0.4, 0.6, 0.8, 1.0, 1.3]
WS = [-1.0, 0.0, 1.0, 2.0]
WORK = Path("/tmp/work")
KENLM = Path("/tmp/kenlm")
VENV = Path("/tmp/py312")
OUT = Path("/kaggle/working")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def sh(cmd: str, check: bool = True) -> str:
    t = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-4:]
    log(f"rc={r.returncode} ({time.time() - t:.0f}s) $ {cmd[:110]}" + ("\n    " + "\n    ".join(tail) if tail else ""))
    if check and r.returncode:
        sys.exit(f"LỖI: {cmd[:80]}")
    return r.stdout


def find(pattern: str) -> Path:
    hits = sorted({p for pre in ("", "*/", "*/*/", "*/*/*/") for p in Path("/kaggle/input").glob(pre + pattern)})
    if not hits:
        sys.exit(f"LỖI: không thấy {pattern}")
    return hits[0]


def setup() -> None:
    """Chạy bằng Python image; phần giải mã chạy lại chính file này trong venv 3.12 (`--inner`)."""
    sh(f"{sys.executable} -m pip install -q uv && uv python install 3.12 && uv venv -q -p 3.12 {VENV}")
    sh(f"uv pip install -q -p {VENV}/bin/python torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu"
       f" && uv pip install -q -p {VENV}/bin/python flashlight-text==0.0.7 numpy sentencepiece jiwer")
    if STAGE == "tune" or FINAL["lm"] == "broad":
        sh("apt-get -qq update && apt-get -qq install -y libboost-program-options-dev libboost-system-dev "
           "libboost-thread-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev > /dev/null")
        sh(f"git clone -q https://github.com/kpu/kenlm {KENLM} && mkdir -p {KENLM}/build && cd {KENLM}/build && "
           "cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1 && make -j4 lmplz build_binary > /dev/null 2>&1")
    bundle = find("gs2-cpu-*/bundle.tar.gz") if list(Path("/kaggle/input").rglob("bundle.tar.gz")) else None
    WORK.mkdir(parents=True, exist_ok=True)
    if bundle:
        with tarfile.open(bundle) as f:
            f.extractall(WORK, filter="data")
    else:  # Kaggle tự giải nén archive khi tạo dataset
        src = next(p for p in Path("/kaggle/input").rglob("src") if (p / "evaluation/lm_decode.py").exists()).parent
        sh(f"cp -r {src}/. {WORK}/")
    # `score_clean_test` chỉ lấy hằng `OURS_VISIBLE_FROM` từ `correct_clean_test` (công cụ hiệu đính, import gradio/torch
    # ở đầu file — venv không có, lỗi 19/10 23:09) → thay bằng file chỉ mang đúng hằng đó. Sửa ở bước này (code kernel)
    # vì phần trong venv chạy bản `decode_cpu.py` cũ trong bundle; khỏi tải lại dataset.
    cct = WORK / "src/data/correct_clean_test.py"
    const = re.search(r'^OURS_VISIBLE_FROM = "[^"]+"', cct.read_text(encoding="utf-8"), re.M).group(0)
    cct.write_text(const + "\n", encoding="utf-8")
    if GRID:  # cùng lý do: bản trong venv là bản cũ trong bundle → sửa thẳng hai dòng lưới của nó
        dc = WORK / "decode_cpu.py"
        text = re.sub(r"^LMW = .*$", f"LMW = {GRID['lmw']}", dc.read_text(encoding="utf-8"), count=1, flags=re.M)
        text = re.sub(r"^WS = .*$", f"WS = {GRID['ws']}", text, count=1, flags=re.M)
        dc.write_text(text, encoding="utf-8")
        log(f"lưới: LMW {GRID['lmw']} × WS {GRID['ws']}")
    log(f"bundle:{sorted(p.name for p in WORK.iterdir())}")


# ---------------------------------------------------------------- phần chạy trong venv 3.12 (cwd = WORK)

def make_decoder(lexicon: str, lm_path: str, lmw: float, ws: float, tokens: list[str]):
    from torchaudio.models.decoder import ctc_decoder
    from src.evaluation import lm_decode as L
    return ctc_decoder(lexicon=lexicon, tokens=tokens, lm=lm_path, nbest=1, beam_size=L.DECODE["beam"],
                       beam_threshold=L.DECODE["beam_threshold"], beam_size_token=L.DECODE["beam_size_token"],
                       lm_weight=lmw, word_score=ws, blank_token=L.BLANK, sil_token=L.SIL, unk_word=L.UNK,
                       unk_score=float("-inf"))


def decode_one(dec, knlm, tok, lp):
    from src.evaluation import lm_decode as L
    g = L.greedy(lp, tok)
    h = L.beam_words(dec, lp)
    if len(h.split()) < L.FALLBACK * len(g.split()):
        h = L.split_sticky(g, knlm)
    return h


def load_emissions(split: str, model: str, keys: set | None = None) -> dict:
    from src.evaluation import lm_decode as L
    path = find(f"gs2-cpu-*/tk_{model}_gs2_{split}.tkz")  # đuôi .tkz: Kaggle tự giải nén .npz (là zip) khi tạo dataset
    return {k: lp for k, lp in L.topk_emissions(path) if keys is None or k in keys}


def build_broad_lm() -> tuple[str, str]:
    import unicodedata
    from src.evaluation import lm_decode as L
    norm = lambda s: " ".join(unicodedata.normalize("NFC", s).upper().split())  # noqa: E731  (= build_lm_text)
    block = set()
    for split in ("dev", "test"):
        for line in open(f"data/gs2/{split}.tsv", encoding="utf-8"):
            block.add(norm(line.rstrip("\n").split("\t", 1)[1]))
    text = Path("/tmp/broad.txt")
    n_in = n_out = 0
    with text.open("w", encoding="utf-8") as f:
        for line in open("checkpoints/lm/lm_text_clean.txt", encoding="utf-8"):  # đã loại câu trùng val/clean-test
            if line.strip():
                f.write(line)
        import gzip
        gz = Path("data/lm/gs2_train_refined.tsv.gz")
        src = gzip.open(gz, "rt", encoding="utf-8") if gz.exists() else open(gz.with_suffix(""), encoding="utf-8")
        for line in src:
            s = norm(line.rstrip("\n").split("\t", 1)[-1])
            n_in += 1
            if s and s not in block:
                f.write(s + "\n")
                n_out += 1
    log(f"văn bản LM rộng: GS2 train {n_in} câu → giữ {n_out} (bỏ {n_in - n_out} trùng dev/test)")
    arpa = "/tmp/broad3.arpa"
    sh(f"{KENLM}/build/bin/lmplz -o 3 -S 60% -T /tmp --discount_fallback < {text} > {arpa}")
    binary = "/tmp/broad3.bin"
    sh(f"{KENLM}/build/bin/build_binary {arpa} {binary}")
    tok = L._Tokens()
    words = set()
    for line in text.open(encoding="utf-8"):
        words.update(line.split())
    lex = "/tmp/broad_lexicon.txt"
    with open(lex, "w", encoding="utf-8") as f:
        for w in sorted(words):
            f.write(w + "\t" + " ".join(tok.sp.encode(w, out_type=str)) + "\n")
    log(f"LM rộng: từ điển {len(words)} từ, ARPA {Path(arpa).stat().st_size / 1e6:.0f} MB")
    return binary, lex


def tune_worker(args):
    cfg, lms, keys = args
    from src.evaluation import lm_decode as L
    tok = L._Tokens()
    knlm = L.load_lm(Path("checkpoints/lm/lm3.pkl"))
    lm_path, lex = lms[cfg["lm"]]
    dec = make_decoder(lex, lm_path, cfg["lmw"], cfg["ws"], tok.tokens)
    em = load_emissions("dev", f"{ENCODER}_seed42", keys)
    return cfg, {k: decode_one(dec, knlm, tok, lp) for k, lp in em.items()}


def score(hyps: dict, refs: dict) -> float:
    from src.evaluation.score_clean_test import hyp_tokens, load_alt, ref_tokens, score_sentence
    alt = load_alt()
    e = w = 0
    for k, h in hyps.items():
        ee, ww = score_sentence(ref_tokens(-1, refs[k], {}, alt), hyp_tokens(h, alt))
        e += sum(ee.values())
        w += sum(ww.values())
    return e / w


def inner() -> None:
    os.chdir(WORK)
    sys.path.insert(0, str(WORK))
    from multiprocessing import Pool

    from src.evaluation import lm_decode as L
    tok = L._Tokens()
    knlm = L.load_lm(Path("checkpoints/lm/lm3.pkl"))
    vss = ("checkpoints/lm/lm3.arpa", "checkpoints/lm/lexicon.txt")

    if STAGE == "tune":
        # 1. Cổng: KenLM + ARPA phải cho cùng giả thuyết như LM Python ở local (cấu hình cố định).
        local = {}
        import csv
        for r in csv.DictReader(open("data/gs2/gs2_hyps_lm_local.tsv", encoding="utf-8"), delimiter="\t"):
            local[r["key"]] = r[f"v2a_{ENCODER}_seed42_lm"]
        dec = make_decoder(vss[1], vss[0], L.DECODE["lm_weight"], L.DECODE["word_score"], tok.tokens)
        em = load_emissions("test", f"{ENCODER}_seed42")
        keys = list(em)[:200]
        same = sum(decode_one(dec, knlm, tok, em[k]) == local[k] for k in keys)
        log(f"CỔNG LM: KenLM(lm3.arpa) trùng LM Python local {same}/{len(keys)} câu")
        gate = {"same": same, "n": len(keys)}
        if same < len(keys) * 0.98:
            (OUT / "tune.json").write_text(json.dumps({"gate": gate, "error": "lệch LM Python"}), encoding="utf-8")
            sys.exit("LỖI: KenLM lệch LM Python — dừng")

        broad = build_broad_lm()
        refs = dict(l.rstrip("\n").split("\t", 1) for l in open("data/gs2/dev.tsv", encoding="utf-8") if "\t" in l)
        by_group = {}
        for k in refs:
            by_group.setdefault(k.split("-")[0], []).append(k)
        per = -(-N_DEV // len(by_group))
        sub = {k for ks in by_group.values() for k in ks[:per]}
        lms = {"vss": vss, "broad": broad}
        grid = [{"lm": lm, "lmw": a, "ws": b} for lm in lms for a in LMW for b in WS]
        t = time.time()
        results = []
        with Pool(4) as pool:
            for cfg, hyps in pool.imap_unordered(tune_worker, [(c, lms, sub) for c in grid]):
                wer = score(hyps, refs)
                results.append({**cfg, "wer": wer})
                log(f"{cfg} → {wer:.4f} ({len(results)}/{len(grid)}, {time.time() - t:.0f}s)")
                (OUT / "tune.json").write_text(json.dumps({"encoder": ENCODER, "gate": gate, "n_dev": len(sub),
                                                           "results": results}, indent=1), encoding="utf-8")
        best = min(results, key=lambda r: r["wer"])
        log(f"TỐT NHẤT ({ENCODER}, dev {len(sub)} câu): {best}")
    else:
        # LM rộng dựng lại ở đây (output kernel tune của tài khoản khác không đọc được); lmplz tất định với cùng văn bản.
        lm_path, lex = build_broad_lm() if FINAL["lm"] == "broad" else vss
        dec = make_decoder(lex, lm_path, FINAL["lmw"], FINAL["ws"], tok.tokens)
        rows = {}
        for seed in (42, 43, 44):
            m = f"{ENCODER}_seed{seed}"
            for k, lp in load_emissions("test", m).items():
                rows.setdefault(k, {})[f"v2a_{m}_lm"] = decode_one(dec, knlm, tok, lp)
            log(f"{m}: xong")
        cols = sorted({c for v in rows.values() for c in v})
        with open(OUT / "test_hyps.tsv", "w", encoding="utf-8") as f:
            f.write("key\t" + "\t".join(cols) + "\n")
            for k, v in rows.items():
                f.write(k + "\t" + "\t".join(v[c] for c in cols) + "\n")


if __name__ == "__main__":
    if "--inner" in sys.argv:
        inner()
    else:
        assert STAGE in ("tune", "final") and ENCODER in ("mamba", "conextbimamba", "conformer_relpos")
        assert STAGE == "tune" or FINAL, "stage final cần FINAL"
        setup()
        env = {**os.environ, "GS2_STAGE": STAGE, "GS2_ENCODER": ENCODER, "GS2_FINAL": json.dumps(FINAL)}
        r = subprocess.run([f"{VENV}/bin/python", str(WORK / "decode_cpu.py"), "--inner"], cwd=WORK, env=env)
        log(f"XONG rc={r.returncode}")
        sys.exit(r.returncode)
