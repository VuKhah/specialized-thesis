"""Kernel Kaggle GPU: nhận diện ngôn ngữ (LID) trên audio cho toàn bộ 67.405
đoạn VietSuperSpeech — căn cứ lọc A1 (TODO.md).

Vì sao cần: nhãn VietSuperSpeech do Zipformer chỉ-tiếng-Việt sinh tự động.
Lọc theo nhãn (tỉ lệ từ có dấu) bắt được đoạn tiếng Anh bị phiên thành chuỗi
giả tiếng Anh, nhưng để lọt audio tiếng Nhật bị phiên thành âm tiết Việt có dấu
(người dùng nghe kiểm video Alive Kicking Ep4, train#15087, 2026-10-03). LID
trên audio không phụ thuộc nhãn.

Nguồn audio: output tar của 5 kernel `make-dataset-asr-*` gắn qua
`kernel_sources` (đọc tuần tự trong tar, không giải nén). 250 câu clean-test
không nằm trong tar val (val = validation trừ clean-test) → tải lẻ từ HF.

Whisper-small, chỉ 1 bước giải mã sau <|startoftranscript|>: softmax trên các
token ngôn ngữ → xác suất từng ngôn ngữ. Đoạn ≤ 15 s nên vừa cửa sổ 30 s.

    kaggle kernels push -p scripts/kaggle/lid

Output: /kaggle/working/lid.csv (split, index, audio, top1, p_top1, top2,
p_top2, p_vi, p_en, p_ja, p_ko, p_zh).
"""

import csv
import io
import json
import os
import queue
import subprocess
import sys
import tarfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.stdout.reconfigure(encoding="utf-8")

MODEL = "openai/whisper-small"
BATCH = 32
HF_DATASET_ID = "thanhnew2001/VietSuperSpeech"
HF_REVISION = "cbf624ae9b30e1c2793a27e95b262115c69601f3"
REPO_URL = "https://github.com/VuKhah/specialized-thesis"
INPUT = Path(os.environ.get("LID_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("LID_WORK", "/kaggle/working"))
LIMIT = int(os.environ.get("LID_LIMIT", "0"))  # >0: chỉ chạy thử N đoạn mỗi nguồn
LANGS = ["vi", "en", "ja", "ko", "zh"]


def index_map():
    """audio path → (split, index) từ các TSV đi kèm tar (data/splits/*.tsv)."""
    m = {}
    for tsv in INPUT.rglob("*.tsv"):
        split = "validation" if tsv.stem == "val" else "train"
        for line in tsv.read_text(encoding="utf-8").splitlines()[1:]:
            i, a, _ = line.split("\t")
            m[a] = (split, int(i))
    return m


def from_tars(q):
    tars = sorted(INPUT.rglob("*.tar"))
    print("tar:", [str(t) for t in tars], flush=True)
    for t in tars:
        n = 0
        with tarfile.open(t, mode="r|") as tar:  # đọc tuần tự, không seek
            for mem in tar:
                if not mem.name.endswith(".wav"):
                    continue
                q.put((mem.name, tar.extractfile(mem).read()))
                n += 1
                if LIMIT and n >= LIMIT:
                    break
        print(f"  xong {t.name}: {n} đoạn", flush=True)


def clean_test_rows():
    """250 câu clean-test: lấy index từ repo, đường dẫn audio từ HF (revision pin)."""
    from datasets import load_dataset

    repo = Path("/tmp/repo")
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)], check=True)
    ct = json.loads((repo / "data/processed/clean_test_manifest.json").read_text(encoding="utf-8"))["samples"]
    audio = load_dataset(HF_DATASET_ID, split="validation", revision=HF_REVISION)["audio"]
    rows = [(audio[s["index"]], s["index"]) for s in ct]
    return rows[:LIMIT] if LIMIT else rows


def from_hf(q, rows):
    from huggingface_hub import hf_hub_download

    def one(a):
        for k in range(6):
            try:
                return a, Path(hf_hub_download(HF_DATASET_ID, a, repo_type="dataset", revision=HF_REVISION,
                                               local_dir="/tmp/ct")).read_bytes()
            except Exception:
                time.sleep(2 ** k)
        return a, None

    with ThreadPoolExecutor(8) as ex:
        for a, b in ex.map(one, [a for a, _ in rows]):
            if b is None:
                print("  LỖI tải", a, flush=True)
            else:
                q.put((a, b))


def main():
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    fe = WhisperFeatureExtractor.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=dtype).to(dev).eval()
    lang_to_id = model.generation_config.lang_to_id  # {"<|vi|>": id, ...}
    names = [k.strip("<|>") for k in lang_to_id]
    ids = torch.tensor(list(lang_to_id.values()), device=dev)
    sot = model.generation_config.decoder_start_token_id

    imap = index_map()
    ct = clean_test_rows()
    for a, i in ct:
        imap[a] = ("validation", i)
    print(f"index map {len(imap)} đoạn (kỳ vọng 67.405 khi đủ 5 tar)", flush=True)

    q = queue.Queue(maxsize=BATCH * 8)
    END = object()

    def producer():
        try:
            from_tars(q)
            from_hf(q, ct)
        finally:
            q.put(END)

    threading.Thread(target=producer, daemon=True).start()

    out = open(WORK / "lid.csv", "w", encoding="utf-8", newline="")
    w = csv.writer(out)
    w.writerow(["split", "index", "audio", "top1", "p_top1", "top2", "p_top2"] + [f"p_{l}" for l in LANGS])
    done, t0, top1s, bad = 0, time.time(), Counter(), 0

    def flush(batch):
        nonlocal done
        paths, wavs = zip(*batch)
        feats = fe(list(wavs), sampling_rate=16000, return_tensors="pt").input_features.to(dev, dtype)
        dec = torch.full((len(wavs), 1), sot, device=dev)
        with torch.no_grad():
            logits = model(input_features=feats, decoder_input_ids=dec).logits[:, -1, :]
        p = torch.softmax(logits[:, ids].float(), dim=-1).cpu().numpy()
        for a, row in zip(paths, p):
            o = np.argsort(-row)
            split, i = imap.get(a, ("?", -1))
            top1s[names[o[0]]] += 1
            w.writerow([split, i, a, names[o[0]], f"{row[o[0]]:.4f}", names[o[1]], f"{row[o[1]]:.4f}"]
                       + [f"{row[names.index(l)]:.4f}" for l in LANGS])
        done += len(batch)
        if done % (BATCH * 100) < len(batch):
            print(f"  {done} đoạn, {time.time() - t0:.0f}s, top1 {top1s.most_common(6)}", flush=True)

    batch = []
    while (item := q.get()) is not END:
        a, b = item
        try:
            x, sr = sf.read(io.BytesIO(b), dtype="float32")
            assert sr == 16000
        except Exception as e:
            print("  LỖI đọc", a, e, flush=True)
            bad += 1
            continue
        batch.append((a, x.mean(1) if x.ndim > 1 else x))
        if len(batch) == BATCH:
            flush(batch)
            batch = []
    if batch:
        flush(batch)
    out.close()
    print(f"\nLID XONG: {done} đoạn, lỗi đọc {bad}, {time.time() - t0:.0f}s", flush=True)
    print("top1:", top1s.most_common(), flush=True)


if __name__ == "__main__":
    main()
