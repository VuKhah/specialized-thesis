"""Kernel Kaggle **GPU** (T4) — đo zero-shot (không fine-tune) 3 mô hình pre-train
trên 250 câu clean-test: đội hình #3-#5 trong docs/notes/survey_model_candidates.md
mục 3 (Parakeet-CTC-0.6B-vi, PhoWhisper-small, wav2vec2-base-vi-250h).

    kaggle kernels push -p scripts/kaggle/zero_shot -t 7200
    kaggle kernels status <username>/zero-shot-asr
    kaggle kernels output <username>/zero-shot-asr -p <thư mục>   (PYTHONUTF8=1)

Output trong /kaggle/working: `results.json` (WER/CER hai mức, tham số, RTF,
VRAM, thiết lập), `predictions.tsv` (ref/hyp từng câu từng mô hình). Ghi lại
sau MỖI mô hình — kernel chết giữa chừng (vd. lúc cài NeMo) vẫn còn kết quả
mô hình trước.

Chạy thử local (CPU, vài câu), từ gốc repo:
    ZS_LIMIT=4 ZS_DEVICE=cpu ZS_MODELS=wav2vec2,phowhisper ZS_OUT=<scratchpad> \
    ZS_AUDIO=<scratchpad>/audio PYTHONIOENCODING=utf-8 python scripts/kaggle/zero_shot/zero_shot.py

Thiết kế đo (ghi vào results.json, cần nêu khi viết khóa luận):
- Batch 1, từng câu; audio đọc sẵn vào RAM nên thời gian KHÔNG gồm đọc đĩa,
  nhưng GỒM trích đặc trưng + mô hình + giải mã ra chuỗi. 3 câu đầu chạy
  warm-up trước rồi mới đo (cả 3 câu vẫn được đo lại trong vòng chính).
- GPU: trọng số fp32 + `torch.autocast(fp16)`; front-end (log-mel/STFT) giữ
  fp32 — cùng chính sách với AMP của pipeline Mamba/Conformer (benchmark.py).
- Giải mã greedy, không LM (Parakeet và wav2vec2 có kèm LM 4-gram trên HF —
  cố ý không dùng để cùng điều kiện với CTC greedy của Mamba/Conformer;
  PhoWhisper num_beams=1).
- Mô hình chạy lần lượt trong cùng tiến trình, mỗi mô hình bọc try/except: lỗi
  cài/tải của một mô hình không làm sập các mô hình khác. Parakeet chạy cuối vì
  `pip install nemo_toolkit` có thể kéo theo đổi thư viện.
- Chuẩn hóa văn bản: `src/evaluation/text_normalize.py`, CHÉP NGUYÊN VĂN vào
  `TEXT_NORMALIZE_SRC` dưới đây (kernel script chỉ upload 1 file). Chạy local
  từ gốc repo thì script tự so khớp với file trong repo và dừng nếu lệch.
"""

import csv
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ON_KAGGLE = Path("/kaggle/working").exists()
_LOCAL_TMP = Path(tempfile.gettempdir()) / "zero_shot"

# Cờ chạy thử: biến môi trường, vì kernel script trên Kaggle không nhận tham số.
LIMIT = int(os.environ.get("ZS_LIMIT", "0"))  # 0 = cả 250 câu
DEVICE_OVERRIDE = os.environ.get("ZS_DEVICE", "")
MODELS = [m for m in os.environ.get("ZS_MODELS", "wav2vec2,phowhisper,parakeet").split(",") if m]
OUT_DIR = Path(os.environ.get("ZS_OUT", "/kaggle/working" if ON_KAGGLE else _LOCAL_TMP / "out"))
AUDIO_DIR = Path(os.environ.get("ZS_AUDIO", "/tmp/clean_test_audio" if ON_KAGGLE else _LOCAL_TMP / "audio"))
N_WARMUP = 3

REPO_RAW = "https://raw.githubusercontent.com/VuKhah/specialized-thesis/master"
MANIFEST_REL = "data/processed/clean_test_manifest.json"
NORMALIZE_REL = "src/evaluation/text_normalize.py"
HF_DATASET_ID = "thanhnew2001/VietSuperSpeech"
HF_REVISION = "cbf624ae9b30e1c2793a27e95b262115c69601f3"
SAMPLE_RATE = 16000

# Pin revision mô hình (sha HF đọc 2026-10-02) để chạy lại ra cùng số.
MODEL_SPECS = {
    "wav2vec2": {
        "repo": "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        "revision": "69e9000591623e5a4fc2f502407860bcdc0de0b2",
        "license": "CC BY-NC 4.0 (theo model card; repo lại kèm file CC-BY-NC-SA-4.0.txt)",
    },
    "phowhisper": {
        "repo": "vinai/PhoWhisper-small",
        "revision": "a86b604c346caf7148c37512eafe783a16420adb",
        "license": "BSD-3-Clause",
    },
    "parakeet": {
        "repo": "nvidia/parakeet-ctc-0.6b-Vietnamese",
        "revision": "b0493142b49458810324e3db8be9e8e07b4ebc17",
        "nemo_file": "parakeet-ctc-0.6b-vi.nemo",
        "license": "NVIDIA Open Model License",
    },
}

# ---- Chép nguyên văn src/evaluation/text_normalize.py (đừng sửa ở đây) ----
TEXT_NORMALIZE_SRC = r'''"""Chuẩn hóa văn bản dùng chung trước khi tính WER/CER — cho MỌI mô hình
(Mamba/Conformer của ta, Parakeet, PhoWhisper, wav2vec2, ...).

Vì sao cần: nhãn VietSuperSpeech viết HOA, không dấu câu, còn Parakeet (train
với nhãn có dấu câu + hoa/thường do Qwen3 sinh) và PhoWhisper ra chữ thường/hoa
lẫn dấu câu. Không chuẩn hóa chung thì WER đo khác biệt định dạng chứ không đo
nhận dạng. Chỉ được có MỘT hàm này: kernel `scripts/kaggle/zero_shot/zero_shot.py`
chép nguyên văn file này (có kiểm tra khớp khi chạy local) — sửa ở đây thì
chép lại sang đó.

Quyết định (đo trên `data/processed/train_transcripts.txt`, 60.656 nhãn, 2026-10-02):
- Nhãn đã NFC 100%, không có chữ số nào (số luôn viết bằng chữ: "BA ĐẾN NĂM
  TRĂM", "MỘT TRĂM PHẦN TRĂM"), ký tự ngoài chữ cái chỉ có `-` (29 lần, vd.
  CHECK-IN, WIN-WIN, X-QUANG, và `-` đứng riêng) và `<` (5 lần, rác).
- Gạch nối → khoảng trắng: nhãn tự mâu thuẫn ("CHECK-IN" / "CHECK IN" /
  "CHECKIN"), tách ra thì hai cách viết đầu khớp nhau.
- Dấu nháy đơn bị xóa (không thành khoảng trắng) để "don't" → "dont" — một từ,
  giống cách Zipformer viết tiếng Anh không dấu nháy.
- **Chữ số giữ nguyên, không đọc thành chữ.** Đọc số tiếng Việt có nhiều cách
  hợp lệ ("hai nghìn không trăm hai mươi tư" / "hai không hai bốn", "nghìn" /
  "ngàn", "mốt" / "một", "linh" / "lẻ"), bộ đọc số nào cũng tự sinh lỗi riêng
  và thiên vị cách đọc của nó. Vì nhãn không bao giờ có chữ số, mỗi chữ số trong
  hyp là một lỗi thật về *định dạng* — kernel zero-shot đếm số câu hyp có chữ số
  (`has_digit`) để lượng hóa phần WER này, thay vì che đi.
"""

import re
import unicodedata

_APOSTROPHES = "'’ʼ‘`"
_WHITESPACE = re.compile(r"\s+")
_DIGIT = re.compile(r"\d")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    for ch in _APOSTROPHES:
        text = text.replace(ch, "")
    # Mọi dấu câu (P*) và ký hiệu (S*: %, <, ⁇ của sentencepiece khi gặp unk, ...)
    # thành khoảng trắng — không xóa hẳn, kẻo dính hai từ "a,b" → "ab".
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return _WHITESPACE.sub(" ", text).strip()


def has_digit(text: str) -> bool:
    return bool(_DIGIT.search(text))


def diacritic_word_ratio(text: str) -> float:
    """Tỉ lệ từ mang dấu tiếng Việt (NFD rồi tìm ký tự combining; `đ` tính là có dấu).

    Heuristic CHƯA KIỂM CHỨNG (docs/notes/survey_model_candidates.md mục 4): đo
    ngôn ngữ của *nhãn*, không phải của audio."""
    words = normalize_text(text).split()
    if not words:
        return 0.0
    marked = sum(
        1 for w in words if "đ" in w or any(unicodedata.combining(ch) for ch in unicodedata.normalize("NFD", w))
    )
    return marked / len(words)


VIETNAMESE_LABEL_MIN_RATIO = 0.2


def is_vietnamese_label(text: str) -> bool:
    return diacritic_word_ratio(text) >= VIETNAMESE_LABEL_MIN_RATIO
'''
# ---- hết phần chép ----

_tn: dict = {}
exec(compile(TEXT_NORMALIZE_SRC, "text_normalize.py", "exec"), _tn)
normalize_text = _tn["normalize_text"]
has_digit = _tn["has_digit"]
is_vietnamese_label = _tn["is_vietnamese_label"]
VI_THRESHOLD = _tn["VIETNAMESE_LABEL_MIN_RATIO"]


def run(cmd, check=True):
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"lệnh trả về {r.returncode}: {' '.join(map(str, cmd))}")
    return r


def fetch_text(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as f:
            return f.read().decode("utf-8")
    except Exception as e:  # noqa: BLE001 — chỉ dùng để so khớp/tải manifest, lỗi thì báo
        print(f"Không tải được {url}: {e}", flush=True)
        return None


def check_normalize_copy() -> str:
    """Bản chép phải trùng file trong repo — WER của mô hình ngoài chỉ so được
    với Mamba/Conformer khi cùng một hàm chuẩn hóa."""
    local = Path(NORMALIZE_REL)
    if local.exists():
        if local.read_text(encoding="utf-8") != TEXT_NORMALIZE_SRC:
            sys.exit(f"LỖI: TEXT_NORMALIZE_SRC lệch {NORMALIZE_REL} — chép lại nguyên văn rồi chạy lại.")
        return "khớp file local"
    remote = fetch_text(f"{REPO_RAW}/{NORMALIZE_REL}")
    if remote is None:
        return "chưa có trên GitHub — dùng bản chép (chưa so khớp được)"
    if remote != TEXT_NORMALIZE_SRC:
        print("CẢNH BÁO: bản chép LỆCH text_normalize.py trên GitHub — kết quả có thể không so được.", flush=True)
        return "LỆCH bản GitHub"
    return "khớp bản GitHub"


def load_manifest() -> dict:
    local = Path(MANIFEST_REL)
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))
    text = fetch_text(f"{REPO_RAW}/{MANIFEST_REL}")
    if text is None:
        sys.exit("LỖI: không tải được clean_test_manifest.json từ GitHub")
    return json.loads(text)


def prepare_samples(manifest: dict) -> list[dict]:
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    samples = manifest["samples"][: LIMIT or None]
    # Manifest chỉ lưu index trong split → tra đường dẫn audio ở đúng revision đã pin;
    # so luôn text để chắc index không lệch (nếu lệch thì ref sai câu).
    ds = load_dataset(HF_DATASET_ID, split=manifest["split"], revision=HF_REVISION)
    rows = [ds[s["index"]] for s in samples]
    mismatch = [s["index"] for s, r in zip(samples, rows) if r["text"] != s["pseudo_label"]]
    if mismatch:
        sys.exit(f"LỖI: pseudo_label lệch text của HF ở index {mismatch[:10]} — index manifest không khớp revision")

    def one(rel):
        return hf_hub_download(HF_DATASET_ID, rel, repo_type="dataset", revision=HF_REVISION, local_dir=AUDIO_DIR)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(one, [r["audio"] for r in rows]))
    print(f"Đã tải {len(paths)} file audio ({time.time() - t0:.0f} s)", flush=True)

    out = []
    for s, path in zip(samples, paths):
        wave, sr = sf.read(path, dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        if sr != SAMPLE_RATE:
            # Pipeline của ta giả định 16 kHz không kiểm tra (ARCHITECTURE.md 8c) — ở đây dừng hẳn.
            sys.exit(f"LỖI: {path} có sample rate {sr}, không phải {SAMPLE_RATE}")
        corrected = s["corrected_text"].strip()
        ref_raw = corrected or s["pseudo_label"]
        out.append({
            "index": s["index"],
            "path": path,
            "wave": np.ascontiguousarray(wave),
            "duration_s": len(wave) / SAMPLE_RATE,
            "ref_source": "corrected_text" if corrected else "pseudo_label",
            "ref": normalize_text(ref_raw),
            "vi_label": is_vietnamese_label(ref_raw),
        })
    return out


# ---------------------------------------------------------------------------
# Mỗi loader trả về (model, transcribe(wave)->str, thông tin thiết lập).


def load_wav2vec2(spec, device, amp):
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(spec["repo"], revision=spec["revision"])
    model = Wav2Vec2ForCTC.from_pretrained(spec["repo"], revision=spec["revision"]).to(device).eval()

    def transcribe(wave):
        x = processor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x).logits
        return processor.batch_decode(logits.argmax(dim=-1))[0]

    # Card khuyên audio < 10 s; clean-test dài 10-15 s → vẫn đưa nguyên câu, ghi lại.
    return model, transcribe, {"decoding": "CTC greedy (argmax), không LM",
                               "note": "model card khuyên audio < 10 s; clean-test 10-15 s, đưa nguyên câu"}


def load_phowhisper(spec, device, amp):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(spec["repo"], revision=spec["revision"])
    model = WhisperForConditionalGeneration.from_pretrained(spec["repo"], revision=spec["revision"]).to(device).eval()

    def transcribe(wave):
        feats = processor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            ids = model.generate(feats, language="vi", task="transcribe", num_beams=1, do_sample=False,
                                 max_new_tokens=200)
        return processor.batch_decode(ids, skip_special_tokens=True)[0]

    return model, transcribe, {"decoding": "greedy (num_beams=1), language=vi, task=transcribe, max_new_tokens=200"}


def install_nemo() -> str:
    import torch

    # Ghim torch đang có: nemo kéo torch mới thì lệch CUDA của Kaggle và làm hỏng
    # import torch đã nạp trong tiến trình này.
    constraints = Path(tempfile.gettempdir()) / "nemo_constraints.txt"
    constraints.write_text(f"torch=={torch.__version__}\n", encoding="utf-8")
    # Card ghi runtime "NeMo 2.6"; PyPI đã có 3.0 (major mới, 2026-10 chưa thử) →
    # thử bản card trước, hỏng mới lấy bản mới nhất. Phiên bản thật ghi vào results.json.
    for req in ["nemo_toolkit[asr]==2.6.2", "nemo_toolkit[asr]"]:
        r = run([sys.executable, "-m", "pip", "install", "-q", req, "-c", str(constraints)], check=False)
        if r.returncode == 0:
            break
    else:
        raise RuntimeError("cài nemo_toolkit[asr] thất bại (cả 2.6.2 lẫn bản mới nhất)")
    r = run([sys.executable, "-c", "import nemo; print(nemo.__version__)"])
    return r.stdout.strip()


def load_parakeet(spec, device, amp):
    import torch
    from huggingface_hub import hf_hub_download

    nemo_version = install_nemo()
    import nemo.collections.asr as nemo_asr

    # Card gợi ý from_pretrained("nvidia/parakeet-ctc-0.6b-vi") nhưng tên đó khác
    # repo HF; tải thẳng file .nemo (repo không gated, tải ẩn danh được) cho chắc.
    path = hf_hub_download(spec["repo"], spec["nemo_file"], revision=spec["revision"])
    model = nemo_asr.models.ASRModel.restore_from(path, map_location=torch.device(device)).eval()

    def decode(log_probs, enc_len):
        hyps = model.decoding.ctc_decoder_predictions_tensor(log_probs, decoder_lengths=enc_len,
                                                             return_hypotheses=False)
        if isinstance(hyps, tuple):  # NeMo cũ trả (best, all)
            hyps = hyps[0]
        h = hyps[0]
        return h.text if hasattr(h, "text") else str(h)

    def transcribe(wave):
        # Gọi forward trực tiếp thay vì model.transcribe(): transcribe() dựng
        # DataLoader + manifest tạm mỗi lần gọi, overhead đó sẽ lọt vào RTF.
        sig = torch.from_numpy(wave)[None].to(device)
        sig_len = torch.tensor([sig.shape[1]], device=device)
        feats, feats_len = model.preprocessor(input_signal=sig, length=sig_len)  # fp32
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            log_probs, enc_len, _ = model(processed_signal=feats, processed_signal_length=feats_len)
        return decode(log_probs.float(), enc_len)

    def reference_transcribe(audio_path):
        out = model.transcribe([audio_path], batch_size=1, verbose=False)
        if isinstance(out, tuple):
            out = out[0]
        h = out[0]
        return h.text if hasattr(h, "text") else str(h)

    return model, transcribe, {"decoding": "CTC greedy (decoding cfg mặc định của .nemo), không LM",
                               "nemo_version": nemo_version, "reference_transcribe": reference_transcribe}


LOADERS = {"wav2vec2": load_wav2vec2, "phowhisper": load_phowhisper, "parakeet": load_parakeet}


def score(refs: list[str], hyps: list[str]) -> dict:
    import jiwer

    if not refs:
        return {"n": 0}
    out = jiwer.process_words(refs, hyps)
    return {"n": len(refs), "wer": out.wer, "cer": jiwer.cer(refs, hyps), "substitutions": out.substitutions,
            "deletions": out.deletions, "insertions": out.insertions, "hits": out.hits}


def run_model(name: str, samples: list[dict], device: str) -> dict:
    import torch

    spec = MODEL_SPECS[name]
    amp = device == "cuda"
    cuda = device == "cuda"
    result = {"model": spec["repo"], "revision": spec["revision"], "license": spec["license"],
              "precision": "fp32 weights + autocast fp16 (front-end fp32)" if amp else "fp32",
              "batch_size": 1, "n_warmup": N_WARMUP}
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, transcribe, info = LOADERS[name](spec, device, amp)
    result["load_time_s"] = round(time.time() - t0, 1)
    reference_transcribe = info.pop("reference_transcribe", None)
    result.update(info)
    result["n_params"] = sum(p.numel() for p in model.parameters())
    if cuda:
        result["vram_weights_gib"] = round(torch.cuda.memory_allocated() / 2**30, 3)
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for s in samples[:N_WARMUP]:
            transcribe(s["wave"])
        raw_hyps, times = [], []
        for s in samples:
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            raw_hyps.append(transcribe(s["wave"]))
            if cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t)

    if cuda:
        result["vram_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        result["vram_peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    if reference_transcribe is not None:
        # Đường forward tự viết phải ra giống model.transcribe() chính thức.
        checks = []
        for s, h in list(zip(samples, raw_hyps))[:3]:
            official = reference_transcribe(s["path"])
            checks.append({"index": s["index"], "same_after_normalize": normalize_text(official) == normalize_text(h),
                           "official": official, "ours": h})
        result["forward_vs_transcribe_check"] = checks

    total_audio = sum(s["duration_s"] for s in samples)
    hyps = [normalize_text(h) for h in raw_hyps]
    refs = [s["ref"] for s in samples]
    vi = [i for i, s in enumerate(samples) if s["vi_label"]]
    result.update({
        "total_infer_s": round(sum(times), 3),
        "total_audio_s": round(total_audio, 1),
        "rtf": sum(times) / total_audio,
        "mean_latency_s": sum(times) / len(times),
        "wer_all": score(refs, hyps),
        "wer_vi_label": score([refs[i] for i in vi], [hyps[i] for i in vi]),
        "n_hyp_empty": sum(1 for h in hyps if not h),
        "n_hyp_with_digits": sum(1 for h in hyps if has_digit(h)),
        "raw_hyps": raw_hyps,
        "hyps": hyps,
    })
    del model, transcribe
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return result


def write_outputs(meta: dict, samples: list[dict], results: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slim = {name: {k: v for k, v in r.items() if k not in ("raw_hyps", "hyps")} for name, r in results.items()}
    (OUT_DIR / "results.json").write_text(json.dumps({"meta": meta, "models": slim}, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    ok = [n for n, r in results.items() if "hyps" in r]
    clean = lambda t: t.replace("\t", " ").replace("\n", " ")  # noqa: E731
    with open(OUT_DIR / "predictions.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(["index", "duration_s", "ref_source", "vi_label", "ref"]
                   + [f"hyp_{n}" for n in ok] + [f"raw_{n}" for n in ok])
        for i, s in enumerate(samples):
            w.writerow([s["index"], f"{s['duration_s']:.2f}", s["ref_source"], int(s["vi_label"]), s["ref"]]
                       + [clean(results[n]["hyps"][i]) for n in ok] + [clean(results[n]["raw_hyps"][i]) for n in ok])


def main():
    if ON_KAGGLE:
        run(["nvidia-smi"], check=False)
        run([sys.executable, "-m", "pip", "install", "-q", "jiwer", "soundfile", "datasets"])
    import torch

    device = DEVICE_OVERRIDE or ("cuda" if torch.cuda.is_available() else "cpu")
    normalize_status = check_normalize_copy()
    manifest = load_manifest()
    samples = prepare_samples(manifest)
    n_corrected = sum(1 for s in samples if s["ref_source"] == "corrected_text")
    n_vi = sum(1 for s in samples if s["vi_label"])
    print(f"Reference: {n_corrected}/{len(samples)} câu là corrected_text, "
          f"{len(samples) - n_corrected} là pseudo_label", flush=True)
    print(f"Tập con nhãn tiếng Việt: {n_vi}/{len(samples)} câu (tỉ lệ từ có dấu ≥ {VI_THRESHOLD})", flush=True)

    meta = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__,
        "dataset": HF_DATASET_ID, "dataset_revision": HF_REVISION, "split": manifest["split"],
        "n": len(samples), "limit": LIMIT or None,
        "n_corrected_text": n_corrected, "n_pseudo_label": len(samples) - n_corrected,
        "n_vi_label": n_vi,
        "normalization": "src/evaluation/text_normalize.py: NFC, chữ thường, bỏ nháy đơn, dấu câu/ký hiệu → "
                         "khoảng trắng, gộp khoảng trắng; chữ số GIỮ NGUYÊN (nhãn không có chữ số → mỗi chữ số "
                         "trong hyp là lỗi, xem n_hyp_with_digits)",
        "normalize_copy_check": normalize_status,
        "vi_label_subset": f"câu có tỉ lệ từ mang dấu tiếng Việt ≥ {VI_THRESHOLD} (NFD, đếm ký tự combining, "
                           "`đ` tính là có dấu) — HEURISTIC CHƯA KIỂM CHỨNG, đo ngôn ngữ của nhãn chứ không "
                           "phải audio (docs/notes/survey_model_candidates.md mục 4)",
        "timing": f"batch 1, audio đã nạp sẵn RAM (không tính đọc đĩa), gồm trích đặc trưng + mô hình + giải mã; "
                  f"{N_WARMUP} câu warm-up không tính; RTF = tổng thời gian / tổng thời lượng audio",
        "vram": "torch.cuda.max_memory_allocated trong lúc suy luận (gồm trọng số); không gồm overhead CUDA context",
    }

    results = {}
    for name in MODELS:
        print(f"\n{'=' * 70}\n{name}: {MODEL_SPECS[name]['repo']}\n{'=' * 70}", flush=True)
        try:
            r = run_model(name, samples, device)
            a, v = r["wer_all"], r["wer_vi_label"]
            print(f"{name}: WER={a['wer']:.4f} CER={a['cer']:.4f} | nhãn Việt ({v['n']}): "
                  f"WER={v.get('wer', float('nan')):.4f} | params={r['n_params']:,} | RTF={r['rtf']:.4f} | "
                  f"hyp rỗng={r['n_hyp_empty']} có chữ số={r['n_hyp_with_digits']}", flush=True)
        except BaseException as e:  # noqa: BLE001 — kể cả SystemExit từ thư viện; lỗi 1 mô hình không làm sập kernel
            if isinstance(e, KeyboardInterrupt):
                raise
            tb = traceback.format_exc()
            print(f"LỖI {name}:\n{tb}", flush=True)
            r = {"model": MODEL_SPECS[name]["repo"], "error": repr(e), "traceback": tb[-4000:]}
            gc.collect()
        results[name] = r
        write_outputs(meta, samples, results)

    summary = {n: ({"wer": r["wer_all"]["wer"], "wer_vi_label": r["wer_vi_label"].get("wer"), "cer": r["wer_all"]["cer"],
                    "rtf": r["rtf"], "n_params": r["n_params"], "vram_peak_gib": r.get("vram_peak_allocated_gib")}
                   if "error" not in r else {"error": r["error"]}) for n, r in results.items()}
    print("\n" + "=" * 70 + "\nKẾT QUẢ\n" + "=" * 70)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nĐã ghi {OUT_DIR / 'results.json'} và {OUT_DIR / 'predictions.tsv'}\nZERO-SHOT XONG", flush=True)


if __name__ == "__main__":
    main()
