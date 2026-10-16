"""Kernel Kaggle **GPU** (T4) — Whisper large-v3 chép 203 câu clean-test làm **bản tham khảo khi hiệu đính**.

Vì sao (2026-10-14, người dùng chốt): người hiệu đính không biết tên công ty/thuật ngữ tiếng Anh viết thế nào và
muốn đối chiếu với một model khác thay vì tự nghe một mình. Whisper **nằm ngoài đội hình so sánh** — tiền lệ: khi
audit nhãn Switchboard, Faria et al. 2022 (arXiv 2206.06192, mục 2.1) cho chuyên gia tham khảo bản chép độc lập
nhưng không cho xem hệ ASR đang được đánh giá. Output chỉ để cố vấn; người dùng tự quyết bằng tai. Lưu ý khi viết:
PhoWhisper (dự bị) fine-tune từ Whisper → có thể hưởng lợi nếu nhãn sửa theo Whisper.

Thử local 14/10 (turbo, 7 câu): Whisper viết đúng tên/tiếng Anh (P&G, tipping point, Gojek) nhưng bỏ lặp từ/từ
đệm, ra chữ số, đôi khi lặp ảo — nên bổ sung cho Zipformer chứ không thay được.

    kaggle kernels push -p scripts/kaggle/whisper_advisor -t 3600

Output /kaggle/working/whisper_advisor.json: {model, revision, decoding, samples: [{split, index, audio, text}]}.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ON_KAGGLE = Path("/kaggle/working").exists()
OUT = Path("/kaggle/working" if ON_KAGGLE else os.environ.get("ADV_OUT", "/tmp/whisper_advisor"))
AUDIO_DIR = Path(os.environ.get("ADV_AUDIO", "/tmp/clean_test_audio"))
MANIFEST_URL = "https://raw.githubusercontent.com/VuKhah/specialized-thesis/master/data/processed/clean_test_manifest.json"
HF_DATASET_ID = "thanhnew2001/VietSuperSpeech"
HF_REVISION = "cbf624ae9b30e1c2793a27e95b262115c69601f3"
MODEL = os.environ.get("ADV_MODEL", "openai/whisper-large-v3")  # local thử: turbo cho nhẹ
LIMIT = int(os.environ.get("ADV_LIMIT", "0"))  # 0 = cả 203 câu
# Beam 5 (khác num_beams=1 của zero-shot): đây là bản tham khảo, cần chất lượng chứ không đo tốc độ.
DECODING = {"language": "vi", "task": "transcribe", "num_beams": 5}


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soundfile"], check=True)
    import soundfile as sf
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import pipeline

    manifest = json.loads(urllib.request.urlopen(MANIFEST_URL, timeout=60).read().decode("utf-8"))
    samples = manifest["samples"][: LIMIT or None]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda rel: hf_hub_download(HF_DATASET_ID, rel, repo_type="dataset",
                                                          revision=HF_REVISION, local_dir=AUDIO_DIR),
                              [s["audio"] for s in samples]))
    print(f"Tải {len(paths)} audio ({time.time() - t0:.0f} s)", flush=True)

    revision = HfApi().model_info(MODEL).sha
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr = pipeline("automatic-speech-recognition", model=MODEL, revision=revision, device=device,
                   dtype=torch.float16 if device == "cuda" else torch.float32)

    out = []
    t0 = time.time()
    for k, (s, p) in enumerate(zip(samples, paths)):
        wave, sr = sf.read(p, dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        text = asr({"raw": wave, "sampling_rate": sr}, generate_kwargs=DECODING)["text"].strip()
        out.append({"split": s["split"], "index": s["index"], "audio": s["audio"], "text": text})
        if k % 20 == 0:
            print(f"{k + 1}/{len(samples)} ({time.time() - t0:.0f} s) {text[:80]}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "whisper_advisor.json").write_text(json.dumps(
        {"model": MODEL, "revision": revision, "decoding": DECODING, "samples": out},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Xong {len(out)} câu ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
