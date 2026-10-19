"""Kernel Kaggle **CPU** (tài khoản B, không tốn quota GPU) — thử trước khi chuyển việc giải mã/LM của GigaSpeech 2 lên
CPU Kaggle (người dùng 19/10: việc CPU chia cho B, C, D, trừ A).

Kiểm 3 điều, in thời gian từng bước:
1. Build KenLM (`lmplz`, `build_binary`) từ mã nguồn — LM rộng 92 triệu từ không vừa LM Python tự viết (`lm_decode.KNLM`
   đếm mọi n-gram trong dict).
2. `torchaudio.models.decoder.ctc_decoder` nhận LM KenLM (flashlight-text bản Linux có KenLM; bản Windows thì không).
3. Giải mã một emission giả bằng LM 3-gram dựng từ văn bản nhỏ.

    KAGGLE_CONFIG_DIR=<B> kaggle kernels push -p scripts/kaggle/lm_probe
"""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
T0 = time.time()
WORK = Path("/tmp/kenlm")
VENV = Path("/tmp/py312")


def run(cmd: str) -> str:
    t = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-5:]
    print(f"[{time.time() - T0:5.0f}s] rc={r.returncode} ({time.time() - t:.0f}s) $ {cmd[:120]}\n    " + "\n    ".join(tail),
          flush=True)
    return r.stdout


def main() -> None:
    run("nproc; free -g | head -2; df -h /tmp | tail -1; python --version; cmake --version | head -1")
    run("python -c 'import torch, torchaudio; print(torch.__version__, torchaudio.__version__)'")
    run("apt-get -qq update && apt-get -qq install -y libboost-program-options-dev libboost-system-dev "
        "libboost-thread-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev > /dev/null")
    run(f"git clone -q https://github.com/kpu/kenlm {WORK} && cd {WORK} && git log -1 --format=%H")
    run(f"mkdir -p {WORK}/build && cd {WORK}/build && cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null && make -j4 lmplz build_binary 2>&1 | tail -2")
    lmplz = WORK / "build/bin/lmplz"
    print("lmplz:", lmplz.exists(), flush=True)
    # flashlight-text không có wheel cp313 và build từ nguồn hỏng (lần thử 1) → Python 3.12 riêng qua uv, torch/torchaudio
    # 2.8 CPU như máy local (bộ giải mã đã kiểm 15–16/10), flashlight-text wheel cp312 (bản Linux kèm KenLM).
    run(f"{sys.executable} -m pip install -q uv && uv python install 3.12 && uv venv -q -p 3.12 {VENV}")
    run(f"uv pip install -q -p {VENV}/bin/python torch==2.8.0 torchaudio==2.8.0 "
        "--index-url https://download.pytorch.org/whl/cpu && "
        f"uv pip install -q -p {VENV}/bin/python flashlight-text==0.0.7 numpy sentencepiece")

    text = Path("/tmp/toy.txt")
    text.write_text("\n".join(["XIN CHÀO CÁC BẠN", "CHÀO CÁC BẠN NHÉ", "CÁC BẠN KHỎE KHÔNG", "XIN CHÀO"] * 50) + "\n",
                    encoding="utf-8")
    run(f"{lmplz} -o 3 --discount_fallback -S 20% -T /tmp < {text} > /tmp/toy.arpa")
    probe = r"""
import torch, sys
from torchaudio.models.decoder import ctc_decoder
tokens = ["-", "|", "X", "I", "N", "C", "H", "À", "O", "Á", "B", "Ạ"]
lex = open("/tmp/lex.txt", "w", encoding="utf-8")
for w in ["XIN", "CHÀO", "CÁC", "BẠN"]:
    lex.write(w + " " + " ".join(w) + " |\n")
lex.close()
dec = ctc_decoder(lexicon="/tmp/lex.txt", tokens=tokens, lm="/tmp/toy.arpa", nbest=1, beam_size=10, lm_weight=0.8,
                  word_score=1.0, blank_token="-", sil_token="|")
e = torch.full((1, 12, len(tokens)), -10.0)
for t, c in enumerate("XIN|CHÀO|CÁC"):
    if c in tokens: e[0, t, tokens.index(c)] = 0.0
print("DECODE OK:", dec(e)[0][0].words)
"""
    Path("/tmp/probe.py").write_text(probe, encoding="utf-8")
    run(f"{VENV}/bin/python /tmp/probe.py")
    print(f"XONG {time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
