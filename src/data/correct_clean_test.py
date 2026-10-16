"""Công cụ nghe & hiệu đính clean-test (thêm 2026-10-08): ghi cột `corrected_text` / `notes` của
`data/processed/clean_test_manifest.json`, mỗi lần bấm Lưu là ghi file ngay.

    python -m src.data.correct_clean_test              # http://127.0.0.1:7861 (demo dùng 7860)

Thay mục 5 của `notebooks/02_dataset_eda.ipynb`: notebook được track trên repo public, nghe audio ở đó dễ để
lại output nhúng audio; sửa JSON bằng tay dễ hỏng file. Lúc khởi động chép một bản sao lưu cạnh manifest
(`data/processed/` bị gitignore trừ chính manifest → bản sao không lên git).

`corrected_text` rỗng = chưa hiệu đính (`eval_clean_test` quay về `pseudo_label`). WER chuẩn hoá bằng
`normalize_text` (chữ thường, bỏ dấu câu) nên hoa/thường và dấu câu không quan trọng; **chữ số thì giữ
nguyên** → viết số bằng chữ như nhãn train.

Tô chữ theo audio (thêm 2026-10-14): forced alignment CTC **chính bản hiệu đính đang gõ** lên log-prob của
Conformer v1 (trung bình epoch 20–29, CPU, khung 10 ms vì v1 không subsampling). Model chỉ cấp mốc thời gian,
không hiện chữ nào của nó → không lộ output mô hình đang so sánh (hướng dẫn mục 8). Từ không khớp audio vẫn bị
"ép" vào một khoảng — dài/ngắn bất thường là gợi ý chỗ cần nghe lại, không phải bằng chứng. Thiếu checkpoint
thì công cụ vẫn chạy, chỉ không tô.

Cố vấn Whisper (thêm 2026-10-14, người dùng chốt): hiện bản chép của Whisper large-v3 (kernel
`scripts/kaggle/whisper_advisor`, output chép về `data/processed/whisper_advisor.json`) — model **ngoài đội hình**,
nên được xem chữ của nó (khác Conformer ở trên). Mỗi chữ của bản hiệu đính được tô theo việc Zipformer/Whisper có
cùng chữ đó không; chữ không model nào có là chỗ nên nghe lại. Thử 14/10: Whisper mạnh ở tên riêng/tiếng Anh, nhưng
bỏ lặp từ/từ đệm và ra chữ số → chữ số và từ đệm hay bị tô đỏ oan.

Output mô hình của đề tài (thêm 2026-10-14 khuya, **người dùng chốt dù đã được báo rủi ro**): hiện chữ của 5 mô hình
đang so sánh (`final_compare_2026-10-08/predictions.tsv`, checkpoint ep28 + Parakeet/PhoWhisper zero-shot) trong một
khung gập. Sửa nhãn theo chữ của một mô hình thì mô hình đó lợi khi chấm → mỗi lần lưu ghi `corrected_at` để lúc
chấm tách được câu hiệu đính trước/sau mốc `OURS_VISIBLE_FROM`.
"""

import argparse
import csv
import difflib
import html
import json
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import yaml

from src.data.vietsuperspeech_dataset import AUDIO_CACHE_DIR
from src.evaluation.text_normalize import normalize_text
from src.tokenizer.bpe_tokenizer import BPETokenizer

MANIFEST = Path("data/processed/clean_test_manifest.json")
ALIGN_CONFIG = Path("configs/model_conformer.yaml")
ALIGN_CHECKPOINT = Path("checkpoints/avg_v1/conformer_avg20-29.pt")
ADVISOR = Path("data/processed/whisper_advisor.json")
OURS = Path("reports/results/final_compare_2026-10-08/predictions.tsv")
OURS_VISIBLE_FROM = "2026-10-14T23:17"  # mốc bắt đầu hiện output mô hình đang so sánh (giờ local)
OURS_NAMES = {"mamba": "Mamba B1", "conextbimamba": "ConExtBiMamba", "conformer": "Conformer-12M",
              "parakeet_fp32": "Parakeet-CTC-0.6B (zero-shot)", "phowhisper": "PhoWhisper-small (zero-shot)"}
FRAME_S = 0.01  # hop 160 mẫu @ 16 kHz, v1 không subsampling
_DIGIT = re.compile(r"\d")

CSS = """
.diff del {background:#fdd; color:#900; text-decoration:line-through; padding:0 2px}
.diff ins {background:#dfd; color:#060; text-decoration:none; padding:0 2px}
.diff {font-size:1.05em; line-height:1.8}
#ct-words {font-size:1.25em; line-height:2; padding:6px 2px}
#ct-words span[data-s] {cursor:pointer; padding:1px 3px; border-radius:4px}
#ct-words span.past {opacity:.55}
#ct-words span.cur {background:#ffd54f; color:#000; opacity:1}
#ct-audio {width:100%}
.gradio-container {max-width:100% !important}
#ct-header {font-size:.95em}
#ct-header h3 {margin:0}
#ct-left {position:sticky; top:6px; align-self:flex-start}
#ct-words-box {max-height:32vh; overflow-y:auto}
#ct-right .tabitem {max-height:78vh; overflow-y:auto}
.agree {font-size:1.1em; line-height:1.9}
.agree .one {background:#ffe0b2; color:#000; padding:0 2px}
.agree .none {background:#ef9a9a; color:#000; padding:0 2px}
"""

# gr.HTML không chạy <script> chèn qua innerHTML → cài một vòng requestAnimationFrame toàn cục qua `head`.
# Tìm phần tử theo id mỗi khung vì Gradio thay DOM mỗi lần đổi câu / sửa chữ.
HEAD_JS = """
<script>
(() => {
  let lastKey = "";
  function update() {
    const a = document.getElementById("ct-audio");
    const spans = document.querySelectorAll("#ct-words span[data-s]");
    if (a && spans.length) {
      const t = a.currentTime;
      let cur = -1;
      for (let k = 0; k < spans.length; k++) { if (+spans[k].dataset.s <= t) cur = k; else break; }
      const key = cur + "|" + spans.length + "|" + spans[0].dataset.s + "|" + spans[spans.length - 1].dataset.s;
      if (key !== lastKey) {
        lastKey = key;
        spans.forEach((sp, k) => { sp.classList.toggle("past", k < cur); sp.classList.toggle("cur", k === cur); });
      }
    }
  }
  // rAF cho mượt; rAF dừng khi tab ẩn nên thêm timeupdate/seeked (không nổi bọt → bắt ở pha capture).
  (function tick() { update(); requestAnimationFrame(tick); })();
  document.addEventListener("timeupdate", update, true);
  document.addEventListener("seeked", update, true);
  // Ctrl+Enter trong ô hiệu đính = bấm Lưu & tiếp (đỡ phải rời bàn phím).
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter" && e.target.closest && e.target.closest("#ct-edited")) {
      e.preventDefault();
      const b = document.querySelector("#ct-save");
      if (b) b.click();
    }
  });
  document.addEventListener("click", (e) => {
    const sp = e.target.closest && e.target.closest("#ct-words span[data-s]");
    const a = document.getElementById("ct-audio");
    if (!sp || !a) return;
    a.currentTime = +sp.dataset.s;
    a.play();
  });
})();
</script>
"""

GUIDE = """
**Cách hiệu đính** (chi tiết: `docs/notes/clean_test_correction_guide.md`)
1. Nghe hết câu, sửa ô *Bản hiệu đính* cho **đúng từng từ được nói** — kể cả từ đệm (thì, là, á, à, ừ) nếu nói rõ.
2. Không cần dấu câu, hoa/thường tuỳ ý. **Số viết bằng chữ** (*hai mươi bảy*, không *27*).
3. Tiếng Anh viết đúng chính tả (*marketing, team*); viết tắt liền (*nft*).
4. Từ bị cắt nửa ở đầu/cuối đoạn: bỏ. Đoạn không nghe rõ: ghi phỏng đoán tốt nhất + ghi chú `khó nghe`.
5. Câu đã đúng: bấm **Lưu & tiếp** luôn (vẫn tính là đã hiệu đính).
"""


def load() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def save(manifest: dict) -> None:
    # Cùng định dạng với make_heldout.py (indent 2, không xuống dòng cuối) → git diff chỉ có dòng đã sửa.
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, MANIFEST)


def diff_html(reference: str, edited: str, name: str = "nhãn giả") -> str:
    a, b = normalize_text(reference).split(), normalize_text(edited).split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(html.escape(" ".join(a[i1:i2])))
            continue
        if i2 > i1:
            out.append(f"<del>{html.escape(' '.join(a[i1:i2]))}</del>")
        if j2 > j1:
            out.append(f"<ins>{html.escape(' '.join(b[j1:j2]))}</ins>")
    n_changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in
                    difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op != "equal")
    head = f"<b>So với {name}:</b> {n_changed} từ khác" if n_changed else f"<b>Giống hệt {name}</b>"
    return f"<div class='diff'>{head}<br>{' '.join(out)}</div>"


def _matched(reference: list[str], edited: list[str]) -> set[int]:
    sm = difflib.SequenceMatcher(a=reference, b=edited, autojunk=False)
    return {j for op, _, _, j1, j2 in sm.get_opcodes() if op == "equal" for j in range(j1, j2)}


def agreement_html(pseudo: str, whisper: str, edited: str) -> str:
    """Tô từng chữ bản hiệu đính: trắng = cả Zipformer lẫn Whisper có, cam = chỉ một bên, đỏ = không bên nào."""
    b = normalize_text(edited).split()
    z, w = normalize_text(pseudo).split(), normalize_text(whisper).split()
    # Whisper large-v3 bịa trọn câu ở 6/203 câu ("hãy subscribe cho kênh ghiền mì gõ…", 14/10) → ngưỡng chung theo
    # độ trùng với nhãn giả thay vì so chuỗi cố định; câu thật trùng ≥ ~0,6 (đo 14/10).
    if len(_matched(z, w)) < 0.3 * max(len(w), 1):
        return ("<div class='agree'><b>⚠️ Whisper gần như không trùng nhãn giả — có thể bịa (ảo giác), "
                "bỏ qua cố vấn ở câu này.</b></div>")
    in_z = _matched(z, b)
    in_w = _matched(w, b)
    out, n_none = [], 0
    for j, w in enumerate(b):
        k = (j in in_z) + (j in in_w)
        n_none += k == 0
        cls = {2: "", 1: " class='one'", 0: " class='none'"}[k]
        out.append(f"<span{cls}>{html.escape(w)}</span>")
    head = (f"<b>Đối chiếu hai model ngoài đội hình:</b> {n_none} chữ không model nào có (đỏ) · "
            "cam = chỉ một model có")
    return f"<div class='agree'>{head}<br>{' '.join(out)}</div>"


def ctc_viterbi(log_probs: np.ndarray, tokens: list[int], blank: int = 0) -> list[int] | None:
    """Đường CTC tốt nhất cho chuỗi token cho trước; mỗi khung trả chỉ số trong `tokens`, -1 = blank.
    Tự viết thay `torchaudio.functional.forced_align` vì hàm đó bị gỡ từ torchaudio 2.9."""
    n_frames = log_probs.shape[0]
    states = np.array([blank] + [x for tok in tokens for x in (tok, blank)])
    n_states = len(states)
    # Nhảy s-2 → s chỉ khi s là token và khác token ở s-2 (hai token giống nhau liền nhau phải có blank chen giữa).
    skip = np.zeros(n_states, dtype=bool)
    skip[3::2] = states[3::2] != states[1:-2:2]
    if n_frames < len(tokens) + int((~skip[3::2]).sum()):
        return None
    emit = log_probs[:, states]
    score = np.full(n_states, -np.inf)
    score[:2] = emit[0, :2]
    back = np.zeros((n_frames, n_states), dtype=np.int8)  # 0: ở yên, 1: từ s-1, 2: từ s-2
    neg = np.full(2, -np.inf)
    for t in range(1, n_frames):
        cand = np.stack([score, np.r_[neg[:1], score[:-1]], np.where(skip, np.r_[neg, score[:-2]], -np.inf)])
        back[t] = cand.argmax(0)
        score = cand.max(0) + emit[t]
    s = n_states - 1 if score[-1] >= score[-2] else n_states - 2
    path = np.empty(n_frames, dtype=np.int64)
    for t in range(n_frames - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    return [(p - 1) // 2 if p % 2 else -1 for p in path.tolist()]


class WordAligner:
    """Log-prob của Conformer v1 tính một lần mỗi câu (giữ trong RAM), rồi align lại mỗi khi chữ đổi —
    Viterbi trên ~1.500 khung chỉ vài chục ms nên chạy theo từng phím gõ được."""

    def __init__(self):
        from src.training.train import build_model  # import nặng (mamba, tensorboard) — chỉ khi có checkpoint
        cfg = yaml.safe_load(ALIGN_CONFIG.read_text(encoding="utf-8"))
        self.tokenizer = BPETokenizer("configs/tokenizer.model")
        self.model = build_model(cfg, self.tokenizer.vocab_size)
        self.model.load_state_dict(torch.load(ALIGN_CHECKPOINT, map_location="cpu")["model"])
        self.model.eval()
        self.cache: dict[str, np.ndarray] = {}

    def log_probs(self, audio_path: str) -> np.ndarray:
        if audio_path not in self.cache:
            wav, sr = sf.read(audio_path, dtype="float32")
            assert sr == 16000, f"{audio_path}: {sr} Hz"
            wav = torch.from_numpy(wav if wav.ndim == 1 else wav.mean(1))[None]
            with torch.no_grad():
                lp, _ = self.model(wav, torch.tensor([wav.shape[1]]))
            self.cache[audio_path] = lp[0].numpy()
        return self.cache[audio_path]

    def prefetch(self, audio_path: str) -> None:
        threading.Thread(target=self.log_probs, args=(audio_path,), daemon=True).start()

    def word_starts(self, audio_path: str, words: list[str]) -> list[float] | None:
        """Chỉ lấy mốc bắt đầu: đỉnh CTC thường rộng 1 khung (đo 14/10) nên "mốc kết thúc" không mang nghĩa."""
        # Encode từng từ riêng (nhãn train toàn chữ HOA) → biết chắc token nào thuộc từ nào.
        pieces = [self.tokenizer.encode(w.upper()) for w in words]
        tokens = [t for p in pieces for t in p]
        owner = [k for k, p in enumerate(pieces) for _ in p]
        path = ctc_viterbi(self.log_probs(audio_path), tokens) if tokens else None
        if path is None:
            return None
        start = [None] * len(words)
        for t, j in enumerate(path):
            if j >= 0 and start[owner[j]] is None:
                start[owner[j]] = t * FRAME_S
        return start


def words_html(aligner: WordAligner | None, audio_path: str, text: str) -> str:
    if aligner is None:
        return "<i>Không có checkpoint Conformer v1 → không tô chữ theo audio.</i>"
    words = normalize_text(text).split()
    if not words:
        return ""
    starts = aligner.word_starts(audio_path, words)
    if starts is None:
        return "<i>Không căn được chữ với audio (bản hiệu đính dài hơn audio?).</i>"
    spans = " ".join(f"<span data-s='{s:.2f}' title='{s:.2f} s'>{html.escape(w)}</span>"
                     for w, s in zip(words, starts))
    return f"<div id='ct-words'>{spans}</div>"


def audio_html(audio_path: str) -> str:
    # Thẻ <audio> riêng thay gr.Audio: JS cần đọc currentTime, player waveform của Gradio không lộ phần tử đó.
    # Query `v` mới mỗi lần: thử 14/10, Chrome giữ một phản hồi 204 cho đúng URL câu mở đầu → <audio> treo ở readyState 0.
    url = "/gradio_api/file=" + quote(Path(audio_path).resolve().as_posix()) + f"?v={time.time_ns()}"
    return f"<audio id='ct-audio' controls autoplay preload='auto' src='{url}'></audio>"


def ours_html(hyps: dict[str, str], edited: str) -> str:
    if not hyps:
        return "<i>Không có output mô hình cho câu này.</i>"
    rows = [diff_html(h, edited, f"<u>{html.escape(OURS_NAMES.get(k, k))}</u>") for k, h in hyps.items()]
    return ("<div style='font-size:.95em'>Đỏ gạch = mô hình có mà bản hiệu đính không có; xanh = ngược lại.</div>"
            + "".join(rows))


def load_ours() -> dict:
    if not OURS.exists():
        return {}
    with OURS.open(encoding="utf-8", newline="") as f:
        return {(r["split"], int(r["index"])): {k: r[f"raw_{k}"] for k in OURS_NAMES if r.get(f"raw_{k}")}
                for r in csv.DictReader(f, delimiter="	")}


def build_app(manifest: dict, aligner: WordAligner | None, advisor: dict, ours: dict) -> gr.Blocks:
    samples = manifest["samples"]
    n = len(samples)

    def whisper_of(i) -> str:
        s = samples[int(i)]
        return advisor.get((s["split"], s["index"]), "")

    def n_done() -> int:
        return sum(bool(s["corrected_text"].strip()) for s in samples)

    def first_todo(start: int = 0) -> int:
        for k in list(range(start, n)) + list(range(0, start)):
            if not samples[k]["corrected_text"].strip():
                return k
        return start

    def view(i: int, message: str = ""):
        i = max(0, min(n - 1, int(i)))
        s = samples[i]
        done = bool(s["corrected_text"].strip())
        text = s["corrected_text"] if done else s["pseudo_label"].lower()
        header = (f"**Câu {i + 1}/{n}** {'✅' if done else '⬜'} · tiến độ **{n_done()}/{n}** · "
                  f"{s['duration_s']:.1f} s · `{s['source'][:60]}`" + (f"  \n{message}" if message else ""))
        audio = str(Path(AUDIO_CACHE_DIR) / s["audio"])
        if aligner is not None and i + 1 < n:
            aligner.prefetch(str(Path(AUDIO_CACHE_DIR) / samples[first_todo(i + 1)]["audio"]))
        whisper = whisper_of(i) or "(chưa có output Whisper — xem data/processed/whisper_advisor.json)"
        return (i, header, audio_html(audio), s["pseudo_label"], whisper, text, s.get("notes", "")) + on_edit(i, text)

    def on_edit(i, text):
        s = samples[int(i)]
        whisper = whisper_of(i)
        adv = (agreement_html(s["pseudo_label"], whisper, text) + diff_html(whisper, text, "Whisper")) if whisper else ""
        return (diff_html(s["pseudo_label"], text),
                words_html(aligner, str(Path(AUDIO_CACHE_DIR) / s["audio"]), text), adv,
                ours_html(ours.get((s["split"], s["index"]), {}), text))

    def do_save(i, text, notes):
        text = unicodedata.normalize("NFC", " ".join(text.split()))
        if not text:
            return view(i, "⚠️ Ô hiệu đính trống — chưa lưu.")
        if _DIGIT.search(text):
            return view(i, "⚠️ Có chữ số — hãy viết số bằng chữ rồi lưu lại.")[:5] + (text, notes) + on_edit(i, text)
        samples[i]["corrected_text"] = text
        samples[i]["notes"] = notes.strip()
        samples[i]["corrected_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save(manifest)
        nxt = first_todo(i + 1) if i + 1 < n else i
        msg = f"💾 Đã lưu câu {i + 1}." + (" 🎉 Xong cả {n} câu!".format(n=n) if n_done() == n else "")
        return view(nxt, msg)

    def do_reset(i):
        samples[i]["corrected_text"] = ""
        save(manifest)
        return view(i, f"↩️ Đã xoá hiệu đính câu {i + 1} (quay về nhãn giả).")

    with gr.Blocks(title="Hiệu đính clean-test") as app:
        idx = gr.State(first_todo())
        # Bố trí 2 cột (14/10, người dùng thấy phải cuộn nhiều): trái = việc nghe + sửa (dính trên màn hình),
        # phải = tài liệu tham khảo chia tab, để không bao giờ phải cuộn qua tham khảo mới tới nút Lưu.
        with gr.Row(equal_height=False):
            header = gr.Markdown(elem_id="ct-header")
            prev_btn = gr.Button("◀ Trước", size="sm", scale=0, min_width=80)
            next_btn = gr.Button("Bỏ qua ▶", size="sm", scale=0, min_width=80)
            jump = gr.Number(label="Đến câu", precision=0, minimum=1, maximum=n, scale=0, min_width=90,
                             container=False)
            jump_btn = gr.Button("Đi", size="sm", scale=0, min_width=50)
            todo_btn = gr.Button("Câu chưa làm", size="sm", scale=0, min_width=110)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, elem_id="ct-left"):
                audio = gr.HTML()
                words = gr.HTML(elem_id="ct-words-box")
                edited = gr.Textbox(label="Bản hiệu đính — sửa ở đây (Ctrl+Enter = Lưu & tiếp)", lines=4,
                                    elem_id="ct-edited")
                with gr.Row():
                    notes = gr.Textbox(label="Ghi chú (khó nghe, nhạc nền, chồng tiếng, tiếng Anh, tên riêng)",
                                       lines=1, scale=3)
                    save_btn = gr.Button("💾 Lưu & tiếp ▶", variant="primary", scale=1, elem_id="ct-save")
                diff = gr.HTML()
            with gr.Column(scale=2, elem_id="ct-right"):
                with gr.Tabs():
                    with gr.Tab("Đối chiếu"):
                        adv = gr.HTML()
                    with gr.Tab("Zipformer"):
                        pseudo = gr.Textbox(label="Nhãn giả (Zipformer)", interactive=False, lines=6)
                    with gr.Tab("Whisper"):
                        whisper = gr.Textbox(label="Whisper large-v3 (ngoài đội hình — mạnh tên riêng/tiếng Anh; "
                                                   "hay bỏ lặp từ/từ đệm, ra chữ số)", interactive=False, lines=6)
                    with gr.Tab("Mô hình đề tài"):
                        gr.Markdown("*Đang so sánh — nên tự nghe và sửa trước, mở tab này sau để đối chiếu.*")
                        ours_box = gr.HTML()
                    with gr.Tab("Hướng dẫn"):
                        gr.Markdown(GUIDE)
                        reset_btn = gr.Button("Xoá hiệu đính câu này", size="sm", variant="stop")

        outputs = [idx, header, audio, pseudo, whisper, edited, notes, diff, words, adv, ours_box]
        app.load(lambda i: view(i), idx, outputs)
        edited.change(on_edit, [idx, edited], [diff, words, adv, ours_box])
        save_btn.click(do_save, [idx, edited, notes], outputs)
        prev_btn.click(lambda i: view(i - 1), idx, outputs)
        next_btn.click(lambda i: view(i + 1), idx, outputs)
        jump_btn.click(lambda j: view((j or 1) - 1), jump, outputs)
        todo_btn.click(lambda i: view(first_todo(i), "" if n_done() < n else "Đã hiệu đính đủ."), idx, outputs)
        reset_btn.click(do_reset, idx, outputs)
    return app


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream.encoding != "utf-8":
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()

    manifest = load()
    backup = MANIFEST.with_name(f"clean_test_manifest.bak-{time.strftime('%Y%m%d-%H%M%S')}.json")
    shutil.copy2(MANIFEST, backup)
    missing = [s["audio"] for s in manifest["samples"] if not (Path(AUDIO_CACHE_DIR) / s["audio"]).exists()]
    print(f"[hiệu đính] {len(manifest['samples'])} câu, sao lưu → {backup}; thiếu audio: {len(missing)}", flush=True)
    aligner = None
    if ALIGN_CHECKPOINT.exists():
        aligner = WordAligner()
        print(f"[hiệu đính] tô chữ theo audio: forced alignment bằng {ALIGN_CHECKPOINT}", flush=True)
    else:
        print(f"[hiệu đính] không thấy {ALIGN_CHECKPOINT} → tắt tô chữ theo audio", flush=True)
    advisor = {}
    if ADVISOR.exists():
        advisor = {(r["split"], r["index"]): r["text"]
                   for r in json.loads(ADVISOR.read_text(encoding="utf-8"))["samples"]}
    print(f"[hiệu đính] cố vấn Whisper: {len(advisor)} câu ({ADVISOR})", flush=True)
    ours = load_ours()
    print(f"[hiệu đính] output mô hình đề tài: {len(ours)} câu ({OURS})", flush=True)
    build_app(manifest, aligner, advisor, ours).launch(server_port=args.port, css=CSS, head=HEAD_JS,
                               allowed_paths=[str(Path(AUDIO_CACHE_DIR).resolve())])


if __name__ == "__main__":
    main()
