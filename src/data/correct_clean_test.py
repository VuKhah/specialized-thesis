"""Công cụ nghe & hiệu đính clean-test (thêm 2026-10-08): ghi cột `corrected_text` / `notes` của
`data/processed/clean_test_manifest.json`, mỗi lần bấm Lưu là ghi file ngay.

    python -m src.data.correct_clean_test              # http://127.0.0.1:7861 (demo dùng 7860)

Thay mục 5 của `notebooks/02_dataset_eda.ipynb`: notebook được track trên repo public, nghe audio ở đó dễ để
lại output nhúng audio; sửa JSON bằng tay dễ hỏng file. Lúc khởi động chép một bản sao lưu cạnh manifest
(`data/processed/` bị gitignore trừ chính manifest → bản sao không lên git).

`corrected_text` rỗng = chưa hiệu đính (`eval_clean_test` quay về `pseudo_label`). WER chuẩn hoá bằng
`normalize_text` (chữ thường, bỏ dấu câu) nên hoa/thường và dấu câu không quan trọng; **chữ số thì giữ
nguyên** → viết số bằng chữ như nhãn train.
"""

import argparse
import difflib
import html
import json
import os
import re
import shutil
import sys
import time
import unicodedata
from pathlib import Path

import gradio as gr

from src.data.vietsuperspeech_dataset import AUDIO_CACHE_DIR
from src.evaluation.text_normalize import normalize_text

MANIFEST = Path("data/processed/clean_test_manifest.json")
_DIGIT = re.compile(r"\d")

CSS = """
.diff del {background:#fdd; color:#900; text-decoration:line-through; padding:0 2px}
.diff ins {background:#dfd; color:#060; text-decoration:none; padding:0 2px}
.diff {font-size:1.05em; line-height:1.8}
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


def diff_html(reference: str, edited: str) -> str:
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
    head = f"<b>So với nhãn giả:</b> {n_changed} từ khác" if n_changed else "<b>Giống hệt nhãn giả</b>"
    return f"<div class='diff'>{head}<br>{' '.join(out)}</div>"


def build_app(manifest: dict) -> gr.Blocks:
    samples = manifest["samples"]
    n = len(samples)

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
        header = (f"### Câu {i + 1}/{n} — {'✅ đã hiệu đính' if done else '⬜ chưa hiệu đính'}\n"
                  f"Tiến độ: **{n_done()}/{n}** · `{s['split']}#{s['index']}` · {s['duration_s']:.1f} s · "
                  f"video `{s['source']}`" + (f"\n\n{message}" if message else ""))
        audio = str(Path(AUDIO_CACHE_DIR) / s["audio"])
        return i, header, audio, s["pseudo_label"], text, s.get("notes", ""), diff_html(s["pseudo_label"], text)

    def do_save(i, text, notes):
        text = unicodedata.normalize("NFC", " ".join(text.split()))
        if not text:
            return view(i, "⚠️ Ô hiệu đính trống — chưa lưu.")
        if _DIGIT.search(text):
            return view(i, "⚠️ Có chữ số — hãy viết số bằng chữ rồi lưu lại.")[:4] + (text, notes,
                                                                                        diff_html(samples[i]["pseudo_label"], text))
        samples[i]["corrected_text"] = text
        samples[i]["notes"] = notes.strip()
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
        gr.Markdown(GUIDE)
        header = gr.Markdown()
        audio = gr.Audio(type="filepath", label="Audio", autoplay=True, interactive=False)
        pseudo = gr.Textbox(label="Nhãn giả (Zipformer, chỉ để tham khảo)", interactive=False, lines=3)
        edited = gr.Textbox(label="Bản hiệu đính — sửa ở đây", lines=4)
        diff = gr.HTML()
        notes = gr.Textbox(label="Ghi chú (vd. khó nghe, nhạc nền, chồng tiếng, tiếng Anh)", lines=1)
        with gr.Row():
            prev_btn = gr.Button("◀ Trước")
            save_btn = gr.Button("💾 Lưu & tiếp ▶", variant="primary")
            next_btn = gr.Button("Bỏ qua ▶")
        with gr.Row():
            jump = gr.Number(label="Đến câu số", precision=0, minimum=1, maximum=n)
            jump_btn = gr.Button("Đi")
            todo_btn = gr.Button("Đến câu chưa làm")
            reset_btn = gr.Button("Xoá hiệu đính câu này")

        outputs = [idx, header, audio, pseudo, edited, notes, diff]
        app.load(lambda i: view(i), idx, outputs)
        edited.change(lambda i, t: diff_html(samples[int(i)]["pseudo_label"], t), [idx, edited], diff)
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
    build_app(manifest).launch(server_port=args.port, css=CSS,
                               allowed_paths=[str(Path(AUDIO_CACHE_DIR).resolve())])


if __name__ == "__main__":
    main()
