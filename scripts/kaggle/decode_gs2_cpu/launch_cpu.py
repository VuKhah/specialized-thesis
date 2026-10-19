"""Đóng gói dữ liệu + đẩy kernel CPU `decode_cpu.py` lên B, C, D (mỗi tài khoản một encoder) — 2026-10-19.

Dataset riêng mỗi tài khoản `gs2-cpu-<encoder>` (B, C, D không đọc được output/dataset private của A): `bundle.tar.gz`
(module chấm/giải mã, tokenizer, LM Vietcetera pkl + ARPA + văn bản, ref dev/test GigaSpeech 2, giả thuyết beam+LM local
cho cổng kiểm, văn bản train GigaSpeech 2 nén) + top-k test 3 seed + dev seed 42 của encoder đó. Tải lên **lần lượt**,
xoá cache upload trước mỗi lần (cache resume của Kaggle CLI không tách theo tài khoản → 403, sự cố 19/10).

    python scripts/kaggle/decode_gs2_cpu/launch_cpu.py bundle                 # dựng thư mục dataset 3 tài khoản
    python scripts/kaggle/decode_gs2_cpu/launch_cpu.py upload                 # tải lên lần lượt
    python scripts/kaggle/decode_gs2_cpu/launch_cpu.py push --stage tune
    python scripts/kaggle/decode_gs2_cpu/launch_cpu.py push --stage final --final '{"lm":"broad","lmw":0.8,"ws":1.0}'
"""

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STAGE_DIR = REPO / "data/raw/gigaspeech2/kaggle_ds"
GS2 = REPO / "data/raw/gigaspeech2/data/vi"
TOPK = REPO / "checkpoints/gs2_topk"
PLAN = {"B": "conformer_relpos", "C": "conextbimamba", "D": "mamba"}
CONFIG_DIRS = {"B": Path("C:/Users/Dell/.kaggle_Duc"), "C": Path("C:/Users/Dell/.kaggle_Nhan"),
               "D": Path("C:/Users/Dell/.kaggle_Lam")}
FILES = ["src/__init__.py", "src/evaluation/__init__.py", "src/evaluation/lm_decode.py", "src/evaluation/vi_syllable.py",
         "src/evaluation/text_normalize.py", "src/evaluation/score_clean_test.py", "src/evaluation/bootstrap.py",
         "src/data/__init__.py", "src/data/correct_clean_test.py", "configs/tokenizer.model",
         "configs/scoring/alt_spellings.tsv", "checkpoints/lm/lexicon.txt", "checkpoints/lm/lm3.pkl",
         "checkpoints/lm/lm3.arpa", "checkpoints/lm/lm_text_clean.txt"]


def user(acc: str) -> str:
    return json.loads((CONFIG_DIRS[acc] / "kaggle.json").read_text(encoding="utf-8"))["username"]


def bundle() -> None:
    tar = STAGE_DIR / "bundle.tar.gz"
    gz = STAGE_DIR / "gs2_train_refined.tsv.gz"
    if not gz.exists():
        with open(GS2 / "train_refined.tsv", "rb") as fi, gzip.open(gz, "wb", compresslevel=6) as fo:
            shutil.copyfileobj(fi, fo)
    with tarfile.open(tar, "w:gz") as t:
        for f in FILES:
            t.add(REPO / f, arcname=f)
        t.add(HERE / "decode_cpu.py", arcname="decode_cpu.py")
        t.add(GS2 / "dev.tsv", arcname="data/gs2/dev.tsv")
        t.add(GS2 / "test.tsv", arcname="data/gs2/test.tsv")
        t.add(REPO / "reports/results/gs2_eval_2026-10-19/40ms/gs2_hyps_lm.tsv", arcname="data/gs2/gs2_hyps_lm_local.tsv")
        t.add(gz, arcname="data/lm/gs2_train_refined.tsv.gz")
    for acc, enc in PLAN.items():
        d = STAGE_DIR / f"cpu_{acc}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        os.link(tar, d / "bundle.tar.gz")
        npz = [f"tk_{enc}_seed{s}_gs2_test.npz" for s in (42, 43, 44)] + [f"tk_{enc}_seed42_gs2_dev.npz"]
        for n in npz:
            if not (TOPK / n).exists():
                sys.exit(f"LỖI: thiếu {TOPK / n}")
            os.link(TOPK / n, d / n.replace(".npz", ".tkz"))  # Kaggle tự giải nén .npz (là zip)
        (d / "dataset-metadata.json").write_text(json.dumps({"title": f"gs2-cpu-{enc.replace('_', '-')}",
                                                             "id": f"{user(acc)}/gs2-cpu-{enc.replace('_', '-')}",
                                                             "licenses": [{"name": "other"}]}), encoding="utf-8")
        size = sum(p.stat().st_size for p in d.iterdir()) / 1e9
        print(f"{acc}: {enc}, {len(list(d.iterdir()))} file, {size:.2f} GB → {d}")


def upload() -> None:
    cache = Path(os.environ.get("TEMP", "C:/Users/Dell/AppData/Local/Temp")) / ".kaggle/uploads"
    for acc in PLAN:
        shutil.rmtree(cache, ignore_errors=True)
        env = {**os.environ, "PYTHONUTF8": "1", "KAGGLE_CONFIG_DIR": str(CONFIG_DIRS[acc])}
        r = subprocess.run(["kaggle", "datasets", "create", "-p", ".", "-r", "skip"], cwd=STAGE_DIR / f"cpu_{acc}",
                           env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        ok = r.returncode == 0 and "403" not in r.stdout + r.stderr
        print(f"{acc}: upload rc={r.returncode} {'OK' if ok else (r.stdout + r.stderr)[-300:].replace(user(acc), '<u>')}",
              flush=True)


def push(stage: str, final: str | None) -> None:
    code = (HERE / "decode_cpu.py").read_text(encoding="utf-8")
    for acc, enc in PLAN.items():
        out = HERE / "_build" / f"{stage}_{acc}"
        out.mkdir(parents=True, exist_ok=True)
        c = (code.replace('os.environ.get("GS2_STAGE", "__STAGE__")', f'os.environ.get("GS2_STAGE", "{stage}")')
             .replace('os.environ.get("GS2_ENCODER", "__ENCODER__")', f'os.environ.get("GS2_ENCODER", "{enc}")')
             .replace("""os.environ.get("GS2_FINAL", '__FINAL__')""", f"""os.environ.get("GS2_FINAL", '{final or "null"}')"""))
        (out / "decode_cpu.py").write_text(c, encoding="utf-8")
        slug = f"gs2-cpu-{stage}-{enc.replace('_', '-')}"
        (out / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user(acc)}/{slug}", "title": slug, "code_file": "decode_cpu.py", "language": "python",
            "kernel_type": "script", "is_private": "true", "enable_gpu": "false", "enable_internet": "true",
            "dataset_sources": [f"{user(acc)}/gs2-cpu-{enc.replace('_', '-')}"], "competition_sources": [],
            "kernel_sources": []}, indent=2), encoding="utf-8")
        env = {**os.environ, "PYTHONUTF8": "1", "KAGGLE_CONFIG_DIR": str(CONFIG_DIRS[acc])}
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(out), "-t", "43200"], env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        print(f"{acc} {slug}: rc={r.returncode} " + (r.stdout + r.stderr).strip().replace(user(acc), f"<{acc}>")[-160:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bundle", "upload", "push"])
    ap.add_argument("--stage", choices=["tune", "final"])
    ap.add_argument("--final", default=None, help='JSON cấu hình cho stage final, vd. {"lm":"vss","lmw":0.8,"ws":1.0}')
    a = ap.parse_args()
    {"bundle": bundle, "upload": upload}.get(a.cmd, lambda: push(a.stage, a.final))()


if __name__ == "__main__":
    main()
