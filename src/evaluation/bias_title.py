"""Bổ sung A — từ trong tên file video (tiêu đề tập, dạng ASCII) làm ngữ cảnh khi giải mã beam + LM; WER / B-WER / U-WER.

Quy tắc chốt trước khi chạy: `docs/notes/ket_qua/bo_sung_AB_quy_tac_2026-10-27.md` mục A (A2 danh sách, A3 ba điều
kiện, A4 chọn b, A5 chấm, A6 thống kê); sửa đổi kỹ thuật ghi ở mục "Sửa đổi" của file đó. B-WER/U-WER theo Le và cs.,
Interspeech 2021 (arXiv:2104.02194): lỗi S/D xếp theo từ ref, lỗi I theo từ hyp, có thuộc danh sách của video hay không.

    python -m src.evaluation.bias_title lists     # A2 → bias_lists.tsv + độ phủ (chỉ đếm trên nhãn, chưa giải mã)
    python -m src.evaluation.bias_title check     # (i) qua đường code mới == cột _lm cũ; (ii) danh sách rỗng == (i)
    python -m src.evaluation.bias_title grid      # A4: nửa tune val_unseen, seed 42 × 3 encoder, b ∈ GRID → grid.json
    python -m src.evaluation.bias_title decode    # A5: clean-test 9 run × (i)/(ii)/(iii), b đọc từ grid.json
    python -m src.evaluation.bias_title score     # A5/A6 → results.json, per_sentence.tsv

Điều kiện riêng (iv) — tiêu đề có dấu + mô tả tập từ trang vietcetera.com (quy tắc A8, cùng file quy tắc):

    python -m src.evaluation.bias_title fetch-pages            # sitemap → URL từng video → tiêu đề + mô tả (không track)
    python -m src.evaluation.bias_title lists-iv               # A8.3 → bias_lists_iv.tsv + coverage_iv.json
    python -m src.evaluation.bias_title grid --variant iv      # A4 với danh sách (iv) → grid_iv.json
    python -m src.evaluation.bias_title decode --variant iv
    python -m src.evaluation.bias_title score --variant iv     # → results_iv.json, per_sentence_iv.tsv
"""

import argparse
import csv
import html
import json
import re
import sys
import time
import unicodedata
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import jiwer
import numpy as np

from src.evaluation.bootstrap import two_level_bootstrap
from src.evaluation.score_clean_test import MANIFEST, TAGS, canonicalize, hyp_tokens, load_alt, ref_tokens
from src.evaluation.text_normalize import normalize_text
from src.evaluation.vi_syllable import is_valid_syllable

OUT = Path("reports/results/bias_title_2026-10-27")
LISTS = OUT / "bias_lists.tsv"
HYPS = OUT / "hyps"
HELDOUT = Path("data/splits/heldout_videos.tsv")
VAL_UNSEEN = Path("data/splits/val_unseen.tsv")
LP_S42 = Path("D:/Model/kltn_checkpoints/logprobs_v2a_s4")  # lp_<m>_{clean_test,val_unseen_subset}.npz
LP_SEEDS = Path("checkpoints/v2a_s4_seeds/logprobs")  # lp_<m>_seed{43,44}_clean_test.npz
# 958 câu val_unseen_subset (khoá, video, nhãn giả) — đúng file agent giải mã 15/10 dùng chọn lmw/ws; nửa tune = video
# thứ 0, 2, 4… theo thứ tự tên (`run_exp.py`: `tune_v = set(vids[::2])`).
TUNE_META = Path("D:/Model/kltn_decoding_2026-10-15/decoding/meta.json")
# Output beam+LM đã báo (điều kiện (i)) để kiểm tái lập: seed 42 cột v2a_<m>_lm, seed 43/44 cột v2a_<m>_seed<N>_lm.
OLD_LM = {42: Path("reports/results/decode_v2a_s4_2026-10-15/cleantest_hyps.tsv"),
          43: Path("reports/results/decode_v2a_s4_seeds/cleantest_hyps.tsv"),
          44: Path("reports/results/decode_v2a_s4_seeds/cleantest_hyps.tsv")}
MODELS = ("mamba", "conextbimamba", "conformer_relpos")
MODEL_NAMES = {"mamba": "B1 Mamba", "conextbimamba": "ConExtBiMamba", "conformer_relpos": "Conformer relPE"}
SEEDS = (42, 43, 44)
GRID = (0, 2, 4, 8, 16)
JOBS = 2  # CPU dùng chung với tác tử khác: ≤ 2 tiến trình giải mã, mỗi tiến trình 2 luồng
SITEMAP = "https://vietcetera.com/sitemap/tree.xml"
PAGES = Path("data/processed/bias_title/episode_pages.json")  # nguyên văn tiêu đề + mô tả: nội dung của trang, không track
LISTS_IV = OUT / "bias_lists_iv.tsv"
VARIANTS = {"iii": {"lists": LISTS, "grid": OUT / "grid.json", "suffix": ""},
            "iv": {"lists": LISTS_IV, "grid": OUT / "grid_iv.json", "suffix": "_iv"}}


# ---------------------------------------------------------------- A2: danh sách thưởng

def video_of(audio: str) -> str:
    return re.sub(r"_seg\d+\.wav$", "", Path(audio).name)


def key_videos() -> dict[str, str]:
    """Khoá "split:index" → video, cho clean-test và val_unseen (câu hai tập đến từ cả hai split HF)."""
    out = {f"{s['split']}:{s['index']}": video_of(s["audio"])
           for s in json.loads(MANIFEST.read_text(encoding="utf-8"))["samples"]}
    with VAL_UNSEEN.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            k = f"{r['split']}:{r['index']}"
            assert out.setdefault(k, video_of(r["audio"])) == video_of(r["audio"]), k
    return out


def title_words(video: str, alt) -> list[str]:
    toks = [w for t in re.split(r"[_\-\s]+", video.lower()) for w in normalize_text(t).split()]
    words = [t[0] for t in canonicalize([(w, "", False) for w in toks], alt)]
    # Âm tiết Việt hợp lệ bị bỏ: tên file mất dấu nên "luc" không khớp được "lực" ở ref (quy tắc A2).
    keep = [w for w in words if re.search(r"[^\W\d_]", w) and len(w) >= 2 and not is_valid_syllable(w)]
    return list(dict.fromkeys(keep))


def build_lists() -> dict[str, list[str]]:
    videos = sorted(set(key_videos().values()))
    with HELDOUT.open(encoding="utf-8") as f:
        assert videos == sorted(r["video"] for r in csv.DictReader(f, delimiter="\t")), "video ≠ heldout_videos.tsv"
    alt = load_alt()
    return {v: title_words(v, alt) for v in videos}


def read_lists(path: Path = LISTS) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as f:
        return {r["video"]: r["words"].split() for r in csv.DictReader(f, delimiter="\t")}


def load_tags() -> dict[str, str]:
    with TAGS.open(encoding="utf-8-sig") as f:
        return {r["key"]: r["label"] for r in csv.DictReader(f, delimiter="\t") if r["label"]}


def clean_test_refs() -> list[tuple[str, str, list]]:
    """[(khoá, video, token ref như `score_clean_test`)] theo thứ tự manifest (số câu #n 1-based cho từ vấp dở)."""
    tags, alt = load_tags(), load_alt()
    return [(f"{s['split']}:{s['index']}", video_of(s["audio"]), ref_tokens(n, s["corrected_text"], tags, alt, True))
            for n, s in enumerate(json.loads(MANIFEST.read_text(encoding="utf-8"))["samples"], 1)]


def tune_refs() -> list[tuple[str, str, list]]:
    """Nửa tune val_unseen_subset, tham chiếu = nhãn giả; cùng cách chấm (không có bảng nhãn từ/từ vấp dở: của clean-test)."""
    meta = json.loads(TUNE_META.read_text(encoding="utf-8"))["meta"]
    tune = set(sorted({m["video"] for m in meta})[::2])
    alt = load_alt()
    return [(f"{m['split']}:{m['index']}", m["video"], ref_tokens(0, m["text"], {}, alt, True))
            for m in meta if m["video"] in tune]


def coverage(refs, lists) -> dict:
    per_word, sents, n_words = {}, 0, 0
    for _, v, ref in refs:
        hit = [t[0] for t in ref if t[0] in set(lists[v])]
        n_words += len(ref)
        sents += bool(hit)
        for w in hit:
            per_word[(v, w)] = per_word.get((v, w), 0) + 1
    return {"ref_words": n_words, "ref_words_in_list": sum(per_word.values()), "sentences": len(refs),
            "sentences_with_list_word": sents, "videos_with_list_word": len({v for v, _ in per_word}),
            "per_word": [{"video": v, "word": w, "count": c} for (v, w), c in sorted(per_word.items())]}


def cmd_lists() -> None:
    lists = build_lists()
    OUT.mkdir(parents=True, exist_ok=True)
    with LISTS.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["video", "n", "words"])
        for v, ws in lists.items():
            w.writerow([v, len(ws), " ".join(ws)])
    report_lists(lists, LISTS, OUT / "coverage.json")


def report_lists(lists: dict, lists_path: Path, cov_path: Path) -> None:
    from src.evaluation.lm_decode import LM_DIR
    lex = {line.split("\t", 1)[0] for line in (LM_DIR / "lexicon.txt").read_text(encoding="utf-8").splitlines()}
    allw = sorted({w for ws in lists.values() for w in ws})
    print(f"{len(lists)} video · {sum(map(len, lists.values()))} mục ({len(allw)} từ khác nhau) → {lists_path}")
    print(f"đã có trong từ điển gốc: {sorted(w for w in allw if w.upper() in lex)}")
    res = {"clean_test": coverage(clean_test_refs(), lists), "val_tune": coverage(tune_refs(), lists)}
    cov_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, c in res.items():
        print(f"[{name}] từ ref thuộc danh sách {c['ref_words_in_list']}/{c['ref_words']} · câu có ≥ 1 từ "
              f"{c['sentences_with_list_word']}/{c['sentences']} · video {c['videos_with_list_word']}")
        tot = {}
        for r in c["per_word"]:
            tot[r["word"]] = tot.get(r["word"], 0) + r["count"]
        print("   ", " ".join(f"{w}×{n}" for w, n in sorted(tot.items(), key=lambda x: -x[1])))


# ---------------------------------------------------------------- A8: điều kiện (iv), tiêu đề có dấu + mô tả tập

# Video không ghép được theo luật A8.1 → URL tìm tay theo tên tập (ghi vào note quy tắc, mục "Sửa đổi").
MANUAL_URLS = {
    "Ap_luc_nao_khien_ta_lo_du_-_Vu_My_Anh_Minh_Tu_Di-tox_EP2":
        "https://vietcetera.com/vn/video/di-tox/ap-luc-nao-khien-ta-lo-du-vu-my-anh-minh-tu-di-tox-ep2",
    "Khui_het_12_tap_trong_12_phut_Have_A_Sip_-_After_Hours_dac_biet":
        "https://vietcetera.com/vn/video/tat-ca-video/khui-het-12-tap-trong-12-phut-have-a-sip-after-hours-dac-biet",
    "Tran_trong_nhung_thang_tien_nho_-_Guy_Truong_Thuy_Minh_Hao_Tran_Dumbbel_Ep2":
        "https://vietcetera.com/vn/video/dumbbell/tran-trong-nhung-thang-tien-nho-guy-truong-thuy-minh-hao-tran-dumbbel-ep2",
    "Trinh_tiet_co_phai_tieu_tiet_VJ_Thuy_Minh_Coi_Mo_S2EP05":
        "https://vietcetera.com/vn/video/coi-mo/trinh-tiet-co-phai-tieu-tiet-vj-thuy-minh-coi-mo-s2ep05",
    "Vao_bar_dung_de_cai_net_di_choi_xa_Coi_Mo_Happy_Hour_EP09_coimohappyhour":
        "https://vietcetera.com/vn/video/coi-mo-happy-hour/vao-bar-dung-de-cai-net-di-choi-xa-coi-mo-happy-hour-ep09"
        "-coimohappyhour",
    "Viet_creative_brief_Dai_ma_ro_y_la_tot_hay_ngan_nhung_truyen_cam_hung_moi_goi_la_hay_YMP_EP10":
        "https://vietcetera.com/vn/video/young-marketers/viet-creative-brief-dai-ma-ro-y-la-tot-hay-ngan-nhung-truyen-cam"
        "-hung-moi-goi-la-hay-ymp-ep10",
}  # Hay_cho_nguoi_co_hoi_yeu_thuong_minh_Yeu_Lanh_SS5_EP4: trang SS5 EP4 hiện mang tiêu đề khác → không nhận, giữ A2
# A8.2: bỏ trước khi tách token — URL, email, hashtag, mốc thời gian.
_NOISE = re.compile(r"https?://\S+|www\.\S+|\S+\.(?:com|vn)\S*|\S+@\S+|#\S+|\b\d{1,2}:\d{2}(?::\d{2})?\b")


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def _slug_tokens(s: str) -> set[str]:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return set(re.split(r"[^a-z0-9]+", s)) - {""}


def sitemap_urls() -> list[str]:
    def locs(xml: str) -> list[str]:
        return re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)
    return [u for sm in locs(_get(SITEMAP)) for u in locs(_get(sm))]


def match_pages(videos, urls) -> dict[str, dict]:
    """A8.1: Jaccard tập token của tên file ↔ đoạn cuối URL; nhận nếu ≥ 0,5. Ưu tiên trang /vn/video/ vì âm thanh lấy
    từ video YouTube và trang video mang nguyên mô tả YouTube."""
    cand = [(u, _slug_tokens(u.rstrip("/").rsplit("/", 1)[-1])) for u in urls]
    out = {}
    for v in videos:
        tv = _slug_tokens(v)
        scored = [(len(tv & tu) / len(tv | tu), u) for u, tu in cand if tu]
        best = max(((j, u) for j, u in scored if "/vn/video/" in u), default=(0.0, None))
        if best[0] < 0.5:
            best = max(scored, default=(0.0, None))
        out[v] = {"url": best[1] if best[0] >= 0.5 else None, "jaccard": round(best[0], 3)}
    return out


def page_text(url: str) -> dict:
    """Tiêu đề + mô tả YouTube của tập (trang podcast mang kèm `episode.video`), thiếu thì của podcast. Không đọc
    `transcriptSSR` hay bất kỳ bản ghi lời nào của tập (trường này thực ra chỉ chứa id tập —
    `docs/notes/du_lieu/vietcetera_transcript_2026-10-27.md`): bản ghi lời là nội dung câu cần nhận dạng."""
    h = _get(url)
    pp = json.loads(re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', h, re.S).group(1))["props"]["pageProps"]
    if "episode" in pp:  # trang podcast
        ep = pp["episode"]
        cands = (("youtube", (ep.get("video") or {}).get("snippet")), ("podcast", ep.get("snippet")))
    else:  # trang /vn/video/
        ep = pp["video"]
        cands = (("youtube", ep.get("snippet")),)
    for src, snip in cands:
        if snip and snip.get("description"):
            return {"source": src, "title": html.unescape(snip.get("title") or ep.get("title") or ""),
                    "description": html.unescape(snip["description"])}
    raise KeyError("snippet")


def cmd_fetch_pages() -> None:
    videos = sorted(read_lists())
    urls = sitemap_urls()
    print(f"sitemap: {len(urls)} URL, {sum('/vn/video/' in u for u in urls)} trang /vn/video/")
    pages = match_pages(videos, urls)
    pages.update({v: {"url": u, "jaccard": None, "manual": True} for v, u in MANUAL_URLS.items()})
    for v, p in pages.items():
        if p["url"]:
            try:
                p.update(page_text(p["url"]))
            except (AttributeError, KeyError, TypeError, ValueError) as e:  # trang không theo cấu trúc trang video
                p["error"] = repr(e)
        print(f"{p['jaccard']}\t{v[:50]}\t{p['url']}\t{len(p.get('description', ''))} ký tự {p.get('error', '')}")
    PAGES.parent.mkdir(parents=True, exist_ok=True)
    PAGES.write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"có mô tả {sum('description' in p for p in pages.values())}/{len(pages)} video → {PAGES}")


def context_words(text: str, alt) -> list[str]:
    """A8.2–A8.3: bỏ URL/email/hashtag/mốc thời gian rồi lọc như A2 — dùng chung cho danh sách (iv) và sản phẩm."""
    return title_words(_NOISE.sub(" ", text), alt)


def cmd_lists_iv() -> None:
    """A8.3: bộ lọc A2 trên tiêu đề có dấu + mô tả, hợp với danh sách A2 của video (video không có trang → chỉ A2)."""
    pages, base, alt = json.loads(PAGES.read_text(encoding="utf-8")), read_lists(), load_alt()
    lists = {}
    for v in base:
        p = pages.get(v, {})
        lists[v] = list(dict.fromkeys(base[v] + context_words(p.get("title", "") + "\n" + p.get("description", ""), alt)))
    with LISTS_IV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["video", "url", "n", "words"])
        for v, ws in lists.items():
            w.writerow([v, pages.get(v, {}).get("url") or "", len(ws), " ".join(ws)])
    report_lists(lists, LISTS_IV, OUT / "coverage_iv.json")


# ---------------------------------------------------------------- giải mã

_STATE: dict = {}


def _init() -> None:
    import torch
    torch.set_num_threads(2)
    from src.evaluation import lm_decode as L
    _STATE.update(tok=L._Tokens(), knlm=L.load_lm(L.LM_DIR / "lm3.pkl"), cache={}, vid=key_videos())


def _decoders(cond: str, b: float):
    """Bộ chọn decoder theo danh sách của video. Mọi decoder trong tiến trình dùng chung bộ nhớ đệm điểm LM thô (điểm
    KNLM không phụ thuộc từ điển hay b; b cộng sau bộ đệm) để RAM không nhân theo số video."""
    from src.evaluation import lm_decode as L
    tok, knlm = _STATE["tok"], _STATE["knlm"]

    def share(dec):
        dec.lm.cache = _STATE["cache"]
        return dec

    if cond == "i":
        base = share(L.make_decoder(L.LM_DIR / "lexicon.txt", knlm, tok))
        return lambda words: base
    pick = L.BiasedDecoders(knlm, tok, bonus=b)  # từ điển theo video ghi vào thư mục tạm (TMPDIR), không vào repo
    built = {}

    def get(words):
        k = tuple(words)
        if k not in built:
            built[k] = share(pick(k))
        return built[k]
    return get


def _run(job: dict) -> str:
    from src.evaluation import lm_decode as L
    out = Path(job["out"])
    if out.exists():
        return f"bỏ qua (đã có) {out.name}"
    t0 = time.time()
    pick = _decoders(job["cond"], job["b"])
    lists = read_lists(Path(job.get("lists", LISTS)))
    vid, tok, knlm = _STATE["vid"], _STATE["tok"], _STATE["knlm"]
    hyps = {k: {} for k in job["keys"]}
    for run, npz in job["runs"].items():
        data = np.load(npz)
        for k in job["keys"]:
            g, h = L.lm_hyp(pick(lists[vid[k]]), knlm, tok, data[k].astype(np.float32))
            hyps[k][f"{run}_greedy"], hyps[k][f"{run}_lm"] = g, h
    tmp = out.with_name(out.name + ".tmp")
    cols = [f"{r}_{m}" for r in job["runs"] for m in ("greedy", "lm")]
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["key", "video"] + cols)
        for k, v in hyps.items():
            w.writerow([k, vid[k]] + [v[c] for c in cols])
    tmp.replace(out)
    return f"{out.name}: {len(job['keys'])} câu × {len(job['runs'])} run, {time.time() - t0:.0f} s"


def cond_label(cond: str, b: float) -> str:
    return cond if cond in ("i", "ii") else f"{cond}_b{b:g}"


def grid_label(var: str, b: float) -> str:
    """Ô b = 0 của lưới A4: với danh sách A2 chính là điều kiện (ii) đã chạy."""
    return "ii" if var == "iii" and b == 0 else cond_label(var, b)


def run_jobs(jobs: list[dict]) -> None:
    HYPS.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not Path(j["out"]).exists()]
    print(f"{len(jobs)} việc, còn {len(todo)}; {JOBS} tiến trình", flush=True)
    t0 = time.time()
    with Pool(min(JOBS, max(len(todo), 1)), initializer=_init) as pool:
        for msg in pool.imap_unordered(_run, todo):
            print(f"[{(time.time() - t0) / 60:5.1f}′] {msg}", flush=True)


def clean_test_npz(m: str, seed: int) -> Path:
    return LP_S42 / f"lp_{m}_clean_test.npz" if seed == 42 else LP_SEEDS / f"lp_{m}_seed{seed}_clean_test.npz"


def read_hyps(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        return {r["key"]: r for r in csv.DictReader(f, delimiter="\t")}


def cmd_check() -> None:
    """Điều kiện (i) qua đường code mới (`lm_hyp` + decoder có bộ đệm chung) phải trùng từng chữ cột _lm đã báo; (ii)
    với danh sách rỗng phải trùng (i) (từ điển không thêm gì → đúng file gốc)."""
    keys = [k for k, _, _ in clean_test_refs()]
    m, seed = "mamba", 43
    out = HYPS / f"check__clean_test__i__{m}_s{seed}.tsv"
    run_jobs([{"out": str(out), "cond": "i", "b": 0.0, "keys": keys, "runs": {f"{m}_s{seed}": str(clean_test_npz(m, seed))}}])
    old, new = read_hyps(OLD_LM[seed]), read_hyps(out)
    col = f"v2a_{m}_seed{seed}_lm" if seed != 42 else f"v2a_{m}_lm"
    same = sum(old[k][col] == new[k][f"{m}_s{seed}_lm"] for k in keys)
    print(f"(i) {m} seed {seed}: trùng từng chữ {same}/{len(keys)} câu với {OLD_LM[seed]}")
    _init()
    from src.evaluation import lm_decode as L
    pick = L.BiasedDecoders(_STATE["knlm"], _STATE["tok"], bonus=0.0)
    data = np.load(clean_test_npz(m, seed))
    same2 = sum(L.lm_hyp(pick(()), _STATE["knlm"], _STATE["tok"], data[k].astype(np.float32))[1] == new[k][f"{m}_s{seed}_lm"]
                for k in keys[:20])
    print(f"(ii) danh sách rỗng: trùng (i) {same2}/20 câu đầu")


def cmd_grid(var: str = "iii") -> None:
    keys = [k for k, _, _ in tune_refs()]
    runs = {f"{m}_s42": str(LP_S42 / f"lp_{m}_val_unseen_subset.npz") for m in MODELS}
    labels = ["i"] + [grid_label(var, b) for b in GRID]
    run_jobs([{"out": str(HYPS / f"val_tune__{c}__s42.tsv"), "cond": c.split("_")[0], "b": b, "keys": keys,
               "runs": runs, "lists": str(VARIANTS[var]["lists"])} for c, b in zip(labels, [0.0] + [float(b) for b in GRID])])
    score_grid(var)


def chosen_b(var: str = "iii") -> float:
    return float(json.loads(VARIANTS[var]["grid"].read_text(encoding="utf-8"))["chosen_b"])


def cmd_decode(var: str = "iii") -> None:
    b = chosen_b(var)
    keys = [k for k, _, _ in clean_test_refs()]
    conds = [("i", 0.0), ("ii", 0.0), ("iii", b)] if var == "iii" else [("i", 0.0), (var, b)]
    jobs = []
    for c, bb in conds:
        for seed in SEEDS:
            jobs.append({"out": str(HYPS / f"clean_test__{cond_label(c, bb)}__s{seed}.tsv"), "cond": c, "b": bb,
                         "keys": keys, "runs": {f"{m}_s{seed}": str(clean_test_npz(m, seed)) for m in MODELS},
                         "lists": str(VARIANTS[var]["lists"])})
    run_jobs(jobs)


# ---------------------------------------------------------------- chấm

def bu_counts(ref, hyp, bias: set) -> np.ndarray:
    """[lỗi B, lỗi U, từ ref B, từ ref U] của một câu (A5): S/D theo từ ref, I theo từ hyp. Từ vấp dở tuỳ chọn như
    `score_clean_test.score_sentence`: bỏ khỏi ref nếu tổng lỗi ít hơn."""
    def u(t) -> int:  # 0 = B, 1 = U; int tường minh — numpy coi chỉ số bool trần là mặt nạ
        return int(t[0] not in bias)

    def count(r):
        e = np.zeros(4)
        for t in r:
            e[2 + u(t)] += 1
        if not r or not hyp:
            for t in (hyp if not r else r):
                e[u(t)] += 1
            return e
        out = jiwer.process_words(" ".join(t[0] for t in r), " ".join(t[0] for t in hyp))
        for ch in out.alignments[0]:
            if ch.type in ("substitute", "delete"):
                for t in r[ch.ref_start_idx:ch.ref_end_idx]:
                    e[u(t)] += 1
            elif ch.type == "insert":
                for t in hyp[ch.hyp_start_idx:ch.hyp_end_idx]:
                    e[u(t)] += 1
        return e

    best = count(ref)
    if any(t[2] for t in ref):
        alt_e = count([t for t in ref if not t[2]])
        if alt_e[:2].sum() < best[:2].sum():
            return alt_e
    return best


def score_hyps(refs, hyps: dict[str, dict], col: str, lists, alt) -> tuple[dict, list]:
    """Đếm theo câu và gộp theo video: {video: [eB, eU, nB, nU]}."""
    per_video, per_sent = {}, []
    for k, v, ref in refs:
        c = bu_counts(ref, hyp_tokens(hyps[k][col], alt, True), set(lists[v]))
        per_video[v] = per_video.get(v, np.zeros(4)) + c
        per_sent.append((k, v, c))
    return per_video, per_sent


def rates(c: np.ndarray) -> dict:
    eB, eU, nB, nU = c
    return {"wer": (eB + eU) / (nB + nU), "b_wer": eB / nB if nB else float("nan"), "u_wer": eU / nU,
            "err_b": int(eB), "err_u": int(eU), "words_b": int(nB), "words_u": int(nU)}


def score_grid(var: str = "iii") -> None:
    refs, lists, alt = tune_refs(), read_lists(VARIANTS[var]["lists"]), load_alt()
    conds = ["i"] + [grid_label(var, b) for b in GRID]
    table = {}
    for c in conds:
        hyps = read_hyps(HYPS / f"val_tune__{c}__s42.tsv")
        for m in MODELS:
            pv, _ = score_hyps(refs, hyps, f"{m}_s42_lm", lists, alt)
            table[f"{c}|{m}"] = rates(sum(pv.values()))
    # A4: một b chung = nhỏ nhất WER tổng trung bình 3 encoder (cùng mẫu số → so tổng lỗi, hoà xét đúng); hoà → b nhỏ.
    grid = {}
    for b in GRID:
        c = grid_label(var, b)
        grid[b] = {"label": c, "errors_sum": sum(table[f"{c}|{m}"]["err_b"] + table[f"{c}|{m}"]["err_u"] for m in MODELS),
                   "wer_mean": float(np.mean([table[f"{c}|{m}"]["wer"] for m in MODELS]))}
    best = min(GRID, key=lambda b: (grid[b]["errors_sum"], b))
    res = {"set": "val_unseen_subset nửa tune (14 video, nhãn giả)", "lists": str(VARIANTS[var]["lists"]),
           "n_sentences": len(refs), "n_videos": len({v for _, v, _ in refs}),
           "grid": {str(b): g for b, g in grid.items()}, "table": table, "chosen_b": best}
    VARIANTS[var]["grid"].write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"Nửa tune: {len(refs)} câu, {res['n_videos']} video. WER / B-WER / U-WER (beam+LM, seed 42):")
    for c in conds:
        print(f"  {c:8}", "  ".join(f"{MODEL_NAMES[m]} {table[f'{c}|{m}']['wer']:.4f}/{table[f'{c}|{m}']['b_wer']:.3f}/"
                                    f"{table[f'{c}|{m}']['u_wer']:.4f}" for m in MODELS))
    for b in GRID:
        print(f"  b = {b:2}: WER tb 3 encoder {grid[b]['wer_mean']:.5f} (tổng lỗi {grid[b]['errors_sum']:.0f})")
    print(f"→ b chọn = {best}")


def two_level(err_a, err_b, words, n_boot=10_000, seed=42) -> dict:
    """A6: đúng phép rút của `two_level_bootstrap` (video có hoàn lại, seed độc lập cho hai vế). Khác duy nhất: mẫu lại
    có tổng mẫu số 0 (không video nào có từ B) bị bỏ và đếm — hàm gốc sẽ chia cho 0. Không bỏ mẫu nào thì kết quả phải
    trùng hàm gốc (kiểm bằng assert)."""
    err_a, err_b, words = np.asarray(err_a, float), np.asarray(err_b, float), np.asarray(words, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(words), size=(n_boot, len(words)))
    sa, sb = rng.integers(0, len(err_a), size=n_boot), rng.integers(0, len(err_b), size=n_boot)
    w = words[idx].sum(1)
    ok = w > 0
    d = (np.take_along_axis(err_a[sa], idx, 1).sum(1) - np.take_along_axis(err_b[sb], idx, 1).sum(1))[ok] / w[ok]
    lo, hi = np.quantile(d, [0.025, 0.975])
    diff = float((err_a.sum(1) / words.sum()).mean() - (err_b.sum(1) / words.sum()).mean())
    if ok.all():
        r = two_level_bootstrap(err_a, err_b, words, n_boot, seed)
        assert np.isclose(r["ci_low"], lo) and np.isclose(r["ci_high"], hi) and np.isclose(r["diff"], diff)
    return {"diff": diff, "ci": [float(lo), float(hi)], "dropped_resamples": int((~ok).sum())}


def paired_seed_bootstrap(err_a, err_b, words, n_boot=10_000, seed=42) -> list[float]:
    """Biến thể mô tả (khai ở mục "Sửa đổi", không dùng để kết luận): cùng một seed cho hai vế vì hai điều kiện giải
    mã chạy trên cùng checkpoint — `two_level_bootstrap` rút seed độc lập vì vốn so hai encoder khác nhau."""
    err_a, err_b, words = np.asarray(err_a, float), np.asarray(err_b, float), np.asarray(words, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(words), size=(n_boot, len(words)))
    s = rng.integers(0, len(err_a), size=n_boot)
    w = words[idx].sum(1)
    ok = w > 0
    d = (np.take_along_axis(err_a[s], idx, 1).sum(1) - np.take_along_axis(err_b[s], idx, 1).sum(1))[ok] / w[ok]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def cmd_score(var: str = "iii") -> None:
    b = chosen_b(var)
    refs, lists, alt = clean_test_refs(), read_lists(VARIANTS[var]["lists"]), load_alt()
    videos = sorted({v for _, v, _ in refs})
    conds = ["i", "ii", cond_label("iii", b)] if var == "iii" else ["i", cond_label(var, b)]
    counts, per_sent = {}, []  # counts[(cond, m, seed)] = [video, 4]
    for c in conds:
        for seed in SEEDS:
            hyps = read_hyps(HYPS / f"clean_test__{c}__s{seed}.tsv")
            for m in MODELS:
                pv, ps = score_hyps(refs, hyps, f"{m}_s{seed}_lm", lists, alt)
                counts[(c, m, seed)] = np.array([pv[v] for v in videos])
                per_sent += [{"cond": c, "model": m, "seed": seed, "key": k, "video": v, "err_b": int(x[0]),
                              "err_u": int(x[1]), "words_b": int(x[2]), "words_u": int(x[3]),
                              "hyp": hyps[k][f"{m}_s{seed}_lm"]} for k, v, x in ps]
    metrics = {"wer": (lambda a: a[:, :, 0] + a[:, :, 1], lambda a: a[0, :, 2] + a[0, :, 3]),
               "b_wer": (lambda a: a[:, :, 0], lambda a: a[0, :, 2]),
               "u_wer": (lambda a: a[:, :, 1], lambda a: a[0, :, 3])}
    table, diffs = {}, []
    for m in MODELS:
        for c in conds:
            seeds = [rates(counts[(c, m, s)].sum(0)) for s in SEEDS]
            table[f"{c}|{m}"] = {k: float(np.mean([r[k] for r in seeds])) for k in ("wer", "b_wer", "u_wer")} | {
                "seeds": seeds}
        for c in conds[1:]:
            A = np.stack([counts[(c, m, s)] for s in SEEDS])
            B = np.stack([counts[("i", m, s)] for s in SEEDS])
            for name, (err, words) in metrics.items():
                ea, eb, w = err(A), err(B), words(A)
                r = two_level(ea, eb, w)
                diffs.append({"model": m, "cond": c, "vs": "i", "metric": name, **r,
                              "per_seed_same_ckpt": ((ea.sum(1) - eb.sum(1)) / w.sum()).tolist(),
                              "ci_paired_seed": paired_seed_bootstrap(ea, eb, w)})
    repro = {}  # (i) giải mã lại qua đường code mới so với cột _lm đã báo, cả 9 run
    for seed in SEEDS:
        old, new = read_hyps(OLD_LM[seed]), read_hyps(HYPS / f"clean_test__i__s{seed}.tsv")
        for m in MODELS:
            col = f"v2a_{m}_lm" if seed == 42 else f"v2a_{m}_seed{seed}_lm"
            repro[f"{m}_s{seed}"] = sum(old[k][col] == new[k][f"{m}_s{seed}_lm"] for k, _, _ in refs)
    # WER của (i) phải bằng WER đã báo ở lần chấm chung (cùng cách chấm, cùng hyps) — kiểm cách đếm B/U cộng về đúng.
    official = json.loads((Path("reports/results/clean_test_scored_2026-10-16_chung") / "results.json")
                          .read_text(encoding="utf-8"))["systems"]
    wer_check = {}
    for m in MODELS:
        for i, seed in enumerate(SEEDS):
            col = f"v2a_{m}_lm" if seed == 42 else f"v2a_{m}_seed{seed}_lm"
            wer_check[f"{m}_s{seed}"] = [table[f"i|{m}"]["seeds"][i]["wer"], official[col]["wer"]]
    cov = coverage(refs, lists)
    res = {"b": b, "lists": str(VARIANTS[var]["lists"]), "conditions": conds,
           "coverage": {k: v for k, v in cov.items() if k != "per_word"},
           "repro_i_identical_of_203": repro, "wer_i_vs_official": wer_check, "table": table, "diffs": diffs}
    sfx = VARIANTS[var]["suffix"]
    (OUT / f"results{sfx}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                            encoding="utf-8")
    with (OUT / f"per_sentence{sfx}.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(per_sent[0]), delimiter="\t")
        w.writeheader()
        w.writerows(per_sent)
    print(f"clean-test {cov['sentences']} câu · từ ref thuộc danh sách {cov['ref_words_in_list']}/{cov['ref_words']} · "
          f"câu có ≥ 1 từ {cov['sentences_with_list_word']} · b = {b:g}")
    print(f"(i) tái lập từng chữ (/{len(refs)}): {repro}")
    print("(i) WER mới / đã báo:", " ".join(f"{k} {a:.4f}/{o:.4f}" for k, (a, o) in wer_check.items()))
    print("WER / B-WER / U-WER (beam+LM, trung bình 3 seed):")
    for m in MODELS:
        print(f"  {MODEL_NAMES[m]:16}", "  ".join(f"({c}) {table[f'{c}|{m}']['wer']:.4f}/{table[f'{c}|{m}']['b_wer']:.4f}/"
                                                f"{table[f'{c}|{m}']['u_wer']:.4f}" for c in conds))
    print("Hiệu so với (i), CI 95% bootstrap hai tầng video × seed:")
    for d in diffs:
        print(f"  {MODEL_NAMES[d['model']]:16} ({d['cond']}) − (i) {d['metric']:5}: {d['diff']:+.4f} "
              f"[{d['ci'][0]:+.4f}; {d['ci'][1]:+.4f}] · từng seed {' '.join(f'{x:+.4f}' for x in d['per_seed_same_ckpt'])}"
              f" · ghép seed [{d['ci_paired_seed'][0]:+.4f}; {d['ci_paired_seed'][1]:+.4f}]")


def cmd_examples() -> None:
    """Ví dụ cho note, luật chọn cố định trước khi xem output (ii)/(iii): seed 42, theo thứ tự manifest — (a) tối đa 4
    câu có từ thuộc danh sách mà (iii) khác (i) ở ít nhất một encoder; (b) 2 câu đầu không có từ thuộc danh sách mà
    (iii) khác (i) (tác dụng phụ)."""
    b = chosen_b()
    refs, lists = clean_test_refs(), read_lists()
    raw = {f"{s['split']}:{s['index']}": (n, s["corrected_text"])
           for n, s in enumerate(json.loads(MANIFEST.read_text(encoding="utf-8"))["samples"], 1)}
    hi, hiii = (read_hyps(HYPS / f"clean_test__{c}__s42.tsv") for c in ("i", cond_label("iii", b)))
    picked = {"có từ danh sách": [], "không có từ danh sách": []}
    for k, v, ref in refs:
        has = any(t[0] in set(lists[v]) for t in ref)
        grp = picked["có từ danh sách" if has else "không có từ danh sách"]
        if len(grp) < (4 if has else 2) and any(hi[k][f"{m}_s42_lm"] != hiii[k][f"{m}_s42_lm"] for m in MODELS):
            grp.append(k)
    for grp, keys in picked.items():
        print(f"### {grp}")
        for k in keys:
            n, text = raw[k]
            print(f"#{n} ({vid_short(refs, k)}) — danh sách: {' '.join(lists[dict((kk, vv) for kk, vv, _ in refs)[k]])}")
            print(f"  ref : {normalize_text(text)}")
            for m in MODELS:
                a, c = hi[k][f"{m}_s42_lm"], hiii[k][f"{m}_s42_lm"]
                print(f"  {MODEL_NAMES[m]:16} (i)  : {normalize_text(a)}")
                if a != c:
                    print(f"  {'':16} (iii): {normalize_text(c)}")


def vid_short(refs, k) -> str:
    return next(v for kk, v, _ in refs if kk == k)[:40]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["lists", "check", "grid", "score-grid", "decode", "score", "examples",
                                    "fetch-pages", "lists-iv"])
    ap.add_argument("--variant", choices=list(VARIANTS), default="iii",
                    help="iii = danh sách từ tên file (A2); iv = + tiêu đề có dấu và mô tả tập (A8)")
    a = ap.parse_args()
    by_var = {"grid": cmd_grid, "score-grid": score_grid, "decode": cmd_decode, "score": cmd_score}
    if a.cmd in by_var:
        by_var[a.cmd](a.variant)
    else:
        {"lists": cmd_lists, "check": cmd_check, "examples": cmd_examples, "fetch-pages": cmd_fetch_pages,
         "lists-iv": cmd_lists_iv}[a.cmd]()


if __name__ == "__main__":
    main()
