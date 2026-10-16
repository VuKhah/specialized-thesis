"""Kiểm âm tiết tiếng Việt hợp lệ (phụ âm đầu + vần + ≤ 1 thanh).

Chép từ script agent giải mã 2026-10-15 (đếm "từ dính"). Dùng thêm để tách từ không phải âm tiết Việt
(tiếng Anh, tên riêng, viết tắt) khi chấm WER theo nhóm từ. Hạn chế: từ tiếng Anh trùng dạng âm tiết Việt
(*in, to, an, hi*) lọt qua — phải duyệt tay.
"""
import re, unicodedata
TONES = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}
ONSETS = "NGH|NG|NH|CH|GH|GI|KH|PH|QU|TH|TR|B|C|D|Đ|G|H|K|L|M|N|P|R|S|T|V|X"
RHYMES = """A AC ACH AI AM AN ANG ANH AO AP AT AU AY
ĂC ĂM ĂN ĂNG ĂP ĂT ÂC ÂM ÂN ÂNG ÂP ÂT ÂU ÂY
E EC EM EN ENG EO EP ET Ê ÊCH ÊM ÊN ÊNH ÊP ÊT ÊU
YNH YT YCH YN YP I IA ICH IM IN INH IP IT IU IÊC IÊM IÊN IÊNG IÊP IÊT IÊU YÊN YÊT YÊU YÊM YÊNG Y
O OC OI OM ON ONG OP OT OONG OOC Ô ÔC ÔI ÔM ÔN ÔNG ÔP ÔT Ơ ƠI ƠM ƠN ƠP ƠT ƠNG
U UA UC UI UM UN UNG UP UT UÔC UÔI UÔM UÔN UÔNG UÔT UY UYA UYCH UYÊN UYÊT UYN UYNH UYP UYT UYU
UÂN UÂT UÂNG UÂY UÊ UÊCH UÊN UÊNH UƠ UƠN UƠI
OA OAC OACH OAI OAM OAN OANG OANH OAO OAP OAT OAU OAY OĂC OĂM OĂN OĂNG OĂT OE OEN OET OEO OEM
Ư ƯA ƯC ƯI ƯM ƯN ƯNG ƯT ƯU ƯƠC ƯƠI ƯƠM ƯƠN ƯƠNG ƯƠP ƯƠT ƯƠU""".split()
_rx = re.compile(f"^(?:{ONSETS})?(?:{'|'.join(sorted(RHYMES, key=len, reverse=True))})$")
VI_CHARS = set("ĂÂĐÊÔƠƯ")

def strip_tone(w: str):
    d = unicodedata.normalize("NFD", w.upper())
    tones = [c for c in d if c in TONES]
    return unicodedata.normalize("NFC", "".join(c for c in d if c not in TONES)), len(tones)

def is_vi_marked(w: str) -> bool:
    """Từ mang dấu/chữ riêng tiếng Việt → chắc chắn được định là âm tiết Việt (không phải từ tiếng Anh)."""
    base, nt = strip_tone(w)
    return nt > 0 or any(c in VI_CHARS for c in base)

def is_valid_syllable(w: str) -> bool:
    base, nt = strip_tone(w)
    return nt <= 1 and bool(_rx.match(base))

def sticky_words(text: str) -> list[str]:
    return [w for w in text.split() if is_vi_marked(w) and not is_valid_syllable(w)]
