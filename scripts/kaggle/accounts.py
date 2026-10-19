"""Tài khoản Kaggle A–E cho các launcher (2026-10-19).

Repo public → **không** ghi username hay thư mục khoá vào file track. Ánh xạ tài khoản → thư mục `KAGGLE_CONFIG_DIR`
nằm ở `scripts/kaggle/accounts.local.json` (gitignore, mẫu: `accounts.example.json`); username đọc từ trường
`username` của `kaggle.json` lúc chạy. Chuỗi tham chiếu tài nguyên Kaggle viết dạng `"{A}/vss-asr-val"` rồi `ref()`.
"""

import json
import os
import re
from functools import cache
from pathlib import Path

LOCAL = Path(__file__).resolve().with_name("accounts.local.json")


@cache
def _dirs() -> dict[str, Path]:
    if not LOCAL.exists():
        raise SystemExit(f"LỖI: thiếu {LOCAL} — chép accounts.example.json rồi điền thư mục khoá từng tài khoản")
    return {k: Path(v).expanduser() for k, v in json.loads(LOCAL.read_text(encoding="utf-8")).items()}


def names() -> list[str]:
    return list(_dirs())


def config_dir(acc: str) -> Path:
    return _dirs()[acc]


@cache
def username(acc: str) -> str:
    return json.loads((config_dir(acc) / "kaggle.json").read_text(encoding="utf-8"))["username"]


def owner(ref_: str) -> str | None:
    """Tài khoản sở hữu một tham chiếu đã giải (`<username>/<slug>`)."""
    return next((a for a in names() if username(a) == ref_.split("/")[0]), None)


def ref(template: str) -> str:
    """`"{A}/vss-asr-val"` → `"<username A>/vss-asr-val"`."""
    return re.sub(r"\{([A-Z])\}", lambda m: username(m.group(1)), template)


def env(acc: str) -> dict[str, str]:
    return {**os.environ, "PYTHONUTF8": "1", "KAGGLE_CONFIG_DIR": str(config_dir(acc))}


def mask(text: str) -> str:
    """Thay mọi username bằng `<A>`… trước khi in/ghi log có thể bị chép vào file track."""
    for a in names():
        if (config_dir(a) / "kaggle.json").exists():
            text = text.replace(username(a), f"<{a}>")
    return text
