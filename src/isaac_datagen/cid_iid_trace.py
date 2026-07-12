
from __future__ import annotations

from pathlib import Path

_PATH: Path | None = None


def init(render_dir: str | Path) -> None:
    global _PATH
    _PATH = Path(render_dir) / "cid_iid_trace.log"


def enabled() -> bool:
    return _PATH is not None


def log(msg: str) -> None:
    if _PATH is None:
        return
    with _PATH.open("a") as f:
        print(msg, file=f, flush=True)


def is_tuna(name: str, cls: str | None = None) -> bool:
    return cls == "fish can" or "tuna" in name.lower()
