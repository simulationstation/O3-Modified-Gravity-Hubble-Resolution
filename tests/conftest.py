from __future__ import annotations

import sys
from pathlib import Path


def _ensure_local_src_first() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if src.is_dir():
        s = str(src)
        if s not in sys.path:
            sys.path.insert(0, s)


_ensure_local_src_first()
