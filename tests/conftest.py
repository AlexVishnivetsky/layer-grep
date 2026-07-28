from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def write_file():
    """Write `content` to `root/rel_path`, creating parent dirs as needed, return the Path -
    shared helper for building synthetic mini-projects under tmp_path across test modules."""
    def _write(root: Path, rel_path: str, content: str) -> Path:
        p = root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p
    return _write
