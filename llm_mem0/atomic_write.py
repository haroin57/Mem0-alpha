"""Atomic file write helpers.

Pattern: write to a uuid-suffixed temp file in the same directory, fsync,
then os.replace() onto the target. The replace step is atomic on POSIX,
so concurrent readers either see the old contents or the new contents —
never a half-written file. uuid suffix avoids name collisions when two
writers race for the same target.

Three flavors mirror the typical payloads in this codebase:
- atomic_write_json: dict/list → JSON file (with indent)
- atomic_write_text: str → text file
- atomic_write_bytes: bytes → binary file

All three use mode 0o600 by default — most state files in `.state/` hold
session IDs, OAuth credentials, or other secrets, so world-readable
defaults would be a foot-gun.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

__all__ = [
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
]


def _tmp_path(target: Path) -> Path:
    return target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")


def atomic_write_bytes(
    path: str | os.PathLike,
    data: bytes,
    *,
    mode: int = 0o600,
) -> None:
    """Atomically write `data` to `path` with file mode `mode`."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(target)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    raw_fd = os.open(tmp, flags, mode)
    fd: int | None = raw_fd
    try:
        with os.fdopen(raw_fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        fd = None  # fdopen took ownership
        os.replace(tmp, target)
    except Exception:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_text(
    path: str | os.PathLike,
    text: str,
    *,
    mode: int = 0o600,
    encoding: str = "utf-8",
) -> None:
    """Atomically write `text` to `path`. Convenience wrapper around bytes."""
    atomic_write_bytes(path, text.encode(encoding), mode=mode)


def atomic_write_json(
    path: str | os.PathLike,
    data: Any,
    *,
    mode: int = 0o600,
    indent: int | None = 2,
    ensure_ascii: bool = False,
) -> None:
    """Atomically write `data` as JSON to `path`."""
    payload = json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)
    atomic_write_text(path, payload, mode=mode)
