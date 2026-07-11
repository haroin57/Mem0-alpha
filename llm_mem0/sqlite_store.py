"""Shared SQLite plumbing for the local index stores.

graph / bm25_index / fact_store / history_index each keep their own sqlite
file under ``STATE_DIR``, and each used to hand-roll the same connection
boilerplate. Worse, three of them re-ran ``PRAGMA journal_mode=WAL`` on
every connection — a DB-level operation whose lock IGNORES busy_timeout, so
~30% of concurrent writes failed with "database is locked" and silently
dropped rows (the bug history_index fixed and the others never inherited).
This module is that fixed pattern, once:

- ``journal_mode=WAL`` + schema run exactly once per DB file, serialized
  under the store lock with a double-check.
- ``busy_timeout=5000`` set per connection (connection-local, lock-free).
- Writes go through :meth:`SqliteStore.write` — serialized under the store
  lock, committed, and retried with a short backoff when a WAL checkpoint
  briefly holds the lock past busy_timeout.

The stores stay synchronous; async callers wrap with ``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

_WRITE_RETRIES = 4


class SqliteStore:
    def __init__(self, filename: str, schema: str, *, timeout: float = 5.0):
        self._filename = filename
        self._schema = schema
        self._timeout = timeout
        # Serializes writes and the one-time WAL/schema init.
        self.lock = threading.Lock()
        self._path: str | None = None
        # Path for which journal_mode=WAL + schema have been applied. Keyed on
        # path so a test tempfile swap or a wipe() re-initialises automatically.
        self._wal_ready_path: str | None = None

    def resolve_path(self) -> str:
        if self._path is None:
            from .settings import settings
            self._path = os.path.join(settings.STATE_DIR, self._filename)
        return self._path

    def reset_path_for_test(self, path: str | None) -> None:
        """Test hook — pytest fixtures point the store at a tempfile."""
        self._path = path
        self._wal_ready_path = None  # new DB file → re-init WAL + schema

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        path = self.resolve_path()
        c = sqlite3.connect(path, check_same_thread=False, timeout=self._timeout)
        c.row_factory = sqlite3.Row
        try:
            c.execute(f"PRAGMA busy_timeout={int(self._timeout * 1000)}")
            c.execute("PRAGMA foreign_keys=ON")
            if self._wal_ready_path != path:
                with self.lock:
                    if self._wal_ready_path != path:
                        c.execute("PRAGMA journal_mode=WAL")
                        c.executescript(self._schema)
                        self._wal_ready_path = path
            yield c
        finally:
            c.close()

    def write(
        self,
        fn: Callable[[sqlite3.Connection], T],
        *,
        what: str = "write",
        default: T | None = None,
    ) -> T | None:
        """Run ``fn`` under the store lock and commit; retry transient locks.

        Returns ``fn``'s result, or ``default`` when every attempt failed
        (the stores are best-effort indices — a failed write must never take
        down ingestion). ``what`` labels the WARNING for grep-ability.
        """
        for attempt in range(_WRITE_RETRIES):
            try:
                with self.connect() as c, self.lock:
                    result = fn(c)
                    c.commit()
                    return result
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() and attempt < _WRITE_RETRIES - 1:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                log.warning("%s.%s failed: %s", self._filename, what, exc)
                return default
            except Exception as exc:
                log.warning("%s.%s failed: %s", self._filename, what, exc)
                return default
        return default

    def read(
        self,
        fn: Callable[[sqlite3.Connection], T],
        *,
        what: str = "read",
        default: T | None = None,
    ) -> T | None:
        """Run a read-only ``fn`` on a fresh connection; ``default`` on error."""
        try:
            with self.connect() as c:
                return fn(c)
        except Exception as exc:
            log.warning("%s.%s failed: %s", self._filename, what, exc)
            return default

    def wipe(self) -> None:
        """Unlink the DB and its WAL/SHM siblings; next connect() recreates.

        Needed by reindex flows: an FTS5 virtual table can't change its
        ``tokenize=``/column set after creation, so a rebuild replaces the file.
        """
        path = self.resolve_path()
        with self.lock:
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.unlink(path + suffix)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    log.warning("%s.wipe: unlink %s%s failed: %s",
                                self._filename, path, suffix, exc)
            self._wal_ready_path = None


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


# FTS5 syntax characters that must be stripped from user-supplied tokens
# before they go into a MATCH expression.
_FTS_SPECIAL_RE = re.compile(r'["\(\)\*\^:]')
_FTS_SPLIT_RE = re.compile(r"[\s、。？！,.?!:;~〜\-—]+")


def build_fts_match_query(query: str, *, max_tokens: int = 10) -> str:
    """Turn a natural-language query into an OR-of-phrases FTS5 MATCH string.

    Tokenizes via Janome (morpho.tokenize_keywords) so each disjunct is a
    single content word; particles / symbols / 1-char noise are dropped.
    Falls back to a punctuation split when Janome is unavailable. Returns
    ``""`` when nothing survives — callers should treat that as no-hits.
    """
    q = (query or "").strip()
    if not q:
        return ""
    try:
        from . import morpho
        tokens = morpho.tokenize_keywords(q)
    except Exception as exc:
        log.warning("fts morpho tokenize failed, falling back: %s", exc)
        tokens = []
    if not tokens:
        cleaned = _FTS_SPECIAL_RE.sub(" ", q)
        tokens = [t.strip() for t in _FTS_SPLIT_RE.split(cleaned)
                  if len(t.strip()) >= 2]
    safe = [_FTS_SPECIAL_RE.sub("", t) for t in tokens[:max_tokens]]
    safe = [t for t in safe if t]
    return " OR ".join(f'"{t}"' for t in safe)
