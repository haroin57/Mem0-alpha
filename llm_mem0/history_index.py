"""BM25 keyword index over the per-channel JSONL conversation transcripts.

The Mem0 BM25 index (services/mem0/bm25_index.py) only contains *curated* facts
— text that survived Haiku extraction + the importance/dedup gate. A proper
noun mentioned once in passing ("東洋水産") never becomes a fact, so neither
vector nor fact-BM25 recall can surface it later.

This module indexes the RAW conversation lines instead. Every line written to
``.state/history/{channel_id}.jsonl`` is
mirrored here so a per-turn literal-keyword search can rescue exact terms that
curation dropped. It is the indexed, full-history counterpart to the forensic
``scripts/search_history.py`` (which re-logs-in and linearly scans).

Store: a sqlite FTS5 virtual table at ``STATE_DIR/history_bm25.sqlite`` — a
SEPARATE file from ``mem0_bm25.sqlite`` (the entity-fact index) so the two can
be rebuilt or wiped independently. Tokenizer is ``unicode61`` (no CJK
segmentation, no substring match): callers pre-tokenize through
``the morpho module`` and store the whitespace-joined content words in the
indexed ``text`` column. The verbatim line stays in the UNINDEXED ``orig_text``
column and is what ``search_history`` returns as ``memory``; the indexed
``text`` column is for matching only and must never be surfaced to an LLM.

Scope: search is keyed by ``channel_id`` so a turn only ever sees its own
channel's history. The JSONL has no author id, so the message-handler
integration gates history search to admin callers (see the plan / ADR-0004).

The module is synchronous; callers wrap with ``asyncio.to_thread`` from event
loops. A single ``_db_lock`` serializes writes; FTS5 reads are concurrent-safe
under WAL.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Iterable

log = logging.getLogger(__name__)


_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS history_fts USING fts5(
    channel_id UNINDEXED,
    ts UNINDEXED,
    role UNINDEXED,
    author UNINDEXED,
    orig_text UNINDEXED,
    text,
    tokenize = 'unicode61 remove_diacritics 1'
);
"""

# FTS5 syntax characters that must be stripped from user-supplied tokens before
# they go into a MATCH expression. Mirrors bm25_index.search_bm25.
_FTS_SPECIAL_RE = re.compile(r'["\(\)\*\^:]')

_db_lock = threading.Lock()
_db_path: str | None = None
# Path for which journal_mode=WAL + schema have been applied. journal_mode is a
# persistent DB-level property whose lock is NOT governed by busy_timeout, so
# running `PRAGMA journal_mode=WAL` on every (concurrent) connection made ~30%
# of concurrent index_line() writes fail with "database is locked" — silently
# dropping a history-search FTS row (observed in prod). We now apply it exactly
# once per DB file, serialized under _db_lock. Keyed on path so a test tempfile
# swap or a reindex_all wipe re-initialises automatically.
_wal_ready_path: str | None = None


def _resolve_db_path() -> str:
    global _db_path
    if _db_path is None:
        from .settings import STATE_DIR
        os.makedirs(STATE_DIR, exist_ok=True)
        _db_path = os.path.join(STATE_DIR, "history_bm25.sqlite")
    return _db_path


def _reset_db_path_for_test(path: str | None) -> None:
    """Test hook — pytest fixtures point the module at a tempfile."""
    global _db_path, _wal_ready_path
    _db_path = path
    _wal_ready_path = None  # new DB file → re-init WAL + schema on next _conn()


@contextmanager
def _conn():
    global _wal_ready_path
    path = _resolve_db_path()
    # timeout=5.0 makes sqlite3 wait up to 5s for a write lock. busy_timeout is
    # the SQLite-level equivalent, set on every connection (it is connection-
    # local and lock-free, so it never contends). journal_mode=WAL, by contrast,
    # is DB-level and its lock IGNORES busy_timeout — running it concurrently was
    # the "database is locked" row-drop bug (ADR-0007). Apply WAL + schema once
    # per DB file, serialized under _db_lock, with a double-check so exactly one
    # thread runs it.
    c = sqlite3.connect(path, check_same_thread=False, timeout=5.0)
    c.row_factory = sqlite3.Row
    try:
        c.execute("PRAGMA busy_timeout=5000")
        if _wal_ready_path != path:
            with _db_lock:
                if _wal_ready_path != path:
                    c.execute("PRAGMA journal_mode=WAL")
                    c.executescript(_SCHEMA)
                    _wal_ready_path = path
        yield c
    finally:
        c.close()


def _tokenize(text: str) -> str:
    try:
        from . import morpho
        tok = morpho.tokenize_for_index(text)
    except Exception as exc:
        log.warning("history morpho tokenize_for_index failed, raw fallback: %s", exc)
        tok = text
    return tok or (text or "").strip()


def index_line(channel_id: int, ts: str, role: str, text: str, author: str = "") -> bool:
    """Insert one conversation line into the FTS5 index.

    Idempotent on the exact line: a DELETE on ``(channel_id, ts, role,
    orig_text)`` runs first, so re-indexing the same line (e.g. a backfill
    re-run overlapping live writes) does not create duplicates, while two
    distinct messages that happen to share a second-granularity timestamp are
    both kept.

    ``role`` is "user"/"assistant" for answered turns or "channel" for a
    message the app observed but did not reply to. ``author`` is the
    speaker's display name (mainly meaningful for "channel" rows, where the
    message can come from anyone in the channel).

    ``text`` is the verbatim line; it is stored in ``orig_text`` and ALSO
    morphologically tokenized into the indexed ``text`` column. Particles /
    fillers / 1-char noise are stripped by morpho.tokenize_for_index.
    """
    if channel_id is None or not text or not role:
        return False
    ts = ts or ""
    author = author or ""
    tokenized = _tokenize(text)
    # Retry on transient "database is locked": WAL still serializes writers and
    # a checkpoint can briefly hold the lock past busy_timeout. Bounded (4 tries,
    # ~0.05/0.1/0.15s backoff) so a genuinely stuck DB returns False rather than
    # blocking the caller. Belt-and-suspenders on top of the WAL-once fix.
    for attempt in range(4):
        try:
            with _conn() as c, _db_lock:
                c.execute(
                    "DELETE FROM history_fts WHERE channel_id = ? AND ts = ? "
                    "AND role = ? AND orig_text = ?",
                    (str(channel_id), ts, role, text),
                )
                c.execute(
                    "INSERT INTO history_fts(channel_id, ts, role, author, orig_text, text) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (str(channel_id), ts, role, author, text, tokenized),
                )
                c.commit()
                return True
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() and attempt < 3:
                time.sleep(0.05 * (attempt + 1))
                continue
            log.warning("history index_line(ch=%s) failed: %s", channel_id, exc)
            return False
        except Exception as exc:
            log.warning("history index_line(ch=%s) failed: %s", channel_id, exc)
            return False
    return False


def search_history(query: str, channel_id: int, limit: int = 5) -> list[dict]:
    """BM25 keyword search over one channel's conversation history.

    Returns ``[{"role": str, "ts": str, "memory": str, "rank": int,
    "bm25_score": float}, ...]`` ordered by rank (1 = most relevant).

    The query is tokenized via Janome (morpho.tokenize_keywords) so each FTS5
    disjunct is a single content word; particles / symbols / 1-char noise are
    dropped. Matching is exact-token (unicode61), not substring — the stored
    line was tokenized the same way, so "東洋水産" matches the standalone token
    indexed from a longer line. Falls back to a punctuation split if Janome is
    unavailable.
    """
    q = (query or "").strip()
    if not q or channel_id is None:
        return []
    try:
        from . import morpho
        tokens = morpho.tokenize_keywords(q)
    except Exception as exc:
        log.warning("history morpho tokenize failed, falling back: %s", exc)
        tokens = []
    if not tokens:
        cleaned = _FTS_SPECIAL_RE.sub(" ", q)
        sub_phrases = re.split(r"[\s、。？！,.?!:;~〜\-—]+", cleaned)
        tokens = [t.strip() for t in sub_phrases if len(t.strip()) >= 2]
    safe_tokens = [_FTS_SPECIAL_RE.sub("", t) for t in tokens[:10]]
    safe_tokens = [t for t in safe_tokens if t]
    if not safe_tokens:
        return []
    fts_query = " OR ".join(f'"{t}"' for t in safe_tokens)
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT role, ts, author, orig_text, bm25(history_fts) AS score "
                "FROM history_fts "
                "WHERE history_fts MATCH ? AND channel_id = ? "
                "ORDER BY bm25(history_fts) "
                "LIMIT ?",
                (fts_query, str(channel_id), max(int(limit), 1)),
            ).fetchall()
    except sqlite3.OperationalError as exc:
        log.warning("history search OperationalError: %s (q=%r)", exc, q[:80])
        return []
    except Exception as exc:
        log.warning("history search failed: %s", exc)
        return []
    out: list[dict] = []
    for i, r in enumerate(rows):
        out.append({
            "role": r["role"],
            "ts": r["ts"],
            "author": r["author"] if "author" in r.keys() else "",
            # Return orig_text verbatim, never the tokenized `text` column.
            "memory": r["orig_text"],
            "rank": i + 1,
            "bm25_score": float(r["score"]),
        })
    return out


def reindex_all(lines: Iterable[tuple[int, str, str, str, str]]) -> int:
    """Wipe-and-rebuild the index from an iterable of
    (channel_id, ts, role, text, author).

    Used by ``scripts/build_history_bm25_index.py`` after reading every
    ``.state/history/*.jsonl``. SQLite virtual tables can't change their
    ``tokenize=``/column set after creation, so we unlink the sqlite + WAL/SHM
    siblings and let the next ``_conn()`` recreate them under the current
    ``_SCHEMA``. Returns the row count inserted.
    """
    global _wal_ready_path
    path = _resolve_db_path()
    with _db_lock:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(path + suffix)
            except FileNotFoundError:
                pass
            except OSError as exc:
                log.warning("history reindex_all: unlink %s%s failed: %s",
                            path, suffix, exc)
        _wal_ready_path = None  # DB wiped → next _conn() recreates WAL + schema

    count = 0
    try:
        with _conn() as c, _db_lock:
            for row in lines:
                channel_id, ts, role, text = row[0], row[1], row[2], row[3]
                author = row[4] if len(row) > 4 else ""
                if channel_id is None or not text or not role:
                    continue
                tokenized = _tokenize(text)
                if not tokenized:
                    tokenized = text  # never index an empty row
                c.execute(
                    "INSERT INTO history_fts(channel_id, ts, role, author, orig_text, text) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (str(channel_id), ts or "", role, author or "", text, tokenized),
                )
                count += 1
            c.commit()
    except Exception as exc:
        log.warning("history reindex_all failed at %d: %s", count, exc)
    return count


def stats() -> dict:
    try:
        with _conn() as c:
            total = c.execute("SELECT count(*) FROM history_fts").fetchone()[0]
    except Exception:
        total = -1
    return {"rows": total, "db_path": _resolve_db_path()}
