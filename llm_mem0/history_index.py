"""BM25 keyword index over the per-channel JSONL conversation transcripts.

The Mem0 BM25 index (bm25_index.py) only contains *curated* facts — text
that survived extraction + the importance/dedup gate. A proper noun
mentioned once in passing ("東洋水産") never becomes a fact, so neither
vector nor fact-BM25 recall can surface it later.

This module indexes the RAW conversation lines instead. Every line the host
application logs can be mirrored here so a per-turn literal-keyword search
can rescue exact terms that curation dropped.

Store: a sqlite FTS5 virtual table at ``STATE_DIR/history_bm25.sqlite`` — a
SEPARATE file from ``mem0_bm25.sqlite`` (the entity-fact index) so the two
can be rebuilt or wiped independently. Tokenizer is ``unicode61`` (no CJK
segmentation, no substring match): callers pre-tokenize through the morpho
module and store the whitespace-joined content words in the indexed
``text`` column. The verbatim line stays in the UNINDEXED ``orig_text``
column and is what ``search_history`` returns as ``memory``; the indexed
``text`` column is for matching only and must never be surfaced to an LLM.

Scope: search is keyed by ``channel_id`` so a turn only ever sees its own
channel's history. The transcript has no author id, so host integrations
should gate history search to trusted callers.

The module is synchronous; callers wrap with ``asyncio.to_thread`` from
event loops. Connection plumbing (WAL-once init — the ADR-0007 fix for the
~30% concurrent-write row-drop bug — busy_timeout, locked writes with
retry) lives in :mod:`llm_mem0.sqlite_store`.
"""

from __future__ import annotations

import logging
from typing import Iterable

from .sqlite_store import SqliteStore, build_fts_match_query

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

_store = SqliteStore("history_bm25.sqlite", _SCHEMA)


def _reset_db_path_for_test(path: str | None) -> None:
    """Test hook — pytest fixtures point the module at a tempfile."""
    _store.reset_path_for_test(path)


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

    def _insert(c) -> bool:
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
        return True

    return bool(_store.write(_insert, what="index_line", default=False))


def search_history(query: str, channel_id: int, limit: int = 5) -> list[dict]:
    """BM25 keyword search over one channel's conversation history.

    Returns ``[{"role": str, "ts": str, "memory": str, "rank": int,
    "bm25_score": float}, ...]`` ordered by rank (1 = most relevant).

    The query goes through :func:`build_fts_match_query` (Janome content
    words when available, punctuation split fallback). Matching is
    exact-token (unicode61), not substring — the stored line was tokenized
    the same way, so "東洋水産" matches the standalone token indexed from a
    longer line.
    """
    q = (query or "").strip()
    if not q or channel_id is None:
        return []
    fts_query = build_fts_match_query(q)
    if not fts_query:
        return []

    def _search(c) -> list[dict]:
        rows = c.execute(
            "SELECT role, ts, author, orig_text, bm25(history_fts) AS score "
            "FROM history_fts "
            "WHERE history_fts MATCH ? AND channel_id = ? "
            "ORDER BY bm25(history_fts) "
            "LIMIT ?",
            (fts_query, str(channel_id), max(int(limit), 1)),
        ).fetchall()
        return [
            {
                "role": r["role"],
                "ts": r["ts"],
                "author": r["author"] if "author" in r.keys() else "",
                # Return orig_text verbatim, never the tokenized `text` column.
                "memory": r["orig_text"],
                "rank": i + 1,
                "bm25_score": float(r["score"]),
            }
            for i, r in enumerate(rows)
        ]

    return _store.read(_search, what="search_history", default=[]) or []


def get_lines_since(channel_id: int, since_ts: str = "", limit: int = 500) -> list[dict]:
    """Return raw conversation lines for one channel, oldest first, strictly
    after ``since_ts`` (lexical comparison — safe because lines are indexed
    with the ISO-8601-ish, zero-padded ``strftime`` format the JSONL writer
    uses).

    Used by ``replay.py``'s replay job to walk history in chronological
    order without depending on the source transcript files directly — this
    module already receives every line via ``index_line`` and the FTS5
    table holds each line's verbatim ``orig_text``, so replay can read
    entirely from here. Not a MATCH query (no keyword filter), so this is a
    plain indexed scan — fine for a nightly batch job, not meant for the hot
    search path.
    """
    if channel_id is None:
        return []

    def _fetch(c) -> list[dict]:
        if since_ts:
            rows = c.execute(
                "SELECT ts, role, author, orig_text FROM history_fts "
                "WHERE channel_id = ? AND ts > ? ORDER BY ts ASC LIMIT ?",
                (str(channel_id), since_ts, max(int(limit), 1)),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT ts, role, author, orig_text FROM history_fts "
                "WHERE channel_id = ? ORDER BY ts ASC LIMIT ?",
                (str(channel_id), max(int(limit), 1)),
            ).fetchall()
        return [
            {"ts": r["ts"], "role": r["role"], "author": r["author"],
             "text": r["orig_text"]}
            for r in rows
        ]

    return _store.read(_fetch, what="get_lines_since", default=[]) or []


def reindex_all(lines: Iterable[tuple[int, str, str, str, str]]) -> int:
    """Wipe-and-rebuild the index from an iterable of
    (channel_id, ts, role, text, author).

    SQLite virtual tables can't change their ``tokenize=``/column set after
    creation, so the underlying file is replaced and the next connection
    recreates it under the current ``_SCHEMA``. Returns the row count.
    """
    _store.wipe()

    def _insert_all(c) -> int:
        count = 0
        for row in lines:
            channel_id, ts, role, text = row[0], row[1], row[2], row[3]
            author = row[4] if len(row) > 4 else ""
            if channel_id is None or not text or not role:
                continue
            c.execute(
                "INSERT INTO history_fts(channel_id, ts, role, author, orig_text, text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(channel_id), ts or "", role, author or "", text, _tokenize(text)),
            )
            count += 1
        return count

    return _store.write(_insert_all, what="reindex_all", default=0) or 0


def stats() -> dict:
    total = _store.read(
        lambda c: c.execute("SELECT count(*) FROM history_fts").fetchone()[0],
        what="stats", default=-1,
    )
    return {"rows": total, "db_path": _store.resolve_path()}
