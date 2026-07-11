"""BM25 keyword index over Mem0 facts (Phase A-1 of mem0-recall-hybrid-hyde-kg).

Vector cosine retrieval is semantically powerful but doesn't guarantee recall
for proper nouns and technical terms. Adding BM25 as a parallel retriever and
fusing the two ranked lists via Reciprocal Rank Fusion lets exact-term matches
rescue facts that cosine missed (e.g. a proper noun like "Deckard" being pulled
toward other characters in embedding space).

The store is a sqlite FTS5 virtual table at ``STATE_DIR/mem0_bm25.sqlite`` —
kept in a separate file from the entity graph so the two indices can be
rebuilt or wiped independently. The tokenizer is ``unicode61`` (NOT trigram):
it has no CJK word segmentation and does not substring-match, so ``index_fact``
pre-tokenizes each fact through ``the morpho module`` and stores the
whitespace-joined content words in the indexed ``text`` column — unicode61's
whitespace split then yields one FTS5 token per Japanese content word. The
original sentence is kept verbatim in the UNINDEXED ``orig_text`` column and is
what ``search_bm25`` returns as ``memory``; the indexed ``text`` column is for
matching only and must never be surfaced to a consumer/LLM.

The module is synchronous; callers wrap with ``asyncio.to_thread`` from event
loops. Connection plumbing (WAL-once init, locked writes with retry) lives in
:mod:`llm_mem0.sqlite_store`.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Iterable

from .sqlite_store import SqliteStore, build_fts_match_query

log = logging.getLogger(__name__)


_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    fact_id UNINDEXED,
    user_id UNINDEXED,
    orig_text UNINDEXED,
    text,
    tokenize = 'unicode61 remove_diacritics 1'
);
"""

_store = SqliteStore("mem0_bm25.sqlite", _SCHEMA)


def _reset_db_path_for_test(path: str | None) -> None:
    """Test hook — pytest fixtures point the module at a tempfile."""
    _store.reset_path_for_test(path)


def _tokenize(text: str) -> str:
    try:
        from . import morpho
        tokenized = morpho.tokenize_for_index(text)
    except Exception as exc:
        log.warning(
            "bm25 morpho tokenize_for_index failed, falling back to raw: %s", exc,
        )
        tokenized = text
    return tokenized or text  # never index an empty row


def index_fact(fact_id: str, text: str, user_id: str) -> bool:
    """Insert (or replace) a fact in the FTS5 index. Idempotent on fact_id.

    The indexed ``text`` column is the morphological tokenization of the
    source fact — whitespace-separated content-word surface forms — so the
    underlying unicode61 tokenizer (which splits on whitespace) yields
    one FTS5 token per Japanese content word. Particles, fillers, and
    1-character noise are stripped by morpho.tokenize_for_index.

    The raw ``text`` argument is ALSO stored verbatim in the UNINDEXED
    ``orig_text`` column and is what ``search_bm25`` returns as ``memory``.
    Callers MUST pass the original fact sentence here, never a pre-tokenized
    string — otherwise ``orig_text`` would hold tokenized garbage too.
    """
    if not fact_id or not text or not user_id:
        return False
    tokenized = _tokenize(text)

    def _insert(c) -> bool:
        c.execute("DELETE FROM facts_fts WHERE fact_id = ?", (fact_id,))
        c.execute(
            "INSERT INTO facts_fts(fact_id, user_id, orig_text, text) "
            "VALUES (?, ?, ?, ?)",
            (fact_id, user_id, text, tokenized),
        )
        return True

    return bool(_store.write(_insert, what="index_fact", default=False))


def remove_fact(fact_id: str) -> int:
    """Drop a fact from the FTS5 index. Returns rowcount removed."""
    if not fact_id:
        return 0

    def _delete(c) -> int:
        cur = c.execute("DELETE FROM facts_fts WHERE fact_id = ?", (fact_id,))
        return cur.rowcount or 0

    return _store.write(_delete, what="remove_fact", default=0) or 0


def search_bm25(query: str, user_id: str, limit: int = 20) -> list[dict]:
    """BM25 keyword search scoped to ``user_id``.

    Returns ``[{"id": fact_id, "rank": int, "bm25_score": float, "memory": text}, ...]``
    ordered by rank (rank=1 most relevant). ``bm25_score`` from sqlite's
    ``bm25()`` is "smaller is better"; callers that just need ranking can
    ignore the absolute value.

    The query goes through :func:`build_fts_match_query`: content-word
    tokenization (Janome when available, punctuation split otherwise), FTS5
    syntax characters stripped, and the survivors OR-joined as phrases.
    Matching is exact-token (unicode61), not substring: morpho splits both
    the stored fact and the query into the same content words, so
    "東京タワー" → "東京" / "タワー" matches the standalone tokens indexed
    from a longer fact.
    """
    q = (query or "").strip()
    if not q or not user_id:
        return []
    fts_query = build_fts_match_query(q)
    if not fts_query:
        return []

    def _search(c) -> list[dict]:
        rows = c.execute(
            "SELECT fact_id, orig_text, bm25(facts_fts) AS score "
            "FROM facts_fts "
            "WHERE facts_fts MATCH ? AND user_id = ? "
            "ORDER BY bm25(facts_fts) "
            "LIMIT ?",
            (fts_query, user_id, max(int(limit), 1)),
        ).fetchall()
        return [
            {
                "id": r["fact_id"],
                # Return the original sentence (orig_text), never the
                # tokenized `text` column — the latter is for FTS5 matching
                # only and would surface as keyword-soup to the LLM reranker.
                "memory": r["orig_text"],
                "rank": i + 1,
                "bm25_score": float(r["score"]),
            }
            for i, r in enumerate(rows)
        ]

    try:
        return _store.read(_search, what="search_bm25", default=[]) or []
    except sqlite3.OperationalError as exc:  # pragma: no cover - defensive
        log.warning("bm25 search OperationalError: %s (q=%r)", exc, q[:80])
        return []


def rrf_merge(
    vector_hits: list[dict],
    bm25_hits: list[dict],
    *,
    k: int = 60,
    limit: int = 20,
) -> list[dict]:
    """Reciprocal Rank Fusion of two ranked lists.

    Each candidate's RRF score is ``Σ 1 / (k + rank)`` across the lists it
    appears in. ``k=60`` is the value originally proposed by Cormack et al.
    (2009) and is the de-facto default across hybrid-search literature.

    Vector hits are expected to carry ``score`` (cosine distance, small =
    relevant). We compute rank from their ordering in the input list — they
    should already be sorted ascending by score upstream. BM25 hits already
    carry a ``rank`` field.

    The merged output keeps the richest representation per id: cosine
    ``score`` if present (so the LLM rerank's existing reinforcement-boost
    math keeps working), otherwise a neutral mid-range placeholder so
    downstream filters don't drop it.
    """
    fused: dict[str, dict] = {}

    def _bump(item: dict, rrf_rank: int) -> None:
        fid = item.get("id") or item.get("memory") or ""
        if not fid:
            return
        score_inc = 1.0 / (k + rrf_rank)
        existing = fused.get(fid)
        if existing is None:
            # Copy so we don't mutate caller's dict.
            merged = dict(item)
            merged["rrf_score"] = score_inc
            # Provide a neutral cosine distance if missing — graph-expand
            # downstream uses 0.8 for the same purpose; we match that.
            if merged.get("score") is None:
                merged["score"] = 0.8
            fused[fid] = merged
        else:
            existing["rrf_score"] = existing.get("rrf_score", 0.0) + score_inc
            # Prefer the cosine score from vector_hits over the placeholder.
            if existing.get("score") in (None, 0.8) and item.get("score") is not None:
                existing["score"] = item["score"]

    # INVARIANT: `memory` must be the richest representation per id — the
    # ChromaDB full sentence, never the FTS5 tokenized form. vector_hits are
    # bumped BEFORE bm25_hits and the existing-id branch in _bump never
    # overwrites `memory`, so a fact that also hit via vector keeps its full
    # text. search_bm25 returns orig_text (the raw sentence) as memory,
    # so a BM25-only id is safe too. Do NOT re-point search_bm25 at the
    # tokenized `text` column or this invariant breaks silently.
    for i, item in enumerate(vector_hits or []):
        _bump(item, i + 1)
    for item in bm25_hits or []:
        r = int(item.get("rank") or 1)
        _bump(item, r)

    # Descending by RRF score — bigger = more relevant.
    merged = sorted(fused.values(), key=lambda x: -x.get("rrf_score", 0.0))
    return merged[: max(int(limit), 1)]


def reindex_all(facts: Iterable[tuple[str, str, str]]) -> int:
    """Wipe-and-rebuild the FTS5 index from an iterable of (fact_id, text, user_id).

    Used by index-rebuild scripts after dumping every Mem0 fact from
    ChromaDB. Each fact's text is morphologically tokenized before
    insertion. Returns the row count.

    SQLite virtual tables can't change their ``tokenize=`` option after
    creation — so when ``_SCHEMA`` changes we must wipe the underlying file,
    not just DELETE the rows; the next connection recreates everything from
    the current schema.
    """
    _store.wipe()

    def _insert_all(c) -> int:
        count = 0
        for fid, text, uid in facts:
            if not fid or not text or not uid:
                continue
            c.execute(
                "INSERT INTO facts_fts(fact_id, user_id, orig_text, text) "
                "VALUES (?, ?, ?, ?)",
                (fid, uid, text, _tokenize(text)),
            )
            count += 1
        return count

    return _store.write(_insert_all, what="reindex_all", default=0) or 0


def stats() -> dict:
    total = _store.read(
        lambda c: c.execute("SELECT count(*) FROM facts_fts").fetchone()[0],
        what="stats", default=-1,
    )
    return {"rows": total, "db_path": _store.resolve_path()}
