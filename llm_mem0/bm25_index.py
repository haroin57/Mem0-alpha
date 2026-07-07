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
loops. A single ``_db_lock`` serializes writes; FTS5 reads are concurrent-safe
under WAL.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterable

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


_db_lock = threading.Lock()
_db_path: str | None = None


def _resolve_db_path() -> str:
    global _db_path
    if _db_path is None:
        from .settings import STATE_DIR
        os.makedirs(STATE_DIR, exist_ok=True)
        _db_path = os.path.join(STATE_DIR, "mem0_bm25.sqlite")
    return _db_path


def _reset_db_path_for_test(path: str | None) -> None:
    """Test hook — pytest fixtures point the module at a tempfile."""
    global _db_path
    _db_path = path


@contextmanager
def _conn():
    path = _resolve_db_path()
    c = sqlite3.connect(path, check_same_thread=False)
    c.row_factory = sqlite3.Row
    try:
        c.execute("PRAGMA journal_mode=WAL")
        with _db_lock:
            c.executescript(_SCHEMA)
        yield c
    finally:
        c.close()


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
    try:
        from . import morpho
        tokenized = morpho.tokenize_for_index(text)
    except Exception as exc:
        log.warning(
            "bm25 morpho tokenize_for_index failed, falling back to raw: %s", exc,
        )
        tokenized = text
    try:
        with _conn() as c, _db_lock:
            c.execute(
                "DELETE FROM facts_fts WHERE fact_id = ?", (fact_id,),
            )
            c.execute(
                "INSERT INTO facts_fts(fact_id, user_id, orig_text, text) "
                "VALUES (?, ?, ?, ?)",
                (fact_id, user_id, text, tokenized),
            )
            c.commit()
            return True
    except Exception as exc:
        log.warning("bm25 index_fact(%s) failed: %s", fact_id, exc)
        return False


def remove_fact(fact_id: str) -> int:
    """Drop a fact from the FTS5 index. Returns rowcount removed."""
    if not fact_id:
        return 0
    try:
        with _conn() as c, _db_lock:
            cur = c.execute(
                "DELETE FROM facts_fts WHERE fact_id = ?", (fact_id,),
            )
            c.commit()
            return cur.rowcount or 0
    except Exception as exc:
        log.warning("bm25 remove_fact(%s) failed: %s", fact_id, exc)
        return 0


def search_bm25(query: str, user_id: str, limit: int = 20) -> list[dict]:
    """BM25 keyword search scoped to ``user_id``.

    Returns ``[{"id": fact_id, "rank": int, "bm25_score": float, "memory": text}, ...]``
    ordered by rank (rank=1 most relevant). ``bm25_score`` from sqlite's
    ``bm25()`` is "smaller is better"; callers that just need ranking can
    ignore the absolute value.

    Sanitization: FTS5 has a small DSL ('AND', 'OR', '"phrase"', column
    filters). Wrapping the whole query in one big "phrase" only works for
    short proper nouns — for natural-language questions ("東京タワーの高さは
    何メートル？") no stored fact contains the entire phrase verbatim, so
    BM25 returns zero hits. We instead split on Japanese + ASCII
    punctuation/whitespace, drop FTS5 special characters, and OR-join
    the resulting sub-phrases. Matching is exact-token (unicode61), not
    substring: morpho splits both the stored fact and the query into the
    same content words, so "東京タワー" → "東京" / "タワー" matches the
    standalone tokens indexed from a longer fact.
    """
    q = (query or "").strip()
    if not q or not user_id:
        return []
    # Phase A-1 follow-up: tokenize the query via Janome so each FTS5
    # disjunct is a single content word ("東京" / "タワー" / "好き"),
    # not a chunk of natural-language soup. Particles, symbols, and
    # 1-char noise are stripped upstream by morpho.tokenize_keywords.
    try:
        from . import morpho
        tokens = morpho.tokenize_keywords(q)
    except Exception as exc:
        log.warning("bm25 morpho tokenize failed, falling back: %s", exc)
        tokens = []
    # Fallback when Janome is unavailable: split on whitespace +
    # Japanese punctuation, drop FTS5 syntax chars, keep tokens >=2 chars.
    if not tokens:
        import re as _re
        cleaned = _re.sub(r'["\(\)\*\^:]', " ", q)
        sub_phrases = _re.split(r"[\s、。？！,.?!:;~〜\-—]+", cleaned)
        tokens = [t.strip() for t in sub_phrases if len(t.strip()) >= 2]
    if not tokens:
        return []
    # Strip FTS5 syntax inside each token in case the morpho layer let
    # one through (e.g. embedded paren in an English token).
    import re as _re
    safe_tokens = [_re.sub(r'["\(\)\*\^:]', "", t) for t in tokens[:10]]
    safe_tokens = [t for t in safe_tokens if t]
    if not safe_tokens:
        return []
    # OR-join each content word as a phrase. unicode61 tokenizer splits
    # the index entries on whitespace, so a phrase like "東京" matches
    # the standalone FTS5 token "東京" in any indexed fact.
    fts_query = " OR ".join(f'"{t}"' for t in safe_tokens)
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT fact_id, orig_text, bm25(facts_fts) AS score "
                "FROM facts_fts "
                "WHERE facts_fts MATCH ? AND user_id = ? "
                "ORDER BY bm25(facts_fts) "
                "LIMIT ?",
                (fts_query, user_id, max(int(limit), 1)),
            ).fetchall()
    except sqlite3.OperationalError as exc:
        # Malformed FTS5 query (e.g. lone parens). Log and fall back to empty.
        log.warning("bm25 search OperationalError: %s (q=%r)", exc, q[:80])
        return []
    except Exception as exc:
        log.warning("bm25 search failed: %s", exc)
        return []
    out: list[dict] = []
    for i, r in enumerate(rows):
        out.append({
            "id": r["fact_id"],
            # Return the original sentence (orig_text), never the tokenized
            # `text` column — the latter is for FTS5 matching only and would
            # surface as keyword-soup to the prompt and the LLM reranker.
            "memory": r["orig_text"],
            "rank": i + 1,
            "bm25_score": float(r["score"]),
        })
    return out


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
    # text. search_bm25 now returns orig_text (the raw sentence) as memory,
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

    Used by ``scripts/build_mem0_bm25_index.py`` after dumping every Mem0 fact
    from ChromaDB. Each fact's text is morphologically tokenized via
    the morpho module before insertion. Returns the row count.

    SQLite virtual tables can't change their ``tokenize=`` option after
    creation — so when the schema in _SCHEMA changes (e.g. trigram →
    unicode61) we must wipe the underlying file, not just DELETE the
    rows. We unlink the sqlite + WAL/SHM siblings so the next _conn()
    call recreates everything from the current _SCHEMA.
    """
    path = _resolve_db_path()
    with _db_lock:
        import os as _os
        for suffix in ("", "-wal", "-shm"):
            try:
                _os.unlink(path + suffix)
            except FileNotFoundError:
                pass
            except OSError as exc:
                log.warning("bm25 reindex_all: unlink %s%s failed: %s",
                            path, suffix, exc)

    count = 0
    try:
        # The first _conn() after the unlink recreates the file under
        # the current schema.
        with _conn() as c, _db_lock:
            try:
                from . import morpho
                _tokenize = morpho.tokenize_for_index
            except Exception:
                _tokenize = lambda s: s  # noqa: E731
            for fid, text, uid in facts:
                if not fid or not text or not uid:
                    continue
                tokenized = _tokenize(text)
                if not tokenized:
                    tokenized = text  # never index an empty row
                c.execute(
                    "INSERT INTO facts_fts(fact_id, user_id, orig_text, text) "
                    "VALUES (?, ?, ?, ?)",
                    (fid, uid, text, tokenized),
                )
                count += 1
            c.commit()
    except Exception as exc:
        log.warning("bm25 reindex_all failed at %d: %s", count, exc)
    return count


def stats() -> dict:
    try:
        with _conn() as c:
            total = c.execute("SELECT count(*) FROM facts_fts").fetchone()[0]
    except Exception:
        total = -1
    return {"rows": total, "db_path": _resolve_db_path()}
