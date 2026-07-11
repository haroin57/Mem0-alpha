"""Obsidian-style knowledge graph layer over ChromaDB Mem0.

A thin SQLite store that mirrors the wikilink/backlink model from
Obsidian:

    fact (ChromaDB row)  ⟷  entity (sqlite row)  via fact_entity_edges

Use cases this enables that pure cosine retrieval can't:
  - "Rust" 関連の話全部 → entity lookup → 1-hop sibling facts
  - 表記揺れ吸収 (Namitape / ナミテープ) via entity_aliases
  - Entity 単位の言及頻度集計

Storage plumbing (WAL-once init, busy_timeout, locked writes with retry)
lives in :mod:`llm_mem0.sqlite_store`. The module is intentionally
synchronous; callers wrap with asyncio.to_thread when invoked from an
event loop.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Iterable

from .sqlite_store import SqliteStore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name  TEXT NOT NULL,
    type            TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    mention_count   INTEGER NOT NULL DEFAULT 0,
    UNIQUE(canonical_name, user_id)
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    entity_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias       TEXT NOT NULL,
    PRIMARY KEY (entity_id, alias)
);

CREATE TABLE IF NOT EXISTS fact_entity_edges (
    fact_id     TEXT NOT NULL,
    entity_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    role        TEXT,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_edges_entity ON fact_entity_edges(entity_id);
CREATE INDEX IF NOT EXISTS idx_edges_fact   ON fact_entity_edges(fact_id);
CREATE INDEX IF NOT EXISTS idx_aliases_alias ON entity_aliases(alias);

-- Phase C-1 (mem0-recall-hybrid-hyde-kg): temporal entity relations.
-- Stores typed edges between entities ("Alice loves Bob") with
-- bitemporal validity columns so a "髪色 黒 → ハイトーン" state change
-- ends up as two rows, one with valid_to=<change time>, the other still
-- open. SQLite forbids expressions inside table-level UNIQUE constraints,
-- so we enforce "at most one currently-open row per (src, rel, dst, user)"
-- via a partial unique index where valid_to IS NULL instead.
CREATE TABLE IF NOT EXISTS entity_relations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    src_entity_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    rel_type        TEXT NOT NULL,
    dst_entity_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    valid_from      INTEGER,
    valid_to        INTEGER,
    source_fact_id  TEXT
);

CREATE INDEX IF NOT EXISTS idx_relations_src        ON entity_relations(src_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_dst        ON entity_relations(dst_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_type       ON entity_relations(rel_type);
CREATE INDEX IF NOT EXISTS idx_relations_user_valid ON entity_relations(user_id, valid_to);

CREATE UNIQUE INDEX IF NOT EXISTS idx_relations_unique_open
    ON entity_relations(src_entity_id, rel_type, dst_entity_id, user_id)
    WHERE valid_to IS NULL;
"""

_store = SqliteStore("mem0_graph.sqlite", _SCHEMA)


def _reset_db_path_for_test(path: str | None) -> None:
    """Test hook — pytest fixtures point the module at a tempfile."""
    _store.reset_path_for_test(path)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Entity:
    id: int
    canonical_name: str
    type: str
    user_id: str
    mention_count: int


def _normalize_name(name: str) -> str:
    """Strip leading/trailing whitespace and collapse internal whitespace."""
    return " ".join(name.split()).strip()


# ---------------------------------------------------------------------------
# Appellation resolution (§2 of docs/mem0-dedup-issues.md)
# ---------------------------------------------------------------------------
# Maps common appellations to their canonical proper noun. The second tuple
# element is a frozenset of user_ids the mapping applies to, or ``None`` for
# all users. Keep this small and unambiguous — when in doubt, leave the
# appellation alone and let the extractor record it as a standalone entity.
_APPELLATION_MAP: dict[str, tuple[str, frozenset[str] | None]] = {
    "boss": ("alice", None),
    "dad": ("bob", None),
    "father": ("bob", None),
}


def _resolve_appellation(name: str, user_id: str) -> tuple[str, list[str]]:
    """Resolve an appellation to its canonical proper noun.

    Returns ``(canonical, extra_aliases)``. When the input is not an
    appellation we know about (or the user_id is out of scope), the name is
    returned unchanged and ``extra_aliases`` is empty.
    """
    entry = _APPELLATION_MAP.get(name)
    if entry is None:
        return name, []
    canonical, allowed_users = entry
    if allowed_users is not None and user_id not in allowed_users:
        return name, []
    return canonical, [name]


# ---------------------------------------------------------------------------
# Entity upsert / lookup
# ---------------------------------------------------------------------------
def upsert_entity(
    name: str, type: str, user_id: str, *, aliases: Iterable[str] = (),
) -> int:
    """Insert-or-update an entity. Returns the entity id (0 on failure).

    Alias-aware lookup order (§1 of docs/mem0-dedup-issues.md):
      1. Resolve appellations (``boss`` → ``alice``) before any DB lookup.
      2. Single OR-merged SELECT: hit if canonical_name matches the candidate
         name OR any candidate alias, OR if any of those surface forms are
         already registered in entity_aliases.
      3. On hit: bump mention_count and register all candidate surface forms
         (name + aliases) as aliases unless they already match the canonical.
      4. On miss: INSERT a new entity, then register the alias rows.
    """
    name = _normalize_name(name)
    if not name or not type or not user_id:
        return 0
    name, appellation_aliases = _resolve_appellation(name, user_id)
    norm_aliases: list[str] = []
    seen: set[str] = {name}
    for raw in list(aliases) + appellation_aliases:
        a = _normalize_name(raw)
        if a and a not in seen:
            norm_aliases.append(a)
            seen.add(a)
    now = int(time.time())
    surface_forms = (name, *norm_aliases)
    placeholders = ",".join("?" for _ in surface_forms)
    lookup_sql = (
        f"SELECT id FROM entities "
        f"WHERE user_id=? AND ("
        f"canonical_name IN ({placeholders}) "
        f"OR id IN (SELECT entity_id FROM entity_aliases "
        f"WHERE alias IN ({placeholders}))"
        f") LIMIT 1"
    )

    def _upsert(c) -> int:
        row = c.execute(
            lookup_sql, (user_id, *surface_forms, *surface_forms),
        ).fetchone()
        if row:
            eid = row["id"]
            c.execute(
                "UPDATE entities SET mention_count = mention_count + 1 "
                "WHERE id=?",
                (eid,),
            )
        else:
            cur = c.execute(
                "INSERT INTO entities(canonical_name, type, user_id, "
                "created_at, mention_count) VALUES (?, ?, ?, ?, 1)",
                (name, type, user_id, now),
            )
            eid = cur.lastrowid
        # Register every surface form (name + aliases) that isn't already the
        # canonical_name of the resolved entity. On the new-INSERT path this
        # naturally skips ``name``; on the merge path it registers ``name``
        # as an alias of the existing canonical too.
        existing_canonical = c.execute(
            "SELECT canonical_name FROM entities WHERE id=?", (eid,),
        ).fetchone()["canonical_name"]
        for sf in surface_forms:
            if sf != existing_canonical:
                c.execute(
                    "INSERT OR IGNORE INTO entity_aliases(entity_id, alias) "
                    "VALUES (?, ?)",
                    (eid, sf),
                )
        return int(eid)

    eid = _store.write(_upsert, what="upsert_entity", default=0) or 0
    if eid:
        _invalidate_top_entities_cache(user_id)
    return eid


def find_entity(query: str, user_id: str, *, fuzzy: bool = True) -> Entity | None:
    """Resolve a free-form name to an entity. Tries (in order):
    1. exact canonical_name match
    2. alias match
    3. (if fuzzy) prefix match against canonical names
    Returns None if nothing reasonable matches.
    """
    q = _normalize_name(query)
    if not q:
        return None

    def _find(c) -> Entity | None:
        row = c.execute(
            "SELECT id, canonical_name, type, user_id, mention_count "
            "FROM entities WHERE canonical_name=? AND user_id=?",
            (q, user_id),
        ).fetchone()
        if row:
            return Entity(**dict(row))
        row = c.execute(
            "SELECT e.id, e.canonical_name, e.type, e.user_id, e.mention_count "
            "FROM entities e JOIN entity_aliases a ON a.entity_id=e.id "
            "WHERE a.alias=? AND e.user_id=?",
            (q, user_id),
        ).fetchone()
        if row:
            return Entity(**dict(row))
        if not fuzzy:
            return None
        # prefix fuzzy — most-mentioned wins
        row = c.execute(
            "SELECT id, canonical_name, type, user_id, mention_count "
            "FROM entities WHERE user_id=? AND canonical_name LIKE ? "
            "ORDER BY mention_count DESC LIMIT 1",
            (user_id, f"{q}%"),
        ).fetchone()
        return Entity(**dict(row)) if row else None

    return _store.read(_find, what="find_entity", default=None)


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------
def add_edge(fact_id: str, entity_id: int, *, role: str | None = None) -> bool:
    """Link a Mem0 fact to an entity. Idempotent."""
    if not fact_id or not entity_id:
        return False
    now = int(time.time())

    def _insert(c) -> bool:
        c.execute(
            "INSERT OR IGNORE INTO fact_entity_edges(fact_id, entity_id, "
            "role, created_at) VALUES (?, ?, ?, ?)",
            (fact_id, int(entity_id), role, now),
        )
        return True

    return bool(_store.write(_insert, what="add_edge", default=False))


def get_related_facts(
    entity_id: int, *, limit: int = 20, exclude_fact_ids: Iterable[str] = (),
) -> list[str]:
    """Return fact ids that link to ``entity_id``, newest first."""
    if not entity_id:
        return []
    excl = set(exclude_fact_ids or ())

    def _fetch(c) -> list[str]:
        rows = c.execute(
            "SELECT fact_id FROM fact_entity_edges WHERE entity_id=? "
            "ORDER BY created_at DESC LIMIT ?",
            (int(entity_id), max(limit + len(excl), 1)),
        ).fetchall()
        out: list[str] = []
        for r in rows:
            fid = r["fact_id"]
            if fid not in excl:
                out.append(fid)
                if len(out) >= limit:
                    break
        return out

    return _store.read(_fetch, what="get_related_facts", default=[]) or []


def get_entity_degree(entity_id: int) -> int:
    """Number of distinct facts linked to this entity — a cheap proxy for
    how "connected"/hub-like an entity is in the fact graph. Used to
    normalize co-occurrence weight in get_co_occurring_entities so a
    high-degree hub entity doesn't dominate every neighborhood purely by
    virtue of appearing everywhere."""
    if not entity_id:
        return 0

    def _fetch(c) -> int:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM fact_entity_edges WHERE entity_id=?",
            (int(entity_id),),
        ).fetchone()
        return int(row["n"]) if row else 0

    return _store.read(_fetch, what="get_entity_degree", default=0) or 0


def get_co_occurring_entities(entity_id: int, *, limit: int = 10) -> list[tuple[int, float]]:
    """Entities that share at least one fact with ``entity_id``, ranked by a
    degree-normalized co-occurrence weight — the graph analogue of cosine
    normalization: ``shared_facts / sqrt(deg(A) * deg(B))``. Without this
    normalization a very high-degree entity (mentioned in hundreds of facts)
    would show up as "related" to almost everything purely from its size,
    the graph equivalent of a stopword dominating a keyword search.

    Used by search._graph_expand_candidates for multi-hop spreading
    activation (Phase ④) — replaces the previous fixed 1-hop, uniform-score
    expansion.

    Returns ``[(other_entity_id, weight), ...]`` sorted by weight descending,
    weight in (0, 1].
    """
    if not entity_id:
        return []
    deg_a = get_entity_degree(entity_id)
    if deg_a == 0:
        return []

    def _fetch(c):
        return c.execute(
            """
            SELECT fe2.entity_id AS other_id, COUNT(*) AS shared
            FROM fact_entity_edges fe1
            JOIN fact_entity_edges fe2
              ON fe1.fact_id = fe2.fact_id AND fe2.entity_id != fe1.entity_id
            WHERE fe1.entity_id = ?
            GROUP BY fe2.entity_id
            ORDER BY shared DESC
            LIMIT ?
            """,
            (int(entity_id), max(limit * 3, limit)),
        ).fetchall()

    rows = _store.read(_fetch, what="get_co_occurring_entities", default=[]) or []
    weighted: list[tuple[int, float]] = []
    for r in rows:
        other_id = r["other_id"]
        deg_b = get_entity_degree(other_id)
        if deg_b == 0:
            continue
        w = r["shared"] / ((deg_a * deg_b) ** 0.5)
        weighted.append((other_id, min(w, 1.0)))
    weighted.sort(key=lambda x: x[1], reverse=True)
    return weighted[:limit]


def get_fact_entities(fact_id: str) -> list[Entity]:
    """All entities a given fact links to."""
    if not fact_id:
        return []

    def _fetch(c) -> list[Entity]:
        rows = c.execute(
            "SELECT e.id, e.canonical_name, e.type, e.user_id, e.mention_count "
            "FROM entities e JOIN fact_entity_edges fe ON fe.entity_id=e.id "
            "WHERE fe.fact_id=?",
            (fact_id,),
        ).fetchall()
        return [Entity(**dict(r)) for r in rows]

    return _store.read(_fetch, what="get_fact_entities", default=[]) or []


def get_aliases(entity_id: int) -> list[str]:
    """All registered aliases (surface forms) for the given entity id.
    Used by search.py to expand a query into canonical + alias variants."""
    if not entity_id:
        return []

    def _fetch(c) -> list[str]:
        rows = c.execute(
            "SELECT alias FROM entity_aliases WHERE entity_id=?",
            (entity_id,),
        ).fetchall()
        return [r["alias"] for r in rows]

    return _store.read(_fetch, what="get_aliases", default=[]) or []


def delete_fact_edges(fact_id: str) -> int:
    """Drop all edges for a fact (call when a Mem0 fact is deleted).
    Returns the number of rows removed."""
    if not fact_id:
        return 0

    def _delete(c) -> int:
        cur = c.execute(
            "DELETE FROM fact_entity_edges WHERE fact_id=?", (fact_id,)
        )
        return cur.rowcount or 0

    return _store.write(_delete, what="delete_fact_edges", default=0) or 0


# ---------------------------------------------------------------------------
# Top-N canonical names — used by extract prompts to anchor canonical forms
# (§5a of docs/mem0-dedup-issues.md).
# ---------------------------------------------------------------------------
_TOP_ENTITIES_TTL_SEC = 300
# {(user_id, limit): (expires_at_epoch, list[str])}
_top_entities_cache: dict[tuple[str, int], tuple[float, list[str]]] = {}
_top_entities_lock = threading.Lock()


def _invalidate_top_entities_cache(user_id: str | None = None) -> None:
    """Drop cached top-entities. Called from upsert_entity on writes."""
    with _top_entities_lock:
        if user_id is None:
            _top_entities_cache.clear()
            return
        for key in [k for k in _top_entities_cache if k[0] == user_id]:
            _top_entities_cache.pop(key, None)


def invalidate_top_entities_cache(user_id: str | None = None) -> None:
    """Public alias for tests and scripts that mutate the DB out-of-band."""
    _invalidate_top_entities_cache(user_id)


def get_top_entities(user_id: str, limit: int = 30) -> list[str]:
    """Return up to ``limit`` canonical_name strings, ordered by mention_count.

    Filters out entities with mention_count < 2 — first-mention noise tends to
    poison the canonical dictionary the extractor sees. Results are cached for
    ``_TOP_ENTITIES_TTL_SEC`` per (user_id, limit) and invalidated on writes.
    """
    if not user_id or limit <= 0:
        return []
    key = (str(user_id), int(limit))
    now = time.time()
    with _top_entities_lock:
        hit = _top_entities_cache.get(key)
        if hit and hit[0] > now:
            return list(hit[1])

    def _fetch(c) -> list[str]:
        rows = c.execute(
            "SELECT canonical_name FROM entities "
            "WHERE user_id=? AND mention_count >= 2 "
            "ORDER BY mention_count DESC, id ASC LIMIT ?",
            (str(user_id), int(limit)),
        ).fetchall()
        return [r["canonical_name"] for r in rows]

    names = _store.read(_fetch, what="get_top_entities", default=[]) or []
    with _top_entities_lock:
        _top_entities_cache[key] = (now + _TOP_ENTITIES_TTL_SEC, list(names))
    return names


# ---------------------------------------------------------------------------
# Stats (used by housekeeping scripts)
# ---------------------------------------------------------------------------
def stats(user_id: str | None = None) -> dict:
    def _fetch(c) -> dict:
        if user_id:
            ents = c.execute(
                "SELECT count(*) FROM entities WHERE user_id=?", (user_id,)
            ).fetchone()[0]
            rels = c.execute(
                "SELECT count(*) FROM entity_relations WHERE user_id=?",
                (user_id,),
            ).fetchone()[0]
        else:
            ents = c.execute("SELECT count(*) FROM entities").fetchone()[0]
            rels = c.execute("SELECT count(*) FROM entity_relations").fetchone()[0]
        edges = c.execute("SELECT count(*) FROM fact_entity_edges").fetchone()[0]
        aliases = c.execute("SELECT count(*) FROM entity_aliases").fetchone()[0]
        return {
            "entities": ents, "edges": edges, "aliases": aliases,
            "relations": rels,
        }

    return _store.read(_fetch, what="stats", default={}) or {}


# ---------------------------------------------------------------------------
# Phase C-1: entity_relations (typed, temporal edges between entities)
# ---------------------------------------------------------------------------
@dataclass
class Relation:
    id: int
    src_entity_id: int
    rel_type: str
    dst_entity_id: int
    user_id: str
    created_at: int
    valid_from: int | None
    valid_to: int | None
    source_fact_id: str | None


def add_relation(
    src_id: int, rel_type: str, dst_id: int, user_id: str, *,
    source_fact_id: str | None = None, valid_from: int | None = None,
) -> int:
    """Insert a typed temporal edge between two entities. Returns the row id
    (0 on failure). The partial unique index ``idx_relations_unique_open``
    (``WHERE valid_to IS NULL``) lets historical versions coexist per
    (src, rel, dst, user) while allowing only one currently-open row.

    When an open row already exists, the INSERT is IGNOREd and the existing
    row's id is returned — ``cur.lastrowid`` is meaningless after an ignored
    insert, so we look the row up explicitly instead of trusting it.
    """
    if not src_id or not dst_id or not rel_type or not user_id:
        return 0
    now = int(time.time())

    def _insert(c) -> int:
        cur = c.execute(
            "INSERT OR IGNORE INTO entity_relations("
            "src_entity_id, rel_type, dst_entity_id, user_id, "
            "created_at, valid_from, valid_to, source_fact_id) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
            (int(src_id), rel_type, int(dst_id), str(user_id),
             now, valid_from, source_fact_id),
        )
        if cur.rowcount:
            return int(cur.lastrowid or 0)
        row = c.execute(
            "SELECT id FROM entity_relations "
            "WHERE src_entity_id=? AND rel_type=? AND dst_entity_id=? "
            "AND user_id=? AND valid_to IS NULL",
            (int(src_id), rel_type, int(dst_id), str(user_id)),
        ).fetchone()
        return int(row["id"]) if row else 0

    return _store.write(_insert, what="add_relation", default=0) or 0


def expire_relation(relation_id: int, *, valid_to: int | None = None) -> bool:
    """Set ``valid_to`` on an open relation. Idempotent: re-running with
    the same valid_to is a no-op. Returns True on a single-row update.
    """
    if not relation_id:
        return False
    vt = int(valid_to or time.time())

    def _expire(c) -> bool:
        cur = c.execute(
            "UPDATE entity_relations SET valid_to=? "
            "WHERE id=? AND (valid_to IS NULL OR valid_to=?)",
            (vt, int(relation_id), vt),
        )
        return bool(cur.rowcount)

    return bool(_store.write(_expire, what="expire_relation", default=False))


def get_relations(
    entity_id: int, *, direction: str = "any",
    rel_type: str | None = None, only_valid: bool = True,
    limit: int = 50,
) -> list[Relation]:
    """Fetch relations involving ``entity_id``.

    direction:
      "src" — entity_id appears as src
      "dst" — entity_id appears as dst
      "any" — either side (default)
    rel_type: optional filter
    only_valid: filter out rows where valid_to <= now (default True)
    """
    if not entity_id:
        return []
    direction = (direction or "any").lower()
    now = int(time.time())
    where = []
    params: list = []
    if direction == "src":
        where.append("src_entity_id = ?")
        params.append(int(entity_id))
    elif direction == "dst":
        where.append("dst_entity_id = ?")
        params.append(int(entity_id))
    else:
        where.append("(src_entity_id = ? OR dst_entity_id = ?)")
        params.extend([int(entity_id), int(entity_id)])
    if rel_type:
        where.append("rel_type = ?")
        params.append(rel_type)
    if only_valid:
        where.append("(valid_to IS NULL OR valid_to > ?)")
        params.append(now)
    sql = (
        "SELECT id, src_entity_id, rel_type, dst_entity_id, user_id, "
        "created_at, valid_from, valid_to, source_fact_id "
        "FROM entity_relations WHERE "
        + " AND ".join(where)
        + " ORDER BY created_at DESC LIMIT ?"
    )
    params.append(int(limit))

    def _fetch(c) -> list[Relation]:
        return [Relation(**dict(r)) for r in c.execute(sql, params).fetchall()]

    return _store.read(_fetch, what="get_relations", default=[]) or []
