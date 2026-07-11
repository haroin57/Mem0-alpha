"""Memory ingestion: P2 extraction-gate path + legacy Mem0-standard fallback.

``add_memories`` is the single entry point; it normalizes the call
signature, then dispatches to the gate path (LLM extraction → filter →
dedup → raw insert + side-effect indexing) or the legacy path (the memory
store's built-in extractor). Every stage lives in its own helper so the
pipeline reads top-to-bottom instead of as one interleaved function.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict

from .client import _get_mem0
from .extract import extract_full_signals
from .search import search_memories
from .settings import settings

log = logging.getLogger(__name__)

# P1 (ADR-0017): process-global guard against concurrent / short-window
# duplicate inserts. The vector dedup only catches a same-text sibling once it
# is COMMITTED; two concurrent add_memories calls (live turns or backfill
# tasks) can each search-then-insert before either commits, producing exact
# dups. This seen-set, guarded by a lock, makes the insert of an identical
# (user_id, text) idempotent across the in-flight window. Bounded FIFO so it
# never grows without limit. Offline hash sweep cleans any historical dups.
_INSERT_SEEN: OrderedDict[str, str] = OrderedDict()
_INSERT_SEEN_LOCK = asyncio.Lock()
_INSERT_SEEN_MAX = 8000

# Caller-supplied metadata is allowlisted before merging so it cannot
# overwrite trust-relevant keys (gate_version, timestamp, importance,
# category, tags). New caller-controlled keys must be added here explicitly.
# `persona` / `episode_id` / `interest_tags` let the recall path carry an
# optional persona label and link a fact back to the episode that produced
# it. `interest_tags` must be a CSV string because ChromaDB rejects
# list-valued metadata; the caller is responsible for the CSV conversion.
_ALLOWED_CALLER_META = {
    "channel_id", "channel_name", "source",
    "persona", "episode_id", "interest_tags",
}

_EMPTY_SKIPPED = {"low_importance": 0, "other_speaker": 0, "near_dup": 0}


def _fact_key(user_id: str, text: str) -> str:
    return hashlib.md5(f"{user_id}\x00{(text or '').strip()}".encode()).hexdigest()


async def _claim_insert(user_id: str, text: str) -> bool:
    """Return True if this (user_id, text) may be inserted now; False if a
    concurrent/recent insert already claimed it (caller should skip)."""
    key = _fact_key(user_id, text)
    async with _INSERT_SEEN_LOCK:
        if key in _INSERT_SEEN:
            return False
        _INSERT_SEEN[key] = "1"
        while len(_INSERT_SEEN) > _INSERT_SEEN_MAX:
            _INSERT_SEEN.popitem(last=False)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def add_memories(
    messages: list[dict] | None = None,
    user_id: str | None = None,
    metadata: dict | None = None,
    *,
    user_text: str | None = None,
    assistant_text: str | None = None,
    context_hint: str = "",
    pre_extracted_facts: list[dict] | None = None,
) -> dict | None:
    """Store conversation for fact extraction.

    Supports both legacy `(messages, user_id, metadata)` and new
    `(user_text=, assistant_text=, user_id=, context_hint=, metadata=)`.

    Facts about the user are extracted from ``user_text`` (with
    ``assistant_text`` / ``context_hint`` as supporting context).

    ``pre_extracted_facts`` — when supplied, skips the LLM extraction
    step and uses the provided facts as-is, so the caller can avoid a
    second extraction pass. Each fact must already be in the same shape
    ``extract_full_signals`` would have returned under its ``facts`` key.
    """
    user_text, assistant_text, context_hint = _normalize_signature(
        messages, user_text, assistant_text, context_hint,
    )

    # 初回 init の ChromaDB heartbeat (同期 requests) でループを止めない。
    mem = await asyncio.to_thread(_get_mem0)
    if not mem:
        return None

    try:
        # The auth backend refreshes access tokens internally, so no explicit
        # token refresh is needed here before touching the store.
        base_meta = _build_base_meta(metadata)
        if settings.MEM0_GATE_ENABLED:
            return await _add_via_gate(
                mem, user_id, user_text, assistant_text, context_hint,
                base_meta, pre_extracted_facts,
            )
        return await _add_legacy(
            mem, messages, user_text, assistant_text, user_id, base_meta,
        )
    except Exception as exc:
        # Surface the reason to callers (MCP server / /remember): the old
        # bare None made every failure look identical and pointed users at
        # "server logs" that a stdio MCP child never gets captured from
        # (2026-07-04 incident: three adds failed invisibly around a bot
        # restart and the WARNING below vanished into a dropped stderr).
        log.warning("Mem0 add error: %s", exc, exc_info=True)
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "results": [], "kept": 0,
            "skipped": dict(_EMPTY_SKIPPED),
        }


def _normalize_signature(
    messages: list[dict] | None,
    user_text: str | None,
    assistant_text: str | None,
    context_hint: str,
) -> tuple[str, str, str]:
    """Fold the legacy ``messages`` list into (user_text, assistant_text,
    context_hint); explicit keyword args win."""
    if messages is not None and user_text is None:
        for m in messages or []:
            r = m.get("role")
            c = m.get("content", "") if isinstance(m.get("content"), str) else ""
            if r == "user" and user_text is None:
                user_text = c
            elif r == "assistant" and assistant_text is None:
                assistant_text = c
            elif r == "system" and not context_hint:
                context_hint = c
    return user_text or "", assistant_text or "", context_hint


def _build_base_meta(metadata: dict | None) -> dict:
    """Timestamp + gate version + allowlisted caller metadata."""
    base_meta = {
        "timestamp": int(time.time()),
        "gate_version": "v2" if settings.MEM0_GATE_ENABLED else "v1",
    }
    if metadata:
        for k, v in metadata.items():
            if k in _ALLOWED_CALLER_META and k not in base_meta:
                base_meta[k] = v
    return base_meta


# ---------------------------------------------------------------------------
# Gate path (P2): LLM extract → filter → dedup → infer=False raw insert
# ---------------------------------------------------------------------------
async def _add_via_gate(
    mem,
    user_id,
    user_text: str,
    assistant_text: str,
    context_hint: str,
    base_meta: dict,
    pre_extracted_facts: list[dict] | None,
) -> dict:
    facts, relations = await _obtain_signals(
        user_id, user_text, assistant_text, context_hint, pre_extracted_facts,
    )

    candidates, skip_low, skip_other, skip_dup = _filter_candidates(facts)

    kept, dedup_dropped, reinforced_ids, conflict_archive = (
        await _dedup_candidates(mem, user_id, candidates, base_meta)
    )
    skip_dup += dedup_dropped

    # Hebbian reinforcement: bump mention_count on each existing fact that
    # absorbed a duplicate candidate. Failures are non-fatal.
    if reinforced_ids:
        from .reinforce import reinforce_existing
        await asyncio.gather(
            *(reinforce_existing(mem, mid) for mid in reinforced_ids),
            return_exceptions=True,
        )

    # Phase A-3: 決定的 conflict ルート。dedup の embedding 近傍ゲート
    # (距離≤0.35)をすり抜けた属性更新(文面の離れた数値更新等)を、
    # attribute 完全一致で捕捉する。dedup が既に archive 対象にした
    # existing_id は二重に積まない。
    if settings.MEM0_ATTRIBUTE_CONFLICT_ENABLED and kept:
        conflict_archive.extend(await _attribute_conflict_check(
            mem, user_id, kept,
            exclude_ids={eid for eid, _ in conflict_archive},
        ))

    # Parallel raw insert: three ChromaDB writes in serial added ~3x
    # latency. gather them and tolerate individual failures.
    insert_results = (
        await asyncio.gather(*(
            _insert_one(mem, user_id, f, base_meta) for f in kept
        ))
        if kept else []
    )
    results = [r for r in insert_results if r is not None]

    # Phase B-2: soft-archive outdated predecessors after the new fact is in
    # storage. Errors are non-fatal — the new fact is already stored, the
    # old one just won't be hidden.
    archived = await _archive_outdated_facts(mem, conflict_archive)

    relations_added = 0
    if settings.MEM0_RELATION_EXTRACT_ENABLED and relations:
        relations_added = await _ingest_relations(user_id, relations)

    log.info(
        "Mem0 gate: facts=%d kept=%d skip(low=%d,other=%d,dup=%d) "
        "archived=%d relations=%d user=...%s",
        len(facts), len(kept), skip_low, skip_other, skip_dup,
        archived, relations_added, str(user_id)[-4:],
    )
    # Phase B-3: structured one-line metric for grep-based weekly health
    # reports. Each key is positional so the parser stays cheap.
    log.info(
        "[MEM0_INGEST_METRICS extracted=%d candidates=%d kept=%d "
        "drop_low=%d drop_other=%d drop_dup=%d archived=%d]",
        len(facts), len(candidates), len(kept),
        skip_low, skip_other, skip_dup, archived,
    )
    return {"results": results, "kept": len(kept), "skipped": {
        "low_importance": skip_low, "other_speaker": skip_other, "near_dup": skip_dup,
    }}


async def _obtain_signals(
    user_id,
    user_text: str,
    assistant_text: str,
    context_hint: str,
    pre_extracted_facts: list[dict] | None,
) -> tuple[list[dict], list[dict]]:
    """Run the extractor (or accept the caller's pre-extracted facts)."""
    if pre_extracted_facts is not None:
        # Caller already ran the extractor and is passing facts in to avoid
        # a duplicate LLM call. We trust the shape because the extractor and
        # the caller share a module-internal contract. Pre-extracted path
        # doesn't carry relations (older callers predate Phase C-2).
        facts, relations = pre_extracted_facts, []
    else:
        # Phase C-2: a single LLM call extracts facts AND entity-entity
        # relations; relations are handed to the graph layer after insert.
        signals = await extract_full_signals(
            user_text, assistant_text, context_hint,
            user_id=str(user_id) if user_id else None,
        )
        facts = signals.get("facts", [])
        relations = signals.get("relations") or []
    # Diagnostic: show every extracted fact text. Helps trace why a turn
    # produced unexpected dedup hits (candidate vs existing).
    if facts:
        previews = [
            f"[{f.get('speaker', '?')}|imp={f.get('importance', '?')}] "
            f"{(f.get('text') or '')[:120]}"
            for f in facts
        ]
        log.info(
            "Mem0 gate: extracted %d facts user=...%s | %s",
            len(facts), str(user_id)[-4:], " || ".join(previews),
        )
    return facts, relations


def _filter_candidates(facts: list[dict]) -> tuple[list[dict], int, int, int]:
    """Cheap synchronous filter pass — drops obvious non-self /
    low-importance facts and intra-batch text duplicates without firing any
    ChromaDB queries. Returns (candidates, skip_low, skip_other, skip_dup).
    """
    min_importance = settings.MEM0_GATE_MIN_IMPORTANCE
    candidates: list[dict] = []
    skip_low = skip_other = skip_dup = 0
    for f in facts:
        if f.get("speaker") != "self":
            skip_other += 1
            continue
        if int(f.get("importance", 0)) < min_importance:
            skip_low += 1
            continue
        candidates.append(f)

    # Intra-batch dedup: two facts with identical text in the same call
    # would both pass the parallel dedup-check (neither is in the store
    # yet) and both get inserted. Drop within-batch duplicates by `text`
    # before firing any ChromaDB queries.
    seen_texts: set[str] = set()
    deduped: list[dict] = []
    for f in candidates:
        t = f.get("text", "")
        if t in seen_texts:
            skip_dup += 1
            continue
        seen_texts.add(t)
        deduped.append(f)
    return deduped, skip_low, skip_other, skip_dup


async def _dedup_candidates(
    mem, user_id, candidates: list[dict], base_meta: dict,
) -> tuple[list[dict], int, list[str], list[tuple[str, str]]]:
    """Near-duplicate suppression against the existing store.

    Returns ``(kept, dropped, reinforced_ids, conflict_archive)``:
      kept             — candidates to insert (a "merge" candidate has its
                         text replaced by the LLM-synthesized fact)
      dropped          — count of candidates dropped as duplicates
      reinforced_ids   — existing fact ids that absorbed a duplicate
      conflict_archive — (existing_id, conflict_type) pairs to soft-archive
                         after the new facts are inserted
    """
    if not settings.MEM0_CLIENT_DEDUP_ENABLED or not candidates:
        return list(candidates), 0, [], []

    # Phase R: scope フィルタ有効時に conversation fact が dedup 候補から
    # 消えると同一会話内の兄弟と dedup できなくなるので、この ingest の
    # 会話 channel_id を渡す。
    dedup_ch = base_meta.get("channel_id")
    dedup_ch = str(dedup_ch) if dedup_ch else None
    near_results = await asyncio.gather(
        *(
            search_memories(
                f["text"], user_id, limit=settings.MEM0_DEDUP_SEARCH_LIMIT,
                current_channel_id=dedup_ch,
            )
            for f in candidates
        ),
        return_exceptions=True,
    )

    if settings.MEM0_DEDUP_TWO_STAGE:
        return await _dedup_two_stage(candidates, near_results)
    return _dedup_single_threshold(candidates, near_results)


async def _dedup_two_stage(
    candidates: list[dict], near_results: list,
) -> tuple[list[dict], int, list[str], list[tuple[str, str]]]:
    """Cosine gate + LLM judgment, with merge (止揚) and conflict routes."""
    from .dedup import check_near_dup

    dry_run = settings.MEM0_DEDUP_DRY_RUN

    async def _check_one(f: dict, near):
        if isinstance(near, Exception):
            return (f, False, None, None, None)
        is_dup, _reason, existing_id, conflict_type, merged_text = await check_near_dup(
            f["text"], f.get("category"), near,
            dry_run=dry_run,
            detect_conflict=settings.MEM0_CONFLICT_DETECTION_ENABLED,
            allow_merge=settings.MEM0_DEDUP_SUPERSET_MERGE,
        )
        return (f, is_dup, existing_id, conflict_type, merged_text)

    check_results = await asyncio.gather(
        *(_check_one(f, near) for f, near in zip(candidates, near_results)),
    )

    kept: list[dict] = []
    dropped = 0
    reinforced_ids: list[str] = []
    # Phase B-2 / 止揚: facts that update / correct / merge with an existing
    # fact get their predecessor soft-archived *after* the new (or
    # synthesized) row is inserted, so we record the pairs here.
    conflict_archive: list[tuple[str, str]] = []
    for f, is_dup, existing_id, conflict_type, merged_text in check_results:
        if is_dup and not dry_run:
            dropped += 1
            if existing_id:
                reinforced_ids.append(existing_id)
            continue
        # 止揚: store the synthesized fact in place of the raw candidate so
        # the merged detail is what lands in ChromaDB; the older row is
        # archived below.
        if conflict_type == "merge" and merged_text and not dry_run:
            f = {**f, "text": merged_text}
        kept.append(f)
        if (
            conflict_type in ("update", "correction", "merge")
            and existing_id
            and not dry_run
        ):
            conflict_archive.append((existing_id, conflict_type))
    return kept, dropped, reinforced_ids, conflict_archive


def _dedup_single_threshold(
    candidates: list[dict], near_results: list,
) -> tuple[list[dict], int, list[str], list[tuple[str, str]]]:
    """Legacy single-threshold dedup. NOTE: score is *distance* (small =
    similar), so we drop when score is BELOW the threshold, not above. The
    settings default of 0.85 is way too loose under this corrected
    semantic; set MEM0_CLIENT_DEDUP_THRESHOLD=0.20 if you actually rely on
    this path."""
    threshold = settings.MEM0_CLIENT_DEDUP_THRESHOLD
    kept: list[dict] = []
    dropped = 0
    for f, near in zip(candidates, near_results):
        if isinstance(near, Exception):
            kept.append(f)
            continue
        if any(
            (r.get("score") if r.get("score") is not None else 99) <= threshold
            for r in near
        ):
            dropped += 1
        else:
            kept.append(f)
    return kept, dropped, [], []


def _build_fact_meta(f: dict, base_meta: dict) -> dict:
    """Assemble the ChromaDB metadata dict for one extracted fact.

    ChromaDB metadata values must be scalar (str/int/float/bool); a list or
    None raises "Cannot convert Python object to MetadataValue" and aborts
    the whole ``mem.add`` — every fact in that turn silently dropped. So:
    tags become CSV, typed values become JSON strings, and None values are
    stripped at the end.
    """
    fact_meta = dict(base_meta)
    imp = int(f.get("importance", 3))
    # importance=2 → force tier=Warm so weekly compact ages it out faster.
    # Otherwise default to tier=1 (Hot); the extract prompt doesn't emit
    # `tier`, so f.get("tier") is normally None. Explicit None check so a
    # pre_extracted_facts caller passing tier=0 (Live) isn't demoted to Hot.
    if imp <= 2:
        tier = 2
    else:
        raw_tier = f.get("tier")
        tier = 1 if raw_tier is None else int(raw_tier)
    raw_tags = f.get("tags") or []
    tags_csv = ",".join(
        str(t).replace(",", " ") for t in raw_tags if str(t).strip()
    )
    fact_meta.update({
        "category":    str(f.get("category") or "meta"),
        "subcategory": str(f.get("subcategory") or ""),
        "importance":  imp,
        "tier":        tier,
        "tags":        tags_csv,
    })
    # Phase A: attribute(統制語彙)を付与。決定的 conflict ルートのキー。
    # 空文字列は付けない(ChromaDB metadata を無駄に増やさない)。
    attr = f.get("attribute") or ""
    if settings.MEM0_ATTRIBUTE_ENABLED and attr:
        fact_meta["attribute"] = attr
    # v5 T7: typed value（quantity）。ChromaDB は dict を持てないので JSON
    # 文字列で保存。読む側は json.loads。
    val = f.get("value")
    if settings.MEM0_TYPED_VALUE_ENABLED and isinstance(val, dict):
        import json
        fact_meta["value_json"] = json.dumps(val, ensure_ascii=False)
    # Phase B: condition(time:/context:)を付与。
    cond = f.get("condition") or ""
    if settings.MEM0_CONDITION_ENABLED and cond:
        fact_meta["condition"] = cond
    # Phase C: scope。global は既定なので保存しない(metadata を節約)。
    # conversation scope には channel_id を紐付け、どの会話限定かを読み出し
    # 時に照合できるようにする。
    scope = f.get("scope") or ""
    if settings.MEM0_SCOPE_ENABLED and scope and scope != "global":
        if scope == "conversation":
            ch = base_meta.get("channel_id")
            fact_meta["scope"] = f"conversation:{ch}" if ch else "conversation"
        else:
            fact_meta["scope"] = scope
    # Final guard: drop any None values base_meta might still contain
    # (e.g. channel_id when called from a script).
    return {k: v for k, v in fact_meta.items() if v is not None}


async def _insert_one(mem, user_id, f: dict, base_meta: dict):
    """Raw-insert one fact (infer=False) and index it into the side stores.

    Side-store indexing (graph / BM25 / fact_store) is best-effort: the
    fact is already in ChromaDB, so those errors log a WARNING and move on.
    """
    # P1: skip insert if an identical (user_id, text) is already in-flight /
    # recently inserted this process (defeats the parallel-ingest exact-dup
    # race). Gated by env.
    if settings.MEM0_INGEST_HASH_DEDUP and not await _claim_insert(
        user_id, f.get("text", "")
    ):
        log.info("ingest hash-dedup: skipped in-flight dup: %r",
                 (f.get("text") or "")[:60])
        return None
    fact_meta = _build_fact_meta(f, base_meta)
    try:
        res = await asyncio.to_thread(
            mem.add,
            [{"role": "user", "content": f["text"]}],
            user_id=user_id, metadata=fact_meta, infer=False,
        )
    except Exception as e:
        log.warning("Mem0 raw insert failed: %s", e)
        return None

    fact_ids: list[str] = []
    if isinstance(res, dict):
        for r in (res.get("results") or []):
            fid = r.get("id") if isinstance(r, dict) else None
            if fid:
                fact_ids.append(fid)
    if fact_ids:
        await _index_side_stores(user_id, f, fact_meta, base_meta, fact_ids)
    return res


async def _index_side_stores(
    user_id, f: dict, fact_meta: dict, base_meta: dict, fact_ids: list[str],
) -> None:
    """Mirror a freshly inserted fact into graph / BM25 / fact_store."""
    # Graph layer: link extracted entities to the new fact.
    try:
        entities = f.get("entities") or []
        if entities:
            from . import graph
            for ent in entities:
                if not isinstance(ent, dict):
                    continue
                name = ent.get("name")
                etype = ent.get("type")
                if not name or not etype:
                    continue
                eid = await asyncio.to_thread(
                    graph.upsert_entity,
                    name, etype, str(user_id),
                    aliases=ent.get("aliases") or [],
                )
                if eid:
                    for fid in fact_ids:
                        await asyncio.to_thread(graph.add_edge, fid, eid)
    except Exception as e:
        log.warning("graph update failed (non-fatal): %s", e)

    # Phase A-1: index the new fact in the BM25 FTS5 store so hybrid
    # retrieval can rescue exact-term matches that cosine missed.
    try:
        from . import bm25_index
        for fid in fact_ids:
            await asyncio.to_thread(
                bm25_index.index_fact, fid, f["text"], str(user_id),
            )
    except Exception as e:
        log.warning("bm25 index update failed (non-fatal): %s", e)

    # v5 T8a: structured fact layer への導出インデックス登録（attribute 付き
    # fact のみ）。正本は Chroma、こちらは best-effort。
    try:
        attr = f.get("attribute") or ""
        if settings.MEM0_FACT_STORE_ENABLED and attr:
            from . import fact_store
            for fid in fact_ids:
                await asyncio.to_thread(
                    fact_store.record_fact,
                    fid, str(user_id), attr,
                    value_text=f.get("text", ""),
                    value_json=fact_meta.get("value_json"),
                    condition=f.get("condition") or "",
                    scope=fact_meta.get("scope", ""),
                    source_conversation_id=(
                        str(base_meta.get("channel_id"))
                        if base_meta.get("channel_id") else None
                    ),
                )
    except Exception as e:
        log.warning("fact_store record failed (non-fatal): %s", e)


async def _ingest_relations(user_id, relations: list[dict]) -> int:
    """Phase C-2: ingest entity-entity relations into the graph layer.

    Resolves src/dst canonical names via graph.find_entity (which knows
    about aliases & appellations) — only inserts a relation when both
    endpoints already exist as entities. We don't auto-create endpoints
    from relations to avoid letting a hallucinated edge spawn ghost
    entities. Returns the number of relations added; non-fatal on failure.
    """
    added = 0
    try:
        from . import graph
        for rel in relations:
            src_name = rel.get("src")
            rel_type = rel.get("rel_type")
            dst_name = rel.get("dst")
            if not (src_name and rel_type and dst_name):
                continue
            src_ent = await asyncio.to_thread(
                graph.find_entity, src_name, str(user_id), fuzzy=False,
            )
            dst_ent = await asyncio.to_thread(
                graph.find_entity, dst_name, str(user_id), fuzzy=False,
            )
            if not (src_ent and dst_ent):
                continue
            rid = await asyncio.to_thread(
                graph.add_relation,
                src_ent.id, rel_type, dst_ent.id, str(user_id),
            )
            if rid:
                added += 1
        if added:
            log.info("[REL_ADDED count=%d user=...%s]", added, str(user_id)[-4:])
    except Exception as exc:
        log.warning("relation ingest failed (non-fatal): %s", exc)
    return added


# ---------------------------------------------------------------------------
# Legacy path: Mem0 standard fact extraction (fallback)
# ---------------------------------------------------------------------------
async def _add_legacy(
    mem,
    messages: list[dict] | None,
    user_text: str,
    assistant_text: str,
    user_id,
    base_meta: dict,
) -> dict:
    """Let the memory store's built-in extractor handle the turn.

    context_hint is INTENTIONALLY excluded from legacy_msgs. The Phase 1
    root cause (other-speaker contamination) was the system-message
    channel-history path; we keep it sealed here regardless of which gate
    is enabled. The caller-supplied `messages` are also filtered to
    user/assistant roles only so a raw list with role=system can't
    re-introduce the contamination vector.
    """
    if messages:
        legacy_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
    else:
        legacy_msgs = []
    if not legacy_msgs:
        if user_text:
            legacy_msgs.append({"role": "user", "content": user_text})
        if assistant_text:
            legacy_msgs.append({"role": "assistant", "content": assistant_text})
    # _classify_for_metadata() was previously fired here, adding a second
    # LLM call on top of the store's own internal extractor. The gate path
    # is the canonical source of taxonomy metadata; in legacy fallback we
    # accept the simpler default and keep this path at 1 LLM call, not 2.
    result = await asyncio.to_thread(
        mem.add, legacy_msgs, user_id=user_id, metadata=base_meta,
    )
    count = len(result.get("results", [])) if isinstance(result, dict) else 0
    if count:
        log.info("Mem0(legacy): extracted %d user=...%s", count, str(user_id)[-4:])
        if log.isEnabledFor(logging.DEBUG):
            previews = " | ".join(
                str(r.get("memory", r.get("text", "")))[:60]
                for r in (result.get("results") or [])[:5]
            )
            log.debug("Mem0(legacy) previews: %s", previews)
    return result


# ---------------------------------------------------------------------------
# Conflict archive + deterministic attribute route
# ---------------------------------------------------------------------------
async def _archive_outdated_facts(mem, archive: list[tuple[str, str]]) -> int:
    """Soft-archive existing facts that were superseded by a new fact.

    Writes ``valid_to=<now>`` and ``conflict_archived_as=<update|correction>``
    to the existing fact's metadata. Retrieval honours valid_to to hide the
    row from candidates (see search.py). The fact is never physically
    deleted, so flipping ``MEM0_HIDE_ARCHIVED=false`` brings them back
    without any data restore.

    Returns the count of successfully archived facts. Failures are logged
    at WARNING and skipped — the new fact stays in storage either way;
    archival is best-effort because we already have the new truth.
    """
    if not archive:
        return 0
    now = int(time.time())
    success = 0
    for existing_id, ctype in archive:
        try:
            existing = await asyncio.to_thread(mem.get, existing_id)
            if not isinstance(existing, dict):
                continue
            text = existing.get("memory") or existing.get("text") or ""
            if not text:
                continue
            partial = {"valid_to": now, "conflict_archived_as": ctype}
            await asyncio.to_thread(
                mem.update, memory_id=existing_id,
                data=text, metadata=partial,
            )
            log.info(
                "[CONFLICT_ARCHIVE existing_id=%s conflict_type=%s valid_to=%d]",
                existing_id, ctype, now,
            )
            # v5 T8a: fact_store 側も supersede（Chroma soft-archive と対）。
            try:
                if settings.MEM0_FACT_STORE_ENABLED:
                    from . import fact_store
                    await asyncio.to_thread(
                        fact_store.supersede_fact, existing_id)
            except Exception as exc2:  # noqa: BLE001
                log.warning("fact_store supersede failed (non-fatal): %s", exc2)
            success += 1
        except Exception as exc:
            log.warning(
                "conflict archive (%s, %s) failed (non-fatal): %s",
                existing_id, ctype, exc,
            )
    return success


async def _get_facts_by_attribute(mem, user_id, attribute: str) -> list[dict]:
    """Phase A-3: attribute 完全一致の valid な既存 fact を取得。

    v5 T8a: MEM0_FACT_STORE_ENABLED なら SQL 索引 (fact_store) を一次に使い、
    行を Chroma 互換 shape ({"id", "memory", "metadata"}) へ変換して返す。
    store が空/未整備なら従来の Chroma 経路へフォールバック（漏れ防止）。

    Chroma 経路: get_episode_facts(search.py) と同じく filters を試し、SDK
    非対応(TypeError) なら全件取得 + Python フィルタにフォールバック。
    valid_to 済み(soft-archive)は除外する。
    """
    if not attribute:
        return []
    if settings.MEM0_FACT_STORE_ENABLED:
        try:
            from . import fact_store
            rows = await asyncio.to_thread(
                fact_store.get_active_facts, str(user_id), attribute)
            if rows:
                return [
                    {
                        "id": r.get("fact_id"),
                        "memory": r.get("value_text") or "",
                        "metadata": {
                            "attribute": r.get("attribute"),
                            "condition": r.get("condition") or "",
                            "value_json": r.get("value_json"),
                        },
                    }
                    for r in rows
                ]
            # store が空 → Chroma へフォールバック（backfill 前の期間の保険）
        except Exception as exc:
            log.warning("fact_store lookup failed, falling back: %s", exc)
    from .client import _rejects_user_id_kwarg, mem_get_all
    try:
        raw = await asyncio.to_thread(
            mem.get_all, user_id=user_id,
            filters={"attribute": attribute}, limit=50,
        )
    except Exception as exc:
        if not (isinstance(exc, TypeError) or _rejects_user_id_kwarg(exc)):
            log.warning("attr conflict get_all failed: %s", exc)
            return []
        # SDK がこの呼び形を受けない（旧SDK: filters 非対応 / 新SDK:
        # トップレベル user_id 非対応）→ user 全件 + Python フィルタ
        try:
            raw = await asyncio.to_thread(
                mem_get_all, mem, user_id=user_id, limit=1000)
        except Exception as exc2:
            log.warning("attr conflict get_all fallback failed: %s", exc2)
            return []
    rows = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
    now = int(time.time())
    out: list[dict] = []
    for r in rows:
        md = r.get("metadata") or {}
        if md.get("attribute") != attribute:
            continue
        vt = md.get("valid_to")
        if vt:
            try:
                if int(vt) <= now:
                    continue
            except (TypeError, ValueError):
                pass
        out.append(r)
    return out


def _compare_typed_values_shadow(ex: dict, cand: dict, attr: str) -> None:
    """v5 T7: 数値属性のシャドー比較ログ（挙動には影響しない）。

    既存 fact の metadata.value_json と候補 fact の value が両方 quantity で
    単位が一致するとき、same/changed を [ATTR_VALUE_COMPARE] で記録する。
    単位が違う場合は unit_mismatch（換算は将来。今は観測のみ）。
    """
    try:
        import json
        cand_val = cand.get("value")
        if not isinstance(cand_val, dict) or cand_val.get("type") != "quantity":
            return
        raw = (ex.get("metadata") or {}).get("value_json")
        if not raw:
            return
        ex_val = json.loads(raw)
        if not isinstance(ex_val, dict) or ex_val.get("type") != "quantity":
            return
        if ex_val.get("unit") != cand_val.get("unit"):
            verdict = "unit_mismatch"
        elif float(ex_val.get("value")) == float(cand_val.get("value")):
            verdict = "same"
        else:
            verdict = "changed"
        log.info(
            "[ATTR_VALUE_COMPARE attribute=%s verdict=%s existing=%s%s "
            "candidate=%s%s]",
            attr, verdict,
            ex_val.get("value"), ex_val.get("unit"),
            cand_val.get("value"), cand_val.get("unit"),
        )
    except Exception as exc:  # noqa: BLE001 - 観測用、絶対に本流を壊さない
        log.debug("typed value compare failed (non-fatal): %s", exc)


async def _attribute_conflict_check(
    mem, user_id, kept: list[dict], *, exclude_ids: set,
) -> list[tuple[str, str]]:
    """Phase A-3: 決定的 conflict ルート（embedding 距離に依存しない）。

    kept の各 fact の attribute で既存 valid fact を metadata 照会し、
    classify_conflict が update/correction を返したら (existing_id, ctype) を
    集めて返す。dedup が既に処理した existing_id は exclude_ids で除く。
    失敗は非致命(空を返す) — 新 fact は既に保存されるので archive は best-effort。
    """
    from .conflict import classify_conflict
    from .attribute_registry import (
        conditions_are_conflict_candidate, get_conflict_group,
        is_update_cardinality,
    )
    out: list[tuple[str, str]] = []
    seen = set(exclude_ids)
    for f in kept:
        attr = f.get("attribute") or ""
        if not attr:
            continue
        cand_text = f.get("text") or ""
        if not cand_text:
            continue
        # v5 T4: 同一 attribute だけでなく conflict group のメンバー全員を照会
        # (likes_coffee vs avoids_caffeine のような cross-attribute 矛盾)。
        # group 未定義なら {attr} 単独で従来挙動。
        existing: list[dict] = []
        try:
            for group_attr in sorted(get_conflict_group(attr)):
                existing.extend(
                    await _get_facts_by_attribute(mem, user_id, group_attr)
                )
        except Exception as exc:
            log.warning("attr conflict lookup failed (non-fatal): %s", exc)
            continue
        for ex in existing:
            eid = ex.get("id")
            if not eid or eid in seen:
                continue
            ex_text = ex.get("memory") or ex.get("text") or ""
            if not ex_text or ex_text == cand_text:
                continue
            # Phase B+/v5: condition 互換判定。disjoint(別条件下=朝の habit vs
            # 夜の habit)なら矛盾ではないので archive しない。same/subset/
            # superset/unknown は重なりうるので conflict 判定へ進める。
            ex_cond = (ex.get("metadata") or {}).get("condition") or ""
            cand_cond = f.get("condition") or ""
            if not conditions_are_conflict_candidate(ex_cond, cand_cond):
                continue
            # v5 T7: typed value のシャドー数値比較。両者が quantity なら結果を
            # ログに残す（classify_conflict の判定と突き合わせて精度を観測し、
            # 将来 LLM 判定の置換可否を決める。今は挙動に影響させない）。
            _compare_typed_values_shadow(ex, f, attr)
            try:
                cls = await classify_conflict(ex_text, cand_text)
            except Exception as exc:
                log.warning("attr conflict classify failed (non-fatal): %s", exc)
                continue
            ctype = cls.get("conflict_type", "addition")
            # v5: cardinality が set_like の attribute は追加型。update 判定でも
            # archive しない(既存の趣味・好みを新しい追加で消さない)。判断基準は
            # 「消される側=既存 fact」の attribute (group 越しで候補と異なりうる)。
            # 無ければ候補側にフォールバック。
            ex_attr = (ex.get("metadata") or {}).get("attribute") or attr
            if ctype in ("update", "correction") and is_update_cardinality(ex_attr):
                log.info(
                    "[ATTR_CONFLICT attribute=%s conflict_type=%s existing_id=%s "
                    "cand=%r]",
                    attr, ctype, eid, cand_text[:60],
                )
                out.append((eid, ctype))
                seen.add(eid)
    return out
