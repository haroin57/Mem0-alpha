"""Memory ingestion: P2 extraction-gate path + legacy Mem0-standard fallback."""

import asyncio
import logging

from .client import _get_mem0
from .extract import (
    extract_facts_for_self,
    extract_full_signals,
)
from .search import search_memories

log = logging.getLogger(__name__)

# P1 (ADR-0017): process-global guard against concurrent / short-window
# duplicate inserts. The vector dedup only catches a same-text sibling once it
# is COMMITTED; two concurrent add_memories calls (live turns or backfill
# tasks) can each search-then-insert before either commits, producing exact
# dups. This seen-set, guarded by a lock, makes the insert of an identical
# (user_id, text) idempotent across the in-flight window. Bounded FIFO so it
# never grows without limit. Offline hash sweep cleans any historical dups.
import hashlib as _hashlib
from collections import OrderedDict as _OrderedDict
_INSERT_SEEN: "_OrderedDict[str, str]" = _OrderedDict()
_INSERT_SEEN_LOCK = asyncio.Lock()
_INSERT_SEEN_MAX = 8000


def _fact_key(user_id: str, text: str) -> str:
    return _hashlib.md5(f"{user_id}\x00{(text or '').strip()}".encode()).hexdigest()


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
    # ── 1. Normalize signature into (user_text, assistant_text, context_hint) ──
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
    user_text = user_text or ""
    assistant_text = assistant_text or ""

    # 初回 init の ChromaDB heartbeat (同期 requests) でループを止めない。
    mem = await asyncio.to_thread(_get_mem0)
    if not mem:
        return None

    # Late import so .env reload (via restart) is honored
    from .settings import (
        MEM0_GATE_ENABLED, MEM0_GATE_MIN_IMPORTANCE,
        MEM0_CLIENT_DEDUP_ENABLED, MEM0_CLIENT_DEDUP_THRESHOLD,
        MEM0_DEDUP_TWO_STAGE, MEM0_DEDUP_SEARCH_LIMIT, MEM0_DEDUP_DRY_RUN,
        MEM0_CONFLICT_DETECTION_ENABLED, MEM0_DEDUP_SUPERSET_MERGE,
        MEM0_RELATION_EXTRACT_ENABLED,
        MEM0_ATTRIBUTE_ENABLED, MEM0_ATTRIBUTE_CONFLICT_ENABLED,  # Phase A
        MEM0_CONDITION_ENABLED, MEM0_SCOPE_ENABLED,  # Phase B/C
        MEM0_TYPED_VALUE_ENABLED,  # v5 T7
        MEM0_FACT_STORE_ENABLED,  # v5 T8a
    )

    try:
        # The auth backend refreshes access tokens internally, so no explicit
        # token refresh is needed here before touching the store.
        import time as _time
        # Caller-supplied metadata is allowlisted before merging so it cannot
        # overwrite trust-relevant keys (gate_version, timestamp, importance,
        # category, tags). New caller-controlled keys must be added here
        # explicitly.
        # `persona` / `episode_id` / `interest_tags` let the recall path carry
        # an optional persona label and link a fact back to the episode that
        # produced it. `interest_tags` must be a CSV string because ChromaDB
        # rejects list-valued metadata (see the tags handling in _insert_one
        # below); the caller is responsible for the CSV conversion.
        _ALLOWED_CALLER_META = {
            "channel_id", "channel_name", "source",
            "persona", "episode_id", "interest_tags",
        }
        base_meta = {
            "timestamp": int(_time.time()),
            "gate_version": "v2" if MEM0_GATE_ENABLED else "v1",
        }
        if metadata:
            for k, v in metadata.items():
                if k in _ALLOWED_CALLER_META and k not in base_meta:
                    base_meta[k] = v

        # ── 2. Gate path (P2): LLM extract → filter → infer=False raw insert ──
        if MEM0_GATE_ENABLED:
            if pre_extracted_facts is not None:
                # Caller already ran the extractor and is passing facts in to
                # avoid a duplicate LLM call. We trust the shape because the
                # extractor and the caller share a module-internal contract.
                facts = pre_extracted_facts
                # Pre-extracted path doesn't carry relations (older callers
                # predate Phase C-2). Treat as empty.
                relations_payload = []
            else:
                # Phase C-2: a single LLM call extracts facts AND entity-entity
                # relations. The facts list is the same shape
                # extract_facts_for_self used to return; relations are handed
                # off to the graph layer in _insert_one below.
                _signals = await extract_full_signals(
                    user_text, assistant_text, context_hint,
                    user_id=str(user_id) if user_id else None,
                )
                facts = _signals.get("facts", [])
                relations_payload = _signals.get("relations") or []
            # Diagnostic: show every extracted fact text. Helps trace why a
            # turn produced unexpected dedup hits (candidate vs existing).
            if facts:
                _previews = [
                    f"[{f.get('speaker','?')}|imp={f.get('importance','?')}] "
                    f"{(f.get('text') or '')[:120]}"
                    for f in facts
                ]
                log.info(
                    "Mem0 gate: extracted %d facts user=...%s | %s",
                    len(facts), str(user_id)[-4:], " || ".join(_previews),
                )

            # Cheap synchronous filter pass first — drops obvious non-self /
            # low-importance facts without firing any ChromaDB queries.
            candidates: list[dict] = []
            skip_low = skip_other = skip_dup = 0
            for f in facts:
                if f.get("speaker") != "self":
                    skip_other += 1
                    continue
                if int(f.get("importance", 0)) < MEM0_GATE_MIN_IMPORTANCE:
                    skip_low += 1
                    continue
                candidates.append(f)

            # Intra-batch dedup: two facts with identical text in the same
            # call would both pass the parallel dedup-check (neither is in
            # the store yet) and both get inserted. Drop within-batch
            # duplicates by `text` before firing any ChromaDB queries.
            seen_texts: set[str] = set()
            deduped: list[dict] = []
            for f in candidates:
                t = f.get("text", "")
                if t in seen_texts:
                    skip_dup += 1
                    continue
                seen_texts.add(t)
                deduped.append(f)
            candidates = deduped

            # Parallel near-dup check with optional 2-stage LLM judgment.
            kept: list[dict] = candidates
            if MEM0_CLIENT_DEDUP_ENABLED and candidates:
                search_limit = MEM0_DEDUP_SEARCH_LIMIT
                # Phase R: scope フィルタ有効時に conversation fact が dedup
                # 候補から消えると同一会話内の兄弟と dedup できなくなるので、
                # この ingest の会話 channel_id を渡す。
                _dedup_ch = base_meta.get("channel_id")
                _dedup_ch = str(_dedup_ch) if _dedup_ch else None
                dedup_results = await asyncio.gather(
                    *(
                        search_memories(
                            f["text"], user_id, limit=search_limit,
                            current_channel_id=_dedup_ch,
                        )
                        for f in candidates
                    ),
                    return_exceptions=True,
                )

                if MEM0_DEDUP_TWO_STAGE:
                    from .dedup import check_near_dup
                    from .reinforce import reinforce_existing
                    kept = []
                    # Run 2-stage checks in parallel
                    async def _check_one(f, near):
                        if isinstance(near, Exception):
                            return (f, False, None, None, None)
                        is_dup, reason, existing_id, conflict_type, merged_text = await check_near_dup(
                            f["text"], f.get("category"), near,
                            dry_run=MEM0_DEDUP_DRY_RUN,
                            detect_conflict=MEM0_CONFLICT_DETECTION_ENABLED,
                            allow_merge=MEM0_DEDUP_SUPERSET_MERGE,
                        )
                        return (f, is_dup, existing_id, conflict_type, merged_text)
                    check_results = await asyncio.gather(
                        *(_check_one(f, near) for f, near in zip(candidates, dedup_results)),
                    )
                    reinforced_ids: list[str] = []
                    # Phase B-2 / 止揚: facts that update / correct / merge with
                    # an existing fact get their predecessor soft-archived
                    # *after* the new (or synthesized) row is inserted, so we
                    # record the (existing_id, conflict_type) pair here.
                    conflict_archive: list[tuple[str, str]] = []
                    for f, is_dup, existing_id, conflict_type, merged_text in check_results:
                        if is_dup and not MEM0_DEDUP_DRY_RUN:
                            skip_dup += 1
                            if existing_id:
                                reinforced_ids.append(existing_id)
                        else:
                            # 止揚: store the synthesized fact in place of the
                            # raw candidate so the merged detail is what lands
                            # in ChromaDB; the older row is archived below.
                            if (
                                conflict_type == "merge"
                                and merged_text
                                and not MEM0_DEDUP_DRY_RUN
                            ):
                                f = {**f, "text": merged_text}
                            kept.append(f)
                            if (
                                conflict_type in ("update", "correction", "merge")
                                and existing_id
                                and not MEM0_DEDUP_DRY_RUN
                            ):
                                conflict_archive.append((existing_id, conflict_type))
                    # Hebbian reinforcement: bump mention_count on each
                    # existing fact that absorbed a duplicate candidate.
                    # Fire-and-await in parallel; failures are non-fatal.
                    if reinforced_ids:
                        await asyncio.gather(
                            *(reinforce_existing(mem, mid) for mid in reinforced_ids),
                            return_exceptions=True,
                        )
                else:
                    # Legacy single-threshold dedup. NOTE: score is *distance*
                    # (small = similar), so we drop when score is BELOW the
                    # threshold, not above. Threshold default in settings.py is
                    # 0.85 which is way too loose under this corrected
                    # semantic; set MEM0_CLIENT_DEDUP_THRESHOLD=0.20 if you
                    # actually rely on this path.
                    kept = []
                    for f, near in zip(candidates, dedup_results):
                        if isinstance(near, Exception):
                            kept.append(f)
                            continue
                        if any(
                            (r.get("score") if r.get("score") is not None else 99) <= MEM0_CLIENT_DEDUP_THRESHOLD
                            for r in near
                        ):
                            skip_dup += 1
                        else:
                            kept.append(f)

            # Parallel raw insert: same rationale — three ChromaDB writes in
            # serial added ~3 x latency. asyncio.gather them and tolerate
            # individual failures via return_exceptions.
            async def _insert_one(f: dict):
                # P1: skip insert if an identical (user_id, text) is already
                # in-flight / recently inserted this process (defeats the
                # parallel-ingest exact-dup race). Gated by env.
                from .settings import MEM0_INGEST_HASH_DEDUP
                if MEM0_INGEST_HASH_DEDUP and not await _claim_insert(
                    user_id, f.get("text", "")
                ):
                    log.info("ingest hash-dedup: skipped in-flight dup: %r",
                             (f.get("text") or "")[:60])
                    return None
                fact_meta = dict(base_meta)
                imp = int(f.get("importance", 3))
                # importance=2 → force tier=Warm so weekly compact ages it out faster.
                # Otherwise default to tier=1 (Hot); Sonnet's extract prompt
                # doesn't emit `tier`, so f.get("tier") is normally None and
                # ChromaDB rejects None values — that previously dropped every
                # importance>=3 fact silently with
                # "Cannot convert Python object to MetadataValue".
                tier = 2 if imp <= 2 else int(f.get("tier") or 1)
                # ChromaDB metadata values must be scalar (str/int/float/bool).
                # Storing the tag list as-is raises
                # "argument 'metadatas': Cannot convert Python object to
                # MetadataValue" and the whole `mem.add` call aborts — every
                # fact in that turn is dropped on the floor. Join with a
                # delimiter ChromaDB can store; search-time consumers can
                # split back on the same separator if/when they need the list
                # shape.
                raw_tags = f.get("tags") or []
                tags_csv = ",".join(
                    str(t).replace(",", " ") for t in raw_tags if str(t).strip()
                )
                # Coerce category/subcategory to strings; None is rejected.
                fact_meta.update({
                    "category":    str(f.get("category") or "meta"),
                    "subcategory": str(f.get("subcategory") or ""),
                    "importance":  imp,
                    "tier":        tier,
                    "tags":        tags_csv,
                })
                # Phase A: attribute(統制語彙)を付与。決定的 conflict ルートの
                # キー。空文字列は付けない(ChromaDB metadata を無駄に増やさない)。
                _attr = f.get("attribute") or ""
                if MEM0_ATTRIBUTE_ENABLED and _attr:
                    fact_meta["attribute"] = _attr
                # v5 T7: typed value（quantity）。ChromaDB は dict を持てないので
                # JSON 文字列で保存。読む側は json.loads。
                _val = f.get("value")
                if MEM0_TYPED_VALUE_ENABLED and isinstance(_val, dict):
                    import json as _json_mod
                    fact_meta["value_json"] = _json_mod.dumps(
                        _val, ensure_ascii=False)
                # Phase B: condition(time:/context:)を付与。
                _cond = f.get("condition") or ""
                if MEM0_CONDITION_ENABLED and _cond:
                    fact_meta["condition"] = _cond
                # Phase C: scope。global は既定なので保存しない(metadata を節約)。
                # conversation scope には channel_id を紐付け、どの会話限定かを
                # 読み出し時に照合できるようにする。
                _scope = f.get("scope") or ""
                if MEM0_SCOPE_ENABLED and _scope and _scope != "global":
                    if _scope == "conversation":
                        _ch = base_meta.get("channel_id")
                        fact_meta["scope"] = (
                            f"conversation:{_ch}" if _ch else "conversation"
                        )
                    else:
                        fact_meta["scope"] = _scope
                # Final guard: drop any None values base_meta might still
                # contain (e.g. channel_id when called from a script).
                fact_meta = {k: v for k, v in fact_meta.items() if v is not None}
                try:
                    res = await asyncio.to_thread(
                        mem.add,
                        [{"role": "user", "content": f["text"]}],
                        user_id=user_id, metadata=fact_meta, infer=False,
                    )
                except Exception as e:
                    log.warning("Mem0 raw insert failed: %s", e)
                    return None
                # Graph layer: link entities extracted by Sonnet to this new
                # fact. Errors are non-fatal — the fact is already stored.
                try:
                    entities = f.get("entities") or []
                    if entities and isinstance(res, dict):
                        fact_ids: list[str] = []
                        for r in (res.get("results") or []):
                            fid = r.get("id") if isinstance(r, dict) else None
                            if fid:
                                fact_ids.append(fid)
                        if fact_ids:
                            from . import graph
                            for ent in entities:
                                if not isinstance(ent, dict):
                                    continue
                                name = ent.get("name")
                                etype = ent.get("type")
                                aliases = ent.get("aliases") or []
                                if not name or not etype:
                                    continue
                                eid = await asyncio.to_thread(
                                    graph.upsert_entity,
                                    name, etype, str(user_id), aliases=aliases,
                                )
                                if eid:
                                    for fid in fact_ids:
                                        await asyncio.to_thread(graph.add_edge, fid, eid)
                except Exception as e:
                    log.warning("graph update failed (non-fatal): %s", e)
                # Phase A-1: index the new fact in the BM25 FTS5 store so
                # hybrid retrieval can rescue exact-term matches that cosine
                # missed. Errors are non-fatal — the fact is already stored.
                try:
                    if isinstance(res, dict):
                        from . import bm25_index
                        for r in (res.get("results") or []):
                            fid = r.get("id") if isinstance(r, dict) else None
                            if fid:
                                await asyncio.to_thread(
                                    bm25_index.index_fact,
                                    fid, f["text"], str(user_id),
                                )
                except Exception as e:
                    log.warning("bm25 index update failed (non-fatal): %s", e)
                # v5 T8a: structured fact layer への導出インデックス登録
                # （attribute 付き fact のみ）。正本は Chroma、こちらは best-effort。
                try:
                    if (
                        MEM0_FACT_STORE_ENABLED
                        and _attr
                        and isinstance(res, dict)
                    ):
                        from . import fact_store
                        for r in (res.get("results") or []):
                            fid = r.get("id") if isinstance(r, dict) else None
                            if fid:
                                await asyncio.to_thread(
                                    fact_store.record_fact,
                                    fid, str(user_id), _attr,
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
                return res

            # Phase A-3: 決定的 conflict ルート。dedup の embedding 近傍ゲート
            # (距離≤0.35)をすり抜けた属性更新(文面の離れた数値更新等)を、
            # attribute 完全一致で捕捉する。dedup が既に archive 対象にした
            # existing_id は二重に積まない。
            if MEM0_ATTRIBUTE_CONFLICT_ENABLED and kept:
                _already = (
                    {eid for eid, _ in conflict_archive}
                    if 'conflict_archive' in locals() else set()
                )
                _attr_pairs = await _attribute_conflict_check(
                    mem, user_id, kept, exclude_ids=_already,
                )
                if _attr_pairs:
                    if 'conflict_archive' not in locals():
                        conflict_archive = []
                    conflict_archive.extend(_attr_pairs)

            insert_results = await asyncio.gather(*(_insert_one(f) for f in kept)) if kept else []
            results = [r for r in insert_results if r is not None]

            # Phase B-2: soft-archive outdated predecessors after the new
            # fact is in storage. Errors are non-fatal — the new fact is
            # already stored, the old one just won't be hidden.
            archived = 0
            if 'conflict_archive' in locals() and conflict_archive:
                archived = await _archive_outdated_facts(mem, conflict_archive)

            # Phase C-2: ingest entity-entity relations into the graph
            # layer. Resolves src/dst canonical names via graph.find_entity
            # (which knows about aliases & appellations) — only inserts a
            # relation when both endpoints already exist as entities. We
            # don't auto-create endpoints from relations to avoid letting
            # a hallucinated edge spawn ghost entities.
            relations_added = 0
            relations_to_add = (
                relations_payload if 'relations_payload' in locals() else []
            )
            if (
                MEM0_RELATION_EXTRACT_ENABLED
                and relations_to_add
            ):
                try:
                    from . import graph
                    for rel in relations_to_add:
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
                            relations_added += 1
                    if relations_added:
                        log.info(
                            "[REL_ADDED count=%d user=...%s]",
                            relations_added, str(user_id)[-4:],
                        )
                except Exception as exc:
                    log.warning("relation ingest failed (non-fatal): %s", exc)

            log.info(
                "Mem0 gate: facts=%d kept=%d skip(low=%d,other=%d,dup=%d) "
                "archived=%d user=...%s",
                len(facts), len(kept), skip_low, skip_other, skip_dup,
                archived, str(user_id)[-4:],
            )
            # Phase B-3: structured one-line metric for grep-based weekly
            # health reports. Each key is positional so the parser stays
            # cheap.
            log.info(
                "[MEM0_INGEST_METRICS extracted=%d candidates=%d kept=%d "
                "drop_low=%d drop_other=%d drop_dup=%d archived=%d]",
                len(facts), len(candidates), len(kept),
                skip_low, skip_other, skip_dup, archived,
            )
            return {"results": results, "kept": len(kept), "skipped": {
                "low_importance": skip_low, "other_speaker": skip_other, "near_dup": skip_dup,
            }}

        # ── 3. Legacy path: Mem0 standard fact extraction (fallback) ──
        # context_hint is INTENTIONALLY excluded from legacy_msgs even when
        # MEM0_LEGACY_CONTEXT is on. The Phase 1 root cause (other-speaker
        # contamination) was the system-message channel-history path; we keep
        # it sealed here regardless of which gate is enabled. Also filter the
        # caller-supplied `messages` to user/assistant roles only so a future
        # caller that passes a raw list with role=system can't re-introduce
        # the contamination vector.
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
        # Haiku call on top of Mem0's own internal extractor. The gate path
        # (above) is the canonical source of taxonomy metadata; in legacy
        # fallback we accept the simpler Mem0 default and skip the redundant
        # classifier so this path costs 1 LLM call, not 2.
        result = await asyncio.to_thread(
            mem.add, legacy_msgs, user_id=user_id, metadata=base_meta
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
            "skipped": {"low_importance": 0, "other_speaker": 0, "near_dup": 0},
        }


async def _archive_outdated_facts(mem, archive: list[tuple[str, str]]) -> int:
    """Soft-archive existing facts that were superseded by a new fact.

    Writes ``valid_to=<now>`` and ``conflict_archived_as=<update|correction>``
    to the existing fact's metadata. Retrieval honours valid_to to hide the
    row from candidates (see search.py). The fact is never
    physically deleted, so flipping ``MEM0_HIDE_ARCHIVED=false`` brings
    them back without any data restore.

    Returns the count of successfully archived facts. Failures are
    logged at WARNING and skipped — the new fact stays in storage either
    way; archival is best-effort because we already have the new truth.
    """
    if not archive:
        return 0
    import time as _time
    now = int(_time.time())
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
                from .settings import MEM0_FACT_STORE_ENABLED as _fs_on
                if _fs_on:
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
    try:
        from .settings import MEM0_FACT_STORE_ENABLED as _fs_on
    except Exception:
        _fs_on = False
    if _fs_on:
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
    import time as _time
    try:
        raw = await asyncio.to_thread(
            mem.get_all, user_id=user_id,
            filters={"attribute": attribute}, limit=50,
        )
    except TypeError:
        # SDK が filters を受け付けない → 全件 + Python フィルタ
        try:
            raw = await asyncio.to_thread(mem.get_all, user_id=user_id, limit=1000)
        except Exception as exc:
            log.warning("attr conflict get_all fallback failed: %s", exc)
            return []
    except Exception as exc:
        log.warning("attr conflict get_all failed: %s", exc)
        return []
    rows = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
    now = int(_time.time())
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
        import json as _json_mod
        cand_val = cand.get("value")
        if not isinstance(cand_val, dict) or cand_val.get("type") != "quantity":
            return
        raw = (ex.get("metadata") or {}).get("value_json")
        if not raw:
            return
        ex_val = _json_mod.loads(raw)
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
            # 夜の habit)なら矛盾ではないので archive しない。same/subset/superset/
            # unknown は重なりうるので conflict 判定へ進める(gpt-5.5-pro 指摘4)。
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
            # archive しない(既存の趣味・好みを新しい追加で消さない。gpt-5.5-pro 指摘3)。
            # 判断基準は「消される側=既存 fact」の attribute (group 越しで候補と
            # 異なりうる)。無ければ候補側にフォールバック。
            _ex_attr = (ex.get("metadata") or {}).get("attribute") or attr
            if ctype in ("update", "correction") and is_update_cardinality(_ex_attr):
                log.info(
                    "[ATTR_CONFLICT attribute=%s conflict_type=%s existing_id=%s "
                    "cand=%r]",
                    attr, ctype, eid, cand_text[:60],
                )
                out.append((eid, ctype))
                seen.add(eid)
    return out
