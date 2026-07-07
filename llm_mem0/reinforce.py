"""Hebbian-style reinforcement: when a duplicate fact is detected, bump
the existing memory's `mention_count` instead of dropping it silently.

This is the read-modify-write helper called from the ``ingest`` module when
``check_near_dup`` returns ``is_dup=True``. Mem0's ``mem.update()`` is a
*full metadata replace*, so we have to load existing metadata, increment
the relevant counters, then write it back.

Field semantics (added to fact.metadata):
    mention_count        int   number of times this fact was reinforced (incl. first ingest)
    reinforced_at        list  unix-seconds timestamps of the most recent N reinforcements
    last_reinforced_at   int   most recent unix-seconds timestamp
    importance           int   bumped by +1 on each reinforcement, capped at 5

Concurrent reinforcement (two workers hitting the same memory simultaneously)
may lose at most one increment; that's acceptable since the bot runs in a
single process and the existing batch ingest path is sequential per fact.
"""

from __future__ import annotations

import logging
import os
import time

log = logging.getLogger(__name__)

# Public so callers / tests can introspect the schema.
MAX_REINFORCED_HISTORY = 10
MAX_IMPORTANCE = 5

# Phase A-2 (mem0-recall-hybrid-hyde-kg): age decay defaults. Read once at
# import so the boost calculation stays a hot-path no-cost lookup.
_DEFAULT_ALPHA_AGE = float(os.environ.get("MEM0_BOOST_ALPHA_AGE", "0.03"))
_DEFAULT_AGE_HALF_LIFE_DAYS = float(os.environ.get("MEM0_BOOST_AGE_HALF_LIFE_DAYS", "90"))


def reinforce_existing_sync(mem, memory_id: str) -> bool:
    """Synchronous reinforcement. Designed to be wrapped with asyncio.to_thread.

    Mem0 SDK's ``mem.update(memory_id, data, metadata=...)`` *partially*
    updates metadata (fields not specified are preserved), but the ``data``
    argument is required. We re-pass the existing memory text unchanged —
    that re-embeds the fact (~1 OpenAI call, negligible cost) but keeps
    the row in place.

    Returns True if metadata was updated, False on any failure (fail-open
    means dedup still drops the candidate; we just lose the mention count
    increment for that one occurrence).
    """
    if not memory_id:
        return False
    try:
        existing = mem.get(memory_id)
    except Exception as exc:
        log.warning("reinforce: get(%s) failed: %s", memory_id, exc)
        return False
    if not isinstance(existing, dict):
        return False

    existing_meta = existing.get("metadata") or {}
    existing_text = existing.get("memory") or existing.get("text") or ""
    if not existing_text:
        return False

    now = int(time.time())
    history = list(existing_meta.get("reinforced_at") or [])
    history.append(now)
    history = history[-MAX_REINFORCED_HISTORY:]
    try:
        cur_imp = int(existing_meta.get("importance", 3))
    except (TypeError, ValueError):
        cur_imp = 3
    try:
        cur_mention = int(existing_meta.get("mention_count", 1))
    except (TypeError, ValueError):
        cur_mention = 1

    # Only specify the fields we're changing — Mem0 preserves the rest.
    partial = {
        "mention_count": cur_mention + 1,
        "reinforced_at": history,
        "last_reinforced_at": now,
        "importance": min(cur_imp + 1, MAX_IMPORTANCE),
    }

    try:
        mem.update(memory_id=memory_id, data=existing_text, metadata=partial)
        return True
    except Exception as exc:
        log.warning("reinforce: update(%s) failed: %s", memory_id, exc)
        return False


async def reinforce_existing(mem, memory_id: str) -> bool:
    """Async wrapper around the sync helper, for ingest call sites that
    already live in an event loop."""
    import asyncio
    return await asyncio.to_thread(reinforce_existing_sync, mem, memory_id)


def reinforcement_boost(metadata: dict | None, *,
                        alpha_mention: float = 0.10,
                        alpha_recency: float = 0.05,
                        recency_half_life_days: float = 30.0,
                        alpha_age: float | None = None,
                        age_half_life_days: float | None = None) -> float:
    """Return the additive score boost for a memory.

    Combines three components:
      - frequency: ``alpha_mention * log(mention_count)`` (log dampening so
        1→10 mentions has the same delta as 10→100; mention=1 yields 0)
      - recency: ``alpha_recency * exp(-days_since_last_reinforce/HL)``
        (boost decays after Hebbian reinforcement falls silent)
      - age (Phase A-2): ``alpha_age * exp(-days_since_creation/HL_age)``
        (new facts get a small bump so a state change like "髪色 = 黒 →
        ハイトーン" is preferred over the older row at retrieval time)

    Used by ``search._rerank_memories`` which subtracts the
    boost from the candidate's cosine distance, effectively bringing
    reinforced/recent facts closer in score-space.
    """
    import math

    if not metadata:
        return 0.0
    try:
        mention = int(metadata.get("mention_count", 1))
    except (TypeError, ValueError):
        mention = 1
    try:
        last_r = int(metadata.get("last_reinforced_at") or 0)
    except (TypeError, ValueError):
        last_r = 0

    boost = alpha_mention * math.log(mention) if mention > 1 else 0.0
    if last_r:
        days_since = max(0.0, (time.time() - last_r) / 86400.0)
        boost += alpha_recency * math.exp(-days_since / recency_half_life_days)

    # Phase A-2: age decay. ingest.py:87 stamps `timestamp` (unix seconds)
    # on every new fact via base_meta; older rows backfilled by
    # scripts/backfill_mem0_timestamps.py also carry it. We accept
    # created_at_unix as an alias for any future caller that writes it
    # under that name.
    aa = _DEFAULT_ALPHA_AGE if alpha_age is None else alpha_age
    ah = _DEFAULT_AGE_HALF_LIFE_DAYS if age_half_life_days is None else age_half_life_days
    if aa and ah > 0:
        try:
            created = int(metadata.get("created_at_unix") or metadata.get("timestamp") or 0)
        except (TypeError, ValueError):
            created = 0
        if created:
            days_since_creation = max(0.0, (time.time() - created) / 86400.0)
            boost += aa * math.exp(-days_since_creation / ah)
    return boost
