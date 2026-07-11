"""Session-scoped priming buffer — Phase ④.

Spreading activation (search._graph_expand_candidates) propagates through
the STORED graph structure built from past facts. Human context-dependent
recall also gets a boost from what was JUST talked about in the current
conversation, independent of any pre-existing graph edge — encoding
specificity: a retrieval cue is more effective when it matches the context
present at recent encoding or recent retrieval, not only when it matches a
statistical co-occurrence pattern accumulated over months.

This module is a tiny in-memory, per-(user_id, channel_id) buffer of
recently-touched entity ids with a short TTL. Both ingest (entities just
extracted from a new fact) and search (entities linked to facts just
surfaced) call ``touch`` so either side of a turn can prime the other within
the same conversation.

Deliberately NOT persisted to disk: priming models working-memory-scale
context (minutes), not long-term storage — a process restart clearing it is
the correct behavior, not a bug. Process-local also means priming does not
generalize across a multi-process deployment; that's an accepted scope
limit for a single-process host.

Knobs (dynamic): ``MEM0_PRIMING_TTL_SEC`` (default 30min),
``MEM0_PRIMING_MAX_ENTITIES`` (per-key cap, default 20).
"""

from __future__ import annotations

import threading
import time

from .settings import settings

_lock = threading.Lock()
# (user_id, channel_id) -> {entity_id: last_touched_monotonic_seconds}
_buffer: dict[tuple[str, str], dict[int, float]] = {}


def _key(user_id: str, channel_id) -> tuple[str, str]:
    return (str(user_id), str(channel_id) if channel_id is not None else "")


def touch(user_id: str, channel_id, entity_ids) -> None:
    """Record that these entities were just active in this (user, channel)
    session. No-op for a falsy/empty ``entity_ids``."""
    if not user_id or not entity_ids:
        return
    key = _key(user_id, channel_id)
    now = time.monotonic()
    max_entities = settings.MEM0_PRIMING_MAX_ENTITIES
    with _lock:
        bucket = _buffer.setdefault(key, {})
        for eid in entity_ids:
            if eid:
                bucket[eid] = now
        overflow = len(bucket) - max_entities
        if overflow > 0:
            stalest = sorted(bucket.items(), key=lambda kv: kv[1])[:overflow]
            for eid, _ in stalest:
                bucket.pop(eid, None)


def get_primed_entities(user_id: str, channel_id) -> set[int]:
    """Entity ids touched within the TTL window for this (user, channel).
    Empty set for an unknown key or an expired/empty buffer."""
    if not user_id:
        return set()
    key = _key(user_id, channel_id)
    now = time.monotonic()
    ttl = settings.MEM0_PRIMING_TTL_SEC
    with _lock:
        bucket = _buffer.get(key)
        if not bucket:
            return set()
        live = {eid: t for eid, t in bucket.items() if now - t < ttl}
        if len(live) != len(bucket):
            _buffer[key] = live  # opportunistic cleanup of expired entries
        return set(live)


def clear(user_id: str | None = None, channel_id=None) -> None:
    """Test/ops hook: clear one (user, channel) key, or the whole buffer
    when ``user_id`` is omitted."""
    with _lock:
        if user_id is None:
            _buffer.clear()
            return
        _buffer.pop(_key(user_id, channel_id), None)
