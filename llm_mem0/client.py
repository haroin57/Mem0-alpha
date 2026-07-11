"""Memory store singleton (mem0/Chroma) + low-level housekeeping (get_all/delete)."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time

from .auth import get_auth_backend
from .settings import settings

log = logging.getLogger(__name__)

# Back-compat module constants. The live values are settings.* (dynamic);
# these freeze at import for callers that import them directly, exactly as
# they always did.
#
# The Chroma adapter returns score = distance (small = similar).
# MEM0_RELEVANCE_MAX_DISTANCE is the pre-rerank gate applied inside
# search_memories() to drop candidates that are physically too far before
# the LLM rerank ever sees them; also the sanity gate for the rerank=False
# path in search_memories_smart.
MEM0_RELEVANCE_MAX_DISTANCE = settings.MEM0_RELEVANCE_MAX_DISTANCE
# DEPRECATED — see settings.py. Kept so env config that sets it still imports.
MEM0_FINAL_RELEVANCE_MAX_DISTANCE = settings.MEM0_FINAL_RELEVANCE_MAX_DISTANCE
# Back-compat alias for any caller still importing the old name. Semantic is
# "max distance" (not "min similarity") — do not flip it.
MEM0_RELEVANCE_THRESHOLD = MEM0_RELEVANCE_MAX_DISTANCE

_mem0_instance = None
_mem0_provider: str | None = None
# AuthBackend.provider_id() snapshot at build time — drift triggers rebuild.
_mem0_provider_id: str | None = None
# Last init failure timestamp (monotonic seconds). When set, _get_mem0()
# returns None without re-trying until _INIT_RETRY_COOLDOWN_SEC has elapsed.
# This allows automatic recovery after a transient ChromaDB outage instead of
# permanently disabling the store after the first failure.
_init_failed_at: float | None = None
_INIT_RETRY_COOLDOWN_SEC = 60.0
# _get_mem0 is called via asyncio.to_thread from several coroutines at once;
# without a lock two threads could both run the (multi-second) init.
_init_lock = threading.Lock()


def _build_config() -> dict:
    """Build mem0 configuration from settings.

    CHROMA_MODE=server (default): connect to a Chroma HTTP server at
    CHROMA_HOST:CHROMA_PORT. This avoids the multi-process write race that
    can corrupt an embedded HNSW index under concurrent writers.

    CHROMA_MODE=embedded: direct on-disk access under STATE_DIR (single
    process only).
    """
    if settings.CHROMA_MODE == "server":
        vector_cfg = {
            "collection_name": settings.MEM0_COLLECTION_NAME,
            "host": settings.CHROMA_HOST,
            "port": settings.CHROMA_PORT,
        }
    else:
        db_path = os.path.join(settings.STATE_DIR, "mem0_db")
        os.makedirs(db_path, exist_ok=True)
        vector_cfg = {
            "collection_name": settings.MEM0_COLLECTION_NAME,
            "path": db_path,
        }

    backend = get_auth_backend()
    llm_model = settings.MEM0_LLM_MODEL or backend.default_model()

    return {
        "llm": backend.mem0_llm_config(model=llm_model),
        "vector_store": {
            "provider": "chroma",
            "config": vector_cfg,
        },
        "embedder": {
            "provider": settings.MEM0_EMBEDDER_PROVIDER,
            "config": {
                "model": settings.MEM0_EMBEDDER_MODEL,
            },
        },
        "version": "v1.1",
    }


def _ping_chromadb_if_server_mode() -> None:
    """When CHROMA_MODE=server, verify the ChromaDB HTTP server is reachable.

    Raises an exception on failure so the caller can mark init as failed
    and trigger the cooldown-based retry path.
    """
    if settings.CHROMA_MODE != "server":
        return
    import requests
    r = requests.get(
        f"http://{settings.CHROMA_HOST}:{settings.CHROMA_PORT}/api/v2/heartbeat",
        timeout=5,
    )
    r.raise_for_status()


def _sync_backend_state(mem) -> None:
    """Give the auth backend a chance to sync rotated credentials onto the
    memory store's internal LLM client. Best-effort — a hook failure must
    never take down store access."""
    try:
        get_auth_backend().refresh_memory_llm(mem)
    except Exception as exc:
        log.warning("auth backend refresh_memory_llm failed (non-fatal): %s", exc)


def _provider_drifted() -> bool:
    """True when the auth backend's effective provider changed since the
    cached instance was built (e.g. OAuth session lost → API-key fallback,
    or restored). Best-effort: an erroring hook never forces a rebuild."""
    try:
        return get_auth_backend().provider_id() != _mem0_provider_id
    except Exception:
        return False


def _get_mem0():
    """Lazy-init singleton mem0 Memory instance with cooldown-based retry.

    On init failure, sets _init_failed_at and returns None for the next
    _INIT_RETRY_COOLDOWN_SEC seconds before retrying, so a transient
    ChromaDB outage recovers automatically instead of disabling the store
    for the rest of the process lifetime.

    Provider drift (``AuthBackend.provider_id`` changing between calls)
    rebuilds the instance so the store's internal LLM follows a runtime
    provider switch; ``AuthBackend.refresh_memory_llm`` runs on every
    hand-out so rotated OAuth tokens propagate to the cached instance.

    NOTE: first init does a synchronous ChromaDB heartbeat +
    Memory.from_config; call via ``asyncio.to_thread`` from event loops.
    """
    global _mem0_instance, _mem0_provider, _mem0_provider_id, _init_failed_at

    if _mem0_instance is not None and not _provider_drifted():
        _sync_backend_state(_mem0_instance)
        return _mem0_instance

    with _init_lock:
        if _mem0_instance is not None:
            if not _provider_drifted():
                _sync_backend_state(_mem0_instance)
                return _mem0_instance
            log.info(
                "mem0 provider drift detected (%s -> %s) — rebuilding instance",
                _mem0_provider_id, get_auth_backend().provider_id(),
            )
            _mem0_instance = None
        if (
            _init_failed_at is not None
            and (time.monotonic() - _init_failed_at) < _INIT_RETRY_COOLDOWN_SEC
        ):
            return None
        try:
            _ping_chromadb_if_server_mode()
            from mem0 import Memory

            cfg = _build_config()
            _mem0_instance = Memory.from_config(cfg)
            _mem0_provider = cfg["llm"]["provider"]
            try:
                _mem0_provider_id = get_auth_backend().provider_id()
            except Exception:
                _mem0_provider_id = None
            _init_failed_at = None
            _sync_backend_state(_mem0_instance)
            vs_cfg = cfg["vector_store"]["config"]
            store_id = vs_cfg.get("path") or f'{vs_cfg.get("host")}:{vs_cfg.get("port")}'
            log.info(
                "mem0 ready — llm=%s provider=%s embedder=%s store=chroma(%s)",
                cfg["llm"]["config"]["model"],
                cfg["llm"]["provider"],
                cfg["embedder"]["config"]["model"],
                store_id,
            )
        except Exception as exc:
            log.error("mem0 init failed: %s", exc, exc_info=True)
            _mem0_instance = None
            _init_failed_at = time.monotonic()

    return _mem0_instance


# ---------------------------------------------------------------------------
# mem0ai API-generation compat
# ---------------------------------------------------------------------------
# mem0ai changed its retrieval API around 1.1: `user_id` moved from a
# top-level kwarg into ``filters={"user_id": ...}`` for search()/get_all()
# (add() still accepts the kwarg). We support both generations without
# pinning: try the legacy call first, and when the SDK rejects it switch
# styles for the rest of the process.
_filters_style: bool | None = None  # None=unknown, False=legacy kwarg, True=filters dict


def _rejects_user_id_kwarg(exc: Exception) -> bool:
    return "Top-level entity parameters" in str(exc)


def mem_search(mem, *, query: str, user_id: str, limit: int):
    """``mem.search`` across mem0ai API generations. Sync — wrap in to_thread."""
    global _filters_style
    if _filters_style:
        return mem.search(query=query, filters={"user_id": user_id}, limit=limit)
    try:
        result = mem.search(query=query, user_id=user_id, limit=limit)
        _filters_style = False
        return result
    except Exception as exc:
        if not _rejects_user_id_kwarg(exc):
            raise
        log.info("mem0 SDK uses filters-style params; switching")
        _filters_style = True
        return mem.search(query=query, filters={"user_id": user_id}, limit=limit)


def mem_get_all(mem, *, user_id: str, limit: int | None = None):
    """``mem.get_all`` across mem0ai API generations. Sync — wrap in to_thread."""
    global _filters_style
    kwargs: dict = {}
    if limit is not None:
        kwargs["limit"] = limit
    if _filters_style:
        return mem.get_all(filters={"user_id": user_id}, **kwargs)
    try:
        result = mem.get_all(user_id=user_id, **kwargs)
        _filters_style = False
        return result
    except Exception as exc:
        if not _rejects_user_id_kwarg(exc):
            raise
        log.info("mem0 SDK uses filters-style params; switching")
        _filters_style = True
        return mem.get_all(filters={"user_id": user_id}, **kwargs)


# ---------------------------------------------------------------------------
# Low-level housekeeping (direct mem0 access — used by housekeeping/compaction
# scripts and by search.py's graph-expansion path)
# ---------------------------------------------------------------------------

async def get_all_memories(user_id: str, limit: int | None = None) -> list[dict]:
    """Retrieve memories for a user.

    `limit` is forwarded to mem0 / ChromaDB so the underlying query can stop
    early instead of fetching the entire collection and slicing in Python.
    Pass None (default) to fetch all (fine for occasional housekeeping;
    avoid on hot paths).
    """
    mem = await asyncio.to_thread(_get_mem0)
    if not mem:
        return []
    try:
        raw = await asyncio.to_thread(mem_get_all, mem, user_id=user_id, limit=limit)
        return raw.get("results", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    except Exception as exc:
        log.warning("mem0 get_all error: %s", exc)
        return []


async def delete_memory(memory_id: str) -> bool:
    """Delete a specific memory by ID."""
    mem = await asyncio.to_thread(_get_mem0)
    if not mem:
        return False
    try:
        await asyncio.to_thread(mem.delete, memory_id)
        return True
    except Exception as exc:
        log.warning("mem0 delete error: %s", exc)
        return False
