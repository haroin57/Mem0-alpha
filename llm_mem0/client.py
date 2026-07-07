"""Memory store singleton (mem0/Chroma) + low-level housekeeping (get_all/delete)."""

import asyncio
import logging
import os
import time

from .auth import get_auth_backend
from .settings import STATE_DIR

log = logging.getLogger(__name__)

# The Chroma adapter returns score = distance (small = similar).
# Pre-rerank gate: applied inside search_memories() to drop candidates
# that are physically too far before the LLM rerank ever sees them. Also
# used as the sanity gate for the rerank=False path in
# search_memories_smart.
MEM0_RELEVANCE_MAX_DISTANCE = float(
    os.environ.get("MEM0_RELEVANCE_MAX_DISTANCE", "1.2")
)
# DEPRECATED. Previously gated the output of search_memories_smart at 0.65 —
# but cosine distance is a poor judge after an LLM rerank has already
# accepted the fact, and 0.65 was pessimistic enough to silently kill a lot
# of conversational (non-English) queries. search_memories_smart no longer
# references this on the rerank=True path. Kept so env config that sets the
# var still imports.
MEM0_FINAL_RELEVANCE_MAX_DISTANCE = float(
    os.environ.get("MEM0_FINAL_RELEVANCE_MAX_DISTANCE", "0.65")
)
# Back-compat alias for any caller still importing the old name. Semantic is
# "max distance" (not "min similarity") — do not flip it.
MEM0_RELEVANCE_THRESHOLD = MEM0_RELEVANCE_MAX_DISTANCE

_mem0_instance = None
# Last init failure timestamp (monotonic seconds). When set, _get_mem0()
# returns None without re-trying until _INIT_RETRY_COOLDOWN_SEC has elapsed.
# This allows automatic recovery after a transient ChromaDB outage instead of
# permanently disabling the store after the first failure.
_init_failed_at: float | None = None
_INIT_RETRY_COOLDOWN_SEC = 60.0
_mem0_provider: str | None = None


def _build_config() -> dict:
    """Build mem0 configuration from env vars.

    CHROMA_MODE=server (default): connect to a Chroma HTTP server at
    CHROMA_HOST:CHROMA_PORT. This avoids the multi-process write race that
    can corrupt an embedded HNSW index under concurrent writers.

    CHROMA_MODE=embedded: direct on-disk access under STATE_DIR (single
    process only).
    """
    chroma_mode = os.environ.get("CHROMA_MODE", "server")
    chroma_host = os.environ.get("CHROMA_HOST", "127.0.0.1")
    chroma_port = int(os.environ.get("CHROMA_PORT", "8765"))
    collection_name = os.environ.get("MEM0_COLLECTION_NAME", "memories")

    if chroma_mode == "server":
        vector_cfg = {
            "collection_name": collection_name,
            "host": chroma_host,
            "port": chroma_port,
        }
    else:
        db_path = os.path.join(STATE_DIR, "mem0_db")
        os.makedirs(db_path, exist_ok=True)
        vector_cfg = {
            "collection_name": collection_name,
            "path": db_path,
        }

    backend = get_auth_backend()
    llm_model = os.environ.get("MEM0_LLM_MODEL", backend.default_model())
    llm_cfg = backend.mem0_llm_config(model=llm_model)

    config: dict = {
        "llm": llm_cfg,
        "vector_store": {
            "provider": "chroma",
            "config": vector_cfg,
        },
        "version": "v1.1",
    }

    # Embedder — defaults to OpenAI text-embedding-3-small.
    embedder_provider = os.environ.get("MEM0_EMBEDDER_PROVIDER", "openai")
    embedder_model = os.environ.get("MEM0_EMBEDDER_MODEL", "text-embedding-3-small")
    config["embedder"] = {
        "provider": embedder_provider,
        "config": {
            "model": embedder_model,
        },
    }

    return config


def _ping_chromadb_if_server_mode() -> None:
    """When CHROMA_MODE=server, verify the ChromaDB HTTP server is reachable.

    Raises an exception on failure so the caller can mark init as failed
    and trigger the cooldown-based retry path.
    """
    if os.environ.get("CHROMA_MODE", "server") != "server":
        return
    host = os.environ.get("CHROMA_HOST", "127.0.0.1")
    port = int(os.environ.get("CHROMA_PORT", "8765"))
    import requests
    r = requests.get(f"http://{host}:{port}/api/v2/heartbeat", timeout=5)
    r.raise_for_status()


def _get_mem0():
    """Lazy-init singleton mem0 Memory instance with cooldown-based retry.

    On init failure, sets _init_failed_at and returns None for the next
    _INIT_RETRY_COOLDOWN_SEC seconds before retrying, so a transient
    ChromaDB outage recovers automatically instead of disabling the store
    for the rest of the process lifetime.
    """
    global _mem0_instance, _mem0_provider, _init_failed_at

    if _mem0_instance is not None:
        return _mem0_instance
    if _init_failed_at is not None and (time.monotonic() - _init_failed_at) < _INIT_RETRY_COOLDOWN_SEC:
        return None

    try:
        _ping_chromadb_if_server_mode()
        from mem0 import Memory

        cfg = _build_config()
        _mem0_instance = Memory.from_config(cfg)
        _mem0_provider = cfg["llm"]["provider"]
        _init_failed_at = None
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
    # First init does a synchronous ChromaDB heartbeat + Memory.from_config,
    # so don't call it directly from the event loop (can stall it for
    # several seconds).
    mem = await asyncio.to_thread(_get_mem0)
    if not mem:
        return []
    try:
        kwargs: dict = {"user_id": user_id}
        if limit is not None:
            kwargs["limit"] = limit
        raw = await asyncio.to_thread(mem.get_all, **kwargs)
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
