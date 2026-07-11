"""Phase 5: semantic (vector) search over the RAW conversation transcript.

Mem0 stores curated facts; ``history_index`` does literal-keyword (FTS5) recall.
Neither lets a *paraphrased* query reach an old raw line ("あの水産会社" →
"東洋水産の決算…"). This module embeds each raw JSONL line into a SEPARATE
ChromaDB collection (``history_transcripts``) on the same server Mem0 uses, so
cosine search can. It augments — never replaces — the keyword path.

Cost note: indexing a line is one embedding call, so the whole thing is gated
behind ``MEM0_HISTORY_EMBED_ENABLED`` (default OFF). Both entry points no-op
when disabled, and every ChromaDB/embedder failure is swallowed — recall must
never break message handling.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading

from .settings import settings

log = logging.getLogger(__name__)

_COLLECTION_NAME = "history_transcripts"
_state: dict = {"col": None}
_state_lock = threading.Lock()


def _enabled() -> bool:
    return settings.MEM0_HISTORY_EMBED_ENABLED


def _get_collection():
    """Lazily get/create the raw-transcript collection (cosine, OpenAI embeds).

    Reuses the same ChromaDB server + embedding model as Mem0 so the vectors
    are comparable. Cached for the process (double-checked lock so two
    threads can't build two clients). Patched out in tests.
    """
    if _state["col"] is not None:
        return _state["col"]
    with _state_lock:
        if _state["col"] is not None:
            return _state["col"]
        import chromadb
        from chromadb.utils import embedding_functions

        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name=settings.MEM0_EMBEDDER_MODEL,
        )
        client = chromadb.HttpClient(
            host=settings.CHROMA_HOST, port=settings.CHROMA_PORT,
        )
        _state["col"] = client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return _state["col"]


def _line_id(channel_id, ts: str, role: str, text: str) -> str:
    h = hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]
    return f"{channel_id}:{ts}:{role}:{h}"


def safe_text(text: str) -> str:
    """Truncate a line to a length safely under the embedder's 8192-token cap.

    The embedding API hard-rejects oversized inputs; for a recall index the
    head of a long line carries the topic, so truncation is fine.
    """
    return (text or "")[:settings.MEM0_HISTORY_EMBED_MAX_CHARS]


def index_line(channel_id, ts: str, role: str, author: str, text: str) -> None:
    """Embed + upsert one raw transcript line. No-op when disabled/empty.

    upsert (not add) so re-indexing the same line is idempotent — the id is
    derived from channel/ts/role/text-hash.
    """
    if not _enabled():
        return
    text = (text or "").strip()
    if not text:
        return
    try:
        col = _get_collection()
        col.upsert(
            ids=[_line_id(channel_id, ts, role, text)],
            documents=[safe_text(text)],
            metadatas=[{
                "channel_id": str(channel_id),
                "ts": ts or "",
                "role": role or "",
                "author": author or "",
            }],
        )
    except Exception as exc:  # best-effort: never break the caller
        log.warning("history_embed.index_line failed (non-fatal): %s", exc)


def search_semantic(query: str, channel_id, limit: int | None = None) -> list[dict]:
    """Cosine search over this channel's raw lines. Returns mem.get-shaped dicts
    ``{memory, ts, role, author, score}`` (score = cosine distance, lower=closer).
    Empty list when disabled or on any failure."""
    if not _enabled():
        return []
    if limit is None:
        limit = settings.MEM0_HISTORY_EMBED_LIMIT
    try:
        col = _get_collection()
        res = col.query(
            query_texts=[query or ""],
            n_results=limit,
            where={"channel_id": str(channel_id)},
        )
    except Exception as exc:
        log.warning("history_embed.search_semantic failed (non-fatal): %s", exc)
        return []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    out: list[dict] = []
    for i, doc in enumerate(docs):
        if not doc:
            continue
        md = metas[i] if i < len(metas) else {}
        out.append({
            "memory": doc,
            "ts": md.get("ts", ""),
            "role": md.get("role", ""),
            "author": md.get("author", ""),
            "score": dists[i] if i < len(dists) else None,
        })
    return out
